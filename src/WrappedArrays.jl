module WrappedArrays

export WrappedArray, WrappedMatrix, WrappedVector

using Base: OneTo

const ArrayDimOrAxis = Union{Integer,AbstractUnitRange{<:Integer}}
const ArrayAxis = AbstractUnitRange{Int}
const ArrayAxes{N} = NTuple{N,ArrayAxis}
const ArrayShape{N} = Union{Dims{N},ArrayAxes{N}}

# Singleton type to indicate that arguments have been checked.
struct UnsafeBuild end

struct WrappedArray{T,N,S<:ArrayShape{N},B} <: DenseArray{T,N}
    # The `pointer` member is such that `unsafe_load(A.pointer,firstindex(A))` loads the
    # first element of a wrapped array `A`. The `parent` member is the object that must be
    # preserved from being garbage collected to warrant that `pointer` is valid. The
    # `shape` member is the array shape: a `N`-tuple of dimensions or a `N`-tuple of index
    # ranges.
    pointer::Ptr{T}
    parent::B
    shape::S

    # The inner constructor is meant to be fast, it does not check its arguments and is
    # thus unsafe. It is the caller's responsibility to carefully check the arguments.
    function WrappedArray{T}(::UnsafeBuild, ptr::Ptr, obj::B,
                             shape::S) where {T,N,S<:ArrayShape{N},B}
        return new{T,N,S,B}(ptr, obj, shape)
    end
end

const WrappedVector{T} = WrappedArray{T,1}
const WrappedMatrix{T} = WrappedArray{T,2}

"""
    A = WrappedArray{T,N}(obj, inds...; offset=0)

builds a dense `N`-dimensional array `A` using object `obj` for backing the storage of the
elements of `A`. Parameter `T` is the type of the elements of `A`. Arguments `inds...`
specify the shape of `A`, each of `inds...` is a dimension length or an index range. The
shape of `A` may also be specified as a `N`-tuple. Parameter `N` may be omitted as it can
be inferred from the shape of `A`.

Keyword `offset` is to specify the offset (in bytes) of the first element of `A` relative
to the base address of `obj` or to the address of its first element if `obj` is an array.

"""
function WrappedArray{T}(obj, shape::ArrayShape{N}; offset::Integer = 0) where {T,N}
    # NOTE: `sizeof(T)` throws an error if `T` is not a concrete type so there there are
    #       no needs to check that.
    obj, ptr, nbytes = storage_parameters(obj)
    off = checked_offset(offset, nbytes)
    maxlen = div(nbytes - off, sizeof(T))
    len = 1
    @inbounds for d in 1:N
        dim = to_dim(shape[d])
        dim ≥ 0 || throw(ArgumentError("dimension(s) must be nonnegative"))
        len *= dim
    end
    len ≤ maxlen || throw(ArgumentError("too many elements"))
    ptr += off + (1 - first_linear_index_from_shape(shape))*sizeof(T)
    return WrappedArray{T}(UnsafeBuild(), ptr, obj, shape)
end

# Convert variable number of arguments in a tuple.
WrappedArray{T}(obj, inds::ArrayDimOrAxis...; kwds...) where {T} =
    WrappedArray{T}(obj, inds; kwds...)
WrappedArray{T,N}(obj, inds::ArrayDimOrAxis...; kwds...) where {T,N} =
    WrappedArray{T}(obj, inds; kwds...)

# Get rid of parameter `N` if consistent.
WrappedArray{T,N}(obj, inds::NTuple{N,ArrayDimOrAxis}; kwds...) where {T,N} =
    WrappedArray{T}(obj, inds; kwds...)

# Convert array shape to one of the standard forms.
WrappedArray{T}(obj, inds::Tuple{ArrayDimOrAxis,Vararg{ArrayDimOrAxis}}; kwds...) where {T} =
    WrappedArray{T}(obj, to_shape(inds); kwds...)

"""
    A = WrappedVector{T}(obj, : ; offset=0)

builds a dense vector `A` using object `obj` for backing the storage of its elements.
Parameter `T` is the type of the elements of `A`. The result is a 1-based vector whose
length is the maximal number of elements that can be stored in the memory occupied by `B`
minus `offset` bytes.

"""
function WrappedVector{T}(obj, ::Colon; offset::Integer = 0) where {T}
    obj, ptr, nbytes = storage_parameters(obj)
    off = checked_offset(offset, nbytes)
    maxlen = div(nbytes - off, sizeof(T))
    shape = (maxlen,)
    ptr += off
    return WrappedArray{T}(UnsafeBuild(), ptr, obj, shape)
end

# `storage_parameters(A)` yields the object to warrant the validity of the storage, the
# storage base address, and the available storage size for object `A`. For an array, the
# storage base address is the address of the first element of the array whatever its index.
storage_parameters(A::DenseArray) = A, pointer(A), length(A)*sizeof(eltype(A))
storage_parameters(A::WrappedArray) = parent(A), pointer(A), length(A)*sizeof(eltype(A))
storage_parameters(A::Any) =
    # This default version is for mutable, non-array objects. If `A` is not mutable
    # an exception will be thrown by `pointer_from_objref`.
    A, pointer_from_objref(A), sizeof(A)

# Yield offset converted to an `Int`, throwing an exception if the specified offset is
# negative or greater than the storage size.
@inline function checked_offset(off::Integer, nbytes::Int)
    off = to_int(off)
    off ≥ 0 || throw(ArgumentError("offset must be nonnegative"))
    off ≤ nbytes || throw(ArgumentError("offset must not exceed storage object size"))
    return off
end

# FIXME: Also extend `reinterpret`, `view`, etc.

# Accessors.
Base.parent(A::WrappedArray) = getfield(A, :parent)
offset_pointer(A::WrappedArray) = getfield(A, :pointer)
shape(A::WrappedArray) = getfield(A, :shape)
shape(A::WrappedArray{T,N}, d::Integer) where {T,N} =
    1 ≤ d ≤ N ? shape(A)[d] :
    d > N ? extra_shape(A) : error("array dimension out of range")

extra_shape(A::WrappedArray{T,N,Dims{N}}) where {T,N} = 1
extra_shape(A::WrappedArray) = OneTo(1)

# In base Julia `abstractarray.jl` and `pointer.jl`, the `Base.pointer` method is defined
# after the `Base.unsafe_convert(Ptr{T},A)` one. For wrapped arrays, we do it in the other
# way because implementing `pointer(A,i)` is the easiest way to account for the possible
# offset between the memorized address and the address of the first element.
Base.unsafe_convert(::Type{Ptr{T}}, A::WrappedArray{T}) where {T} = pointer(A)

# NOTE: `pointer(A) and `pointer(A,firstindex(A))` yields the same address for regular and
# offset arrays, that is the address of the first element of `A`.
#
# Override the 2 `Base.pointer` methods defined in base Julia `abstractarray.jl` for
# wrapped arrays. `pointer(A,i)` yields the address of the element at index `i`, of the
# first element if `i` is not specified.
Base.pointer(A::WrappedArray, i::Integer = first_linear_index(A)) =
    offset_pointer(A) + (to_int(i) - 1)*elsize(A)

# Implement abstract array API for buffers.
Base.eachindex(::IndexLinear, A::WrappedArray) = linear_indices(A)
Base.firstindex(A::WrappedArray) = first_linear_index(A)
Base.lastindex(A::WrappedArray) = last_linear_index(A)
Base.length(A::WrappedArray) = prod(size(A))
Base.size(A::WrappedArray) = to_size(shape(A))
Base.size(A::WrappedArray, d::Integer) = to_dim(shape(A, d))
Base.axes(A::WrappedArray) = to_axes(shape(A))
Base.axes(A::WrappedArray, d::Integer) = to_axis(shape(A, d))
Base.IndexStyle(::Type{<:WrappedArray}) = IndexLinear()
Base.has_offset_axes(A::WrappedArray{T,N,<:Dims{N},B}) where {T,N,B} = false

# NOTE: We assume that it is sufficient to preserve the wrapped object itself even though
#       it is immutable, to also preserve the storage object. If this would not be the
#       case, `LinearAlgebra.Adjoint` would not work properly.
@inline function Base.getindex(A::WrappedArray, i::Int)
    @boundscheck checkbounds(A, i)
    return GC.@preserve A unsafe_load(offset_pointer(A), i)
end

@inline function Base.setindex!(A::WrappedArray, x, i::Int)
    @boundscheck checkbounds(A, i)
    GC.@preserve A unsafe_store!(offset_pointer(A), x, i)
    return A
end

to_int(x::Int) = x
to_int(x::Integer) = convert(Int, x)::Int

to_dim(x::Integer) = to_int(x)
to_dim(x::AbstractUnitRange{<:Integer}) = length(x)::Int

to_axis(dim::Integer) = OneTo{Int}(dim)
to_axis(rng::AbstractUnitRange{Int}) = rng
to_axis(rng::OneTo{<:Integer}) = to_axis(length(rng))
to_axis(rng::AbstractUnitRange{<:Integer}) =
    convert(AbstractUnitRange{Int}, rng)::AbstractUnitRange{Int}

to_axes(x::ArrayAxes) = x
to_axes(x::Tuple{}) = x
to_axes(x::Tuple) = map(to_axis, x)

to_size(x::Dims) = x
to_size(x::Tuple{}) = x
to_size(x::Tuple) = map(to_dim, x)

to_shape(x::ArrayShape) = x
to_shape(x::Tuple{}) = x
to_shape(x::Tuple{ArrayDimOrAxis,Vararg{ArrayDimOrAxis}}) = map(to_axis, x)
to_shape(x::Tuple{Union{Integer,OneTo{<:Integer}},Vararg{Union{Integer,OneTo{<:Integer}}}}) =
    map(to_dim, x)

# For any Julia array `A`, the linear indices of `A` are 1 to `length(A)` if `A` is
# multi-dimensional, otherwise, if `A` is an uni-dimensional array (a.k.a. vector), its
# first linear index is given by: `first(axes(A,1))`.
first_linear_index(A::AbstractArray) = 1
first_linear_index(A::AbstractVector) = first(linear_indices(A))

last_linear_index(A::AbstractArray) = length(A)
last_linear_index(A::AbstractVector) = last(linear_indices(A))

# Yields the range of linear indices of an array.
linear_indices(A::AbstractArray) = OneTo(length(A))
linear_indices(A::AbstractVector) = Base.axes1(A)

first_linear_index_from_shape(shape::ArrayShape) = 1
first_linear_index_from_shape(shape::ArrayAxes{1}) = first(shape[1])

elsize(A::AbstractArray) = elsize(typeof(A))
elsize(::Type{T}) where {T<:AbstractArray} = sizeof(eltype(T)) # FIXME: should be aligned_sizeof

end # module
