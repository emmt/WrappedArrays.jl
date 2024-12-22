using WrappedArrays
using Test

mutable struct MutableStaticBuffer{T,N}
    data::NTuple{N,T}
    MutableStaticBuffer{T,N}() where {T,N} = new{T,N}()
end

Base.getindex(A::MutableStaticBuffer, i::Int) = A.data[i]
Base.firstindex(A::MutableStaticBuffer) = 1
Base.lastindex(A::MutableStaticBuffer) = length(A)
Base.length(A::MutableStaticBuffer{T,N}) where {T,N} = N
Base.size(A::MutableStaticBuffer) = (length(A),)

const ArrayLike{T} = Union{AbstractArray{T},Tuple{Vararg{T}},MutableStaticBuffer{T}}

function have_same_values(x::ArrayLike{Tx}, y::ArrayLike{Ty}) where {Tx,Ty}
    (length(x) == length(y)) || return false
    flag = true
    j = firstindex(y)
    for i in firstindex(x):lastindex(x)
        flag &= (x[i] == y[j])
        j += 1
    end
    return flag
end

function have_identical_values(x::ArrayLike{Tx}, y::ArrayLike{Ty}) where {Tx,Ty}
    ((Tx === Ty) && (length(x) == length(y))) || return false
    flag = true
    j = firstindex(y)
    for i in firstindex(x):lastindex(x)
        flag &= (x[i] === y[j])
        j += 1
    end
    return flag
end

to_axis(x::AbstractUnitRange{Int}) = x
to_axis(x::AbstractUnitRange{<:Integer}) = convert(AbstractUnitRange{Int}, x)
to_axis(x::Integer) = Base.OneTo{Int}(x)

@testset "WrappedArrays.jl" begin
    @testset "Utilities" begin
        @test WrappedArrays.to_int(2) === 2
        @test WrappedArrays.to_int(Int16(2)) === 2
        @test WrappedArrays.to_dim(3) === 3
        @test WrappedArrays.to_dim(Int16(7)) === 7
        @test WrappedArrays.to_dim(0:8) === 9
        @test WrappedArrays.to_dim(-Int16(1):Int16(4)) === 6
        @test WrappedArrays.to_dim(Base.OneTo(9)) === 9
        @test_throws Exception WrappedArrays.to_dim(1:2:11)
        @test WrappedArrays.to_axis(3) === Base.OneTo(3)
        @test WrappedArrays.to_axis(Int16(7)) === Base.OneTo(7)
        @test WrappedArrays.to_axis(0:8) === 0:8
        @test WrappedArrays.to_axis(-Int16(1):Int16(4)) === -1:4
        @test WrappedArrays.to_axis(Base.OneTo(9)) === Base.OneTo(9)
        @test_throws Exception WrappedArrays.to_axis(1:2:11)
        @test WrappedArrays.to_axes((5,2,3)) === (Base.OneTo(5), Base.OneTo(2), Base.OneTo(3))
        @test WrappedArrays.to_axes((Int8(11),Int8(8))) === (Base.OneTo(11), Base.OneTo(8))
        @test WrappedArrays.to_axes(()) === ()
        @test WrappedArrays.to_axes((-1:5, 2:4, 0:6)) === (-1:5, 2:4, 0:6)
        @test WrappedArrays.to_axes((Int16(-1):Int16(6),)) === (-1:6,)
        @test WrappedArrays.to_size((5,2,3)) === (5,2,3)
        @test WrappedArrays.to_size((Int8(11),Int8(8))) === (11,8)
        @test WrappedArrays.to_size(()) === ()
        @test WrappedArrays.to_size((-1:5, 2:4, 0:6)) === (7,3,7)
        @test WrappedArrays.to_size((Int16(-1):Int16(6),)) === (8,)
        @test WrappedArrays.to_shape(()) === ()
        @test WrappedArrays.to_shape((2,)) === (2,)
        @test WrappedArrays.to_shape((0,7)) === (0,7,)
        @test WrappedArrays.to_shape((Int8(2),Base.OneTo(5))) === (2,5)
        @test WrappedArrays.to_shape((8,Base.OneTo{Int16}(3))) === (8,3)
        @test WrappedArrays.to_shape((1:8,Base.OneTo{Int16}(3),4)) === (1:8,Base.OneTo(3),Base.OneTo(4))
        @test WrappedArrays.to_shape((0:4,-1:5,Base.OneTo(3))) === (0:4,-1:5,Base.OneTo(3))
    end

    # WrappedArrays.
    T = Float32
    dims = (2, 3, 4)
    len = prod(dims)
    @testset "Wrapped array (storage: $(typeof(B)))" for B in (MutableStaticBuffer{T,len}(),
                                                               Array{T}(undef,dims))
        # Wrapped vector with 1-based indices.
        A = @inferred WrappedVector{T}(B,:)
        @test eltype(A) === T
        @test length(A) === len
        @test size(A) === (length(A),)
        @test axes(A) === map(Base.OneTo, size(A))
        flag = true
        for i in 1:length(A)
            A[i] = i
            flag &= (B[i] == i)
        end
        @test flag
        @test have_identical_values(A, B)
        @test pointer(A) === Base.unsafe_convert(Ptr{T}, A)
        @test firstindex(A) === 1
        @test lastindex(A) === length(A)
        @test WrappedArrays.linear_indices(A) == firstindex(A):lastindex(A)
        @test Base.has_offset_axes(A) === false

        # Wrapped vector with offset axis.
        @test_throws ArgumentError WrappedVector{T}(B, -1:len) # too many elements
        rng = -1:len-2
        X = @inferred WrappedVector{T}(B, rng)
        @test eltype(X) === T
        @test length(X) === length(rng)
        @test size(X) === (length(X),)
        @test axes(X) === (rng,)
        flag = true
        for (i,j) in enumerate(eachindex(X))
            flag &= (X[j] == i)
        end
        @test flag
        @test have_identical_values(X, B)
        @test pointer(X) === Base.unsafe_convert(Ptr{T}, X)
        @test WrappedArrays.linear_indices(X) == firstindex(X):lastindex(X)
        @test firstindex(X) === first(rng)
        @test lastindex(X) === last(rng)
        @test WrappedArrays.linear_indices(X) == firstindex(X):lastindex(X)
        @test Base.has_offset_axes(X) === true

        # Multi-dimensional wrapped array.
        C = @inferred WrappedArray{T}(B, dims)
        @test eltype(C) === T
        @test size(C) === dims
        @test axes(C) === map(Base.OneTo, size(C))
        for d in 1:ndims(C)+1
            @test size(C,d) === (d ≤ ndims(C) ? size(C)[d] : 1)
            @test axes(C,d) === (d ≤ ndims(C) ? axes(C)[d] : Base.OneTo(1))
        end
        @test have_identical_values(C, B)
        @test have_identical_values(C, A)
        @test pointer(C) === Base.unsafe_convert(Ptr{T}, C)
        @test firstindex(C) === 1
        @test lastindex(C) === length(C)
        @test WrappedArrays.linear_indices(C) == firstindex(C):lastindex(C)
        @test Base.has_offset_axes(C) === false
        val = @inferred convert(T, 759)
        I = (1,2,3)
        i = LinearIndices(C)[I...]
        C[I...] = val
        @test A[i] === val
        @test have_identical_values(C, A)
        @test C === @inferred WrappedArray{eltype(C)}(A, dims...)
        @test C === @inferred WrappedArray{eltype(C)}(B, dims...)
        @test C === @inferred WrappedArray{eltype(C)}(C, dims...)
        @test C === @inferred WrappedArray{eltype(C),ndims(C)}(B, dims)
        @test C === @inferred WrappedArray{eltype(C),ndims(C)}(B, dims...)
        shape = (Base.OneTo(dims[1]), Int16(dims[2]), dims[3:end]...,)
        @test C === @inferred WrappedArray{eltype(C)}(B, shape)
        @test C === @inferred WrappedArray{eltype(C)}(B, shape...)
        @test C === @inferred WrappedArray{eltype(C),ndims(C)}(B, shape)
        @test C === @inferred WrappedArray{eltype(C),ndims(C)}(B, shape...)
        shape = (0:dims[1]-1, Int16(-1):Int16(dims[2]-2), map(Int16, dims[3:end])...,)
        D = @inferred WrappedArray{T}(B, shape...)
        @test eltype(D) === T
        @test size(D) == dims
        @test length(D) === prod(size(D),)
        @test axes(D) == map(to_axis, shape)
        @test firstindex(D) === 1
        @test lastindex(D) === length(D)
        @test WrappedArrays.linear_indices(D) == firstindex(D):lastindex(D)
        @test Base.has_offset_axes(D) === true
        @test D[1] === A[1]
        D[3] = 42
        @test A[3] == 42
    end
end
