# WrappedArrays

*Arrays built on other storage objects*

[![Build Status](https://github.com/emmt/WrappedArrays.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/emmt/WrappedArrays.jl/actions/workflows/CI.yml?query=branch%3Amain) [![Build Status](https://ci.appveyor.com/api/projects/status/github/emmt/WrappedArrays.jl?svg=true)](https://ci.appveyor.com/project/emmt/WrappedArrays-jl) [![Coverage](https://codecov.io/gh/emmt/WrappedArrays.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/emmt/WrappedArrays.jl)

`WrappedArrays` is a [Julia](https://julialang.org/) package to build dense arrays (i.e.
with contiguous elements) whose elements are stored in another object. This generalizes
the principle of [`StaticArrays`](https://github.com/JuliaArrays/StaticArrays.jl) which
wrap small arrays over tuples of values.

A wrapped array is typically built by:

``` julia
A = WrappedArray{T}(obj, inds...; offset=0)
```

which yields a dense array `A` using object `obj` for backing the storage of the elements
of `A`. Parameter `T` is the type of the elements of `A`. Arguments `inds...` specify the
shape of `A`, each of `inds...` is a dimension length or an index range. The shape of `A`
may also be specified as a tuple. Keyword `offset` is the number of bytes between the
first element of `A` and the base address of `obj` or the address of the first element of
`obj` if it is an array.

The constructor may also be called as `WrappedArray{T,N}(...)` with `N` the number of
dimensions which is usually omitted as it can be inferred from the given array shape.

`WrappedVector{T}` and `WrappedMatrix{T}` are aliases for `WrappedArray{T,N}` with `N`
equal to `1` and `2` respectively.

A wrapped vector with as much elements of type `T` as can be stored by `obj` (minus
`offset` bytes if this keyword is specified) can be created with:

``` julia
A = WrappedVector{T}(obj, :)
```

Example:

``` julia
mutable struct NamedData{T,L}
    name::Symbol
    data::NTuple{L,T}
    serial::UInt
end
data_eltype(B::NamedData{T,L}) where {T,L} = T
data_length(B::NamedData{T,L}) where {T,L} = L
B = NamedData(:someName, (1.0, 0.0, 11.7, -2.4), UInt(0x796));
A = WrappedArray{data_eltype(B)}(B, data_length(B), offset=fieldoffset(typeof(B), 2));
```

builds a wrapped array `A` over the entries of the `data` member of object `B`. These
entries are then accessible for reading, with the syntax `A[i]`, or writing, , with the
syntax `A[i] = x`, and with the same speed and safety (i.e., bound checking) as for
regular arrays. Array `A` may also be passed to `ccall` or `@ccall` for calling a
C-function in a shared library.
