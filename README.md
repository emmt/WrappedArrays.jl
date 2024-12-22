# WrappedArrays [![Build Status](https://github.com/emmt/WrappedArrays.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/emmt/WrappedArrays.jl/actions/workflows/CI.yml?query=branch%3Amain) [![Build Status](https://ci.appveyor.com/api/projects/status/github/emmt/WrappedArrays.jl?svg=true)](https://ci.appveyor.com/project/emmt/WrappedArrays-jl) [![Coverage](https://codecov.io/gh/emmt/WrappedArrays.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/emmt/WrappedArrays.jl)

Package `WrappedArrays` builds dense arrays (i.e. with contiguous elements) using a
mutable object for backing the storage of their elements.

Example:

``` julia
A = WrappedArray{T}(obj, inds...; offset=0)
```

builds a dense array `A` using object `obj` for backing the storage of the elements of
`A`. Parameter `T` is the type of the elements of `A`. Arguments `inds...` specify the
shape of `A`, each of `inds...` is a dimension length or an index range. The shape of `A`
may also be specified as a tuple.

Keyword `offset` is to specify the offset (in bytes) of the first element of `A` relative
to the base address of `obj` or to the address of its first element if `obj` is an array.

The constructor may also be called as `WrappedArray{T,N}(...)` with `N` the number of
dimensions which is usually omitted as it can be inferred from the given array shape.

`WrappedVector{T}` and `WrappedMatrix{T}` are aliases for `WrappedArray{T,N}` with `N`
equal to `1` and `2` respectively.

A wrapped vector with as much elements of type `T` as can be stored by `obj` (minus
`offset` bytes if this keyword is specified) can be created with:

``` julia
A = WrappedVector{T}(obj, :)
```
