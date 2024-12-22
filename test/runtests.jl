using WrappedArrays
using Test

mutable struct MutableStaticBuffer{N,T}
    data::NTuple{N,T}
    MutableStaticBuffer{N,T}() where {N,T} = new{N,T}()
end

to_axis(x::AbstractUnitRange{Int}) = x
to_axis(x::AbstractUnitRange{<:Integer}) = convert(AbstractUnitRange{Int}, x)
to_axis(x::Integer) = Base.OneTo{Int}(x)

@testset "WrappedArrays.jl" begin
    # Utilities.
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

    # WrappedArrays.
    T = Float32
    dims = (2, 3, 4)
    len = prod(dims)
    B = MutableStaticBuffer{len,T}()
    A = @inferred WrappedVector{T}(B,:)
    @test eltype(A) === T
    @test length(A) === len
    @test size(A) === (length(A),)
    @test axes(A) === map(Base.OneTo, size(A))
    flag = true
    for i in 1:length(A)
        A[i] = i
        flag &= (B.data[i] == i)
    end
    @test flag
    flag = true
    for i in 1:length(A)
        flag &= (A[i] === B.data[i])
    end
    @test flag
    C = @inferred WrappedArray{T}(B, dims)
    @test eltype(C) === T
    @test size(C) === dims
    @test axes(C) === map(Base.OneTo, size(C))
    flag = true
    for (i, j) in enumerate(eachindex(C))
        flag &= (C[j] == i)
    end
    @test flag
    val = @inferred convert(T, 759)
    I = (1,2,3)
    i = LinearIndices(C)[I...]
    C[I...] = val
    @test A[i] === val
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
    @test D[1] === A[1]
    D[3] = 42
    @test A[3] == 42
end
