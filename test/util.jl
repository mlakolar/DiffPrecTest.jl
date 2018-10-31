module UtilTest

using Test
using DiffPrecTest

@testset "kron_sub" begin
    A = randn(10, 5)
    B = randn(7, 9)

    k = kron(A, B)

    I = [1]
    out = zeros(length(I), length(I))
    @test DiffPrecTest.kron_sub!(out, A, B, I) == k[I, I]

    I = [1, 2, 3]
    out = zeros(length(I), length(I))
    @test DiffPrecTest.kron_sub!(out, A, B, I) == k[I, I]
end

end
