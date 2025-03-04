using ITensors
using ITensorNetworks
using Test
using Random

@testset "Linsolve" begin
  @testset "Linsolve Basics" begin
    cutoff = 1E-11
    maxdim = 8
    nsweeps = 2

    N = 8
    s = siteinds("S=1/2", N; conserve_qns=true)

    os = OpSum()
    for j in 1:(N - 1)
      os += 0.5, "S+", j, "S-", j + 1
      os += 0.5, "S-", j, "S+", j + 1
      os += "Sz", j, "Sz", j + 1
    end
    H = MPO(os, s)

    state = [isodd(n) ? "Up" : "Dn" for n in 1:N]

    ## Correct x is x_c
    #x_c = randomMPS(s, state; linkdims=4)
    ## Compute b
    #b = apply(H, x_c; cutoff)

    #x0 = randomMPS(s, state; linkdims=10)
    #x = linsolve(H, b, x0; cutoff, maxdim, nsweeps, ishermitian=true, solver_tol=1E-6)

    #@show norm(x - x_c)
    #@test norm(x - x_c) < 1E-4

    #
    # Test complex case
    #
    Random.seed!(1234)
    x_c = randomMPS(s, state; linkdims=4) + 0.1im * randomMPS(s, state; linkdims=2)
    b = apply(H, x_c; cutoff)

    x0 = randomMPS(s, state; linkdims=10)
    x = linsolve(H, b, x0; cutoff, maxdim, nsweeps, ishermitian=true, solver_tol=1E-6)

    @show norm(x - x_c)
    @test norm(x - x_c) < 1E-3
  end
end

nothing
