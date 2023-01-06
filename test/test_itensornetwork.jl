using Dictionaries
using GraphsFlows
using ITensors
using ITensorNetworks
using NamedGraphs
using Random
using Test

@testset "ITensorNetwork tests" begin
  @testset "ITensorNetwork Basics" begin
    Random.seed!(1234)
    g = named_grid((4,))
    s = siteinds("S=1/2", g)

    @test s isa IndsNetwork
    @test nv(s) == 4
    @test ne(s) == 3
    @test neighbors(s, (2,)) == [(1,), (3,)]

    tn = ITensorNetwork(s; link_space=2)

    @test nv(tn) == 4
    @test ne(tn) == 3
    @test tn isa ITensorNetwork
    @test neighbors(tn, (2,)) == [(1,), (3,)]

    # TODO: How to support this syntax?
    @test_broken tn[1] isa ITensor
    @test tn[(1,)] isa ITensor

    # TODO: How to support this syntax?
    @test_broken order(tn[1]) == 2
    @test order(tn[(1,)]) == 2

    # TODO: How to support this syntax?
    @test_broken tn[2] isa ITensor
    @test tn[(2,)] isa ITensor

    # TODO: How to support this syntax?
    @test_broken order(tn[2]) == 3
    @test order(tn[(2,)]) == 3

    # XXX: Slicing syntax is no long supported, use `induced_subgraph`.
    @test_broken tn[1:2] isa ITensorNetwork
    # TODO: Support this syntax, maybe rename `subgraph`.
    @test_broken induced_subgraph(tn, [(1,), (2,)]) isa ITensorNetwork

    randn!.(vertex_data(tn))
    tn′ = sim(dag(tn); sites=[])

    @test tn′ isa ITensorNetwork
    inner_tn = tn ⊗ tn′
    @test inner_tn isa ITensorNetwork
    sequence = contraction_sequence(inner_tn)
    @test sequence isa Vector
    inner_res = contract(inner_tn; sequence)[]
    @test inner_res isa Float64
  end

  @testset "Contract edge (regression test for issue #5)" begin
    dims = (2, 2)
    g = named_grid(dims)
    s = siteinds("S=1/2", g)
    ψ = ITensorNetwork(s, v -> "↑")
    tn = inner_network(ψ, ψ)
    tn_2 = contract(tn, ((1, 2), 2) => ((1, 2), 1))
    @test !has_vertex(tn_2, ((1, 2), 2))
    @test tn_2[((1, 2), 1)] ≈ tn[((1, 2), 2)] * tn[((1, 2), 1)]
  end

  @testset "Remove edge (regression test for issue #5)" begin
    dims = (2, 2)
    g = named_grid(dims)
    s = siteinds("S=1/2", g)
    ψ = ITensorNetwork(s, v -> "↑")
    rem_vertex!(ψ, (1, 2))
    tn = inner_network(ψ, ψ)
    @test !has_vertex(tn, ((1, 2), 1))
    @test !has_vertex(tn, ((1, 2), 2))
    @test has_vertex(tn, ((1, 1), 1))
    @test has_vertex(tn, ((1, 1), 2))
    @test has_vertex(tn, ((2, 1), 1))
    @test has_vertex(tn, ((2, 1), 2))
    @test has_vertex(tn, ((2, 2), 1))
    @test has_vertex(tn, ((2, 2), 2))
  end

  @testset "Custom element type" for eltype in (Float32, Float64, ComplexF32, ComplexF64),
    link_space in (nothing, 3),
    g in (
      grid((4,)),
      named_grid((3, 3)),
      siteinds("S=1/2", grid((4,))),
      siteinds("S=1/2", named_grid((3, 3))),
    )
    ψ = ITensorNetwork(g; link_space) do v, inds...
      return itensor(randn(eltype, dims(inds)...), inds...)
    end
    @test Base.eltype(ψ[first(vertices(ψ))]) == eltype
    ψ = ITensorNetwork(g; link_space) do v, inds...
      return itensor(randn(dims(inds)...), inds...)
    end
    @test Base.eltype(ψ[first(vertices(ψ))]) == Float64
    ψ = randomITensorNetwork(eltype, g; link_space)
    @test Base.eltype(ψ[first(vertices(ψ))]) == eltype
    ψ = randomITensorNetwork(g; link_space)
    @test Base.eltype(ψ[first(vertices(ψ))]) == Float64
    ψ = ITensorNetwork(eltype, undef, g; link_space)
    @test Base.eltype(ψ[first(vertices(ψ))]) == eltype
    ψ = ITensorNetwork(undef, g)
    @test Base.eltype(ψ[first(vertices(ψ))]) == Float64
  end

  @testset "orthogonalize" begin
    tn = randomITensorNetwork(named_grid(4); link_space=2)
    Z = contract(inner_network(tn, tn))[]

    tn_ortho = factorize(tn, 4 => 3)

    # TODO: Error here in arranging the edges. Arrange by hash?
    Z̃ = contract(inner_network(tn_ortho, tn_ortho))[]
    @test nv(tn_ortho) == 5
    @test nv(tn) == 4
    @test Z ≈ Z̃

    tn_ortho = orthogonalize(tn, 4 => 3)
    Z̃ = contract(inner_network(tn_ortho, tn_ortho))[]
    @test nv(tn_ortho) == 4
    @test nv(tn) == 4
    @test Z ≈ Z̃

    tn_ortho = orthogonalize(tn, 1)
    Z̃ = contract(inner_network(tn_ortho, tn_ortho))[]
    @test Z ≈ Z̃
    Z̃ = contract(inner_network(tn_ortho, tn))[]
    @test Z ≈ Z̃
  end

  @testset "dijkstra_shortest_paths" begin
    tn = ITensorNetwork(named_grid(4); link_space=2)
    paths = dijkstra_shortest_paths(tn, [1])
    @test paths.dists == Dictionary([0, 1, 2, 3])
    @test paths.parents == Dictionary([1, 1, 2, 3])
    @test paths.pathcounts == Dictionary([1.0, 1.0, 1.0, 1.0])
  end

  @testset "mincut" begin
    tn = ITensorNetwork(named_grid(4); link_space=3);
    w = weights(tn)
    @test w isa Dictionary{Tuple{Int,Int},Float64}
    @test length(w) ≈ ne(tn)
    @test w[(1, 2)] ≈ log2(3)
    @test w[(2, 3)] ≈ log2(3)
    @test w[(3, 4)] ≈ log2(3)

    p1, p2, wc = GraphsFlows.mincut(tn, 2, 3)
    @test issetequal(p1, [1, 2])
    @test issetequal(p2, [3, 4])
    @test isone(wc)

    p1, p2, wc = GraphsFlows.mincut(tn, 2, 3, w)
    @test issetequal(p1, [1, 2])
    @test issetequal(p2, [3, 4])
    @test wc ≈ log2(3)
  end
end
