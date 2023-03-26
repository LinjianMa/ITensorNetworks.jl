using ITensors, Graphs, NamedGraphs, TimerOutputs
using Distributions, Random, DataStructures
using KaHyPar
using ITensors: contract
using ITensorNetworks
using ITensorNetworks: delta_network, _delta_inds_disjointsets
using Random
using StatsBase

include("exact_contract.jl")
include("3d_cube.jl")
include("sweep_contractor.jl")
include("randregulargraph.jl")
include("bench.jl")

Random.seed!(1234)
TimerOutputs.enable_debug_timings(ITensorNetworks)
TimerOutputs.enable_debug_timings(@__MODULE__)
ITensors.set_warn_order(100)

function random_corner_double_network(N::Tuple)
  tn = ITensorNetwork(named_grid((N..., 2)); link_space=2)
  ranges = [1:n for n in N]
  coords = vec(collect(Iterators.product(ranges...)))
  subgraph_vertices = [[(coord..., 1), (coord..., 2)] for coord in coords]
  tn_partition = partition(tn, subgraph_vertices)
  tn_partition = rename_vertices(v -> coords[v], tn_partition)
  delta_tn = ITensorNetwork()
  out_tn = ITensorNetwork()
  for v in vertices(tn_partition)
    inds = noncommoninds(Vector{ITensor}(tn_partition[v])...)
    inds = Random.shuffle(inds)
    inds_partition = collect(Iterators.partition(inds, 2))
    ts = []
    for (i, inds) in enumerate(inds_partition)
      add_vertex!(delta_tn, (v, i))
      delta_tn[(v, i)] = delta(inds...)
      push!(ts, delta_tn[(v, i)])
    end
    add_vertex!(out_tn, v)
    out_tn[v] = contract(ts...)
  end
  @info out_tn
  return delta_tn, out_tn
end

N = (8, 8, 1)
delta_tn, tn = random_corner_double_network(N)
ds = _delta_inds_disjointsets(Vector{ITensor}(delta_tn), Vector{Index}())
deltainds = [ds...]
roots = [find_root!(ds, i) for i in deltainds]

@info countmap(roots)
@info length(countmap(roots))
@info length(countmap(roots)) * log(2)

tntree = build_tntree(N, tn; block_size=(1, 1, 1), snake=false, env_size=(5, 1, 1))
@time bench_lnZ(
  tntree;
  num_iter=1,
  cutoff=1e-12,
  maxdim=64,
  ansatz="mps",
  algorithm="density_matrix",
  use_cache=true,
  ortho=false,
  swap_size=10000,
)

@info countmap(roots)
@info length(countmap(roots))
@info length(countmap(roots)) * log(2)
