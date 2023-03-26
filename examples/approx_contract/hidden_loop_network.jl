using ITensors, Graphs, NamedGraphs, TimerOutputs
using Distributions, Random, DataStructures
using KaHyPar
using ITensorNetworks
using ITensorNetworks: delta_network, _delta_inds_disjointsets

include("exact_contract.jl")
include("3d_cube.jl")
include("sweep_contractor.jl")
include("randregulargraph.jl")
include("bench.jl")

Random.seed!(1234)
TimerOutputs.enable_debug_timings(ITensorNetworks)
TimerOutputs.enable_debug_timings(@__MODULE__)
ITensors.set_warn_order(100)

nv = 1000
deg = 2
network = delta_network(IndsNetwork(random_regular_graph(nv, deg)); link_space=2)
@info network
ds = _delta_inds_disjointsets(Vector{ITensor}(network), Vector{Index}())
deltainds = [ds...]
for i in deltainds
  @info i, find_root!(ds, i)
end

# @info ds
@info DataStructures.num_groups(ds)
@info length(ds)
@info length(Set([find_root!(ds, i) for i in deltainds]))
