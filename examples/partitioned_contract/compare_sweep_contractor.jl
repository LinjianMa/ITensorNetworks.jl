using ITensors, Graphs, NamedGraphs, TimerOutputs
using Distributions, Random
using KaHyPar, Metis
using ITensorNetworks
using ITensorNetworks: ising_network, contract, _recursive_bisection
using OMEinsumContractionOrders
using AbstractTrees

include("exact_contract.jl")
include("3d_cube.jl")
include("sweep_contractor.jl")
include("bench.jl")

Random.seed!(1234)
TimerOutputs.enable_debug_timings(ITensorNetworks)
TimerOutputs.enable_debug_timings(@__MODULE__)
ITensors.set_warn_order(100)

# 2D grid 
# N = (8, 8)
# beta = 0.44
# maxdim = 256
# distribution = Uniform{Float64}(-1.0, 1.0)
# tn = randomITensorNetwork(distribution, named_grid(N); link_space=16)
# # tn = ising_network(named_grid(N), beta; h=0.0, szverts=nothing)
# tntree = build_tntree(N, tn; block_size=(1, 1), env_size=(1, 1))
# @time bench_lnZ(
#   tntree;
#   num_iter=1,
#   cutoff=1e-15,
#   maxdim=maxdim,
#   ansatz="mps",
#   approx_itensornetwork_alg="density_matrix",
#   swap_size=1,
#   contraction_sequence_alg="sa_bipartite",
#   contraction_sequence_kwargs=(;),
#   linear_ordering_alg="bottom_up",
# )

# ltn = sweep_contractor_tensor_network(tn, (i, j) -> (i, j))
# @time lnz = contract_w_sweep(ltn; rank=maxdim)
# @info "lnZ of SweepContractor is", lnz

# 3D grid, not work well
N = (5, 5, 5)
beta = 0.44
maxdim = 512
distribution = Uniform{Float64}(-1.0, 1.0)
tn = randomITensorNetwork(distribution, named_grid(N); link_space=4)
# tn = ising_network(named_grid(N), beta; h=0.0, szverts=nothing)
tntree = build_tntree(N, tn; block_size=(1, 1, 1), env_size=(1, 1, 1))
@time bench_lnZ(
  tntree;
  num_iter=1,
  cutoff=1e-15,
  maxdim=maxdim,
  ansatz="mps",
  approx_itensornetwork_alg="density_matrix",
  swap_size=1,
  contraction_sequence_alg="sa_bipartite",
  contraction_sequence_kwargs=(;),
  linear_ordering_alg="bottom_up",
)

ltn = sweep_contractor_tensor_network(tn, (i, j, k) -> (i * N[1] + j, k + 0.1 * randn()))
@time lnz = contract_w_sweep(ltn; rank=maxdim)
@info "lnZ of SweepContractor is", lnz

# random regular graph

# function linear_path(tn::Vector)
#   if length(tn) <= 2
#     return tn
#   end
#   return [linear_path(tn[1:(end - 1)]), [tn[end]]]
# end

# nvertices = 100
# deg = 3
# maxdim = 256
# distribution = Uniform{Float64}(-1.0, 1.0)
# tn = randomITensorNetwork(distribution, random_regular_graph(nvertices, deg); link_space=8)
# tensors = collect(Leaves(_recursive_bisection(tn, Vector{ITensor}(tn))))

# @time bench_lnZ(
#   linear_path(tensors);
#   num_iter=1,
#   cutoff=1e-15,
#   maxdim=maxdim,
#   ansatz="mps",
#   approx_itensornetwork_alg="density_matrix",
#   swap_size=1,
#   contraction_sequence_alg="sa_bipartite",
#   contraction_sequence_kwargs=(;),
#   linear_ordering_alg="bottom_up",
# )

# ltn = sweep_contractor_tensor_network(ITensorNetwork(tensors), i -> (0.0001 * randn(), i))
# @time lnz = contract_w_sweep(ltn; rank=maxdim)
# @info "lnZ of SweepContractor is", lnz
