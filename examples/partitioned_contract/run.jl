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
# include("random_circuit.jl")
include("bench.jl")

Random.seed!(1234)
TimerOutputs.enable_debug_timings(ITensorNetworks)
TimerOutputs.enable_debug_timings(@__MODULE__)
ITensors.set_warn_order(100)

#=
Exact contraction of ising_network
=#
# N = (28, 28)
# beta = 0.44
# distribution = Uniform{Float64}(-0.5, 1.0)
# network = randomITensorNetwork(distribution, named_grid(N); link_space=2)
# # network = ising_network(named_grid(N), beta; h=0.0, szverts=nothing)
# exact_contract(network; sc_target=28)

#=
Exact contraction of randomITensorNetwork
=#
# N = (5, 5, 5)
# distribution = Uniform{Float64}(-0.4, 1.0)
# network = randomITensorNetwork(named_grid(N); link_space=2, distribution=distribution)
# exact_contract(network; sc_target=30) # 47.239753244708396

#=
Bugs
=#
# TODO: (6, 6, 6), env_size=(2, 1, 1) is buggy (cutoff=1e-12, maxdim=256, ansatz="comb", algorithm="density_matrix",)
# TODO below is buggy
# @time bench_3d_cube_lnZ(
#   (3, 8, 10);
#   block_size=(1, 1, 1),
#   beta=0.3,
#   h=0.0,
#   num_iter=2,
#   cutoff=1e-20,
#   maxdim=128,
#   ansatz="mps",
#   algorithm="density_matrix",
#   use_cache=true,
#   ortho=false,
#   env_size=(3, 1, 1),
# )

#=
bench_2d_cube_lnZ
=#
# N = (28, 28)
# beta = 0.44
# maxdim = 64
# tn = ising_network(named_grid(N), beta; h=0.0, szverts=nothing)
# # distribution = Uniform{Float64}(-0.5, 1.0)
# # tn = randomITensorNetwork(distribution, named_grid(N); link_space=2)
# tntree = build_tntree(N, tn; block_size=(1, 1), env_size=(28, 1))
# @time bench_lnZ(
#   tntree;
#   num_iter=1,
#   cutoff=1e-15,
#   maxdim=maxdim,
#   ansatz="mps",
#   approx_itensornetwork_alg="density_matrix",
#   swap_size=10000,
#   contraction_sequence_alg="sa_bipartite",
#   contraction_sequence_kwargs=(;),
#   linear_ordering_alg="bottom_up",
# )

# ltn = sweep_contractor_tensor_network(tn, (i, j) -> (i, j))
# @time lnz = contract_w_sweep(ltn; rank=maxdim)
# @info "lnZ of SweepContractor is", lnz

#=
bench_3d_cube_lnZ
=#
# N = (5, 5, 5)
# beta = 0.3
# maxdim = 64
# ansatz= "mps"
# tn = ising_network(named_grid(N), beta; h=0.0, szverts=nothing)
# # distribution = Uniform{Float64}(-0.4, 1.0)
# # tn = randomITensorNetwork(distribution, named_grid(N); link_space=2)
# tntree = build_tntree(N, tn; block_size=(1, 1, 1), env_size=(5, 2, 1))
# @time bench_lnZ(
#   tntree;
#   num_iter=1,
#   cutoff=1e-14,
#   maxdim=maxdim,
#   ansatz=ansatz,
#   approx_itensornetwork_alg="density_matrix",
#   swap_size=10000,
#   contraction_sequence_alg="sa_bipartite",
#   contraction_sequence_kwargs=(;),
#   linear_ordering_alg="bottom_up",
# )

# ltn = sweep_contractor_tensor_network(tn, (i, j, k) -> (i * N[1] + j, k + 0.1 * randn()))
# @time lnz = contract_w_sweep(ltn; rank=maxdim)
# @info "lnZ of SweepContractor is", lnz

#=
bench_3d_cube_magnetization
=#
# N = (6, 6, 6)
# beta = 0.22
# h = 0.050
# szverts = [(3, 3, 3)]
# network1 = ising_network(named_grid(N), beta; h=h, szverts=szverts)
# network2 = ising_network(named_grid(N), beta; h=h, szverts=nothing)
# # e1 = exact_contract(network1; sc_target=28)
# # e2 = exact_contract(network2; sc_target=28)
# # @info exp(e1[2] - e2[2])
# tntree1 = build_tntree(N, network1; block_size=(1, 1, 1), env_size=(3, 1, 1))
# tntree2 = build_tntree(N, network2; block_size=(1, 1, 1), env_size=(3, 1, 1))
# @time bench_magnetization(
#   tntree1 => tntree2;
#   num_iter=1,
#   cutoff=1e-12,
#   maxdim=256,
#   ansatz="mps",
#   algorithm="density_matrix",
#   use_cache=true,
#   ortho=false,
#   swap_size=10000,
#   warmup=false,
# )

#=
SweepContractor
=#
# reset_timer!(ITensors.timer)
# # N = (5, 5, 5)
# # beta = 0.3
# # network = ising_network(named_grid(N), beta; h=0.0, szverts=nothing)
# N = (3, 3, 3)
# distribution = Uniform{Float64}(-0.4, 1.0)
# network = randomITensorNetwork(named_grid(N); link_space=2, distribution=distribution)
# ltn = sweep_contractor_tensor_network(
#   network, (i, j, k) -> (j + 0.01 * randn(), k + 0.01 * randn())
# )
# @time lnz = contract_w_sweep(ltn; rank=256)
# @info "lnZ of SweepContractor is", lnz
# show(ITensors.timer)

#=
random regular graph
=#
nvertices = 220
deg = 3
beta = 0.65
maxdim = 128
# swap_size = 4
num_iter = 1
# distribution = Uniform{Float64}(-0.2, 1.0)
# network = randomITensorNetwork(
#   distribution, random_regular_graph(nvertices, deg); link_space=2
# )
tn = ising_network(
  NamedGraph(random_regular_graph(nvertices, deg)), beta; h=0.0, szverts=nothing
)
# exact_contract(tn; sc_target=30)
# -26.228887728408008 (-1.0, 1.0)
# 5.633462619348083 (-0.2, 1.0)
# nvertices_per_partition = 5 # works 15/20 not work
# tntree = partitioned_contraction_sequence(network; nvertices_per_partition=10)

output_dict = Dict()
for nvertices_per_partition in [10]
  for swap_size in [8]
    out = @time bench_lnZ(
      tn;
      num_iter=num_iter,
      nvertices_per_partition=nvertices_per_partition,
      backend="Metis",
      cutoff=1e-12,
      maxdim=maxdim,
      ansatz="mps",
      approx_itensornetwork_alg="density_matrix",
      swap_size=swap_size,
      contraction_sequence_alg="sa_bipartite",
      contraction_sequence_kwargs=(;),
      linear_ordering_alg="bottom_up",
    )
    output_dict[("par:$(nvertices_per_partition)", "swap:$(swap_size)")] = out
    @info output_dict
  end
end

# tensors = collect(Leaves(_recursive_bisection(tn, Vector{ITensor}(tn))))
# ltn = sweep_contractor_tensor_network(ITensorNetwork(tensors), i -> (0.0001 * randn(), i))
# @time lnz = contract_w_sweep(ltn; rank=maxdim)
# @info "lnZ of SweepContractor is", lnz

#=
Simulation of random quantum circuit
=#
# N = (6, 6)
# depth = 6
# sequence = random_circuit_line_partition_sequence(N, depth)
# @time bench_lnZ(
#   sequence;
#   num_iter=1,
#   cutoff=1e-12,
#   maxdim=256,
#   ansatz="mps",
#   approx_itensornetwork_alg="density_matrix",
#   swap_size=8,
#   contraction_sequence_alg="sa_bipartite",
#   contraction_sequence_kwargs=(;),
#   linear_ordering_alg="bottom_up",
# )
