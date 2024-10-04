using ITensors, Graphs, NamedGraphs, TimerOutputs
using Distributions, Random
using KaHyPar, Metis
using ITensorNetworks
using ITensorNetworks: ising_network, contract, _recursive_bisection
using OMEinsumContractionOrders
using AbstractTrees

include("utils.jl")
include("exact_contract.jl")
include("3d_cube.jl")
include("sweep_contractor.jl")
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
# network = ising_network(named_grid(N), beta; h=0.0, szverts=nothing)  # 718.5111735461454
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
# use_linear_ordering = false
# accurate_lnZ = 718.5111735461454
# cutoff = 1e-18

# tn = ising_network(named_grid(N), beta; h=0.0, szverts=nothing)
# # distribution = Uniform{Float64}(-0.5, 1.0)
# # tn = randomITensorNetwork(distribution, named_grid(N); link_space=2)
# tntree = build_tntree(N, tn; block_size=(1, 1), env_size=(28, 1))
# out = @time bench_lnZ(
#   tntree,
#   accurate_lnZ;
#   num_iter=1,
#   cutoff=cutoff,
#   maxdim=maxdim,
#   ansatz="mps",
#   approx_itensornetwork_alg="density_matrix",
#   swap_size=10000,
#   contraction_sequence_alg="sa_bipartite",
#   contraction_sequence_kwargs=(;),
#   linear_ordering_alg="bottom_up",
#   use_linear_ordering=use_linear_ordering,
# )
# output_dict[("model=$(model)", "ansatz: $(ansatz)", "env:$(env)", "maxdim:$(maxdim)", "cutoff:$(cutoff)", "use_linear_ordering=$(use_linear_ordering)")] = out
# @info output_dict

# ltn = sweep_contractor_tensor_network(tn, (i, j) -> (i, j))
# @time lnz = contract_w_sweep(ltn; rank=maxdim)
# @info "lnZ of SweepContractor is", lnz

#=
bench_3d_cube_lnZ
=#
N = (5, 5, 5)
cutoff = 1e-16
ansatz= "mps"
model = "ising"
use_linear_ordering = false

if model == "ising"
  accurate_lnZ = 103.704066573293
  tn = ising_network(named_grid(N), 0.3; h=0.0, szverts=nothing)
else
  accurate_lnZ = 47.239753244708396
  distribution = Uniform{Float64}(-0.4, 1.0)
  tn = randomITensorNetwork(distribution, named_grid(N); link_space=2)
end

output_dict = Dict()
for maxdim in [512]
  for env in [(5, 1, 1)]
    tntree = build_tntree(N, tn; block_size=(1, 1, 1), env_size=env)
    out = @time bench_lnZ(
      tntree,
      accurate_lnZ;
      num_iter=1,
      cutoff=cutoff,
      maxdim=maxdim,
      ansatz=ansatz,
      approx_itensornetwork_alg="density_matrix",
      swap_size=10000,
      contraction_sequence_alg="sa_bipartite",
      contraction_sequence_kwargs=(;),
      linear_ordering_alg="bottom_up",
      use_linear_ordering=use_linear_ordering,
    )
    output_dict[("model=$(model)", "ansatz: $(ansatz)", "env:$(env)", "maxdim:$(maxdim)", "cutoff:$(cutoff)", "use_linear_ordering=$(use_linear_ordering)")] = out
    @info output_dict
  end
end

N = (6, 6, 6)
cutoff = 1e-16
ansatz= "mps"
model = "ising"
use_linear_ordering = false

accurate_lnZ = 181.79986585678668
tn = ising_network(named_grid(N), 0.3; h=0.0, szverts=nothing)

output_dict = Dict()
for maxdim in [384]
  for env in [(6, 1, 1)]
    tntree = build_tntree(N, tn; block_size=(1, 1, 1), env_size=env)
    out = @time bench_lnZ(
      tntree,
      accurate_lnZ;
      num_iter=1,
      cutoff=cutoff,
      maxdim=maxdim,
      ansatz=ansatz,
      approx_itensornetwork_alg="density_matrix",
      swap_size=10000,
      contraction_sequence_alg="sa_bipartite",
      contraction_sequence_kwargs=(;),
      linear_ordering_alg="bottom_up",
      use_linear_ordering=use_linear_ordering,
    )
    output_dict[("model=$(model)", "ansatz: $(ansatz)", "env:$(env)", "maxdim:$(maxdim)", "cutoff:$(cutoff)", "use_linear_ordering=$(use_linear_ordering)")] = out
    @info output_dict
    show(ITensors.timer)
  end
end

# ltn = sweep_contractor_tensor_network(tn, (i, j, k) -> (i * N[1] + j, k + 0.1 * randn()))
# @time lnz = contract_w_sweep(ltn; rank=512)
# @info "lnZ of SweepContractor is", abs((lnz - accurate_lnZ) / lnz)

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
