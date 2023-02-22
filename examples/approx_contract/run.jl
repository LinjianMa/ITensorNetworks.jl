using ITensors, Graphs, NamedGraphs, TimerOutputs
using Distributions, Random
using ITensorNetworks
using ITensorNetworks: ising_network

include("3d_cube.jl")
include("sweep_contractor.jl")

Random.seed!(1234)
TimerOutputs.enable_debug_timings(ITensorNetworks)
TimerOutputs.enable_debug_timings(@__MODULE__)

#=
Exact contraction of ising_network
=#
# N = (3, 3, 3)
# beta = 0.3
# network = ising_network(named_grid(N), beta=beta)
# exact_contract(N, network; sc_target=28)

#=
Exact contraction of randomITensorNetwork
=#
# N = (5, 5, 5)
# distribution = Uniform{Float64}(-0.4, 1.0)
# network = randomITensorNetwork(named_grid(N); link_space=2, distribution=distribution)
# exact_contract(N, network; sc_target=30) # 47.239753244708396

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
#   snake=false,
#   use_cache=true,
#   ortho=false,
#   env_size=(3, 1, 1),
# )

#=
bench_3d_cube_lnZ
=#
# N = (3, 3, 3)
# beta = 0.3
# network = ising_network(named_grid(N), beta; h=0.0, szverts=nothing)
# @time bench_3d_cube_lnZ(
#   N,
#   network;
#   block_size=(1, 1, 1),
#   num_iter=2,
#   cutoff=1e-12,
#   maxdim=64,
#   ansatz="mps",
#   algorithm="density_matrix",
#   snake=false,
#   use_cache=true,
#   ortho=false,
#   env_size=(3, 1, 1),
# )

#=
bench_3d_cube_magnetization
=#
# N = (1, 6, 6)
# beta = 0.44
# h = 0.0001
# szverts = [(1, 3, 3)]
# network1 = ising_network(named_grid(N), beta; h=h, szverts=szverts)
# network2 = ising_network(named_grid(N), beta; h=h, szverts=nothing)
# @time bench_3d_cube_magnetization(
#   N,
#   network1 => network2;
#   block_size=(1, 1, 1),
#   num_iter=2,
#   cutoff=1e-12,
#   maxdim=64,
#   ansatz="mps",
#   algorithm="density_matrix",
#   snake=false,
#   use_cache=true,
#   ortho=false,
#   env_size=(1, 6, 1),
# )

#=
SweepContractor
=#
reset_timer!(ITensors.timer)
# N = (5, 5, 5)
# beta = 0.3
# network = ising_network(named_grid(N), beta; h=0.0, szverts=nothing)
N = (3, 3, 3)
distribution = Uniform{Float64}(-0.4, 1.0)
network = randomITensorNetwork(named_grid(N); link_space=2, distribution=distribution)
ltn = sweep_contractor_tensor_network(
  network, (i, j, k) -> (j + 0.01 * randn(), k + 0.01 * randn())
)
@time lnz = contract_w_sweep(ltn; rank=256)
@info "lnZ of SweepContractor is", lnz
show(ITensors.timer)
