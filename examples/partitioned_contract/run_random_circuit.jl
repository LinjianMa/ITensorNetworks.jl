using ITensors, Graphs, NamedGraphs, TimerOutputs
using Distributions, Random
using KaHyPar, Metis
using ITensorNetworks
using ITensorNetworks: ising_network, contract, _recursive_bisection
using OMEinsumContractionOrders
using AbstractTrees

include("utils.jl")
include("exact_contract.jl")
include("sweep_contractor.jl")
include("random_circuit.jl")
include("bench.jl")

Random.seed!(1234)
TimerOutputs.enable_debug_timings(ITensorNetworks)
TimerOutputs.enable_debug_timings(@__MODULE__)
ITensors.set_warn_order(100)


#=
Simulation of random quantum circuit
=#
N = (6, 6)
depth = 6
nvertices_per_partition = 5
max_dim = 512

# build the circuit
@assert N isa Number || length(N) <= 2
# Each layer contains multiple two-qubit gates, followed by one-qubit
# gates, each applying on one qubit.
layers = randomcircuit(
  N; depth=depth, twoqubitgates="CX", onequbitgates="Ry"
)
# @info "layers", layers
# exit()
hilbert1, hilbert2, part1, part2 = random_circuit_four_part_partition(N, depth)
@info length(hilbert1)
@info length(hilbert2)
@info length(part1)
@info length(part2)
left_inds = Set(noncommoninds(hilbert1...))
right_inds = Set(noncommoninds(hilbert2..., part2...))
ordered_part1 = collect(Leaves(_recursive_bisection(ITensorNetwork(part1), part1; left_inds=left_inds, right_inds=right_inds)))
@info length(ordered_part1)

left_inds = Set(noncommoninds(hilbert1..., part1...))
right_inds = Set(noncommoninds(hilbert2...))
ordered_part2 = collect(Leaves(_recursive_bisection(ITensorNetwork(part2), part2; left_inds=left_inds, right_inds=right_inds)))
@info length(ordered_part2)
tensors = [hilbert1..., ordered_part1..., ordered_part2..., hilbert2...]
sequence = linear_path(tensors, nvertices_per_partition)

# sequence = random_circuit_line_partition_sequence(N, depth)
out = @time bench_lnZ(
  sequence, 0.0;
  num_iter=1,
  cutoff=1e-10, # 1e-14
  maxdim=max_dim,
  ansatz="mps",
  approx_itensornetwork_alg="density_matrix",
  swap_size=8,
  contraction_sequence_alg="sa_bipartite",
  contraction_sequence_kwargs=(;),
  linear_ordering_alg="bottom_up",
  use_linear_ordering=false,
)
@info out

# ltn = sweep_contractor_tensor_network(ITensorNetwork(tensors), i -> (0.0001 * randn(), i))
# @time lnz = contract_w_sweep(ltn; rank=max_dim)
# @info "lnZ of SweepContractor is", lnz
