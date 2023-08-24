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
include("bench.jl")

Random.seed!(1234)
TimerOutputs.enable_debug_timings(ITensorNetworks)
TimerOutputs.enable_debug_timings(@__MODULE__)
ITensors.set_warn_order(100)

#=
random regular graph
=#
nvertices = 220
deg = 3
beta = 0.65
maxdim = 32
cutoff = 1e-14
# swap_size = 4
num_iter = 1
ansatz = "mps"
model = "ising"
path = "linear"
use_linear_ordering = true

if model == "ising"
  accurate_lnZ = 223.53205177989938
  tn = ising_network(
    NamedGraph(random_regular_graph(nvertices, deg)), beta; h=0.0, szverts=nothing
  )
else
  accurate_lnZ = 5.633462619348083
  distribution = Uniform{Float64}(-0.2, 1.0)
  tn = randomITensorNetwork(
    distribution, random_regular_graph(nvertices, deg); link_space=2
  )
end
# exact_contract(tn; sc_target=30)
# -26.228887728408008 (-1.0, 1.0)
# 5.633462619348083 (-0.2, 1.0)
output_dict = Dict()
for nvertices_per_partition in [3]
  for swap_size in [16, 64]
    if path == "balanced"
      out = @time bench_lnZ(
        tn,
        accurate_lnZ;
        num_iter=num_iter,
        nvertices_per_partition=nvertices_per_partition,
        backend="KaHyPar",
        cutoff=cutoff,
        maxdim=maxdim,
        ansatz=ansatz,
        approx_itensornetwork_alg="density_matrix",
        swap_size=swap_size,
        contraction_sequence_alg="sa_bipartite",
        contraction_sequence_kwargs=(;),
        linear_ordering_alg="bottom_up",
        use_linear_ordering=use_linear_ordering,
      )
    else
      tensors = collect(Leaves(_recursive_bisection(tn, Vector{ITensor}(tn))))
      out = @time bench_lnZ(
        linear_path(tensors, nvertices_per_partition),
        accurate_lnZ;
        num_iter=num_iter,
        cutoff=cutoff,
        maxdim=maxdim,
        ansatz=ansatz,
        approx_itensornetwork_alg="density_matrix",
        swap_size=swap_size,
        contraction_sequence_alg="sa_bipartite",
        contraction_sequence_kwargs=(;),
        linear_ordering_alg="bottom_up",
        use_linear_ordering=use_linear_ordering,
      )
    end
    output_dict[("model=$(model)", "path=$(path)", "ansatz: $(ansatz)", "par:$(nvertices_per_partition)", "swap:$(swap_size)", "maxdim:$(maxdim)", "cutoff:$(cutoff)", "use_linear_ordering=$(use_linear_ordering)")] = out
    @info output_dict
  end
end

# tensors = collect(Leaves(_recursive_bisection(tn, Vector{ITensor}(tn))))
# ltn = sweep_contractor_tensor_network(ITensorNetwork(tensors), i -> (0.0001 * randn(), i))
# @time lnz = contract_w_sweep(ltn; rank=maxdim)
# @info "lnZ of SweepContractor is", abs((lnz - accurate_lnZ) / lnz)
