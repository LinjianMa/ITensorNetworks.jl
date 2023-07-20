using ITensors, Graphs, NamedGraphs, TimerOutputs
using Distributions, Random
using KaHyPar, Metis
using ITensorNetworks
using ITensorNetworks:
  ising_network, contract, _nested_vector_to_digraph, print_flops, reset_flops
using OMEinsumContractionOrders
using AbstractTrees

include("utils.jl")

function balanced_binary_tree_tn(num_leaves; link_space, physical_spaces)
  sequence = bipartite_sequence(collect(1:num_leaves))
  g = NamedGraph()
  for v in PostOrderDFS(sequence)
    add_vertex!(g, collect(Leaves(v)))
    if v isa Vector
      for c in v
        add_edge!(g, collect(Leaves(c)) => collect(Leaves(v)))
      end
    end
  end
  inds_net = IndsNetwork(g; link_space=link_space)
  inds_leaves = [Index(physical_spaces[i], "$(i)") for i in 1:num_leaves]
  for i in 1:num_leaves
    inds_net[[i]] = [inds_leaves[i]]
  end
  distribution = Uniform{Float64}(-1.0, 1.0)
  return randomITensorNetwork(distribution, inds_net), inds_leaves
end

function unbalanced_binary_tree_tn(num_leaves; link_space, physical_spaces)
  inds_net = IndsNetwork(named_grid((num_leaves)); link_space=link_space)
  inds_leaves = [Index(physical_spaces[i], "$(i)") for i in 1:num_leaves]
  for i in 1:num_leaves
    inds_net[i] = [inds_leaves[i]]
  end
  distribution = Uniform{Float64}(-1.0, 1.0)
  return randomITensorNetwork(distribution, inds_net), inds_leaves
end

Random.seed!(1234)
TimerOutputs.enable_debug_timings(ITensorNetworks)
# TimerOutputs.enable_debug_timings(@__MODULE__)
ITensors.set_warn_order(100)

# balanced binary tree
n = 30
link_space = 480
physical_spaces = [200 for i in 1:n]
maxdim = 50
tn, inds_leaves = balanced_binary_tree_tn(
  n; link_space=link_space, physical_spaces=physical_spaces
)
btree = _nested_vector_to_digraph(bipartite_sequence(inds_leaves))

# unbalanced binary tree
# n = 30
# link_space = 4096
# physical_spaces = [2 for i in 1:n]
# physical_spaces[1] = link_space
# physical_spaces[end] = link_space
# maxdim = 100 #div(link_space, 2)
# tn, inds_leaves = unbalanced_binary_tree_tn(
#   n; link_space=link_space, physical_spaces=physical_spaces
# )
# btree = _nested_vector_to_digraph(linear_sequence(inds_leaves))

@info "link_space", link_space, "maxdim", maxdim
@info inds_leaves
# @info tn
for alg in ["ttn_svd", "density_matrix"]
  reset_flops()
  @info "alg is", alg
  reset_timer!(ITensors.timer)
  out = @time bench(tn, btree; alg=alg, maxdim=maxdim)
  @info "out norm is", out[2]
  show(ITensors.timer)
  @info ""
  print_flops()
end
