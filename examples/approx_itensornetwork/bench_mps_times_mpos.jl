using ITensors, Graphs, NamedGraphs, TimerOutputs
using Distributions, Random
using KaHyPar, Metis
using ITensorNetworks
using ITensorNetworks:
  ising_network,
  contract,
  _nested_vector_to_digraph,
  print_flops,
  reset_flops,
  map_data,
  _root
using OMEinsumContractionOrders
using AbstractTrees

include("utils.jl")

function embed(tn, btree, external_inds, num_mpo)
  external_sim_inds = [sim(ind) for ind in external_inds]
  tn = map_data(t -> replaceinds(t, external_inds => external_sim_inds), tn; edges=[])
  new_deltas = [
    delta(external_inds[i], external_sim_inds[i]) for i in 1:length(external_inds)
  ]
  subgraph_vs = Vector{Vector{Tuple}}()
  leaves = collect(leaf_vertices(btree))
  leaf_i = 1
  nonleaf_i = 1
  for v in post_order_dfs_vertices(btree, _root(btree))
    if v in leaves
      push!(subgraph_vs, Vector{Tuple}([(leaf_i, 2)]))
      leaf_i += 1
    else
      if nonleaf_i == 1
        vs = [((i, j), 1) for i in 1:2 for j in 1:num_mpo]
        push!(subgraph_vs, Vector{Tuple}(vs))
        nonleaf_i += 2
      else
        vs = [((nonleaf_i, j), 1) for j in 1:num_mpo]
        push!(subgraph_vs, Vector{Tuple}(vs))
        nonleaf_i += 1
      end
    end
  end
  return partition(disjoint_union(tn, ITensorNetwork(new_deltas)), subgraph_vs)
end

function mps_times_mpos(len_mps, num_mpo; link_space, physical_spaces)
  inds_net = IndsNetwork(named_grid((len_mps, num_mpo)))
  for j in 1:num_mpo
    for i in 1:(len_mps - 1)
      rank = min(link_space, 2^i, 2^(len_mps - i))
      inds_net[(i, j) => (i + 1, j)] = [Index(rank, "$(i)x$(j),$(i+1)x$(j)")]
    end
  end
  for i in 1:len_mps
    for j in 1:(num_mpo - 1)
      inds_net[(i, j) => (i, j + 1)] = [Index(physical_spaces[2], "$(i)x$(j),$(i)x$(j+1)")]
    end
  end
  inds_leaves = [Index(physical_spaces[i], "$(i)x$(num_mpo),out") for i in 1:len_mps]
  for i in 1:len_mps
    inds_net[(i, num_mpo)] = [inds_leaves[i]]
  end
  distribution = Uniform{Float64}(-1.0, 1.0)
  return randomITensorNetwork(distribution, inds_net), inds_leaves
end

Random.seed!(1234)
TimerOutputs.enable_debug_timings(ITensorNetworks)
# TimerOutputs.enable_debug_timings(@__MODULE__)
ITensors.set_warn_order(100)

len_mps = 40
num_mpo = 2
link_space = 64 # the largest seems 80, beyond which would get too slow.
maxdim = link_space
physical_spaces = [2 for i in 1:len_mps]
physical_spaces[1] = maxdim
physical_spaces[end] = maxdim
tree = "mps"

tn, inds_leaves = mps_times_mpos(
  len_mps, num_mpo; link_space=link_space, physical_spaces=physical_spaces
)
if tree == "mps"
  btree = _nested_vector_to_digraph(linear_sequence(inds_leaves))
else
  btree = _nested_vector_to_digraph(bipartite_sequence(inds_leaves))
end
par = embed(tn, btree, inds_leaves, num_mpo)
# @info "tn", tn
@info "number of vertices in tn", length(vertices(tn))
@info "inds_leaves is", inds_leaves
for alg in ["ttn_svd", "density_matrix"]
  reset_flops()
  @info "alg is [$(alg)] with automatic embedding"
  reset_timer!(ITensors.timer)
  out = @time bench(
    tn, btree; alg=alg, maxdim=maxdim, contraction_sequence_alg="sa_bipartite"
  )
  @info "out norm is", out[2]
  show(ITensors.timer)
  @info ""
  print_flops()

  reset_flops()
  @info "alg is [$(alg)] with manual embedding"
  reset_timer!(ITensors.timer)
  out = @time bench_embed(
    par, _root(btree); alg=alg, maxdim=maxdim, contraction_sequence_alg="optimal"
  )
  @info "out norm is", out[2]
  show(ITensors.timer)
  @info ""
  print_flops()
end
