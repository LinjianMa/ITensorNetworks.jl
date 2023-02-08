struct _DensityMatrix
  tensor::ITensor
  children::Vector
end

struct _PartialDensityMatrix
  tensor::ITensor
  child::Union{<:Number,Tuple}
end

struct _DensityMatrixAlgCaches
  v_to_cdm::Dict{Union{<:Number,Tuple},_DensityMatrix}
  v_to_cpdms::Dict{Union{<:Number,Tuple},Vector{_PartialDensityMatrix}}
end

function _DensityMatrixAlgCaches()
  v_to_cdm = Dict{Union{<:Number,Tuple},_DensityMatrix}()
  v_to_cpdms = Dict{Union{<:Number,Tuple},Vector{_PartialDensityMatrix}}()
  return _DensityMatrixAlgCaches(v_to_cdm, v_to_cpdms)
end

function _remove_cpdms(cpdms::Vector, children)
  return filter(pdm -> !(pdm.child in children), cpdms)
end

struct _DensityMartrixAlgGraph
  partition::DataGraph
  out_tree::NamedGraph
  root::Union{<:Number,Tuple}
  innerinds_to_sim::Pair{Vector,Vector}
  caches::_DensityMatrixAlgCaches
end

function _DensityMartrixAlgGraph(
  partition::DataGraph, out_tree::NamedGraph, root::Union{<:Number,Tuple}
)
  innerinds = _get_inner_inds(partition)
  sim_innerinds = [sim(ind) for ind in innerinds]
  return _DensityMartrixAlgGraph(
    partition, out_tree, root, innerinds => sim_innerinds, _DensityMatrixAlgCaches()
  )
end

function _get_inner_inds(partition::DataGraph)
  networks = [Vector{ITensor}(partition[v]) for v in vertices(partition)]
  network = vcat(networks...)
  outinds = noncommoninds(network...)
  allinds = mapreduce(t -> [i for i in inds(t)], vcat, network)
  return Vector(setdiff(allinds, outinds))
end

function _get_out_inds(partition::DataGraph)
  networks = [Vector{ITensor}(partition[v]) for v in vertices(partition)]
  network = vcat(networks...)
  return noncommoninds(network...)
end

function _optcontract(network::Vector)
  @timeit_debug ITensors.timer "[densitymatrxalg]: _optcontract" begin
    if length(network) == 0
      return ITensor(1.0)
    end
    @assert network isa Vector{ITensor}
    # @info "start contract, size is", size(network)
    # for t in network
    #   @info "size of t is", size(t)
    # end
    @timeit_debug ITensors.timer "[densitymatrxalg]: contraction_sequence" begin
      seq = contraction_sequence(network; alg="sa_bipartite")
    end
    output = contract(network; sequence=seq)
    return output
  end
end

function _get_low_rank_projector(tensor, inds1, inds2; cutoff, maxdim)
  t00 = time()
  @info "eigen input size", size(tensor)
  @timeit_debug ITensors.timer "[densitymatrxalg]: eigen" begin
    diag, U = eigen(tensor, inds1, inds2; cutoff=cutoff, maxdim=maxdim, ishermitian=true)
  end
  t11 = time() - t00
  @info "size of U", size(U), "size of diag", size(diag), "costs", t11
  return U
end

function _densitymatrix_outinds_to_sim(partition, root)
  outinds = _get_out_inds(partition)
  outinds_root = intersect(outinds, noncommoninds(Vector{ITensor}(partition[root])...))
  outinds_root_to_sim = outinds_root => [sim(ind) for ind in outinds_root]
  return outinds_root_to_sim
end

function _sim(partial_dm_tensor::ITensor, indices, simindices)
  inds_to_siminds = Dict(zip(indices, simindices))
  siminds_to_inds = Dict(zip(simindices, indices))
  indices = intersect(indices, inds(partial_dm_tensor))
  simindices = setdiff(inds(partial_dm_tensor), indices)
  reorder_inds = [indices..., simindices...]
  reorder_siminds = vcat(
    [inds_to_siminds[i] for i in indices], [siminds_to_inds[i] for i in simindices]
  )
  return replaceinds(partial_dm_tensor, reorder_inds => reorder_siminds)
end

function _get_pdm(
  partial_dms::Vector{_PartialDensityMatrix}, child_v, child_dm_tensor, network
)
  for partial_dm in partial_dms
    if partial_dm.child == child_v
      return partial_dm
    end
  end
  tensor = _optcontract([child_dm_tensor, network...])
  return _PartialDensityMatrix(tensor, child_v)
end

function _update!(
  caches::_DensityMatrixAlgCaches,
  v::Union{<:Number,Tuple},
  children::Vector,
  root::Union{<:Number,Tuple},
  network::Vector{ITensor},
  inds_to_sim,
)
  if haskey(caches.v_to_cdm, v) && caches.v_to_cdm[v].children == children && v != root
    @assert haskey(caches.v_to_cdm, v)
    return nothing
  end
  child_to_dm = [c => caches.v_to_cdm[c].tensor for c in children]
  if !haskey(caches.v_to_cpdms, v)
    caches.v_to_cpdms[v] = []
  end
  cpdms = [
    _get_pdm(caches.v_to_cpdms[v], child_v, dm_tensor, network) for
    (child_v, dm_tensor) in child_to_dm
  ]
  if length(cpdms) == 0
    sim_network = Vector{ITensor}(map(t -> replaceinds(t, inds_to_sim), network))
    density_matrix = _optcontract([network..., sim_network...])
  elseif length(cpdms) == 1
    sim_network = Vector{ITensor}(map(t -> replaceinds(t, inds_to_sim), network))
    density_matrix = _optcontract([cpdms[1].tensor, sim_network...])
  else
    simtensor = _sim(cpdms[2].tensor, inds_to_sim.first, inds_to_sim.second)
    density_matrix = _optcontract([cpdms[1].tensor, simtensor])
  end
  caches.v_to_cdm[v] = _DensityMatrix(density_matrix, children)
  caches.v_to_cpdms[v] = cpdms
  return nothing
end

function _rem_vertex!(alg_graph::_DensityMartrixAlgGraph, root; kwargs...)
  caches = alg_graph.caches
  outinds_root_to_sim = _densitymatrix_outinds_to_sim(alg_graph.partition, root)
  inds_to_sim =
    [alg_graph.innerinds_to_sim.first..., outinds_root_to_sim.first...] =>
      [alg_graph.innerinds_to_sim.second..., outinds_root_to_sim.second...]
  dm_dfs_tree = dfs_tree(alg_graph.out_tree, root)
  @assert length(child_vertices(dm_dfs_tree, root)) == 1
  for v in post_order_dfs_vertices(dm_dfs_tree, root)
    children = sort(child_vertices(dm_dfs_tree, v))
    @assert length(children) <= 2
    network = Vector{ITensor}(alg_graph.partition[v])
    _update!(caches, v, children, root, Vector{ITensor}(network), inds_to_sim)
  end
  U = _get_low_rank_projector(
    caches.v_to_cdm[root].tensor,
    outinds_root_to_sim.second,
    outinds_root_to_sim.first;
    kwargs...,
  )
  # update partition and tree
  root_tensor = _optcontract([Vector{ITensor}(alg_graph.partition[root])..., U])
  new_root = child_vertices(dm_dfs_tree, root)[1]
  new_tn = disjoint_union(alg_graph.partition[new_root], ITensorNetwork([root_tensor]))
  alg_graph.partition[new_root] = ITensorNetwork{Any}(new_tn)
  rem_vertex!(alg_graph.partition, root)
  rem_vertex!(alg_graph.out_tree, root)
  # update v_to_cpdms[new_root]
  delete!(caches.v_to_cpdms, root)
  truncate_dfs_tree = dfs_tree(alg_graph.out_tree, alg_graph.root)
  caches.v_to_cpdms[new_root] = _remove_cpdms(
    caches.v_to_cpdms[new_root], child_vertices(truncate_dfs_tree, new_root)
  )
  @assert length(caches.v_to_cpdms[new_root]) <= 1
  caches.v_to_cpdms[new_root] = [
    _PartialDensityMatrix(cpdm.tensor * root_tensor, cpdm.child) for
    cpdm in caches.v_to_cpdms[new_root]
  ]
  return U
end

function approx_itensornetwork!(
  partition::DataGraph, out_tree::NamedGraph; root=1, cutoff=1e-15, maxdim=10000
)
  @assert sort(vertices(partition)) == sort(vertices(out_tree))
  alg_graph = _DensityMartrixAlgGraph(partition, out_tree, root)
  output_tn = ITensorNetwork()
  for approx_v in post_order_dfs_vertices(out_tree, root)[1:(end - 1)]
    U = _rem_vertex!(alg_graph, approx_v; cutoff=cutoff, maxdim=maxdim)
    # update output_tn
    add_vertex!(output_tn, approx_v)
    output_tn[approx_v] = U
  end
  @assert length(vertices(partition)) == 1
  add_vertex!(output_tn, root)
  root_tensor = _optcontract(Vector{ITensor}(partition[root]))
  root_norm = norm(root_tensor)
  root_tensor /= root_norm
  output_tn[root] = root_tensor
  # TODO: only useful for the binary tree case
  _rem_leaf_vertices!(output_tn, root)
  return output_tn, log(root_norm)
end

function approx_itensornetwork!(
  binary_tree_partition::DataGraph; root=1, cutoff=1e-15, maxdim=10000
)
  @assert is_tree(binary_tree_partition)
  @assert root in vertices(binary_tree_partition)
  # TODO: explain this
  partition_wo_deltas = _remove_deltas(binary_tree_partition; r=root)
  return approx_itensornetwork!(
    partition_wo_deltas,
    underlying_graph(binary_tree_partition);
    root=root,
    cutoff=cutoff,
    maxdim=maxdim,
  )
end

function _rem_leaf_vertices!(tn::ITensorNetwork, root)
  dfs_t = dfs_tree(tn, root)
  leaves = leaf_vertices(dfs_t)
  parents = [parent_vertex(dfs_t, leaf) for leaf in leaves]
  for (l, p) in zip(leaves, parents)
    tn[p] = tn[p] * tn[l]
    rem_vertex!(tn, l)
  end
end
