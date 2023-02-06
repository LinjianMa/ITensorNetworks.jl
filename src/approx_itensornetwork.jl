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
  @timeit_debug ITensors.timer "[ITensorNetworks]: _optcontract" begin
    if length(network) == 0
      return ITensor(1.0)
    end
    @assert network isa Vector{ITensor}
    # @info "start contract, size is", size(network)
    # for t in network
    #   @info "size of t is", size(t)
    # end
    @timeit_debug ITensors.timer "[ITensorNetworks]: contraction_sequence" begin
      seq = contraction_sequence(network; alg="sa_bipartite")
    end
    output = contract(network; sequence=seq)
    return output
  end
end

function _get_low_rank_projector(tensor, inds1, inds2; cutoff, maxdim)
  t00 = time()
  @info "eigen input size", size(tensor)
  @timeit_debug ITensors.timer "[ITensorNetworks]: eigen" begin
    diag, U = eigen(tensor, inds1, inds2; cutoff=cutoff, maxdim=maxdim, ishermitian=true)
  end
  t11 = time() - t00
  @info "size of U", size(U), "size of diag", size(diag), "costs", t11
  return U
end

function _binary_partition_with_truncate!(
  partition::DataGraph,
  tree::NamedGraph,
  v_to_density_matrix::Dict,
  v_to_children::Dict,
  root,
  innerinds_to_sim;
  kwargs...,
)
  @info "root is", root
  outinds = _get_out_inds(partition)
  outinds_root = intersect(outinds, noncommoninds(Vector{ITensor}(partition[root])...))
  outinds_root_to_sim = outinds_root => [sim(ind) for ind in outinds_root]
  directed_tree = dfs_tree(tree, root)
  @assert length(child_vertices(directed_tree, root)) == 1
  for v in post_order_dfs_vertices(directed_tree, root)
    children = sort(child_vertices(directed_tree, v))
    if haskey(v_to_children, v) && v_to_children[v] == children && v != root
      @assert haskey(v_to_density_matrix, v)
      continue
    end
    children_density_matrices = [v_to_density_matrix[c] for c in children]
    v_network = Vector{ITensor}(partition[v])
    @info "v", v
    @info "v_network", length(v_network)
    # @assert length(v_network) > 0
    v_network_sim = map(t -> replaceinds(t, innerinds_to_sim), v_network)
    if v == root
      v_network_sim = map(t -> replaceinds(t, outinds_root_to_sim), v_network_sim)
    end
    v_to_density_matrix[v] = _optcontract([
      children_density_matrices..., v_network..., v_network_sim...
    ])
    v_to_children[v] = children
  end
  U = _get_low_rank_projector(
    v_to_density_matrix[root],
    outinds_root_to_sim.second,
    outinds_root_to_sim.first;
    kwargs...,
  )
  # update partition and tree
  root_tensor = _optcontract([Vector{ITensor}(partition[root])..., U])
  child = child_vertices(directed_tree, root)[1]
  new_tn = disjoint_union(partition[child], ITensorNetwork([root_tensor]))
  partition[child] = ITensorNetwork{Any}(new_tn)
  rem_vertex!(partition, root)
  rem_vertex!(tree, root)
  return U
end

function approx_itensornetwork!(
  binary_tree_partition::DataGraph; root=1, cutoff=1e-15, maxdim=10000
)
  @assert is_tree(binary_tree_partition)
  tree = underlying_graph(binary_tree_partition)
  # vs = post_order_dfs_vertices(tree, root)
  partition_wo_deltas = _remove_deltas(binary_tree_partition; r=root)
  v_to_density_matrix = Dict{Union{Number,Tuple},ITensor}()
  v_to_children = Dict{Union{Number,Tuple},Vector}()
  output_tn = ITensorNetwork()
  innerinds = _get_inner_inds(partition_wo_deltas)
  sim_innerinds = [sim(ind) for ind in innerinds]
  for approx_v in post_order_dfs_vertices(tree, root)[1:(end - 1)]
    U = _binary_partition_with_truncate!(
      partition_wo_deltas,
      tree,
      v_to_density_matrix,
      v_to_children,
      approx_v,
      innerinds => sim_innerinds;
      cutoff=cutoff,
      maxdim=maxdim,
    )
    # update output_tn
    add_vertex!(output_tn, approx_v)
    output_tn[approx_v] = U
  end
  @assert length(vertices(partition_wo_deltas)) == 1
  add_vertex!(output_tn, root)
  root_tensor = _optcontract(Vector{ITensor}(partition_wo_deltas[root]))
  root_norm = norm(root_tensor)
  root_tensor /= root_norm
  output_tn[root] = root_tensor
  return output_tn, log(root_norm)
end
