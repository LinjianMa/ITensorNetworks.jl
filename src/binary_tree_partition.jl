"""
Partition the input network containing both `tn` and `deltas` (a vector of delta tensors)
into two partitions, one adjacent to source_inds and the other adjacent to other external
inds of the network.
"""
function _binary_partition(tn::ITensorNetwork, source_inds::Vector{<:Index})
  external_inds = noncommoninds(Vector{ITensor}(tn)...)
  # add delta tensor to each external ind
  external_sim_ind = [sim(ind) for ind in external_inds]
  tn = map_data(t -> replaceinds(t, external_inds => external_sim_ind), tn; edges=[])
  tn_wo_deltas = rename_vertices(v -> v[1], subgraph(v -> v[2] == 1, tn))
  deltas = Vector{ITensor}(subgraph(v -> v[2] == 2, tn))
  new_deltas = [
    delta(external_inds[i], external_sim_ind[i]) for i in 1:length(external_inds)
  ]
  deltas = [deltas..., new_deltas...]
  tn = disjoint_union(tn_wo_deltas, ITensorNetwork(deltas))
  p1, p2 = _mincut_partition_maxweightoutinds(
    tn, source_inds, setdiff(external_inds, source_inds)
  )
  source_tn = _contract_deltas(subgraph(tn, p1))
  remain_tn = _contract_deltas(subgraph(tn, p2))
  @assert (
    length(external_inds) ==
    length(noncommoninds(Vector{ITensor}(source_tn)..., Vector{ITensor}(remain_tn)...))
  )
  return source_tn, remain_tn
end

"""
Given an input tn and a rooted binary tree of indices, return a partition of tn with the
same binary tree structure as inds_btree.
Note: in the output partition, we add multiple delta tensors to the network so that
  the output graph is guaranteed to be the same binary tree as inds_btree.
Note: in the output partition, tensor vertex names will be changed. For a given input
  tensor with vertex name `v``, its name in the output partition will be `(v, 1)`, and any
  delta tensor will have name `(v, 2)`.
Note: for a given binary tree with n indices, the output partition will contain 2n-1 vertices,
  with each leaf vertex corresponding to a sub tn adjacent to one output index. Keeping these
  leaf vertices in the partition makes later `approx_itensornetwork` algorithms more efficient.
"""
function binary_tree_partition(tn::ITensorNetwork, inds_btree::Vector)
  output_tns = Vector{ITensorNetwork}()
  output_deltas_vector = Vector{Vector{ITensor}}()
  # Mapping each vertex of the binary tree to a tn representing the partition
  # of the subtree containing this vertex and its descendant vertices.
  v_to_subtree_tn = Dict{Union{Vector,Index},ITensorNetwork}()
  v_to_subtree_tn[inds_btree] = disjoint_union(tn, ITensorNetwork())
  for v in PreOrderDFS(inds_btree)
    @assert haskey(v_to_subtree_tn, v)
    input_tn = v_to_subtree_tn[v]
    if !(v isa Index)
      tn1, input_tn = _binary_partition(input_tn, collect(Leaves(v[1])))
      v_to_subtree_tn[v[1]] = tn1
      tn1, input_tn = _binary_partition(input_tn, collect(Leaves(v[2])))
      v_to_subtree_tn[v[2]] = tn1
    end
    tn = rename_vertices(u -> u[1], subgraph(u -> u[2] == 1, input_tn))
    deltas = Vector{ITensor}(subgraph(u -> u[2] == 2, input_tn))
    push!(output_tns, tn)
    push!(output_deltas_vector, deltas)
  end
  # In subgraph_vertices, each element is a vector of vertices to be
  # grouped in one partition.
  subgraph_vs = Vector{Vector{Tuple}}()
  delta_num = 0
  for (tn, deltas) in zip(output_tns, output_deltas_vector)
    vs = Vector{Tuple}([(v, 1) for v in vertices(tn)])
    vs = vcat(vs, [(i + delta_num, 2) for i in 1:length(deltas)])
    push!(subgraph_vs, vs)
    delta_num += length(deltas)
  end
  out_tn = ITensorNetwork()
  for tn in output_tns
    for v in vertices(tn)
      add_vertex!(out_tn, v)
      out_tn[v] = tn[v]
    end
  end
  tn_deltas = ITensorNetwork(vcat(output_deltas_vector...))
  return partition(ITensorNetwork(disjoint_union(out_tn, tn_deltas)), subgraph_vs)
end
