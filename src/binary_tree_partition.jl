
function _introot_union!(s, x, y; left_root=true)
  parents = s.parents
  rks = s.ranks
  @inbounds xrank = rks[x]
  @inbounds yrank = rks[y]
  if !left_root
    x, y = y, x
  end
  @inbounds parents[y] = x
  s.ngroups -= 1
  return x
end

function _root_union!(s, x, y; left_root=true)
  return s.revmap[_introot_union!(s.internal, s.intmap[x], s.intmap[y]; left_root=true)]
end

"""
partition the input network containing both tn and deltas (a vector of delta tensors) into two partitions,
one adjacent to source_inds and the other adjacent to other external inds of the network.
"""
# TODO: rewrite replaceinds
function _binary_partition(
  tn::ITensorNetwork, deltas::Vector{ITensor}, source_inds::Vector{<:Index}
)
  all_tensors = [Vector{ITensor}(tn)..., deltas...]
  external_inds = noncommoninds(all_tensors...)
  # add delta tensor to each external ind
  external_sim_ind = [sim(ind) for ind in external_inds]
  new_deltas = [
    delta(external_inds[i], external_sim_ind[i]) for i in 1:length(external_inds)
  ]
  deltas = map(t -> replaceinds(t, external_inds => external_sim_ind), deltas)
  deltas = [deltas..., new_deltas...]
  tn = map_data(t -> replaceinds(t, external_inds => external_sim_ind), tn; edges=[])
  p1, p2 = _mincut_partition_maxweightoutinds(
    disjoint_union(tn, ITensorNetwork(deltas)),
    source_inds,
    setdiff(external_inds, source_inds),
  )
  tn_vs = [v[1] for v in p1 if v[2] == 1]
  source_tn = subgraph(tn, tn_vs)
  delta_indices = [v[1] for v in p1 if v[2] == 2]
  source_deltas = Vector{ITensor}([deltas[i] for i in delta_indices])
  source_tn, source_deltas = _simplify_deltas(source_tn, source_deltas)
  tn_vs = [v[1] for v in p2 if v[2] == 1]
  remain_tn = subgraph(tn, tn_vs)
  delta_indices = [v[1] for v in p2 if v[2] == 2]
  remain_deltas = Vector{ITensor}([deltas[i] for i in delta_indices])
  remain_tn, remain_deltas = _simplify_deltas(remain_tn, remain_deltas)
  @assert (
    length(noncommoninds(all_tensors...)) == length(
      noncommoninds(
        Vector{ITensor}(source_tn)...,
        source_deltas...,
        Vector{ITensor}(remain_tn)...,
        remain_deltas...,
      ),
    )
  )
  return source_tn, source_deltas, remain_tn, remain_deltas
end

# TODO: add explanation: it's better to keep leave nodes in the partition since 
# we can remove more deltas in `_remove_deltas`, which is more efficient.
function binary_tree_partition(tn::ITensorNetwork, inds_btree::Vector)
  # network = Vector{ITensor}(tn)
  output_tns = Vector{ITensorNetwork}()
  output_deltas_vector = Vector{Vector{ITensor}}()
  btree_to_input_tn_deltas = Dict{Union{Vector,Index},Tuple}()
  btree_to_input_tn_deltas[inds_btree] = (tn, Vector{ITensor}())
  for node in PreOrderDFS(inds_btree)
    @assert haskey(btree_to_input_tn_deltas, node)
    input_tn, input_deltas = btree_to_input_tn_deltas[node]
    if node isa Index
      push!(output_tns, input_tn)
      push!(output_deltas_vector, input_deltas)
      continue
    end
    tn1, deltas1, input_tn, input_deltas = _binary_partition(
      input_tn, input_deltas, collect(Leaves(node[1]))
    )
    btree_to_input_tn_deltas[node[1]] = (tn1, deltas1)
    tn1, deltas1, input_tn, input_deltas = _binary_partition(
      input_tn, input_deltas, collect(Leaves(node[2]))
    )
    btree_to_input_tn_deltas[node[2]] = (tn1, deltas1)
    push!(output_tns, input_tn)
    push!(output_deltas_vector, input_deltas)
    # @info "btree_to_output_tn[node]", btree_to_output_tn[node]
  end
  # form subgraph_vertices
  subgraph_vs = Vector{Vector{Tuple}}()
  delta_num = 0
  for (tn, deltas) in zip(output_tns, output_deltas_vector)
    vs = Vector{Tuple}([(v, 1) for v in vertices(tn)])
    vs = vcat(vs, [(i + delta_num, 2) for i in 1:length(deltas)])
    push!(subgraph_vs, vs)
    delta_num += length(deltas)
  end
  # all_deltas = [deltas... for _, deltas in output_tn_deltas_pair]
  tn1 = ITensorNetwork()
  for tn in output_tns
    for v in vertices(tn)
      add_vertex!(tn1, v)
      tn1[v] = tn[v]
    end
  end
  tn_deltas = ITensorNetwork(vcat(output_deltas_vector...))
  return partition(ITensorNetwork{Any}(disjoint_union(tn1, tn_deltas)), subgraph_vs)
end

function _simplify_deltas(tn::ITensorNetwork, deltas::Vector{ITensor})
  out_delta_inds = Vector{Pair}()
  network = [Vector{ITensor}(tn)..., deltas...]
  outinds = noncommoninds(network...)
  inds_list = map(t -> collect(inds(t)), deltas)
  deltainds = collect(Set(vcat(inds_list...)))
  ds = DisjointSets(deltainds)
  for t in deltas
    i1, i2 = inds(t)
    if find_root!(ds, i1) in outinds && find_root!(ds, i2) in outinds
      push!(out_delta_inds, find_root!(ds, i1) => find_root!(ds, i2))
    end
    if find_root!(ds, i1) in outinds
      _root_union!(ds, find_root!(ds, i1), find_root!(ds, i2))
    else
      _root_union!(ds, find_root!(ds, i2), find_root!(ds, i1))
    end
  end
  tn = map_data(
    t -> replaceinds(t, deltainds => [find_root!(ds, i) for i in deltainds]), tn; edges=[]
  )
  out_deltas = Vector{ITensor}([delta(i.first, i.second) for i in out_delta_inds])
  return tn, out_deltas
end

_is_delta(t) = (t.tensor.storage.data == 1.0)

# remove deltas to improve the performance
function _remove_deltas(partition::DataGraph; r=1)
  partition = copy(partition)
  leaves = leaf_vertices(dfs_tree(partition, r))
  # only remove deltas in intermediate vertices
  nonleaf_vertices = setdiff(vertices(partition), leaves)
  network = vcat([Vector{ITensor}(partition[v]) for v in nonleaf_vertices]...)
  outinds = noncommoninds(network...)
  all_deltas = []
  for tn_v in nonleaf_vertices
    tn = partition[tn_v]
    deltas = [tn[v] for v in vertices(tn) if _is_delta(tn[v])]
    all_deltas = vcat(all_deltas, deltas)
  end
  if length(all_deltas) == 0
    return partition
  end
  inds_list = map(t -> collect(inds(t)), all_deltas)
  deltainds = collect(Set(vcat(inds_list...)))
  ds = DisjointSets(deltainds)
  for t in all_deltas
    i1, i2 = inds(t)
    if find_root!(ds, i1) in outinds
      _root_union!(ds, find_root!(ds, i1), find_root!(ds, i2))
    else
      _root_union!(ds, find_root!(ds, i2), find_root!(ds, i1))
    end
  end
  sim_deltainds = [find_root!(ds, ind) for ind in deltainds]
  for tn_v in nonleaf_vertices
    tn = partition[tn_v]
    nondelta_vertices = [v for v in vertices(tn) if !_is_delta(tn[v])]
    new_tn = ITensorNetwork()
    for v in nondelta_vertices
      add_vertex!(new_tn, v)
      new_tn[v] = replaceinds(tn[v], deltainds, sim_deltainds)
    end
    partition[tn_v] = new_tn
  end
  return partition
end
