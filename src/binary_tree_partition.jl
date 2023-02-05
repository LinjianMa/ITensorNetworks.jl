struct UF
  parent_map::Dict
end

function UF(values::Vector)
  parent_map = Dict()
  for value in values
    parent_map[value] = value
  end
  return UF(parent_map)
end

function root(uf::UF, n)
  while uf.parent_map[n] != n
    n = uf.parent_map[n]
  end
  return n
end

function connect(uf, n1, n2)
  rootn1 = root(uf, n1)
  rootn2 = root(uf, n2)
  if rootn1 == rootn2
    # Already connected
    return nothing
  end
  return uf.parent_map[rootn1] = rootn2
end

"""
partition the input network containing both tn and deltas (a vector of delta tensors) into two partitions,
one adjacent to source_inds and the other adjacent to other external inds of the network.
"""
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
  @info "p1 is", p1
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

function binary_tree_partition(tn::ITensorNetwork, inds_btree::Vector; algorithm)
  # network = Vector{ITensor}(tn)
  output_tns = Vector{ITensorNetwork}()
  output_deltas_vector = Vector{Vector{ITensor}}()
  btree_to_input_tn_deltas = Dict{Union{Vector,Index},Tuple}()
  btree_to_input_tn_deltas[inds_btree] = (tn, Vector{ITensor}())
  for node in PreOrderDFS(inds_btree)
    @info "node is", node
    @assert haskey(btree_to_input_tn_deltas, node)
    input_tn, input_deltas = btree_to_input_tn_deltas[node]
    # @info "node", node
    if node isa Index
      push!(output_tns, input_tn)
      push!(output_deltas_vector, input_deltas)
      continue
    end
    tn1, deltas1, tn2, deltas2 = _binary_partition(
      input_tn, input_deltas, collect(Leaves(node[1]))
    )
    btree_to_input_tn_deltas[node[1]] = (tn1, deltas1)
    tn1, deltas1, tn2, deltas2 = _binary_partition(tn2, deltas2, collect(Leaves(node[2])))
    btree_to_input_tn_deltas[node[2]] = (tn1, deltas1)
    push!(output_tns, tn2)
    push!(output_deltas_vector, deltas2)
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

# _is_delta(t) = (t.tensor.storage.data == 1.0)

function _simplify_deltas(tn::ITensorNetwork, deltas::Vector{ITensor})
  out_delta_inds = Vector{Pair}()
  network = [Vector{ITensor}(tn)..., deltas...]
  outinds = noncommoninds(network...)
  inds_list = map(t -> collect(inds(t)), deltas)
  deltainds = collect(Set(vcat(inds_list...)))
  uf = UF(deltainds)
  for t in deltas
    i1, i2 = inds(t)
    if root(uf, i1) in outinds && root(uf, i2) in outinds
      push!(out_delta_inds, root(uf, i1) => root(uf, i2))
    end
    if root(uf, i1) in outinds
      connect(uf, i2, i1)
    else
      connect(uf, i1, i2)
    end
  end
  sim_dict = Dict([ind => root(uf, ind) for ind in deltainds])
  tn = map_data(
    t -> replaceinds(t, deltainds => [root(uf, i) for i in deltainds]), tn; edges=[]
  )
  out_deltas = Vector{ITensor}([delta(i.first, i.second) for i in out_delta_inds])
  return tn, out_deltas
end

# # remove deltas to improve the performance
# function remove_deltas(tnets_dict::Dict)
#   # only remove deltas in intermediate nodes
#   ks = filter(k -> (length(k) > 1), collect(keys(tnets_dict)))
#   network = vcat([tnets_dict[k] for k in ks]...)
#   # outinds will always be the roots in union-find
#   outinds = noncommoninds(network...)

#   deltas = filter(t -> _is_delta(t), network)
#   inds_list = map(t -> collect(inds(t)), deltas)
#   deltainds = collect(Set(vcat(inds_list...)))
#   uf = UF(deltainds)
#   for t in deltas
#     i1, i2 = inds(t)
#     if root(uf, i1) in outinds
#       connect(uf, i2, i1)
#     else
#       connect(uf, i1, i2)
#     end
#   end
#   sim_dict = Dict([ind => root(uf, ind) for ind in deltainds])
#   for k in ks
#     net = tnets_dict[k]
#     net = setdiff(net, deltas)
#     tnets_dict[k] = replaceinds(net, sim_dict)
#     # @info "$(k), $(TreeTensor(net...))"
#   end
#   return tnets_dict
# end
