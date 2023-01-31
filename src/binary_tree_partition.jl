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

function mincut_subnetwork_insert_deltas(
  network::Vector{ITensor}, source_inds::Vector{<:Index}
)
  @info "mincut_subnetwork_insert_deltas", source_inds, length(network)
  out_inds = noncommoninds(network...)
  deltas, networkprime, _ = split_deltas(noncommoninds(network...), network)
  network = Vector{ITensor}(vcat(deltas, networkprime))
  @info length(network)
  @info source_inds, setdiff(out_inds, source_inds)
  p1, _ = _mincut_partition_maxweightoutinds(ITensorNetwork(network), source_inds, setdiff(out_inds, source_inds))
  @info "p1 is", p1
  source_subnetwork = [network[i] for i in p1]
  remain_network = setdiff(network, source_subnetwork)
  source_subnetwork = simplify_deltas(source_subnetwork)
  remain_network = simplify_deltas(remain_network)
  @assert (
    length(noncommoninds(network...)) ==
    length(noncommoninds(source_subnetwork..., remain_network...))
  )
  return source_subnetwork, remain_network
end

function binary_tree_partition(network::Vector{ITensor}, inds_btree::Vector; algorithm)
  btree_to_output_tn = Dict{Union{Vector,Index},Vector{ITensor}}()
  btree_to_input_tn = Dict{Union{Vector,Index},Vector{ITensor}}()
  btree_to_input_tn[inds_btree] = network
  for node in collect(PreOrderDFS(inds_btree))
    @info "node is", node
    @assert haskey(btree_to_input_tn, node)
    input_tn = btree_to_input_tn[node]
    # @info "node", node
    if node isa Index
      btree_to_output_tn[node] = input_tn
      continue
    end
    net1, input_tn = mincut_subnetwork_insert_deltas(input_tn, collect(Leaves(node[1])))
    btree_to_input_tn[node[1]] = net1
    net1, input_tn = mincut_subnetwork_insert_deltas(input_tn, collect(Leaves(node[2])))
    btree_to_input_tn[node[2]] = net1
    btree_to_output_tn[node] = input_tn
    # @info "btree_to_output_tn[node]", btree_to_output_tn[node]
  end
  if algorithm == "svd"
    return btree_to_output_tn
  else
    return remove_deltas(btree_to_output_tn)
  end
end

is_delta(t) = (t.tensor.storage.data == 1.0)

function simplify_deltas(network::Vector{ITensor})
  @info "simplify_deltas", network
  out_delta_inds = Vector{Pair}()
  # outinds will always be the roots in union-find
  outinds = noncommoninds(network...)
  deltas = filter(t -> is_delta(t), network)
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
  network = setdiff(network, deltas)
  network = replaceinds(network, sim_dict)
  out_delta = [delta(i.first, i.second) for i in out_delta_inds]
  return Vector{ITensor}([network..., out_delta...])
end

# remove deltas to improve the performance
function remove_deltas(tnets_dict::Dict)
  # only remove deltas in intermediate nodes
  ks = filter(k -> (length(k) > 1), collect(keys(tnets_dict)))
  network = vcat([tnets_dict[k] for k in ks]...)
  # outinds will always be the roots in union-find
  outinds = noncommoninds(network...)

  deltas = filter(t -> is_delta(t), network)
  inds_list = map(t -> collect(inds(t)), deltas)
  deltainds = collect(Set(vcat(inds_list...)))
  uf = UF(deltainds)
  for t in deltas
    i1, i2 = inds(t)
    if root(uf, i1) in outinds
      connect(uf, i2, i1)
    else
      connect(uf, i1, i2)
    end
  end
  sim_dict = Dict([ind => root(uf, ind) for ind in deltainds])
  for k in ks
    net = tnets_dict[k]
    net = setdiff(net, deltas)
    tnets_dict[k] = replaceinds(net, sim_dict)
    # @info "$(k), $(TreeTensor(net...))"
  end
  return tnets_dict
end

function split_deltas(inds, subnet)
  sim_dict = Dict([ind => sim(ind) for ind in inds])
  deltas = [delta(i, sim_dict[i]) for i in inds]
  subnet = replaceinds(subnet, sim_dict)
  return deltas, subnet, collect(values(sim_dict))
end

function ITensors.replaceinds(network::Array{ITensor}, sim_dict::Dict)
  if length(network) == 0
    return network
  end
  indices = collect(keys(sim_dict))
  function siminds(tensor)
    sim_inds = [ind for ind in inds(tensor) if ind in indices]
    if (length(sim_inds) == 0)
      return tensor
    end
    outinds = map(i -> sim_dict[i], sim_inds)
    return replaceinds(tensor, sim_inds => outinds)
  end
  return map(x -> siminds(x), network)
end