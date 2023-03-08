using Graphs, GraphsFlows, Combinatorics
using OMEinsumContractionOrders
using ITensorNetworks: contraction_sequence, TTN, IndsNetwork

function Base.show(io::IO, tensor::ITensor)
  return print(io, string(inds(tensor)))
end

include("tree_utils.jl")
include("index_group.jl")
include("index_adjacency_tree.jl")
include("approx_contract_cache.jl")

approximate_contract(tn::ITensor, inds_groups; kwargs...) = [tn], 0.0

function approximate_contract(tn::Vector{ITensor}, inds_btree=nothing; kwargs...)
  out, log_norm = approximate_contract(tn, inds_btree; kwargs...)
  return out, log_norm
end

function approximate_contract_ctree_to_tensor(
  tn::Vector{ITensor},
  inds_btree=nothing;
  cutoff,
  maxdim,
  ansatz="mps",
  algorithm="density_matrix",
)
  uncontract_inds = noncommoninds(tn...)
  allinds = collect(Set(mapreduce(t -> collect(inds(t)), vcat, tn)))
  innerinds = setdiff(allinds, uncontract_inds)
  if length(uncontract_inds) <= 2
    if inds_btree == nothing
      inds_btree = [[i] for i in uncontract_inds]
    end
    return Dict{Vector,ITensor}(inds_btree => _optcontract(tn)), 0.0
  end
  # # cases where tn is a tree, or contains 2 disconnected trees
  # if length(innerinds) <= length(tn) - 1
  #   # TODO
  #   return tn
  # end
  # # TODO: may want to remove this
  # if inds_groups != nothing
  #   deltainds = vcat(filter(g -> length(g) > 1, inds_groups)...)
  #   deltas, tnprime, _ = split_deltas(deltainds, tn)
  #   tn = Vector{ITensor}(vcat(deltas, tnprime))
  # end
  if inds_btree == nothing
    inds_btree = inds_binary_tree(tn, nothing; algorithm=ansatz)
  end
  inds_btree = _change_leaves_type(inds_btree)
  par = binary_tree_partition(ITensorNetwork(tn), inds_btree)
  inds_btree = _change_leaves_type_back(inds_btree)
  tn = vcat([Vector{ITensor}(par[v]) for v in vertices(par)]...)
  i2 = noncommoninds(tn...)
  @assert (length(uncontract_inds) == length(i2))
  @timeit_debug ITensors.timer "tree_approximation" begin
    return tree_approximation(
      par, inds_btree; cutoff=cutoff, maxdim=maxdim, algorithm=algorithm
    )
  end
end

function _change_leaves_type(btree)
  if length(btree) == 1
    return btree[1]
  end
  return [_change_leaves_type(btree[1]), _change_leaves_type(btree[2])]
end

function _change_leaves_type_back(btree)
  if !(btree isa Vector)
    return [btree]
  end
  return [_change_leaves_type_back(btree[1]), _change_leaves_type_back(btree[2])]
end

function uncontractinds(tn)
  if tn isa ITensor
    return inds(tn)
  else
    return noncommoninds(vectorize(tn)...)
  end
end

function get_ancestors(ctree)
  @timeit_debug ITensors.timer "get_ancestors" begin
    ctree_to_ancestors = Dict{Vector,Vector}()
    queue = [ctree]
    ctree_to_ancestors[ctree] = []
    while queue != []
      node = popfirst!(queue)
      if node isa Vector{ITensor}
        continue
      end
      for (i, child) in enumerate(node)
        queue = [queue..., child]
        ctree_to_ancestors[child] = [(i, node), ctree_to_ancestors[node]...]
      end
    end
    return ctree_to_ancestors
  end
end

function _approximate_contract_pre_process(tn_leaves, ctrees)
  @timeit_debug ITensors.timer "_approximate_contract_pre_process" begin
    # mapping each contraction tree to its uncontracted index groups
    ctree_to_igs = Dict{Vector,Vector{IndexGroup}}()
    index_groups = get_index_groups(ctrees[end])
    for c in vcat(tn_leaves, ctrees)
      ctree_to_igs[c] = neighbor_index_groups(c, index_groups)
    end
    ctree_to_ancestors = get_ancestors(ctrees[end])
    # mapping each contraction tree to its index adjacency tree
    ctree_to_adj_tree = Dict{Vector,IndexAdjacencyTree}()
    for leaf in tn_leaves
      ctree_to_adj_tree[leaf] = generate_adjacency_tree(
        leaf, ctree_to_ancestors[leaf], ctree_to_igs
      )
      minswap_adjacency_tree!(ctree_to_adj_tree[leaf])
    end
    for c in ctrees
      ancestors = ctree_to_ancestors[c]
      adj_tree = generate_adjacency_tree(c, ancestors, ctree_to_igs)
      if adj_tree != nothing
        ctree_to_adj_tree[c] = adj_tree
      end
    end
    # mapping each contraction tree to its contract igs
    ctree_to_contract_igs = Dict{Vector,Vector{IndexGroup}}()
    for c in ctrees
      contract_igs = intersect(ctree_to_igs[c[1]], ctree_to_igs[c[2]])
      ctree_to_contract_igs[c[1]] = contract_igs
      ctree_to_contract_igs[c[2]] = contract_igs
    end
    # special case when the network contains uncontracted inds
    ctree_to_contract_igs[ctrees[end]] = ctree_to_igs[ctrees[end]]
    # mapping each index group to a linear ordering
    ig_to_linear_order = Dict{IndexGroup,Vector}()
    for leaf in tn_leaves
      for ig in ctree_to_igs[leaf]
        if !haskey(ig_to_linear_order, ig)
          inds_order = _mps_partition_inds_order(ITensorNetwork(leaf), ig.data)
          ig_to_linear_order[ig] = [[i] for i in inds_order]
        end
      end
    end
    return ctree_to_igs, ctree_to_adj_tree, ctree_to_contract_igs, ig_to_linear_order
  end
end

function ordered_igs_to_binary_tree(ordered_igs, contract_igs, ig_to_linear_order; ansatz)
  @assert ansatz in ["comb", "mps"]
  @timeit_debug ITensors.timer "ordered_igs_to_binary_tree" begin
    @assert contract_igs != []
    left_igs, right_igs = split_igs(ordered_igs, contract_igs)
    if ansatz == "comb"
      return ordered_igs_to_binary_tree_comb(
        left_igs, right_igs, contract_igs, ig_to_linear_order
      )
    elseif ansatz == "mps"
      return ordered_igs_to_binary_tree_mps(
        left_igs, right_igs, contract_igs, ig_to_linear_order
      )
    end
  end
end

function ordered_igs_to_binary_tree(igs, ig_to_linear_order; ansatz)
  @assert ansatz in ["comb", "mps"]
  if ansatz == "comb"
    return line_to_tree([line_to_tree(ig_to_linear_order[ig]) for ig in igs])
  end
  order = vcat([ig_to_linear_order[ig] for ig in igs]...)
  return line_to_tree(order)
end

function ordered_igs_to_binary_tree_mps(
  left_igs, right_igs, contract_igs, ig_to_linear_order
)
  left_order = get_leaves([ig_to_linear_order[ig] for ig in left_igs])
  right_order = get_leaves([ig_to_linear_order[ig] for ig in right_igs])
  contract_order = get_leaves([ig_to_linear_order[ig] for ig in contract_igs])
  if length(left_order) <= length(right_order)
    left_order = [left_order..., contract_order...]
  else
    right_order = [contract_order..., right_order...]
  end
  return merge_tree(line_to_tree(left_order), line_to_tree(reverse(right_order)))
end

function ordered_igs_to_binary_tree_comb(
  left_igs, right_igs, contract_igs, ig_to_linear_order
)
  tree_1 = ordered_igs_to_binary_tree(left_igs, ig_to_linear_order; ansatz="comb")
  tree_contract = ordered_igs_to_binary_tree(
    contract_igs, ig_to_linear_order; ansatz="comb"
  )
  tree_2 = ordered_igs_to_binary_tree(reverse(right_igs), ig_to_linear_order; ansatz="comb")
  # make the binary tree more balanced to save tree approximation cost
  if tree_1 == []
    return merge_tree(merge_tree(tree_1, tree_contract), tree_2)
  end
  if tree_2 == []
    return merge_tree(tree_1, merge_tree(tree_contract, tree_2))
  end
  if length(vectorize(tree_1)) <= length(vectorize(tree_2))
    return merge_tree(merge_tree(tree_1, tree_contract), tree_2)
  else
    return merge_tree(tree_1, merge_tree(tree_contract, tree_2))
  end
end

function update_tn_tree_keys!(tn_tree, inds_btree, pairs::Vector{Pair})
  @timeit_debug ITensors.timer "update_tn_tree_keys!" begin
    current_to_update_key = Dict{Vector,Vector}(pairs...)
    nodes = topo_sort(inds_btree; type=Vector{<:Vector})
    for n in nodes
      @assert haskey(tn_tree, n)
      new_key = n
      if haskey(current_to_update_key, n[1])
        new_key = [current_to_update_key[n[1]], n[2]]
      end
      if haskey(current_to_update_key, n[2])
        new_key = [new_key[1], current_to_update_key[n[2]]]
      end
      if new_key != n
        tn_tree[new_key] = tn_tree[n]
        delete!(tn_tree, n)
        current_to_update_key[n] = new_key
      end
    end
  end
end

function get_child_tn(ctree_to_tn_tree, ctree::Vector)
  if !haskey(ctree_to_tn_tree, ctree)
    @assert ctree isa Vector{ITensor}
    return ctree
  elseif ctree_to_tn_tree[ctree] isa Vector{ITensor}
    return ctree_to_tn_tree[ctree]
  else
    return vcat(collect(values(ctree_to_tn_tree[ctree]))...)
  end
end

_index_less(a::Index, b::Index) = tags(a)[1] < tags(b)[1]

function _replaceinds(t1::ITensor, t2::ITensor)
  inds1 = sort(inds(t1); lt=_index_less)
  inds2 = sort(inds(t2); lt=_index_less)
  inds1_tags = [tags(i) for i in inds1]
  inds2_tags = [tags(i) for i in inds2]
  @assert inds1_tags == inds2_tags
  return replaceinds(t1, inds1, inds2)
end

function orthogonalize!(ctree_to_tn_tree::Dict, environments::Vector, c::Vector)
  @timeit_debug ITensors.timer "orthogonalize" begin
    index = []
    if c[1] in environments
      push!(index, 1)
    elseif c[2] in environments
      push!(index, 2)
    end
    if index == []
      return nothing
    end
    environments = setdiff(environments, c)
    if length(environments) == 0
      return nothing
    end
    @info "start orthogonalize with env size", length(environments)
    network = vcat([get_child_tn(ctree_to_tn_tree, env) for env in environments]...)
    env_boundary = get_child_tn(ctree_to_tn_tree, c[index[1]])
    source_tensor = env_boundary[1]
    @assert !(source_tensor in network)
    push!(network, source_tensor)
    ctree_to_tn_tree[c[index[1]]] = env_boundary[2:end]
    orth_tn = orthogonalize(ITensorNetwork(network), length(network))
    tensor_to_ortho_tensor = Dict{ITensor,ITensor}()
    for i in 1:length(network)
      new_tensor = _replaceinds(orth_tn[i], network[i])
      tensor_to_ortho_tensor[network[i]] = new_tensor
    end
    for env in environments
      ortho_tensors = Vector{ITensor}([
        tensor_to_ortho_tensor[t] for t in get_child_tn(ctree_to_tn_tree, env)
      ])
      ctree_to_tn_tree[env] = ortho_tensors
    end
    push!(ctree_to_tn_tree[c[index[1]]], tensor_to_ortho_tensor[source_tensor])
  end
end

# ctree: contraction tree
# tn: vector of tensors representing a tensor network
# tn_tree: a dict maps each index tree in the tn to a tensor
# adj_tree: index adjacency tree
# ig: index group
# contract_ig: the index group to be contracted next
# ig_tree: an index group with a tree hierarchy 
function approximate_contract(
  ctree::Vector;
  cutoff,
  maxdim,
  ansatz="mps",
  use_cache=true,
  orthogonalize=false,
  algorithm="density_matrix",
  swap_size=4,
)
  @timeit_debug ITensors.timer "approximate_contract" begin
    tn_leaves = get_leaves(ctree)
    environments = tn_leaves
    ctrees = topo_sort(ctree; leaves=tn_leaves)
    ctree_to_igs, ctree_to_adj_tree, ctree_to_contract_igs, ig_to_linear_order = _approximate_contract_pre_process(
      tn_leaves, ctrees
    )
    # mapping each contraction tree to a tensor network
    ctree_to_tn_tree = Dict{Vector,Union{Dict{Vector,ITensor},Vector{ITensor}}}()
    # accumulate norm
    log_accumulated_norm = 0.0
    for (ii, c) in enumerate(ctrees)
      @info "orthogonalize", orthogonalize
      if orthogonalize == true
        orthogonalize!(ctree_to_tn_tree, environments, c)
        environments = setdiff(environments, c)
      end
      @info ii, "th tree approximation"
      if ctree_to_igs[c] == []
        @assert c == ctrees[end]
        tn1 = get_child_tn(ctree_to_tn_tree, c[1])
        tn2 = get_child_tn(ctree_to_tn_tree, c[2])
        tn = vcat(tn1, tn2)
        return [_optcontract(tn)], log_accumulated_norm
      end
      tree_noswap, ctree_to_adj_tree[c] = minswap_adjacency_tree(
        ctree_to_adj_tree[c], ctree_to_adj_tree[c[1]], ctree_to_adj_tree[c[2]]
      )
      @info "target_tree", ctree_to_adj_tree[c]
      @info "tree_noswap", tree_noswap
      # caching is not used here
      if use_cache == false
        tn1 = get_child_tn(ctree_to_tn_tree, c[1])
        tn2 = get_child_tn(ctree_to_tn_tree, c[2])
        # TODO: change new_igs into a vector of igs
        inds_btree = ordered_igs_to_binary_tree(
          ctree_to_adj_tree[c].children,
          ctree_to_contract_igs[c],
          ig_to_linear_order;
          ansatz=ansatz,
        )
        ctree_to_tn_tree[c], log_root_norm = approximate_contract_ctree_to_tensor(
          [tn1..., tn2...], inds_btree; cutoff=cutoff, maxdim=maxdim, algorithm=algorithm
        )
        log_accumulated_norm += log_root_norm
        continue
      end
      # caching
      # Note: cache_igs_right has a reversed ordering
      center_igs, cache_igs_left, cache_igs_right = get_igs_cache_info(
        [ctree_to_adj_tree[i].children for i in [c, c[1], c[2]]],
        [ctree_to_contract_igs[i] for i in [c, c[1], c[2]]],
      )
      # it's possible that `tree_noswap` has different boundaries than
      # `ctree_to_adj_tree[c]`
      if !_has_boundary(tree_noswap.children, cache_igs_left, cache_igs_right)
        interpolate_igs = _interpolate(
          tree_noswap.children, ctree_to_adj_tree[c].children; size=swap_size
        )
        tn1 = get_child_tn(ctree_to_tn_tree, c[1])
        tn2 = get_child_tn(ctree_to_tn_tree, c[2])
        uncached_tn = [tn1..., tn2...]
        for (jj, c_igs) in enumerate(interpolate_igs)
          @info "approximate contract with no cache", jj, c_igs, length(c_igs)
          if jj == length(interpolate_igs)
            inds_btree = ordered_igs_to_binary_tree(
              c_igs, ctree_to_contract_igs[c], ig_to_linear_order; ansatz
            )
          else
            inds_btree = ordered_igs_to_binary_tree(c_igs, ig_to_linear_order; ansatz)
          end
          new_tn_tree, log_root_norm = approximate_contract_ctree_to_tensor(
            uncached_tn, inds_btree; cutoff=cutoff, maxdim=maxdim, algorithm=algorithm
          )
          log_accumulated_norm += log_root_norm
          uncached_tn = vcat(collect(values(new_tn_tree))...)
        end
        ctree_to_tn_tree[c] = new_tn_tree
      else
        noswap_center_igs = _get_center_igs(
          tree_noswap.children, cache_igs_left, cache_igs_right
        )
        interpolate_center_igs = _interpolate(noswap_center_igs, center_igs; size=swap_size)
        @info "interpolate_center_igs has size", length(interpolate_center_igs)
        @assert interpolate_center_igs[end] == center_igs
        cached_tn_tree, uncached_tn, new_ig_left, new_ig_right = get_tn_cache_info(
          ctree_to_tn_tree,
          c[1],
          c[2],
          cache_igs_left,
          cache_igs_right,
          ig_to_linear_order;
          ansatz,
        )
        if new_ig_right == nothing && new_ig_left == nothing
          @info "Caching is not used in this approximation"
          @assert length(cached_tn_tree) == 0
        else
          @info "Caching is used in this approximation", new_ig_left, new_ig_right
        end
        new_ig_to_linear_order = _cached_ig_to_linear_order(
          new_ig_left, new_ig_right, ig_to_linear_order
        )
        new_index_to_btree = _cached_new_index_to_btree(
          cache_igs_left,
          cache_igs_right,
          new_ig_left,
          new_ig_right,
          ig_to_linear_order;
          ansatz,
        )
        new_tn_tree = nothing
        for (jj, c_igs) in enumerate(interpolate_center_igs)
          contract_igs = _cached_ig_order(
            c_igs, cache_igs_left, cache_igs_right, new_ig_left, new_ig_right
          )
          @info "approximate contract", jj
          if jj == length(interpolate_center_igs)
            inds_btree = ordered_igs_to_binary_tree(
              contract_igs, ctree_to_contract_igs[c], new_ig_to_linear_order; ansatz
            )
          else
            inds_btree = ordered_igs_to_binary_tree(
              contract_igs, new_ig_to_linear_order; ansatz
            )
          end
          # @info "start approximate_contract_ctree_to_tensor"
          new_tn_tree, log_root_norm = approximate_contract_ctree_to_tensor(
            uncached_tn, inds_btree; cutoff=cutoff, maxdim=maxdim, algorithm=algorithm
          )
          log_accumulated_norm += log_root_norm
          uncached_tn = vcat(collect(values(new_tn_tree))...)
        end
        if length(new_index_to_btree) != 0
          update_tn_tree_keys!(new_tn_tree, inds_btree, new_index_to_btree)
        end
        ctree_to_tn_tree[c] = merge(new_tn_tree, cached_tn_tree)
      end
      # release the memory
      delete!(ctree_to_tn_tree, c[1])
      delete!(ctree_to_tn_tree, c[2])
    end
    tn = vcat(collect(values(ctree_to_tn_tree[ctrees[end]]))...)
    return tn, log_accumulated_norm
  end
end

function tree_approximation(
  par::DataGraph, inds_btree::Vector; cutoff=1e-15, maxdim=10000, algorithm="density_matrix"
)
  @assert algorithm in ["density_matrix", "density_matrix_contract_first", "svd"]
  if algorithm == "density_matrix"
    tn, log_norm = _approx_binary_tree_itensornetwork(par; cutoff=cutoff, maxdim=maxdim)
    _rem_leaf_vertices!(tn; root=1)
    ctree_to_tensor = Dict{Vector,ITensor}()
    iii = 1
    for node in PreOrderDFS(inds_btree)
      if !(node isa Vector)
        continue
      end
      if length(node) == 1
        iii += 1
        continue
      end
      ctree_to_tensor[node] = tn[iii]
      iii += 1
    end
    return ctree_to_tensor, log_norm
  end
  # if algorithm == "density_matrix_contract_first"
  #   @info "density_matrix_contract_first"
  #   btree_to_contracted_tn = Dict{Vector,Vector{ITensor}}()
  #   for (btree, ts) in embedding
  #     btree_to_contracted_tn[btree] = [optcontract(ts)]
  #   end
  #   return tree_approximation_density_matrix(
  #     btree_to_contracted_tn, inds_btree; cutoff=cutoff, maxdim=maxdim
  #   )
  # end
  # if algorithm == "svd"
  #   return tree_approximation_svd(
  #     embedding, inds_btree; cutoff=cutoff, maxdim=maxdim
  #   )
  # end
end

# function tree_approximation_svd(
#   embedding::Dict, inds_btree::Vector; cutoff=1e-15, maxdim=10000
# )
#   @info "start tree_approximation_svd", inds_btree
#   @info "cutoff", cutoff, "maxdim", maxdim
#   network = Vector{ITensor}()
#   btree_to_order = Dict{Vector,Int}()
#   root_vertex = nothing
#   for (btree, ts) in embedding
#     # use dense to convert Diag type to dense for QR decomposition TODO: raise an error in ITensors
#     push!(network, dense(optcontract(ts)))
#     btree_to_order[btree] = length(network)
#     if btree == inds_btree
#       root_vertex = length(network)
#     end
#   end
#   @assert root_vertex != nothing
#   ttn = TTN(ITensorNetwork(network))
#   @timeit_debug ITensors.timer "truncate" begin
#     truncate_ttn = truncate(ttn; cutoff=cutoff, maxdim=maxdim, root_vertex=root_vertex)
#   end
#   out_network = [truncate_ttn[i] for i in 1:length(network)]
#   inds1 = sort(uncontractinds(out_network); lt=_index_less)
#   inds2 = sort(uncontractinds(network); lt=_index_less)
#   out_network = replaceinds(out_network, Dict(zip(inds1, inds2)))
#   root_norm = norm(out_network[root_vertex])
#   out_network[root_vertex] /= root_norm
#   ctree_to_tensor = Dict{Vector,ITensor}()
#   for node in topo_sort(inds_btree; type=Vector{<:Vector})
#     children_tensors = []
#     if !(node[1] isa Vector{<:Vector})
#       push!(children_tensors, out_network[btree_to_order[node[1]]])
#     end
#     if !(node[2] isa Vector{<:Vector})
#       push!(children_tensors, out_network[btree_to_order[node[2]]])
#     end
#     t = out_network[btree_to_order[node]]
#     if children_tensors == []
#       ctree_to_tensor[node] = t
#     else
#       ctree_to_tensor[node] = optcontract([t, children_tensors...])
#     end
#   end
#   return ctree_to_tensor, log(root_norm)
# end
