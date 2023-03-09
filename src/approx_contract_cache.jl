function get_igs_cache_info(igs_list, contract_igs_list)
  function split_boundary(list1::Vector{IndexGroup}, list2::Vector{IndexGroup})
    index = 1
    boundary = Vector{IndexGroup}()
    while list1[index] == list2[index]
      push!(boundary, list2[index])
      index += 1
      if index > length(list1) || index > length(list2)
        break
      end
    end
    if index <= length(list1)
      remain_list1 = list1[index:end]
    else
      remain_list1 = Vector{IndexGroup}()
    end
    return boundary, remain_list1
  end
  function split_boundary(igs::Vector{IndexGroup}, lists::Vector{Vector{IndexGroup}})
    if length(igs) <= 1
      return Vector{IndexGroup}(), igs
    end
    for l in lists
      if length(l) >= 2 && igs[1] == l[1] && igs[2] == l[2]
        return split_boundary(igs, l)
      end
    end
    return Vector{IndexGroup}(), igs
  end
  @timeit_debug ITensors.timer "get_igs_cache_info" begin
    out, input1, input2 = igs_list
    contract_out, contract_input1, contract_input2 = contract_igs_list
    out_left, out_right = split_igs(out, contract_out)
    out_right = reverse(out_right)
    input1_left, input1_right = split_igs(input1, contract_input1)
    input2_left, input2_right = split_igs(input2, contract_input2)
    inputs = [input1_left, reverse(input1_right), input2_left, reverse(input2_right)]
    boundary_left, remain_left = split_boundary(out_left, inputs)
    boundary_right, remain_right = split_boundary(out_right, inputs)
    return [remain_left..., contract_out..., reverse(remain_right)...],
    boundary_left,
    boundary_right
  end
end

function get_tn_cache_sub_info(tn_tree::Dict{Vector,ITensor}, cache_binary_trees::Vector)
  cached_tn = []
  cached_tn_tree = Dict{Vector,ITensor}()
  new_igs = []
  for binary_tree in cache_binary_trees
    if binary_tree == [] || !haskey(tn_tree, binary_tree)
      push!(new_igs, nothing)
    else
      binary_tree = Vector{Vector}(binary_tree)
      nodes = topo_sort(binary_tree; type=Vector{<:Vector})
      sub_tn = [tn_tree[n] for n in nodes]
      sub_tn_tree = Dict([n => tn_tree[n] for n in nodes]...)
      index_leaves = vectorize(binary_tree)
      new_indices = setdiff(noncommoninds(sub_tn...), index_leaves)
      @assert length(new_indices) == 1
      new_indices = Vector{<:Index}(new_indices)
      push!(new_igs, IndexGroup(new_indices))
      cached_tn = vcat(cached_tn, sub_tn)
      cached_tn_tree = merge(cached_tn_tree, sub_tn_tree)
    end
  end
  tn = vcat(collect(values(tn_tree))...)
  uncached_tn = setdiff(tn, cached_tn)
  return cached_tn_tree, uncached_tn, new_igs
end

function get_tn_cache_info(
  ctree_to_tn_tree,
  ctree_1::Vector,
  ctree_2::Vector,
  cache_igs_left,
  cache_igs_right,
  ig_to_linear_order;
  ansatz,
)
  @assert ansatz in ["comb", "mps"]
  cache_binary_tree_left = ordered_igs_to_binary_tree(
    cache_igs_left, ig_to_linear_order; ansatz=ansatz, direction="left"
  )
  cache_binary_tree_right = ordered_igs_to_binary_tree(
    cache_igs_right, ig_to_linear_order; ansatz=ansatz, direction="right"
  )
  cache_binary_trees = [cache_binary_tree_left, cache_binary_tree_right]
  @timeit_debug ITensors.timer "get_tn_cache_info" begin
    if haskey(ctree_to_tn_tree, ctree_1) && ctree_to_tn_tree[ctree_1] isa Dict
      tn_tree_1 = ctree_to_tn_tree[ctree_1]
      cached_tn_tree1, uncached_tn1, new_igs_1 = get_tn_cache_sub_info(
        tn_tree_1, cache_binary_trees
      )
    else
      cached_tn_tree1 = Dict{Vector,ITensor}()
      uncached_tn1 = get_child_tn(ctree_to_tn_tree, ctree_1)
      new_igs_1 = [nothing, nothing]
    end
    if haskey(ctree_to_tn_tree, ctree_2) && ctree_to_tn_tree[ctree_2] isa Dict
      tn_tree_2 = ctree_to_tn_tree[ctree_2]
      cached_tn_tree2, uncached_tn2, new_igs_2 = get_tn_cache_sub_info(
        tn_tree_2, cache_binary_trees
      )
    else
      cached_tn_tree2 = Dict{Vector,ITensor}()
      uncached_tn2 = get_child_tn(ctree_to_tn_tree, ctree_2)
      new_igs_2 = [nothing, nothing]
    end
    uncached_tn = [uncached_tn1..., uncached_tn2...]
    new_igs_left = [i for i in [new_igs_1[1], new_igs_2[1]] if i != nothing]
    @assert length(new_igs_left) <= 1
    if length(new_igs_left) == 1
      new_ig_left = new_igs_left[1]
    else
      new_ig_left = nothing
    end
    new_igs_right = [i for i in [new_igs_1[2], new_igs_2[2]] if i != nothing]
    @assert length(new_igs_right) <= 1
    if length(new_igs_right) == 1
      new_ig_right = new_igs_right[1]
    else
      new_ig_right = nothing
    end
    return merge(cached_tn_tree1, cached_tn_tree2), uncached_tn, new_ig_left, new_ig_right
  end
end

function _cached_ig_order(
  center_igs, cache_igs_left, cache_igs_right, new_ig_left, new_ig_right
)
  new_igs = center_igs
  if new_ig_left == nothing
    new_igs = [cache_igs_left..., new_igs...]
  else
    new_igs = [new_ig_left, new_igs...]
  end
  if new_ig_right == nothing
    new_igs = [new_igs..., cache_igs_right...]
  else
    new_igs = [new_igs..., new_ig_right]
  end
  return new_igs
end

function _cached_ig_to_linear_order(new_ig_left, new_ig_right, ig_to_linear_order)
  new_ig_to_linear_order = ig_to_linear_order
  if new_ig_left != nothing
    new_ig_to_linear_order = merge(
      new_ig_to_linear_order, Dict(new_ig_left => [new_ig_left.data])
    )
  end
  if new_ig_right != nothing
    new_ig_to_linear_order = merge(
      new_ig_to_linear_order, Dict(new_ig_right => [new_ig_right.data])
    )
  end
  return new_ig_to_linear_order
end

function _cached_new_index_to_btree(
  cache_igs_left, cache_igs_right, new_ig_left, new_ig_right, ig_to_linear_order; ansatz
)
  @assert ansatz in ["comb", "mps"]
  cache_binary_tree_left = ordered_igs_to_binary_tree(
    cache_igs_left, ig_to_linear_order; ansatz=ansatz, direction="left"
  )
  cache_binary_tree_right = ordered_igs_to_binary_tree(
    cache_igs_right, ig_to_linear_order; ansatz=ansatz, direction="right"
  )
  new_index_to_btree = Vector{Pair}()
  if new_ig_left != nothing
    push!(new_index_to_btree, new_ig_left.data => cache_binary_tree_left)
  end
  if new_ig_right != nothing
    push!(new_index_to_btree, new_ig_right.data => cache_binary_tree_right)
  end
  return new_index_to_btree
end

function _has_boundary(igs, igs_left, igs_right)
  return igs[1:length(igs_left)] == igs_left &&
         reverse(igs)[1:length(igs_right)] == igs_right
end

function _get_center_igs(igs, igs_left, igs_right)
  @assert _has_boundary(igs, igs_left, igs_right)
  return Vector{IndexGroup}(setdiff(igs, [igs_left..., igs_right...]))
end
