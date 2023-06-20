mutable struct IndexGroup
  data::Vector
end

function Base.show(io::IO, ig::IndexGroup)
  return print(io, ig.data[1].tags)
end

# TODO: general tags are not comparable
Base.isless(a::Index, b::Index) = id(a) < id(b) || (id(a) == id(b) && plev(a) < plev(b)) # && tags(a) < tags(b)

function IndexGroup(indices::Vector{<:Index})
  return IndexGroup(sort(indices), false)
end

function get_index_groups(tn_tree::Vector)
  @timeit_debug ITensors.timer "get_index_groups" begin
    tn_leaves = get_leaves(tn_tree)
    tn = vcat(tn_leaves...)
    uncontract_inds = noncommoninds(tn...)
    igs = []
    for leaf in tn_leaves
      inds = intersect(noncommoninds(leaf...), uncontract_inds)
      if length(inds) >= 1
        push!(igs, IndexGroup(inds))
      end
    end
    for (t1, t2) in powerset(tn_leaves, 2, 2)
      inds = intersect(noncommoninds(t1...), noncommoninds(t2...))
      if length(inds) >= 1
        push!(igs, IndexGroup(inds))
      end
    end
    return igs
  end
end

function neighbor_index_groups(contraction, index_groups)
  @timeit_debug ITensors.timer "get_index_groups" begin
    inds = noncommoninds(vectorize(contraction)...)
    nigs = []
    for ig in index_groups
      if issubset(ig.data, inds)
        push!(nigs, ig)
      end
    end
    return nigs
  end
end

function split_igs(igs::Vector{IndexGroup}, inter_igs::Vector{IndexGroup})
  igs_left = Vector{IndexGroup}()
  igs_right = Vector{IndexGroup}()
  target_array = igs_left
  for i in igs
    if i in inter_igs
      target_array = igs_right
      continue
    end
    push!(target_array, i)
  end
  return igs_left, igs_right
end

function _mps_partition_inds_set_order(tn::ITensorNetwork, igs::Vector{IndexGroup})
  tn = ITensorNetwork{Any}(tn)
  newinds = Vector{Index}()
  ind_to_ig = Dict{Index,IndexGroup}()
  for (i, ig) in enumerate(igs)
    outinds = ig.data
    new_ind = Index(dim(outinds), "temp" * string(i))
    push!(newinds, new_ind)
    ind_to_ig[new_ind] = ig
    new_t = ITensor(outinds..., new_ind)
    add_vertex!(tn, ("temp", i))
    tn[("temp", i)] = new_t
  end
  inds_order = _mps_partition_inds_order(tn, newinds)
  return [ind_to_ig[i] for i in inds_order]
end
