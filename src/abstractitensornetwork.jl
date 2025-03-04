abstract type AbstractITensorNetwork{V} <: AbstractDataGraph{V,ITensor,ITensor} end

# Field access
data_graph_type(::Type{<:AbstractITensorNetwork}) = not_implemented()
data_graph(graph::AbstractITensorNetwork) = not_implemented()

# TODO: Define a generic fallback for `AbstractDataGraph`?
edge_data_type(::Type{<:AbstractITensorNetwork}) = ITensor

# Graphs.jl overloads
function weights(graph::AbstractITensorNetwork)
  V = vertextype(graph)
  es = Tuple.(edges(graph))
  ws = Dictionary{Tuple{V,V},Float64}(es, undef)
  for e in edges(graph)
    w = log2(dim(commoninds(graph, e)))
    ws[(src(e), dst(e))] = w
  end
  return ws
end

# Copy
copy(tn::AbstractITensorNetwork) = not_implemented()

# Overload if needed
is_directed(::Type{<:AbstractITensorNetwork}) = false

# Derived interface, may need to be overloaded
function underlying_graph_type(G::Type{<:AbstractITensorNetwork})
  return underlying_graph_type(data_graph_type(G))
end

# AbstractDataGraphs overloads
function vertex_data(graph::AbstractITensorNetwork, args...)
  return vertex_data(data_graph(graph), args...)
end
edge_data(graph::AbstractITensorNetwork, args...) = edge_data(data_graph(graph), args...)

underlying_graph(tn::AbstractITensorNetwork) = underlying_graph(data_graph(tn))
function vertex_to_parent_vertex(tn::AbstractITensorNetwork, vertex)
  return vertex_to_parent_vertex(underlying_graph(tn), vertex)
end

#
# Iteration
#

# TODO: iteration

# TODO: different `map` functionalities as defined for ITensors.AbstractMPS

# TODO: broadcasting

function union(tn1::AbstractITensorNetwork, tn2::AbstractITensorNetwork; kwargs...)
  tn = ITensorNetwork(union(data_graph(tn1), data_graph(tn2)); kwargs...)
  # Add any new edges that are introduced during the union
  for v1 in vertices(tn1)
    for v2 in vertices(tn2)
      if hascommoninds(tn[v1], tn[v2])
        add_edge!(tn, v1 => v2)
      end
    end
  end
  return tn
end

function rename_vertices(f::Function, tn::AbstractITensorNetwork)
  return ITensorNetwork(rename_vertices(f, data_graph(tn)))
end

#
# Data modification
#

function setindex_preserve_graph!(tn::AbstractITensorNetwork, value, vertex)
  data_graph(tn)[vertex] = value
  return tn
end

function hascommoninds(tn::AbstractITensorNetwork, edge::Pair)
  return hascommoninds(tn, edgetype(tn)(edge))
end

function hascommoninds(tn::AbstractITensorNetwork, edge::AbstractEdge)
  return hascommoninds(tn[src(edge)], tn[dst(edge)])
end

function setindex!(tn::AbstractITensorNetwork, value, v)
  # v = to_vertex(tn, index...)
  setindex_preserve_graph!(tn, value, v)
  for edge in incident_edges(tn, v)
    rem_edge!(tn, edge)
  end
  for vertex in vertices(tn)
    if v ≠ vertex
      edge = v => vertex
      if hascommoninds(tn, edge)
        add_edge!(tn, edge)
      end
    end
  end
  return tn
end

# Convert to a collection of ITensors (`Vector{ITensor}`).
function Vector{ITensor}(tn::AbstractITensorNetwork)
  return [tn[v] for v in vertices(tn)]
end

# Convenience wrapper
itensors(tn::AbstractITensorNetwork) = Vector{ITensor}(tn)

#
# Promotion and conversion
#

function LinearAlgebra.promote_leaf_eltypes(tn::AbstractITensorNetwork)
  return LinearAlgebra.promote_leaf_eltypes(itensors(tn))
end

function ITensors.promote_itensor_eltype(tn::AbstractITensorNetwork)
  return LinearAlgebra.promote_leaf_eltypes(tn)
end

ITensors.scalartype(tn::AbstractITensorNetwork) = LinearAlgebra.promote_leaf_eltypes(tn)

# TODO: eltype(::AbstractITensorNetwork) (cannot behave the same as eltype(::ITensors.AbstractMPS))

# TODO: mimic ITensors.AbstractMPS implementation using map
function ITensors.convert_leaf_eltype(eltype::Type, tn::AbstractITensorNetwork)
  tn = copy(tn)
  vertex_data(tn) .= convert_eltype.(Ref(eltype), vertex_data(tn))
  return tn
end

# TODO: mimic ITensors.AbstractMPS implementation using map
function NDTensors.convert_scalartype(eltype::Type{<:Number}, tn::AbstractITensorNetwork)
  tn = copy(tn)
  vertex_data(tn) .= ITensors.adapt.(Ref(eltype), vertex_data(tn))
  return tn
end

#
# Conversion to Graphs
#

function Graph(tn::AbstractITensorNetwork)
  return Graph(Vector{ITensor}(tn))
end

function NamedGraph(tn::AbstractITensorNetwork)
  return NamedGraph(Vector{ITensor}(tn))
end

#
# Conversion to IndsNetwork
#

# Convert to an IndsNetwork
function IndsNetwork(tn::AbstractITensorNetwork)
  is = IndsNetwork(underlying_graph(tn))
  for v in vertices(tn)
    is[v] = uniqueinds(tn, v)
  end
  for e in edges(tn)
    is[e] = commoninds(tn, e)
  end
  return is
end

function siteinds(tn::AbstractITensorNetwork)
  is = IndsNetwork(underlying_graph(tn))
  for v in vertices(tn)
    is[v] = uniqueinds(tn, v)
  end
  return is
end

function linkinds(tn::AbstractITensorNetwork)
  is = IndsNetwork(underlying_graph(tn))
  for e in edges(tn)
    is[e] = commoninds(tn, e)
  end
  return is
end

#
# Index access
#

function neighbor_itensors(tn::AbstractITensorNetwork, vertex)
  return [tn[vn] for vn in neighbors(tn, vertex)]
end

function uniqueinds(tn::AbstractITensorNetwork, vertex)
  return uniqueinds(tn[vertex], neighbor_itensors(tn, vertex)...)
end

function uniqueinds(tn::AbstractITensorNetwork, edge::AbstractEdge)
  return uniqueinds(tn[src(edge)], tn[dst(edge)])
end

function uniqueinds(tn::AbstractITensorNetwork, edge::Pair)
  return uniqueinds(tn, edgetype(tn)(edge))
end

function siteinds(tn::AbstractITensorNetwork, vertex)
  return uniqueinds(tn, vertex)
end

function commoninds(tn::AbstractITensorNetwork, edge)
  e = edgetype(tn)(edge)
  return commoninds(tn[src(e)], tn[dst(e)])
end

function linkinds(tn::AbstractITensorNetwork, edge)
  return commoninds(tn, edge)
end

# Priming and tagging (changing Index identifiers)
function replaceinds(tn::AbstractITensorNetwork, is_is′::Pair{<:IndsNetwork,<:IndsNetwork})
  tn = copy(tn)
  is, is′ = is_is′
  @assert underlying_graph(is) == underlying_graph(is′)
  for v in vertices(is)
    isassigned(is, v) || continue
    setindex_preserve_graph!(tn, replaceinds(tn[v], is[v] => is′[v]), v)
  end
  for e in edges(is)
    isassigned(is, e) || continue
    for v in (src(e), dst(e))
      setindex_preserve_graph!(tn, replaceinds(tn[v], is[e] => is′[e]), v)
    end
  end
  return tn
end

function map_inds(f, tn::AbstractITensorNetwork, args...; kwargs...)
  is = IndsNetwork(tn)
  is′ = map_inds(f, is, args...; kwargs...)
  return replaceinds(tn, is => is′)
end

const map_inds_label_functions = [
  :prime,
  :setprime,
  :noprime,
  :replaceprime,
  # :swapprime, # TODO: add @test_broken as a reminder
  :addtags,
  :removetags,
  :replacetags,
  :settags,
  :sim,
  :swaptags,
  :dag,
  # :replaceind,
  # :replaceinds,
  # :swapind,
  # :swapinds,
]

for f in map_inds_label_functions
  @eval begin
    function $f(n::Union{IndsNetwork,AbstractITensorNetwork}, args...; kwargs...)
      return map_inds($f, n, args...; kwargs...)
    end

    function $f(
      ffilter::typeof(linkinds),
      n::Union{IndsNetwork,AbstractITensorNetwork},
      args...;
      kwargs...,
    )
      return map_inds($f, n, args...; sites=[], kwargs...)
    end

    function $f(
      ffilter::typeof(siteinds),
      n::Union{IndsNetwork,AbstractITensorNetwork},
      args...;
      kwargs...,
    )
      return map_inds($f, n, args...; links=[], kwargs...)
    end
  end
end

adjoint(tn::Union{IndsNetwork,AbstractITensorNetwork}) = prime(tn)

dag(tn::AbstractITensorNetwork) = map_vertex_data(dag, tn)

# TODO: should this make sure that internal indices
# don't clash?
function ⊗(
  tn1::AbstractITensorNetwork,
  tn2::AbstractITensorNetwork,
  tn_tail::AbstractITensorNetwork...;
  kwargs...,
)
  return ⊔(tn1, tn2, tn_tail...; kwargs...)
end

function ⊗(
  tn1::Pair{<:Any,<:AbstractITensorNetwork},
  tn2::Pair{<:Any,<:AbstractITensorNetwork},
  tn_tail::Pair{<:Any,<:AbstractITensorNetwork}...;
  kwargs...,
)
  return ⊔(tn1, tn2, tn_tail...; kwargs...)
end

# TODO: how to define this lazily?
#norm(tn::AbstractITensorNetwork) = sqrt(inner(tn, tn))

function contract(tn::AbstractITensorNetwork; sequence=vertices(tn), kwargs...)
  sequence_linear_index = deepmap(v -> vertex_to_parent_vertex(tn, v), sequence)
  return contract(Vector{ITensor}(tn); sequence=sequence_linear_index, kwargs...)
end

function contract(tn::AbstractITensorNetwork, edge::Pair; kwargs...)
  return contract(tn, edgetype(tn)(edge); kwargs...)
end

# Contract the tensors at vertices `src(edge)` and `dst(edge)`
# and store the results in the vertex `dst(edge)`, removing
# the vertex `src(edge)`.
# TODO: write this in terms of a more generic function
# `Graphs.merge_vertices!` (https://github.com/mtfishman/ITensorNetworks.jl/issues/12)
function contract(tn::AbstractITensorNetwork, edge::AbstractEdge; merged_vertex=dst(edge))
  V = promote_type(vertextype(tn), typeof(merged_vertex))
  # TODO: Check `ITensorNetwork{V}`, shouldn't need a copy here.
  tn = ITensorNetwork{V}(copy(tn))
  neighbors_src = setdiff(neighbors(tn, src(edge)), [dst(edge)])
  neighbors_dst = setdiff(neighbors(tn, dst(edge)), [src(edge)])
  new_itensor = tn[src(edge)] * tn[dst(edge)]

  # The following is equivalent to:
  #
  # tn[dst(edge)] = new_itensor
  #
  # but without having to search all vertices
  # to update the edges.
  rem_vertex!(tn, src(edge))
  rem_vertex!(tn, dst(edge))
  add_vertex!(tn, merged_vertex)
  for n_src in neighbors_src
    add_edge!(tn, merged_vertex => n_src)
  end
  for n_dst in neighbors_dst
    add_edge!(tn, merged_vertex => n_dst)
  end
  setindex_preserve_graph!(tn, new_itensor, merged_vertex)

  return tn
end

function tags(tn::AbstractITensorNetwork, edge)
  is = linkinds(tn, edge)
  return commontags(is)
end

function svd(tn::AbstractITensorNetwork, edge::Pair; kwargs...)
  return svd(tn, edgetype(tn)(edge))
end

function svd(
  tn::AbstractITensorNetwork,
  edge::AbstractEdge;
  U_vertex=src(edge),
  S_vertex=(edge, "S"),
  V_vertex=(edge, "V"),
  u_tags=tags(tn, edge),
  v_tags=tags(tn, edge),
  kwargs...,
)
  tn = copy(tn)
  left_inds = uniqueinds(tn, edge)
  U, S, V = svd(tn[src(edge)], left_inds; lefttags=u_tags, right_tags=v_tags, kwargs...)

  rem_vertex!(tn, src(edge))
  add_vertex!(tn, U_vertex)
  tn[U_vertex] = U

  add_vertex!(tn, S_vertex)
  tn[S_vertex] = S

  add_vertex!(tn, V_vertex)
  tn[V_vertex] = V

  return tn
end

function qr(
  tn::AbstractITensorNetwork,
  edge::AbstractEdge;
  Q_vertex=src(edge),
  R_vertex=(edge, "R"),
  tags=tags(tn, edge),
  kwargs...,
)
  tn = copy(tn)
  left_inds = uniqueinds(tn, edge)
  Q, R = factorize(tn[src(edge)], left_inds; tags, kwargs...)

  rem_vertex!(tn, src(edge))
  add_vertex!(tn, Q_vertex)
  tn[Q_vertex] = Q

  add_vertex!(tn, R_vertex)
  tn[R_vertex] = R

  return tn
end

function factorize(
  tn::AbstractITensorNetwork,
  edge::AbstractEdge;
  X_vertex=src(edge),
  Y_vertex=("Y", edge),
  tags=tags(tn, edge),
  kwargs...,
)
  # Promote vertex type
  V = promote_type(vertextype(tn), typeof(X_vertex), typeof(Y_vertex))

  # TODO: Check `ITensorNetwork{V}`, shouldn't need a copy here.
  tn = ITensorNetwork{V}(copy(tn))

  neighbors_X = setdiff(neighbors(tn, src(edge)), [dst(edge)])
  left_inds = uniqueinds(tn, edge)
  X, Y = factorize(tn[src(edge)], left_inds; tags, kwargs...)

  rem_vertex!(tn, src(edge))
  add_vertex!(tn, X_vertex)
  add_vertex!(tn, Y_vertex)

  add_edge!(tn, X_vertex => Y_vertex)
  for nX in neighbors_X
    add_edge!(tn, X_vertex => nX)
  end
  add_edge!(tn, Y_vertex => dst(edge))

  # tn[X_vertex] = X
  setindex_preserve_graph!(tn, X, X_vertex)

  # tn[Y_vertex] = Y
  setindex_preserve_graph!(tn, Y, Y_vertex)

  return tn
end

function factorize(
  tn::AbstractITensorNetwork,
  edge::Pair;
  kwargs...,
)
  return factorize(tn, edgetype(tn)(edge); kwargs...)
end

# For ambiguity error; TODO: decide whether to use graph mutating methods when resulting graph is unchanged?
function _orthogonalize_edge(tn::AbstractITensorNetwork, edge::AbstractEdge; kwargs...)
  # tn = factorize(tn, edge; kwargs...)
  # # TODO: Implement as `only(common_neighbors(tn, src(edge), dst(edge)))`
  # new_vertex = only(neighbors(tn, src(edge)) ∩ neighbors(tn, dst(edge)))
  # return contract(tn, new_vertex => dst(edge))
  tn = copy(tn)
  left_inds = uniqueinds(tn, edge)
  ltags = tags(tn, edge)
  X, Y = factorize(tn[src(edge)], left_inds; tags=ltags, ortho="left", kwargs...)
  tn[src(edge)] = X
  tn[dst(edge)] *= Y
  return tn
end

function orthogonalize(tn::AbstractITensorNetwork, edge::AbstractEdge; kwargs...)
  return _orthogonalize_edge(tn, edge; kwargs...)
end

function orthogonalize(tn::AbstractITensorNetwork, edge::Pair; kwargs...)
  return orthogonalize(tn, edgetype(tn)(edge); kwargs...)
end

# Orthogonalize an ITensorNetwork towards a source vertex, treating
# the network as a tree spanned by a spanning tree.
# TODO: Rename `tree_orthogonalize`.
function orthogonalize(ψ::AbstractITensorNetwork, source_vertex)
  spanning_tree_edges = post_order_dfs_edges(bfs_tree(ψ, source_vertex), source_vertex)
  for e in spanning_tree_edges
    ψ = orthogonalize(ψ, e)
  end
  return ψ
end

# TODO: decide whether to use graph mutating methods when resulting graph is unchanged?
function _truncate_edge(tn::AbstractITensorNetwork, edge::AbstractEdge; kwargs...)
  tn = copy(tn)
  left_inds = uniqueinds(tn, edge)
  ltags = tags(tn, edge)
  U, S, V = svd(tn[src(edge)], left_inds; lefttags=ltags, ortho="left", kwargs...)
  tn[src(edge)] = U
  tn[dst(edge)] *= (S * V)
  return tn
end

function truncate(tn::AbstractITensorNetwork, edge::AbstractEdge; kwargs...)
  return _truncate_edge(tn, edge; kwargs...)
end

function truncate(tn::AbstractITensorNetwork, edge::Pair; kwargs...)
  return truncate(tn, edgetype(tn)(edge); kwargs...)
end

function Base.:*(c::Number, ψ::AbstractITensorNetwork)
  v₁ = first(vertices(ψ))
  cψ = copy(ψ)
  cψ[v₁] *= c
  return cψ
end

# Return a list of vertices in the ITensorNetwork `ψ`
# that share indices with the ITensor `T`
function neighbor_vertices(ψ::AbstractITensorNetwork, T::ITensor)
  ψT = ψ ⊔ ITensorNetwork([T])
  v⃗ = neighbors(ψT, (1, 2))
  return first.(v⃗)
end

function linkinds_combiners(tn::AbstractITensorNetwork)
  combiners = DataGraph(directed_graph(underlying_graph(tn)), ITensor, ITensor)
  for e in edges(tn)
    C = combiner(linkinds(tn, e); tags=edge_tag(e))
    combiners[e] = C
    combiners[reverse(e)] = dag(C)
  end
  return combiners
end

function combine_linkinds(tn::AbstractITensorNetwork, combiners=linkinds_combiners(tn))
  combined_tn = copy(tn)
  for e in edges(tn)
    combined_tn[src(e)] = combined_tn[src(e)] * combiners[e]
    combined_tn[dst(e)] = combined_tn[dst(e)] * combiners[reverse(e)]
  end
  return combined_tn
end

function flatten_networks(
  tn1::AbstractITensorNetwork,
  tn2::AbstractITensorNetwork;
  map_bra_linkinds=sim,
  flatten=true,
  combine_linkinds=true,
  kwargs...,
)
  @assert issetequal(vertices(tn1), vertices(tn2))
  tn1 = map_bra_linkinds(tn1; sites=[])
  flattened_net = ⊗(tn1, tn2; kwargs...)
  if flatten
    for v in vertices(tn1)
      flattened_net = contract(flattened_net, (v, 2) => (v, 1); merged_vertex=v)
    end
  end
  if combine_linkinds
    flattened_net = ITensorNetworks.combine_linkinds(flattened_net)
  end
  return flattened_net
end

function flatten_networks(
  tn1::AbstractITensorNetwork,
  tn2::AbstractITensorNetwork,
  tn3::AbstractITensorNetwork,
  tn_tail::AbstractITensorNetwork...;
  kwargs...,
)
  return flatten_networks(flatten_networks(tn1, tn2; kwargs...), tn3, tn_tail...; kwargs...)
end

# TODO: Use or replace with `flatten_networks`
function inner_network(
  tn1::AbstractITensorNetwork,
  tn2::AbstractITensorNetwork;
  map_bra_linkinds=sim,
  combine_linkinds=false,
  flatten=combine_linkinds,
  kwargs...,
)
  @assert issetequal(vertices(tn1), vertices(tn2))
  tn1 = map_bra_linkinds(tn1; sites=[])
  inner_net = ⊗(dag(tn1), tn2; kwargs...)
  if flatten
    for v in vertices(tn1)
      inner_net = contract(inner_net, (v, 2) => (v, 1); merged_vertex=v)
    end
  end
  if combine_linkinds
    inner_net = ITensorNetworks.combine_linkinds(inner_net)
  end
  return inner_net
end

# TODO: Rename `inner`.
function contract_inner(
  ϕ::AbstractITensorNetwork,
  ψ::AbstractITensorNetwork;
  sequence=nothing,
  contraction_sequence_kwargs=(;),
)
  tn = inner_network(ϕ, ψ; combine_linkinds=true)
  if isnothing(sequence)
    sequence = contraction_sequence(tn; contraction_sequence_kwargs...)
  end
  return contract(tn; sequence)[]
end

# TODO: rename `sqnorm` to match https://github.com/JuliaStats/Distances.jl,
# or `norm_sqr` to match `LinearAlgebra.norm_sqr`
norm_sqr(ψ::AbstractITensorNetwork; sequence) = contract_inner(ψ, ψ; sequence)

norm_sqr_network(ψ::AbstractITensorNetwork; kwargs...) = inner_network(ψ, ψ; kwargs...)

#
# Printing
#

function show(io::IO, mime::MIME"text/plain", graph::AbstractITensorNetwork)
  println(io, "$(typeof(graph)) with $(nv(graph)) vertices:")
  show(io, mime, vertices(graph))
  println(io, "\n")
  println(io, "and $(ne(graph)) edge(s):")
  for e in edges(graph)
    show(io, mime, e)
    println(io)
  end
  println(io)
  println(io, "with vertex data:")
  show(io, mime, inds.(vertex_data(graph)))
  return nothing
end

show(io::IO, graph::AbstractITensorNetwork) = show(io, MIME"text/plain"(), graph)

function visualize(
  tn::AbstractITensorNetwork,
  args...;
  vertex_labels_prefix=nothing,
  vertex_labels=nothing,
  kwargs...,
)
  if !isnothing(vertex_labels_prefix)
    vertex_labels = [vertex_labels_prefix * string(v) for v in vertices(tn)]
  end
  return visualize(Vector{ITensor}(tn), args...; vertex_labels, kwargs...)
end

# 
# Link dimensions
# 

function maxlinkdim(tn::AbstractITensorNetwork)
  md = 1
  for e in edges(tn)
    md = max(md, linkdim(tn, e))
  end
  return md
end

function linkdim(tn::AbstractITensorNetwork, edge::Pair)
  return linkdim(tn, edgetype(tn)(edge))
end

function linkdim(tn::AbstractITensorNetwork{V}, edge::AbstractEdge{V}) where {V}
  ls = linkinds(tn, edge)
  return prod([isnothing(l) ? 1 : dim(l) for l in ls])
end

function linkdims(tn::AbstractITensorNetwork{V}) where {V}
  ld = DataGraph{V,Any,Int}(copy(underlying_graph(tn)))
  for e in edges(ld)
    ld[e] = linkdim(tn, e)
  end
  return ld
end

# 
# Common index checking
# 

function hascommoninds(
  ::typeof(siteinds), A::AbstractITensorNetwork{V}, B::AbstractITensorNetwork{V}
) where {V}
  for v in vertices(A)
    !hascommoninds(siteinds(A, v), siteinds(B, v)) && return false
  end
  return true
end

function check_hascommoninds(
  ::typeof(siteinds), A::AbstractITensorNetwork{V}, B::AbstractITensorNetwork{V}
) where {V}
  N = nv(A)
  if nv(B) ≠ N
    throw(
      DimensionMismatch(
        "$(typeof(A)) and $(typeof(B)) have mismatched number of vertices $N and $(nv(B))."
      ),
    )
  end
  for v in vertices(A)
    !hascommoninds(siteinds(A, v), siteinds(B, v)) && error(
      "$(typeof(A)) A and $(typeof(B)) B must share site indices. On vertex $v, A has site indices $(siteinds(A, v)) while B has site indices $(siteinds(B, v)).",
    )
  end
  return nothing
end

function hassameinds(
  ::typeof(siteinds), A::AbstractITensorNetwork{V}, B::AbstractITensorNetwork{V}
) where {V}
  nv(A) ≠ nv(B) && return false
  for v in vertices(A)
    !ITensors.hassameinds(siteinds(A, v), siteinds(B, v)) && return false
  end
  return true
end

# 
# Site combiners
# 

# TODO: will be broken, fix this
function site_combiners(tn::AbstractITensorNetwork{V}) where {V}
  Cs = DataGraph{V,ITensor}(copy(underlying_graph(tn)))
  for v in vertices(tn)
    s = siteinds(tn, v)
    Cs[v] = combiner(s; tags=commontags(s))
  end
  return Cs
end

## # TODO: should this make sure that internal indices
## # don't clash?
## function hvncat(
##   dim::Int, tn1::AbstractITensorNetwork, tn2::AbstractITensorNetwork; new_dim_names=(1, 2)
## )
##   dg = hvncat(dim, data_graph(tn1), data_graph(tn2); new_dim_names)
## 
##   # Add in missing edges that may be shared
##   # across `tn1` and `tn2`.
##   vertices1 = vertices(dg)[1:nv(tn1)]
##   vertices2 = vertices(dg)[(nv(tn1) + 1):end]
##   for v1 in vertices1, v2 in vertices2
##     if hascommoninds(dg[v1], dg[v2])
##       add_edge!(dg, v1 => v2)
##     end
##   end
## 
##   # TODO: Allow customization of the output type.
##   ## return promote_type(typeof(tn1), typeof(tn2))(dg)
##   ## return contract_output(typeof(tn1), typeof(tn2))(dg)
## 
##   return ITensorNetwork(dg)
## end
