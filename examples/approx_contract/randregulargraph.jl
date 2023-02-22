function build_tntree(tn::ITensorNetwork; nvertices_per_partition=2, backend="KaHyPar")
  @assert is_connected(tn)
  g_parts = partition(tn; nvertices_per_partition=nvertices_per_partition, backend=backend)
  @assert is_connected(g_parts)
  root = 1
  tree = bfs_tree(g_parts, root)
  tntree = nothing
  queue = [root]
  while queue != []
    v = popfirst!(queue)
    queue = vcat(queue, child_vertices(tree, v))
    if tntree == nothing
      tntree = Vector{ITensor}(g_parts[v])
    else
      tntree = [tntree, Vector{ITensor}(g_parts[v])]
    end
  end
  return tntree
end
