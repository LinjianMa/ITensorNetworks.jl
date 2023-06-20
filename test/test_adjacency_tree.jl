using Graphs, ITensors, AbstractTrees, NamedGraphs
using ITensorNetworks:
  _generate_adjacency_tree, _get_paths, mindist_ordering, IndexGroup, _add_vertex_edges!

@testset "test _generate_adjacency_tree" begin
  v, u1, u2, u3, u4, u5 = [ITensor(i) for i in 1:6]
  ctree = [[[[[[v], [u1]], [u2]], [u3]], [u4]], [u5]]
  path = _get_paths(ctree)[[v]]
  ctree_to_open_edges = Dict()
  ctree_to_open_edges[[v]] = [1, 2, 3, 4, 5]
  ctree_to_open_edges[[u1]] = [2, 3, 6, 7, 8]
  ctree_to_open_edges[[u2]] = [4, 8]
  ctree_to_open_edges[[u3]] = [1, 6, 9]
  ctree_to_open_edges[[u4]] = [5, 7, 10]
  ctree_to_open_edges[[u5]] = [9, 10]
  ctree_to_open_edges[[[v], [u1]]] = [1, 4, 5, 6, 7, 8]
  ctree_to_open_edges[[[[v], [u1]], [u2]]] = [1, 5, 6, 7]
  ctree_to_open_edges[[[[[v], [u1]], [u2]], [u3]]] = [5, 7, 9]
  ctree_to_open_edges[[[[[[v], [u1]], [u2]], [u3]], [u4]]] = [9, 10]
  ctree_to_open_edges[[[[[[[v], [u1]], [u2]], [u3]], [u4]], [u5]]] = []
  adj_tree = _generate_adjacency_tree([v], path, ctree_to_open_edges)
  for v in vertices(adj_tree)
    if Set(Leaves(v[1])) == Set([2, 3])
      @test v[2] == "unordered"
    end
    if Set(Leaves(v[1])) == Set([1, 2, 3, 4])
      @test v[2] == "ordered"
    end
    if Set(Leaves(v[1])) == Set([1, 2, 3, 4, 5])
      @test v[2] == "unordered"
    end
  end
end

@testset "test mindist_ordering" begin
  is = [Index(2, string(i)) for i in 1:8]
  T = randomITensor(is...)
  M = MPS(T, Tuple(is); cutoff=1e-5, maxdim=2)
  tensors = M[:]

  reference_order = map(x -> IndexGroup([x]), is)
  adj_tree = NamedDiGraph{Tuple{Tuple,String}}()
  vs = [((i,), "unordered") for i in reference_order]
  for v in vs
    add_vertex!(adj_tree, v)
  end
  v23 = ((vs[2][1], vs[3][1]), "unordered")
  v45 = ((vs[4][1], vs[5][1]), "unordered")
  v67 = ((vs[7][1], vs[6][1]), "unordered")
  v45678 = ((vs[8][1], v67[1], v45[1]), "ordered")
  root = ((v45678[1], v23[1], vs[1][1]), "ordered")
  _add_vertex_edges!(adj_tree, v23; children=[vs[2], vs[3]])
  _add_vertex_edges!(adj_tree, v45; children=[vs[4], vs[5]])
  _add_vertex_edges!(adj_tree, v67; children=[vs[6], vs[7]])
  _add_vertex_edges!(adj_tree, v45678; children=[vs[8], v45, v67])
  _add_vertex_edges!(adj_tree, root; children=[v45678, vs[1], v23])
  out_order, nswap = mindist_ordering(adj_tree, reference_order, tensors)
  @test out_order == reference_order
  @test nswap == 0
end
