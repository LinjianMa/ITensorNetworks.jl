using Graphs, ITensors, AbstractTrees
using ITensorNetworks: _generate_adjacency_tree, _get_paths

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
