function bipartite_sequence(vec::Vector)
  if length(vec) == 1
    return vec[1]
  end
  if length(vec) == 2
    return vec
  end
  middle = floor(Int64, length(vec) / 2)
  return [bipartite_sequence(vec[1:middle]), bipartite_sequence(vec[(middle + 1):end])]
end

function linear_sequence(vec::Vector)
  if length(vec) <= 2
    return vec
  end
  return [linear_sequence(vec[1:(end - 1)]), vec[end]]
end

function bench(tn, btree; alg, maxdim, contraction_sequence_alg="sa_bipartite")
  @timeit ITensors.timer "approx_itensornetwork" begin
    return approx_itensornetwork(
      tn,
      btree;
      alg,
      cutoff=1e-15,
      maxdim=maxdim,
      contraction_sequence_alg=contraction_sequence_alg,
      contraction_sequence_kwargs=(;),
    )
  end
end

function bench_embed(
  partitioned_tn, root; alg, maxdim, contraction_sequence_alg="sa_bipartite"
)
  @timeit ITensors.timer "approx_itensornetwork with embedding" begin
    approx_itensornetwork(
      partitioned_tn::DataGraph;
      alg,
      root=root,
      cutoff=1e-15,
      maxdim=maxdim,
      contraction_sequence_alg=contraction_sequence_alg,
      contraction_sequence_kwargs=(;),
    )
  end
end
