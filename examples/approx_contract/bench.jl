function bench_lnZ(tntree; num_iter, cutoff, maxdim, ansatz, algorithm, use_cache, ortho)
  reset_timer!(ITensors.timer)
  function _run()
    out, log_acc_norm = approximate_contract(
      tntree;
      cutoff=cutoff,
      maxdim=maxdim,
      ansatz=ansatz,
      algorithm=algorithm,
      use_cache=use_cache,
      orthogonalize=ortho,
    )
    log_acc_norm = log(norm(out)) + log_acc_norm
    @info "out is", log_acc_norm
    return log_acc_norm
  end
  out_list = []
  for _ in 1:num_iter
    push!(out_list, _run())
  end
  show(ITensors.timer)
  # after warmup, start to benchmark
  reset_timer!(ITensors.timer)
  for _ in 1:num_iter
    push!(out_list, _run())
  end
  @info "lnZ results are", out_list, "mean is", sum(out_list) / (num_iter * 2)
  return show(ITensors.timer)
end

function bench_magnetization(
  tntree_pair; num_iter, cutoff, maxdim, ansatz, algorithm, use_cache, ortho
)
  reset_timer!(ITensors.timer)
  tntree1, tntree2 = tntree_pair
  function _run()
    out, log_acc_norm = approximate_contract(
      tntree1;
      cutoff=cutoff,
      maxdim=maxdim,
      ansatz=ansatz,
      algorithm=algorithm,
      use_cache=use_cache,
      orthogonalize=ortho,
    )
    lognorm1 = log(out[1][1]) + log_acc_norm
    out, log_acc_norm = approximate_contract(
      tntree2;
      cutoff=cutoff,
      maxdim=maxdim,
      ansatz=ansatz,
      algorithm=algorithm,
      use_cache=use_cache,
      orthogonalize=ortho,
    )
    lognorm2 = log(out[1][1]) + log_acc_norm
    return lognorm1 / lognorm2
  end
  out_list = []
  for _ in 1:num_iter
    push!(out_list, _run())
  end
  show(ITensors.timer)
  # after warmup, start to benchmark
  reset_timer!(ITensors.timer)
  for _ in 1:num_iter
    push!(out_list, _run())
  end
  @info "magnetization results are", out_list, "mean is", sum(out_list) / (num_iter * 2)
  return show(ITensors.timer)
end
