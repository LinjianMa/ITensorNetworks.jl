using ITensorNetworks

function bench_lnZ(tntree::Vector; num_iter, kwargs...)
  function _run()
    out, log_acc_norm = partitioned_contract(tntree; kwargs...)
    out = Vector{ITensor}(out)
    log_acc_norm = log(norm(out)) + log_acc_norm
    @info "out value is", out[1].tensor
    @info "out norm is", log_acc_norm
    return log_acc_norm
  end
  out_list = []
  # after warmup, start to benchmark
  reset_timer!(ITensors.timer)
  for _ in 1:num_iter
    push!(out_list, _run())
  end
  @info "lnZ results are", out_list, "mean is", sum(out_list) / (num_iter * 2)
  return show(ITensors.timer)
end

function get_computation_time(timer)
  t1 = TimerOutputs.time(
    ITensors.timer["partitioned_contract"]["approx_itensornetwork w/ density matrix"]["[approx_binary_tree_itensornetwork]: _optcontract"],
  )
  t2 = TimerOutputs.time(
    ITensors.timer["partitioned_contract"]["approx_itensornetwork w/ density matrix"]["[approx_binary_tree_itensornetwork]: eigen"],
  )
  t3 = TimerOutputs.time(
    ITensors.timer["partitioned_contract"]["approx_itensornetwork w/ density matrix"]["factorize"],
  )
  return (t1 + t2 + t3) / 1000000000
end

function bench_lnZ(tn::ITensorNetwork; num_iter, kwargs...)
  accurate_lnZ = 223.53205177989938
  function _run()
    out, log_acc_norm = contract(tn; alg="partitioned_contract", kwargs...)
    out = Vector{ITensor}(out)
    log_acc_norm = log(norm(out)) + log_acc_norm
    @info "out value is", out[1].tensor
    @info "out norm is", log_acc_norm
    return abs((log_acc_norm - accurate_lnZ) / accurate_lnZ)
  end
  out_list = []
  time_list = []
  # after warmup, start to benchmark
  for (i, _) in enumerate(1:num_iter)
    reset_timer!(ITensors.timer)
    push!(out_list, _run())
    push!(time_list, get_computation_time(ITensors.timer))
    @info "$(i)/$(num_iter), $(out_list), $(time_list)"
  end
  avg_output = sum(out_list) / (num_iter)
  avg_time = sum(time_list) / (num_iter)
  @info "lnZ results are", out_list, "mean is", avg_output
  @info "the average time is", avg_time
  return avg_output, avg_time
end

function bench_magnetization(
  tntree_pair;
  num_iter,
  cutoff,
  maxdim,
  ansatz,
  approx_itensornetwork_alg,
  swap_size,
  contraction_sequence_alg,
  contraction_sequence_kwargs,
  linear_ordering_alg,
  warmup,
)
  reset_timer!(ITensors.timer)
  tntree1, tntree2 = tntree_pair
  function _run()
    out, log_acc_norm = approximate_contract(
      tntree1;
      cutoff,
      maxdim,
      ansatz,
      approx_itensornetwork_alg,
      swap_size,
      contraction_sequence_alg,
      contraction_sequence_kwargs,
      linear_ordering_alg,
    )
    out = Vector{ITensor}(out)
    lognorm1 = log(norm(out)) + log_acc_norm
    out, log_acc_norm = approximate_contract(
      tntree2;
      cutoff,
      maxdim,
      ansatz,
      approx_itensornetwork_alg,
      swap_size,
      contraction_sequence_alg,
      contraction_sequence_kwargs,
      linear_ordering_alg,
    )
    out = Vector{ITensor}(out)
    lognorm2 = log(norm(out)) + log_acc_norm
    @info "magnetization is", exp(lognorm1 - lognorm2)
    return exp(lognorm1 - lognorm2)
  end
  out_list = []
  total_iter = num_iter
  if warmup
    for _ in 1:num_iter
      push!(out_list, _run())
    end
    show(ITensors.timer)
    total_iter += num_iter
  end
  @info "warmup finished"
  # after warmup, start to benchmark
  reset_timer!(ITensors.timer)
  for _ in 1:num_iter
    push!(out_list, _run())
  end
  @info "magnetization results are", out_list, "mean is", sum(out_list) / total_iter
  return show(ITensors.timer)
end
