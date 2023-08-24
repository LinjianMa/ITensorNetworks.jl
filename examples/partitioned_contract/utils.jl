function linear_path(tn::Vector, par_size)
  if length(tn) <= max(2, par_size)
    return tn
  end
  return [linear_path(tn[1:(end - par_size)], par_size), tn[end - par_size + 1:end]]
end