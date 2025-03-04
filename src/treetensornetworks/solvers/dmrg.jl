function eigsolve_solver(; kwargs...)
  function solver(H, t, psi0; kws...)
    howmany = 1
    which = get(kwargs, :solver_which_eigenvalue, :SR)
    solver_kwargs = (;
      ishermitian=get(kwargs, :ishermitian, true),
      tol=get(kwargs, :solver_tol, 1E-14),
      krylovdim=get(kwargs, :solver_krylovdim, 3),
      maxiter=get(kwargs, :solver_maxiter, 1),
      verbosity=get(kwargs, :solver_verbosity, 0),
    )
    vals, vecs, info = eigsolve(H, psi0, howmany, which; solver_kwargs...)
    psi = vecs[1]
    return psi, info
  end
  return solver
end

function dmrg(H, psi0::IsTreeState; kwargs...)
  t = Inf # DMRG is TDVP with an infinite timestep and no reverse step
  reverse_step = false
  psi = tdvp(eigsolve_solver(; kwargs...), H, t, psi0; reverse_step, kwargs...)
  return psi
end

# Alias for DMRG
function eigsolve(H, psi0::IsTreeState; kwargs...)
  return dmrg(H, psi0; kwargs...)
end
