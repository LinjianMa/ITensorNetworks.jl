
"""
$(TYPEDSIGNATURES)

Compute a solution x to the linear system:

(a₀ + a₁ * A)*x = b

using starting guess x₀. Leaving a₀, a₁
set to their default values solves the 
system A*x = b.

To adjust the balance between accuracy of solution
and speed of the algorithm, it is recommed to first try
adjusting the `solver_tol` keyword argument descibed below.

Keyword arguments:
  - `ishermitian::Bool=false` - should set to true if the MPO A is Hermitian
  - `solver_krylovdim::Int=30` - max number of Krylov vectors to build on each solver iteration
  - `solver_maxiter::Int=100` - max number outer iterations (restarts) to do in the solver step
  - `solver_tol::Float64=1E-14` - tolerance or error goal of the solver

Overload of `KrylovKit.linsolve`.
"""
function linsolve(A::MPO, b::MPS, x₀::MPS, a₀::Number=0, a₁::Number=1; kwargs...)
  function linsolve_solver(
    P::ProjMPO_MPS2,
    t,
    x₀;
    ishermitian=false,
    solver_tol=1E-14,
    solver_krylovdim=30,
    solver_maxiter=100,
    solver_verbosity=0,
    kwargs...,
  )
    solver_kwargs = (;
      ishermitian=ishermitian,
      tol=solver_tol,
      krylovdim=solver_krylovdim,
      maxiter=solver_maxiter,
      verbosity=solver_verbosity,
    )
    b = dag(only(proj_mps(P)))
    x, info = KrylovKit.linsolve(P, b, x₀, a₀, a₁; solver_kwargs...)
    return x, nothing
  end

  t = Inf
  P = ProjMPO_MPS2(A, b)
  return tdvp(linsolve_solver, P, t, x₀; reverse_step=false, kwargs...)
end
