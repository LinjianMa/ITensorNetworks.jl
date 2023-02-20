"""
!!! warning
    This helper macro covers only the simple common cases.
    It does not support `where`-clauses.

Code modified from https://github.com/JuliaDiff/ChainRulesCore.jl/blob/ed9a0073ff83cb3b1f4619303e41f4dd5d8c4825/src/rule_definition_tools.jl#L362
"""
function no_overdub(sig_expr::Expr)
  Meta.isexpr(sig_expr, :call) || error("Invalid use of `@no_overdub`")
  has_vararg = _isvararg(sig_expr.args[end])

  primal_name, orig_args = Iterators.peel(sig_expr.args)

  primal_name_sig, primal_name = _split_primal_name(primal_name)
  constrained_args = _constrain_and_name.(orig_args, :Any)
  primal_sig_parts = [primal_name_sig, constrained_args...]

  unconstrained_args = esc.(_unconstrain.(constrained_args))
  primal_name = esc(primal_name)

  primal_invoke = if !has_vararg
    :($(primal_name)($(unconstrained_args...); kwargs...))
  else
    normal_args = unconstrained_args[1:(end - 1)]
    var_arg = unconstrained_args[end]
    :($(primal_name)($(normal_args...), $(var_arg)...; kwargs...))
  end

  return quote
    $(_no_overdub(primal_sig_parts, primal_invoke))
  end
end

function _no_overdub(primal_sig_parts, primal_invoke)
  # @gensym kwargs
  return quote
    function Cassette.overdub(::CounterCtx, $(map(esc, primal_sig_parts)...); kwargs...)
      return $(primal_invoke)
    end
  end
end

"""
    _isvararg(expr)
returns true if the expression could represent a vararg
```
julia> _isvararg(:(x...))
true
julia> _isvararg(:(x::Int...))
true
julia> _isvararg(:(::Int...))
true
julia> _isvararg(:(x::Vararg))
true
julia> _isvararg(:(x::Vararg{Int}))
true
julia> _isvararg(:(::Vararg))
true
julia> _isvararg(:(::Vararg{Int}))
true
julia> _isvararg(:(x))
false
````
"""
_isvararg(expr) = false
function _isvararg(expr::Expr)
  Meta.isexpr(expr, :...) && return true
  if Meta.isexpr(expr, :(::))
    constraint = last(expr.args)
    constraint === :Vararg && return true
    Meta.isexpr(constraint, :curly) && first(constraint.args) === :Vararg && return true
  end
  return false
end

"""
splits the first arg of the `call` expression into an expression to use in the signature
and one to use for calling that function
"""
function _split_primal_name(primal_name)
  # e.g. f(x, y)
  is_plain = primal_name isa Symbol
  is_qualified = Meta.isexpr(primal_name, :(.))
  is_parameterized = Meta.isexpr(primal_name, :curly)
  if is_plain || is_qualified || is_parameterized
    primal_name_sig = :(::$Core.Typeof($primal_name))
    return primal_name_sig, primal_name
  elseif Meta.isexpr(primal_name, :(::))  # e.g. (::T)(x, y)
    _primal_name = gensym(Symbol(:instance_, primal_name.args[end]))
    primal_name_sig = Expr(:(::), _primal_name, primal_name.args[end])
    return primal_name_sig, _primal_name
  else
    error("invalid primal name: `$primal_name`")
  end
end

"turn both `a` and `a::S` into `a`"
_unconstrain(arg::Symbol) = arg
function _unconstrain(arg::Expr)
  Meta.isexpr(arg, :(::), 2) && return arg.args[1]  # drop constraint.
  Meta.isexpr(arg, :(...), 1) && return _unconstrain(arg.args[1])
  return error("malformed arguments: $arg")
end

"turn both `a` and `::constraint` into `a::constraint` etc"
function _constrain_and_name(arg::Expr, _)
  Meta.isexpr(arg, :(::), 2) && return arg  # it is already fine.
  Meta.isexpr(arg, :(::), 1) && return Expr(:(::), gensym(), arg.args[1]) # add name
  Meta.isexpr(arg, :(...), 1) && return Expr(:(...), _constrain_and_name(arg.args[1], :Any))
  return error("malformed arguments: $arg")
end
_constrain_and_name(name::Symbol, constraint) = Expr(:(::), name, constraint)  # add type
