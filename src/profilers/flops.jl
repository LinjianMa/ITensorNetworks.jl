# function flops()

"""
!!! warning
    This helper macro covers only the simple common cases.
    It does not support `where`-clauses.

Code modified from https://github.com/JuliaDiff/ChainRulesCore.jl/blob/ed9a0073ff83cb3b1f4619303e41f4dd5d8c4825/src/rule_definition_tools.jl#L362
"""
macro no_flops(sig_expr)
  return no_overdub(sig_expr)
end

# macro flops(sig_expr)
#   ctx, primal_sig_parts, primal_invoke = _pre_process_overdub_expr(expr)
#   return quote
#     $(_no_overdub(ctx, primal_sig_parts, primal_invoke))
#     $(_flops(ctx, primal_sig_parts, primal_invoke))
#   end
# end

# # TODO
# function _flops(ctx, primal_sig_parts, primal_invoke)
#   return quote
#     function Cassette.overdub(::$(ctx), $(primal_sig_parts...); kwargs...)
#       @show $(primal_invoke), kwargs, "kwargs"
#       return $(primal_invoke)
#     end
#   end
# end

# Modified from https://github.com/triscale-innov/GFlops.jl/blob/a2ad017e880e908381d384c04c030db0042b37c5/src/count_ops.jl#L1-L36
prepare_call(expr) = expr
prepare_call(s::Symbol) = esc(s)

function prepare_call(expr::Expr)
  expr_list = map(x -> prepare_call(x), expr.args)
  return Expr(expr.head, expr_list...)
end

function count_flops(expr)
  Meta.isexpr(expr, :tuple) || error("Invalid use of `count_flops`")
  length(expr.args) == 2 || error("Invalid use of `count_flops`")
  count_ctx, expr = expr.args
  expr = prepare_call(expr)
  return quote
    ctx = $(count_ctx)(; metadata=Dict())
    Cassette.overdub(ctx, () -> $expr)
    ctx.metadata
  end
end

macro count_flops(expr)
  return count_flops(expr)
end
