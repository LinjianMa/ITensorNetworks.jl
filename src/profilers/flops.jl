Cassette.@context CounterCtx

function Cassette.prehook(
  ctx::CounterCtx, contract::typeof(contract), t1::ITensor, t2::ITensor
)
  @info "prehook contrct", t1, t2, "type", typeof(contract)
  if !haskey(ctx.metadata, "contract")
    ctx.metadata["contract"] = 0
  end
  return ctx.metadata["contract"] += 2 * dim(union(inds(t1), inds(t2)))
end

"""
!!! warning
    This helper macro covers only the simple common cases.
    It does not support `where`-clauses.

Code modified from https://github.com/JuliaDiff/ChainRulesCore.jl/blob/ed9a0073ff83cb3b1f4619303e41f4dd5d8c4825/src/rule_definition_tools.jl#L362
"""
macro no_flops(sig_expr)
  return no_overdub(sig_expr)
end

# macro flops
# end

# macro count_flops
# end

# Modified from https://github.com/triscale-innov/GFlops.jl/blob/a2ad017e880e908381d384c04c030db0042b37c5/src/count_ops.jl#L1-L36
prepare_call(expr) = expr
prepare_call(s::Symbol) = esc(s)

function prepare_call(expr::Expr)
  expr_list = map(x -> prepare_call(x), expr.args)
  return Expr(expr.head, expr_list...)
end

function count_ops(expr)
  expr = prepare_call(expr)
  return quote
    ctx = CounterCtx(; metadata=Dict())
    Cassette.overdub(ctx, () -> $expr)
    ctx.metadata
  end
end

macro count_ops(expr)
  return count_ops(expr)
end

@no_flops noncommoninds(args...)
# @no_flops commoninds(args...; kwargs...)
@no_flops contract(::ITensor, ::ITensor) # TODO
