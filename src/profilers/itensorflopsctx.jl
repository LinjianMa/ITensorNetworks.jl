Cassette.@context ITensorFlopsCtx

function Cassette.prehook(
  ctx::ITensorFlopsCtx, contract::typeof(contract), t1::ITensor, t2::ITensor
)
  @info "prehook contrct", t1, t2, "type", typeof(contract)
  if !haskey(ctx.metadata, "contract")
    ctx.metadata["contract"] = 0
  end
  return ctx.metadata["contract"] += 2 * dim(union(inds(t1), inds(t2)))
end

# TODO: refactor, put these into file `itensorflopsctx.jl`, and make flops function independent of Ctx
@no_flops ITensorFlopsCtx, noncommoninds(args...)
@no_flops ITensorFlopsCtx, commoninds(args...)
@no_flops ITensorFlopsCtx, contract(::ITensor, ::ITensor)
