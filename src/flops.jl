Cassette.@context CounterCtx;

@eval function Cassette.prehook(ctx::CounterCtx, contract, t1::ITensor, t2::ITensor)
    if !haskey(ctx.metadata, "contract")
        ctx.metadata["contract"] = 0
    end
    ctx.metadata["contract"] += 2 * dim(union(inds(t1), inds(t2)))
end

# From https://github.com/triscale-innov/GFlops.jl/blob/a2ad017e880e908381d384c04c030db0042b37c5/src/count_ops.jl#L1-L36
prepare_call!(vars, expr) = expr
prepare_call!(vars, s::Symbol) = esc(s)

function prepare_call!(vars, e::Expr)
    e.head == :$ || return Expr(e.head, map(x->prepare_call!(vars, x), e.args)...)
    var = gensym()
    push!(vars, :($var = $(prepare_call!(vars, e.args[1]))))
    var
end

prepare_call(e) = let v=[]
    e2 = prepare_call!(v, e)
    v, e2
end

function count_ops(funcall)
    v, e = prepare_call(funcall)
    quote
        let
            ctx = CounterCtx(metadata=Dict())
            $(v...)
            Cassette.overdub(ctx, ()->begin
                             $e
                             end)
            ctx.metadata
        end
    end
end

macro count_ops(funcall)
    count_ops(funcall)
end

function Cassette.overdub(ctx::CounterCtx, f, args...; kwargs...)
    if typeof(f) == typeof(noncommoninds)
        return
    end
    @info f
    @info typeof(f)
    return Cassette.recurse(ctx::CounterCtx, f, args...; kwargs...)
end