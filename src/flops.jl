Cassette.@context CounterCtx;

function Cassette.prehook(ctx::CounterCtx, contract::typeof(contract), t1::ITensor, t2::ITensor)
    @info "prehook contrct", t1, t2, "type", typeof(contract)
    if !haskey(ctx.metadata, "contract")
        ctx.metadata["contract"] = 0
    end
    ctx.metadata["contract"] += 2 * dim(union(inds(t1), inds(t2)))
end

# Modified from https://github.com/triscale-innov/GFlops.jl/blob/a2ad017e880e908381d384c04c030db0042b37c5/src/count_ops.jl#L1-L36
prepare_call(expr) = expr
prepare_call(s::Symbol) = esc(s)

function prepare_call(expr::Expr)
    expr_list = map(x->prepare_call(x), expr.args)
    return Expr(expr.head, expr_list...)
end

function count_ops(expr)
    expr = prepare_call(expr)
    return quote
        ctx = CounterCtx(metadata=Dict())
        Cassette.overdub(ctx, ()-> $expr)
        ctx.metadata
    end
end

macro count_ops(expr)
    count_ops(expr)
end

macro no_flops(expr)
    @assert Meta.isexpr(expr, :call)
    @assert expr.args[1] isa Symbol
    func = expr.args[1]
    func2 = esc(expr.args[1])
    @info func
    func_args = []
    func_kwargs = []
    for arg in expr.args[2:end]
        if arg isa Symbol || (arg isa Expr && arg.head == :...)
            push!(func_args, arg)
        else
            # kwargs
            @assert arg isa Expr && arg.head == :parameters
            for p in arg.args
                push!(func_kwargs, p)
            end
        end
    end
    out = Expr(:call, esc(func), func_args...)
    return quote
         begin
            # f = $func
            # @info $(func_args)
            # @info $(func_kwargs)
            #args = esc(:(
            # args = $($(func_args)...)
            # args = $(func_args)
            # #))
            # if length($(func_kwargs)) == 0
            #     kwargs = []
            # else
            #   kwargs = esc(:($($(func_kwargs)...)))
            # end
            # @info "f is", f
            # @info "args is", args 
            # @info "kwargs is", kwargs
            function Cassette.overdub(::CounterCtx, ::typeof($func), $(func_args...); $(func_kwargs...))
                @info "func", $func
                # @info "args", args
                # @info "kwargs", kwargs
              return $(out)#$func($(func_args...); $(func_kwargs...))
            end
        end
    end
end

@no_flops noncommoninds(args...; kwargs...)
# @no_flops commoninds(args...; kwargs...)
@no_flops contract(t1, t2) # TODO
