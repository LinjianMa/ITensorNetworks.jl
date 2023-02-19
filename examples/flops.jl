using ITensors, Cassette
using ITensorNetworks: @count_ops
# @no_flops noncommoninds(args...; kwargs...)

###### 
num = 3
indices = [Index(2, "$(i)") for i in 1:num]
tensors = [ITensor(i) for i in indices]
# @count_ops a * b

function main(tensors::Vector{ITensor}; use=true)
    out = noncommoninds(tensors[1] * tensors[2], tensors[3])
    @info out
    @info typeof(out)
end

@info methods(Cassette.overdub)
out = @count_ops main(tensors::Vector{ITensor}; use=true)
@info "out", out