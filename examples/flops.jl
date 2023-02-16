using ITensors
using ITensorNetworks: @count_ops

num = 3
indices = [Index(2, "$(i)") for i in 1:num]
tensors = [ITensor(i) for i in indices]
# @count_ops a * b
@count_ops noncommoninds(tensors...)