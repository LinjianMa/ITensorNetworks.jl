using ITensors

ninds = 101
indices = [Index(2, "$(i)") for i in 1:ninds]
new_indices = [Index(2, "new$(i)") for i in 1:ninds]

a = ITensor(indices[1])
replaceinds(a, indices, new_indices)
replaceinds(a, indices => new_indices)
