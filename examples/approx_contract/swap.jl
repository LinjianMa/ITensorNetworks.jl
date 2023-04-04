using ITensors, ITensorNetworks
using ITensorNetworks: line_to_tree, _remove_non_leaf_deltas

sites = [Index(2, "$(i)") for i in 1:10]
sites[1] = Index(256, "1")
sites[10] = Index(256, "10")
mps = randomMPS(sites; linkdims=256)

temp = sites[9]
sites[9] = sites[2]
sites[2] = temp
@info sites

inds_btree = line_to_tree(sites)
@info inds_btree

partition = binary_tree_partition(ITensorNetwork(mps.data), inds_btree)
partition = _remove_non_leaf_deltas(partition)
@info partition
