using ITensors, Graphs, NamedGraphs, TimerOutputs
using Random
using PastaQ
using KaHyPar
using ITensorNetworks
using ITensorNetworks: vectorize

include("exact_contract.jl")
include("randregulargraph.jl")
include("bench.jl")

Random.seed!(1234)
TimerOutputs.enable_debug_timings(ITensorNetworks)
ITensors.set_warn_order(100)

function gate_tensors(hilbert::Vector{<:Index}, gates; applydag=false)
  hilbert = copy(hilbert)
  Us = ITensor[]
  for g in gates
    if applydag == true
      t = dag(gate(hilbert, g))
      inds1 = [hilbert[n] for n in g[2]]
      t = swapinds(t, inds1, [i' for i in inds1])
    else
      t = gate(hilbert, g)
    end
    if length(g[2]) == 2
      ind = hilbert[g[2][1]]
      L, R = factorize(t, ind, ind'; cutoff=1e-16)
      push!(Us, L, R)
    else
      push!(Us, t)
    end
    for n in g[2]
      hilbert[n] = hilbert[n]'
    end
  end
  @info "gates", gates
  @info "hilbert", hilbert
  return Us, hilbert
end

function line_partition_gates(N, gates; order=1)
  @assert order in [1, 2, 3, 0]
  nrow, ncol = N
  if order in [0, 1]
    partition_qubits = [[(c - 1) * nrow + r for r in 1:nrow] for c in 1:ncol]
  elseif order == 2
    # TODO: add c = 1
    partition_qubits = [
      [(c1 - 1) * nrow + r for r in 1:nrow for c1 in [c, c + 1]] for c in 2:2:(ncol - 1)
    ]
  else
    partition_qubits = [
      [(c1 - 1) * nrow + r for r in 1:nrow for c1 in [c, c + 1]] for c in 1:2:ncol
    ]
  end
  @info partition_qubits
  partition_gates = [filter(g -> all(q -> q in p, g[2]), gates) for p in partition_qubits]
  return partition_gates
end

function random_circuit_tn(N, depth; twoqubitgates="CX", onequbitgates="Ry")
  @assert N isa Number || length(N) <= 2
  layers = randomcircuit(
    N; depth=depth, twoqubitgates=twoqubitgates, onequbitgates=onequbitgates
  )
  for (i, layer) in enumerate(layers)
    @info i
    @info layer
  end
  gates = reduce(vcat, layers)
  hilbert = qubits(prod(N))
  init_state = Vector{ITensor}([ITensor([1.0, 0.0], i) for i in hilbert])
  circuit_tensors, hilbert = gate_tensors(hilbert, gates)
  circuit_tensors_dag, hilbert = gate_tensors(hilbert, reverse(gates); applydag=true)
  final_state = Vector{ITensor}([ITensor([1.0, 0.0], i) for i in hilbert])
  return init_state, circuit_tensors, circuit_tensors_dag, final_state
end

function random_circuit_tn_line_partition_tree(
  N, depth; twoqubitgates="CX", onequbitgates="Ry"
)
  @assert N isa Number || length(N) <= 2
  layers = randomcircuit(
    N; depth=depth, twoqubitgates=twoqubitgates, onequbitgates=onequbitgates
  )
  partition_gates = []
  for (i, l) in enumerate(layers)
    push!(partition_gates, line_partition_gates(N, l; order=i % 4)...)
  end
  hilbert = qubits(prod(N))
  tensor_partitions = [productstate(hilbert)[:]]
  for gates in partition_gates
    tensors, hilbert = gate_tensors(hilbert, gates)
    push!(tensor_partitions, tensors)
  end
  for gates in reverse(partition_gates)
    tensors, hilbert = gate_tensors(hilbert, reverse(gates); applydag=true)
    push!(tensor_partitions, tensors)
  end
  push!(tensor_partitions, productstate(hilbert)[:])
  return line_network(tensor_partitions)
end

function line_network(network::Vector)
  tntree = network[1]
  for i in 2:length(network)
    if network[i] isa Vector
      tntree = [tntree, network[i]]
    else
      tntree = [tntree, [network[i]]]
    end
  end
  return tntree
end

function element_grouping(init_state, circuit_tensors, circuit_tensors_dag, final_state)
  tntree = line_network([init_state, circuit_tensors...])
  tntree = line_network([tntree, circuit_tensors_dag...])
  return [tntree, final_state]
end

function insert_tntree(tntree2, tntree1, t::ITensor)
  if t in tntree2
    return [setdiff(tntree2, [t]), tntree1]
  end
  if t in vectorize(tntree2[1])
    return [insert_tntree(tntree2[1], tntree1, t), tntree2[2]]
  else
    @assert t in vectorize(tntree2[2])
    return [tntree2[1], insert_tntree(tntree2[2], tntree1, t)]
  end
end

function random_tntree(
  init_state, circuit_tensors, circuit_tensors_dag, final_state; nvertices_per_partition
)
  net = Vector{ITensor}([init_state..., circuit_tensors...])
  tntree1 = build_tntree_balanced(
    ITensorNetwork(net); nvertices_per_partition=nvertices_per_partition
  )
  new_t = ITensor(noncommoninds(net...)...)
  net = Vector{ITensor}([new_t, circuit_tensors_dag...])
  tntree2 = build_tntree_balanced(
    ITensorNetwork(net); nvertices_per_partition=nvertices_per_partition
  )
  tntree = insert_tntree(tntree2, tntree1, new_t)
  return [tntree, final_state]
end

function boundary_to_center_tntree(
  init_state, circuit_tensors, circuit_tensors_dag, final_state; nvertices_per_partition
)
  network1 = vcat(init_state, circuit_tensors)
  network2 = vcat(final_state, circuit_tensors_dag)
  tntree1 = build_tntree_balanced(
    ITensorNetwork(network1); nvertices_per_partition=nvertices_per_partition
  )
  tntree2 = build_tntree_balanced(
    ITensorNetwork(network2); nvertices_per_partition=nvertices_per_partition
  )
  return [tntree1, tntree2]
end

N = (8, 8)
depth = 6
# init_state, circuit_tensors, circuit_tensors_dag, final_state = random_circuit_tn(N, depth)
# tntree = boundary_to_center_tntree(init_state, circuit_tensors, circuit_tensors_dag, final_state; nvertices_per_partition=4)
tntree = random_circuit_tn_line_partition_tree(N, depth)
@time bench_lnZ(
  tntree;
  num_iter=1,
  cutoff=1e-12,
  maxdim=512,
  ansatz="mps",
  algorithm="density_matrix",
  use_cache=false,
  ortho=false,
  swap_size=8,
)
