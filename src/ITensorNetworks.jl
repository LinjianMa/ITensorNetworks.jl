module ITensorNetworks

using AbstractTrees
using Combinatorics
using Compat
using DataGraphs
using DataStructures
using Dictionaries
using Distributions
using DocStringExtensions
using Graphs
using GraphsFlows
using Graphs.SimpleGraphs # AbstractSimpleGraph
using IsApprox
using ITensors
using ITensors.ContractionSequenceOptimization
using ITensors.ITensorVisualizationCore
using ITensors.LazyApply
using IterTools
using KrylovKit: KrylovKit
using NamedGraphs
using Observers
using Observers.DataFrames: select!
using Printf
using Requires
using SimpleTraits
using SparseArrayKit
using SplitApplyCombine
using StaticArrays
using Suppressor
using TimerOutputs

using DataGraphs: IsUnderlyingGraph, edge_data_type, vertex_data_type
using Graphs: AbstractEdge, AbstractGraph, Graph, add_edge!
using ITensors:
  @Algorithm_str,
  @debug_check,
  @timeit_debug,
  AbstractMPS,
  Algorithm,
  OneITensor,
  check_hascommoninds,
  commontags,
  dim,
  orthocenter,
  ProjMPS,
  set_nsite!
using KrylovKit: exponentiate, eigsolve, linsolve
using NamedGraphs:
  AbstractNamedGraph,
  parent_graph,
  vertex_to_parent_vertex,
  parent_vertices_to_vertices,
  not_implemented

include("imports.jl")

# TODO: Move to `DataGraphs.jl`
edge_data_type(::AbstractNamedGraph) = Any
isassigned(::AbstractNamedGraph, ::Any) = false
function iterate(::AbstractDataGraph)
  return error(
    "Iterating data graphs is not yet defined. We may define it in the future as iterating through the vertex and edge data.",
  )
end

include("observers.jl")
include("visualize.jl")
include("graphs.jl")
include("itensors.jl")
include("partition.jl")
include("lattices.jl")
include("abstractindsnetwork.jl")
include("indextags.jl")
include("indsnetwork.jl")
include("opsum.jl")
include("sitetype.jl")
include("abstractitensornetwork.jl")
include("contraction_sequences.jl")
include("apply.jl")
include("expect.jl")
include("models.jl")
include("tebd.jl")
include("itensornetwork.jl")
include("mincut.jl")
include("contract_deltas.jl")
include("binary_tree_partition.jl")
include(joinpath("approx_itensornetwork", "utils.jl"))
include(joinpath("approx_itensornetwork", "density_matrix.jl"))
include(joinpath("approx_itensornetwork", "ttn_svd.jl"))
include(joinpath("approx_itensornetwork", "approx_itensornetwork.jl"))
include(joinpath("partitioned_contract", "utils.jl"))
include(joinpath("partitioned_contract", "adjacency_tree.jl"))
include(joinpath("partitioned_contract", "constrained_ordering.jl"))
include(joinpath("partitioned_contract", "partitioned_contract.jl"))
include("contract.jl")
include("utility.jl")
include("specialitensornetworks.jl")
include("renameitensornetwork.jl")
include("boundarymps.jl")
include("beliefpropagation.jl")
include("contraction_tree_to_graph.jl")
include("gauging.jl")
include("utils.jl")
include("tensornetworkoperators.jl")
include(joinpath("ITensorsExt", "itensorutils.jl"))
include(joinpath("Graphs", "abstractgraph.jl"))
include(joinpath("Graphs", "abstractdatagraph.jl"))
include(joinpath("treetensornetworks", "abstracttreetensornetwork.jl"))
include(joinpath("treetensornetworks", "ttn.jl"))
include(joinpath("treetensornetworks", "opsum_to_ttn.jl"))
include(joinpath("treetensornetworks", "projttns", "abstractprojttn.jl"))
include(joinpath("treetensornetworks", "projttns", "projttn.jl"))
include(joinpath("treetensornetworks", "projttns", "projttnsum.jl"))
include(joinpath("treetensornetworks", "projttns", "projttn_apply.jl"))
include(joinpath("treetensornetworks", "solvers", "solver_utils.jl"))
include(joinpath("treetensornetworks", "solvers", "applyexp.jl"))
include(joinpath("treetensornetworks", "solvers", "update_step.jl"))
include(joinpath("treetensornetworks", "solvers", "alternating_update.jl"))
include(joinpath("treetensornetworks", "solvers", "tdvp.jl"))
include(joinpath("treetensornetworks", "solvers", "dmrg.jl"))
include(joinpath("treetensornetworks", "solvers", "dmrg_x.jl"))
include(joinpath("treetensornetworks", "solvers", "contract.jl"))
include(joinpath("treetensornetworks", "solvers", "linsolve.jl"))
include(joinpath("treetensornetworks", "solvers", "tree_sweeping.jl"))

include("exports.jl")

function __init__()
  @require KaHyPar = "2a6221f6-aa48-11e9-3542-2d9e0ef01880" include(
    joinpath("requires", "kahypar.jl")
  )
  @require Metis = "2679e427-3c69-5b7f-982b-ece356f1e94b" include(
    joinpath("requires", "metis.jl")
  )
  @require OMEinsumContractionOrders = "6f22d1fd-8eed-4bb7-9776-e7d684900715" include(
    joinpath("requires", "omeinsumcontractionorders.jl")
  )
end

# TODO: the flop counter below is just for benchmark purposes.
# It is not supposed to be merged into the main branch.
QR_FLOPS = 0
SVD_FLOPS = 0
CONTRACT_FLOPS = 0
function reset_flops()
  QR_FLOPS = 0
  SVD_FLOPS = 0
  return CONTRACT_FLOPS = 0
end

function print_flops()
  flops = QR_FLOPS + SVD_FLOPS + CONTRACT_FLOPS
  @info "The number of QR Gflops is $(QR_FLOPS / 10^9)"
  @info "The number of SVD Gflops is $(SVD_FLOPS / 10^9)"
  @info "The number of contract Gflops is $(CONTRACT_FLOPS / 10^9)"
  @info "The number of total Gflops is $(flops / 10^9)"
end

end
