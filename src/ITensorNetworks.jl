module ITensorNetworks

using Compat
using DataGraphs
using Dictionaries
using DocStringExtensions
using Graphs
using Graphs.SimpleGraphs # AbstractSimpleGraph
using ITensors
using ITensors.ContractionSequenceOptimization
using ITensors.ITensorVisualizationCore
using KrylovKit: KrylovKit
using NamedGraphs
using Observers
using Printf
using Requires
using SimpleTraits
using SplitApplyCombine
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
  check_hascommoninds,
  commontags,
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

include("utils.jl")
include("visualize.jl")
include("graphs.jl")
include("itensors.jl")
include("partition.jl")
include("lattices.jl")
include("abstractindsnetwork.jl")
include("indextags.jl")
include("indsnetwork.jl")
include("opsum.jl") # Requires IndsNetwork
include("sitetype.jl")
include("abstractitensornetwork.jl")
include("contraction_sequences.jl")
include("apply.jl")
include("expect.jl")
include("models.jl")
include("tebd.jl")
include("itensornetwork.jl")
include("specialitensornetworks.jl")
include("renameitensornetwork.jl")
include("boundarymps.jl")
include("beliefpropagation.jl")
include(joinpath("treetensornetworks", "treetensornetwork.jl"))
# Compatibility of ITensor observer and Observers
# TODO: Delete this
include(joinpath("treetensornetworks", "solvers", "update_observer.jl"))
# Utilities for making it easier
# to define solvers (like ODE solvers)
# for TDVP
include(joinpath("treetensornetworks", "solvers", "solver_utils.jl"))
include(joinpath("treetensornetworks", "solvers", "applyexp.jl"))
include(joinpath("treetensornetworks", "solvers", "tdvporder.jl"))
include(joinpath("treetensornetworks", "solvers", "tdvpinfo.jl"))
include(joinpath("treetensornetworks", "solvers", "tdvp_step.jl"))
include(joinpath("treetensornetworks", "solvers", "tdvp_generic.jl"))
include(joinpath("treetensornetworks", "solvers", "tdvp.jl"))
include(joinpath("treetensornetworks", "solvers", "dmrg.jl"))
include(joinpath("treetensornetworks", "solvers", "dmrg_x.jl"))
include(joinpath("treetensornetworks", "solvers", "projmpo_apply.jl"))
include(joinpath("treetensornetworks", "solvers", "contract_mpo_mps.jl"))
include(joinpath("treetensornetworks", "solvers", "projmps2.jl"))
include(joinpath("treetensornetworks", "solvers", "projmpo_mps2.jl"))
include(joinpath("treetensornetworks", "solvers", "linsolve.jl"))

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

end
