#!/usr/bin/env julia
# TFTS Architecture Demo
# Demonstrates the Topological Field-Theoretic Semantics Architecture

using Pkg
Pkg.activate("..")

using Revise
using .TFTS
using LinearAlgebra
using Plots

function main()
    println("Starting TFTS Architecture Demo")
    
    # Initialize a new TFTS network with 3D hyperbolic embeddings
    println("Initializing TFTS network...")
    network = TFTS.TFTSNetwork(3)
    
    # Add some nodes
    println("Adding nodes to the network...")
    add_node!(network.graph, "node1")
    add_node!(network.graph, "node2")
    add_node!(network.graph, "node3")
    
    # Add connections
    println("Creating network connections...")
    add_edge!(network.graph, "node1", "node2")
    add_edge!(network.graph, "node2", "node3")
    add_edge!(network.graph, "node1", "node3")
    
    # Initialize tokens for nodes
    println("Initializing economic layer...")
    network.tokens["node1"] = TFTS.TGeo(100.0)
    network.tokens["node2"] = TFTS.TGeo(100.0)
    network.tokens["node3"] = TFTS.TGeo(100.0)
    
    # Evolve the network
    println("Evolving network...")
    for i in 1:10
        TFTS.evolve_network!(network, 0.01)
        println("Evolution step ", i, " completed")
    end
    
    println("TFTS Demo completed successfully!")
end

# Run the demo
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
