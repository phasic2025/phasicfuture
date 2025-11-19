#!/usr/bin/env julia
# Quick launcher for Neural Network UI
# Run: julia run_neural_network_ui.jl

println("=" ^ 70)
println("Neural Network UI Launcher")
println("=" ^ 70)

# Check if GLMakie is installed
try
    using GLMakie
    println("✅ GLMakie found")
catch
    println("⚠️  GLMakie not found. Installing...")
    using Pkg
    Pkg.add("GLMakie")
    using GLMakie
    println("✅ GLMakie installed")
end

println("\nLoading neural network UI...")
include("NEURAL_NETWORK_UI.jl")

println("\n" * "=" ^ 70)
println("Starting Neural Network Monitor...")
println("=" ^ 70)
println("\nYou will see:")
println("  • Neurons in hyperbolic space (Poincaré disk)")
println("  • Connection strengths (learned via Hebbian)")
println("  • Activations (color-coded)")
println("  • Wave phases (arrows)")
println("  • Synchronization order parameter")
println("  • Real-time metrics")
println("\nClose the window to stop.\n")

Random.seed!(42)
run_interactive_ui()

