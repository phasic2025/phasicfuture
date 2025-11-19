#!/usr/bin/env julia
# Install required packages for Neural Network Web Monitor

println("=" ^ 70)
println("Installing Required Packages")
println("=" ^ 70)
println("")

using Pkg

# Activate project
Pkg.activate(".")

println("Installing packages...")
println("")

# Install required packages
packages = ["HTTP", "JSON"]

for pkg in packages
    println("Installing $pkg...")
    try
        Pkg.add(pkg)
        println("✅ $pkg installed")
    catch e
        println("⚠️  Error installing $pkg: $e")
    end
end

println("")
println("=" ^ 70)
println("Package Installation Complete!")
println("=" ^ 70)
println("")
println("Now you can run:")
println("  julia --project=. web_neural_network_server.jl")
println("")
println("Or use:")
println("  ./run_server.sh")
println("")

