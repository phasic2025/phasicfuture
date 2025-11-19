#!/usr/bin/env julia
# Standalone web server launcher
# Run: julia start_web_monitor.jl

println("=" ^ 70)
println("Neural Network Web Monitor - Starting...")
println("=" ^ 70)

# Check and install dependencies
println("\nChecking dependencies...")
try
    using HTTP
    using JSON
    println("✅ HTTP and JSON found")
catch
    println("Installing HTTP and JSON...")
    using Pkg
    Pkg.add("HTTP")
    Pkg.add("JSON")
    using HTTP
    using JSON
    println("✅ Dependencies installed")
end

using Random
using LinearAlgebra
using Sockets

# Include network code (or define inline if file doesn't exist)
if isfile("NEURAL_NETWORK_UI.jl")
    include("NEURAL_NETWORK_UI.jl")
else
    println("⚠️  NEURAL_NETWORK_UI.jl not found, using inline definitions...")
    # Minimal network definitions
    mutable struct Neuron
        activation::Float64
        phase::Float64
        frequency::Float64
        position::Vector{Float64}
        morphology::Float64
    end
    
    mutable struct NeuralNetwork
        neurons::Vector{Neuron}
        connections::Matrix{Float64}
        n_neurons::Int
        time::Float64
    end
    
    function create_network(n_neurons::Int = 50)
        neurons = Vector{Neuron}()
        for i in 1:n_neurons
            r = sqrt(rand()) * 0.9
            theta = rand() * 2π
            position = [r * cos(theta), r * sin(theta)]
            neuron = Neuron(rand() * 0.5, rand() * 2π, 1.0 + 0.2 * randn(), position, rand() * 0.5 + 0.5)
            push!(neurons, neuron)
        end
        connections = zeros(n_neurons, n_neurons)
        for i in 1:n_neurons
            for j in 1:n_neurons
                if i != j
                    dist = norm(neurons[i].position - neurons[j].position)
                    if dist < 0.3
                        connections[i, j] = 0.1 * exp(-dist / 0.1)
                    end
                end
            end
        end
        return NeuralNetwork(neurons, connections, n_neurons, 0.0)
    end
    
    function kuramoto_update!(network::NeuralNetwork, dt::Float64)
        lambda = 0.2
        for i in 1:network.n_neurons
            dphi_dt = network.neurons[i].frequency
            for j in 1:network.n_neurons
                if network.connections[i, j] > 0
                    dist = norm(network.neurons[i].position - network.neurons[j].position)
                    distance_weight = exp(-dist / lambda)
                    phase_diff = network.neurons[j].phase - network.neurons[i].phase
                    coupling = network.connections[i, j] * sin(phase_diff) * distance_weight
                    dphi_dt += coupling
                end
            end
            network.neurons[i].phase += dt * dphi_dt
            network.neurons[i].phase = mod(network.neurons[i].phase, 2π)
        end
    end
    
    function wave_propagation!(network::NeuralNetwork, dt::Float64)
        for i in 1:network.n_neurons
            network.neurons[i].activation = 0.5 * (1 + sin(network.neurons[i].phase))
            for j in 1:network.n_neurons
                if network.connections[i, j] > 0
                    dist = norm(network.neurons[i].position - network.neurons[j].position)
                    network.neurons[i].activation += 0.1 * network.connections[i, j] * 
                        network.neurons[j].activation * exp(-dist / 0.2)
                end
            end
            network.neurons[i].activation *= 0.95
            network.neurons[i].activation = clamp(network.neurons[i].activation, 0.0, 1.0)
        end
    end
    
    function hebbian_update!(network::NeuralNetwork, learning_rate::Float64 = 0.01)
        for i in 1:network.n_neurons
            for j in 1:network.n_neurons
                if i != j && network.connections[i, j] > 0
                    hebbian_term = learning_rate * 
                        network.neurons[i].activation * 
                        network.neurons[j].activation * 
                        cos(network.neurons[i].phase - network.neurons[j].phase)
                    network.connections[i, j] += hebbian_term
                    network.connections[i, j] = clamp(network.connections[i, j], 0.0, 1.0)
                end
            end
        end
    end
    
    function order_parameter(network::NeuralNetwork)
        n = network.n_neurons
        z = sum(exp(im * neuron.phase) for neuron in network.neurons) / n
        return abs(z), angle(z)
    end
end

# Server state
mutable struct ServerState
    network::NeuralNetwork
    running::Bool
    update_count::Int
end

state = ServerState(create_network(50), true, 0)

# HTML Dashboard (same as before)
const HTML_DASHBOARD = read("web_neural_network_server.jl") do f
    # Extract HTML from the file or use default
    include_string(@__MODULE__, read(f, String))
end

# Use the HTML from web_neural_network_server.jl
include("web_neural_network_server.jl")

println("\n" * "=" ^ 70)
println("Server starting on http://localhost:8080")
println("Open this URL in your browser!")
println("=" ^ 70)
println("\nPress Ctrl+C to stop\n")

# Start server
Random.seed!(42)
run_web_server(8080)

