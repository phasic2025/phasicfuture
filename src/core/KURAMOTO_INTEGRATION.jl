# Kuramoto Model Integration for Synchronized Neuron Firing
# Demonstrates phase synchronization in hyperbolic neural networks

using LinearAlgebra
using Statistics

# ============================================================================
# 1. HYPERBOLIC KURAMOTO MODEL
# ============================================================================

"""
    hyperbolic_distance(u, v)

Compute hyperbolic distance in Poincaré disk model.
"""
function hyperbolic_distance(u::Vector{Float64}, v::Vector{Float64})
    # Poincaré disk metric
    num = norm(u - v)^2
    den = (1 - norm(u)^2) * (1 - norm(v)^2)
    return acosh(1 + 2 * num / den)
end

"""
    kuramoto_phase_update(phases, frequencies, coupling_matrix, positions, dt, lambda)

Update phases according to hyperbolic Kuramoto model:
    dφᵢ/dt = ωᵢ + Σⱼ (Kᵢⱼ/|N(i)|) · sin(φⱼ - φᵢ) · exp(-dᵢⱼ/λ)

Where:
- phases: current phases φᵢ(t)
- frequencies: natural frequencies ωᵢ
- coupling_matrix: Kᵢⱼ (learned via Hebbian)
- positions: neuron positions in hyperbolic space
- lambda: distance decay parameter
"""
function kuramoto_phase_update(phases::Vector{Float64},
                                frequencies::Vector{Float64},
                                coupling_matrix::Matrix{Float64},
                                positions::Matrix{Float64},
                                dt::Float64,
                                lambda::Float64 = 1.0)
    n = length(phases)
    new_phases = copy(phases)
    
    for i in 1:n
        # Natural frequency term
        dphi_dt = frequencies[i]
        
        # Coupling term (only neighbors within boundary)
        neighbors = find_neighbors(i, positions, coupling_matrix)
        
        if length(neighbors) > 0
            coupling_sum = 0.0
            for j in neighbors
                # Hyperbolic distance
                d_ij = hyperbolic_distance(positions[i, :], positions[j, :])
                
                # Distance-dependent coupling
                distance_weight = exp(-d_ij / lambda)
                
                # Kuramoto coupling
                phase_diff = phases[j] - phases[i]
                coupling_sum += (coupling_matrix[i, j] / length(neighbors)) * 
                                sin(phase_diff) * distance_weight
            end
            dphi_dt += coupling_sum
        end
        
        # Update phase
        new_phases[i] = phases[i] + dt * dphi_dt
        
        # Wrap to [0, 2π)
        new_phases[i] = mod(new_phases[i], 2π)
    end
    
    return new_phases
end

"""
    find_neighbors(i, positions, coupling_matrix)

Find neighbors of neuron i within topological boundary.
In practice, this would use persistent homology to determine boundaries.
For now, use coupling matrix (non-zero entries indicate neighbors).
"""
function find_neighbors(i::Int, positions::Matrix{Float64}, coupling_matrix::Matrix{Float64})
    neighbors = Int[]
    for j in 1:size(coupling_matrix, 2)
        if i != j && coupling_matrix[i, j] > 0
            push!(neighbors, j)
        end
    end
    return neighbors
end

# ============================================================================
# 2. SYNCHRONIZATION ORDER PARAMETER
# ============================================================================

"""
    order_parameter(phases)

Compute synchronization order parameter:
    r(t) = |(1/N) Σⱼ exp(i·φⱼ(t))|

Returns:
- r: synchronization strength [0, 1]
- psi: average phase
"""
function order_parameter(phases::Vector{Float64})
    n = length(phases)
    
    # Complex representation
    z = sum(exp(im * phi) for phi in phases) / n
    
    # Magnitude = synchronization strength
    r = abs(z)
    
    # Average phase
    psi = angle(z)
    
    return r, psi
end

# ============================================================================
# 3. KURAMOTO-ENHANCED HEBBIAN LEARNING
# ============================================================================

"""
    kuramoto_hebbian_update(coupling_matrix, activations, phases, learning_rate, sync_weight)

Update coupling strengths via Kuramoto-enhanced Hebbian learning:
    ΔKᵢⱼ = η · [sᵢ·sⱼ·cos(φᵢ - φⱼ) + α·rᵢⱼ]

Where:
- rᵢⱼ = cos(φᵢ - φⱼ) = local synchronization measure
"""
function kuramoto_hebbian_update!(coupling_matrix::Matrix{Float64},
                                    activations::Vector{Float64},
                                    phases::Vector{Float64},
                                    learning_rate::Float64,
                                    sync_weight::Float64 = 0.2)
    n = length(phases)
    
    for i in 1:n
        for j in 1:n
            if i != j
                # Classic Hebbian term
                hebbian_term = activations[i] * activations[j] * cos(phases[i] - phases[j])
                
                # Synchronization term
                local_sync = cos(phases[i] - phases[j])
                sync_term = sync_weight * local_sync
                
                # Update coupling strength
                delta_K = learning_rate * (hebbian_term + sync_term)
                coupling_matrix[i, j] += delta_K
                
                # Keep non-negative (or allow negative for inhibition)
                coupling_matrix[i, j] = max(0.0, coupling_matrix[i, j])
            end
        end
    end
    
    return coupling_matrix
end

"""
    adaptive_learning_rate(base_rate, order_param, boost_factor)

Adapt learning rate based on global synchronization:
    η(t) = η₀ · (1 + β·r(t))

When synchronized, learning accelerates.
"""
function adaptive_learning_rate(base_rate::Float64, order_param::Float64, boost_factor::Float64 = 0.5)
    return base_rate * (1.0 + boost_factor * order_param)
end

# ============================================================================
# 4. STATE-PHASE COUPLING
# ============================================================================

"""
    activation_from_phase(phases, amplitudes, offsets)

Couple activation to phase:
    sᵢ(t) = Aᵢ · sin(φᵢ(t) + θᵢ)

Where amplitudes and offsets come from wave equation and morphology.
"""
function activation_from_phase(phases::Vector{Float64},
                                amplitudes::Vector{Float64},
                                offsets::Vector{Float64})
    activations = zeros(length(phases))
    
    for i in 1:length(phases)
        activations[i] = amplitudes[i] * sin(phases[i] + offsets[i])
    end
    
    return activations
end

# ============================================================================
# 5. COMPLETE INTEGRATION EXAMPLE
# ============================================================================

"""
    simulate_kuramoto_network(n_neurons, n_steps, dt)

Complete simulation of Kuramoto-synchronized neural network.
"""
function simulate_kuramoto_network(n_neurons::Int = 100,
                                    n_steps::Int = 1000,
                                    dt::Float64 = 0.01)
    
    # Initialize
    phases = 2π * rand(n_neurons)  # Random initial phases
    frequencies = 1.0 .+ 0.1 * randn(n_neurons)  # Natural frequencies (mean=1, std=0.1)
    
    # Random positions in Poincaré disk (within unit circle)
    positions = 0.8 * (rand(n_neurons, 2) .- 0.5)  # Keep within disk
    
    # Initial coupling matrix (sparse, random)
    coupling_matrix = 0.1 * rand(n_neurons, n_neurons)
    coupling_matrix[diagind(coupling_matrix)] .= 0.0  # No self-connections
    
    # Activation parameters
    amplitudes = ones(n_neurons)
    offsets = zeros(n_neurons)
    
    # Storage
    order_params = Float64[]
    phases_history = Matrix{Float64}(undef, n_steps, n_neurons)
    
    # Simulation loop
    for step in 1:n_steps
        # 1. Update phases (Kuramoto dynamics)
        phases = kuramoto_phase_update(phases, frequencies, coupling_matrix, 
                                       positions, dt)
        
        # 2. Compute synchronization
        r, psi = order_parameter(phases)
        push!(order_params, r)
        
        # 3. Compute activations from phases
        activations = activation_from_phase(phases, amplitudes, offsets)
        
        # 4. Adaptive learning rate
        learning_rate = adaptive_learning_rate(0.01, r, 0.5)
        
        # 5. Update coupling strengths (Hebbian learning)
        kuramoto_hebbian_update!(coupling_matrix, activations, phases, learning_rate)
        
        # Store history
        phases_history[step, :] = phases
    end
    
    return phases_history, order_params, coupling_matrix
end

# ============================================================================
# 6. DEMONSTRATION
# ============================================================================

function demonstrate_kuramoto()
    println("=" ^ 70)
    println("Kuramoto Model Integration Demonstration")
    println("=" ^ 70)
    
    println("\nSimulating network with 100 neurons...")
    phases_history, order_params, final_coupling = simulate_kuramoto_network(100, 1000, 0.01)
    
    println("\nResults:")
    println("  Initial synchronization: r(0) = $(round(order_params[1], digits=3))")
    println("  Final synchronization: r(T) = $(round(order_params[end], digits=3))")
    
    if order_params[end] > 0.7
        println("  ✓ Network achieved strong synchronization!")
    elseif order_params[end] > 0.3
        println("  ⚠ Partial synchronization (clusters formed)")
    else
        println("  ✗ Network remained desynchronized")
    end
    
    println("\nCoupling matrix statistics:")
    println("  Mean coupling: $(round(mean(final_coupling), digits=4))")
    println("  Max coupling: $(round(maximum(final_coupling), digits=4))")
    println("  Sparsity: $(round(100 * (1 - count(!iszero, final_coupling) / length(final_coupling)), digits=1))%")
    
    println("\n" * "=" ^ 70)
    println("Key Insights:")
    println("  1. Phases synchronize when coupling exceeds critical value")
    println("  2. Synchronization enables coherent wave propagation")
    println("  3. Hebbian learning strengthens synchronized connections")
    println("  4. Learning accelerates when network is synchronized")
    println("=" ^ 70)
end

# Run demonstration
if abspath(PROGRAM_FILE) == @__FILE__
    demonstrate_kuramoto()
end

# ============================================================================
# 7. INTEGRATION WITH TOPOLOGICAL BOUNDARIES
# ============================================================================

"""
    restrict_kuramoto_to_boundary(phases, frequencies, coupling_matrix, boundary_indices)

Restrict Kuramoto dynamics to neurons on topological boundaries.
This dramatically reduces computation while preserving synchronization.
"""
function restrict_kuramoto_to_boundary(phases::Vector{Float64},
                                        frequencies::Vector{Float64},
                                        coupling_matrix::Matrix{Float64},
                                        boundary_indices::Vector{Int})
    
    # Extract boundary components
    boundary_phases = phases[boundary_indices]
    boundary_frequencies = frequencies[boundary_indices]
    boundary_coupling = coupling_matrix[boundary_indices, boundary_indices]
    
    return boundary_phases, boundary_frequencies, boundary_coupling
end

"""
    compute_complexity_reduction(n_total, n_boundary)

Demonstrate computational savings from boundary restriction.
"""
function compute_complexity_reduction(n_total::Int, n_boundary::Int)
    println("\n" * "=" ^ 70)
    println("Computational Complexity Reduction via Boundaries")
    println("=" ^ 70)
    
    # Full network
    full_ops = n_total^2  # O(n²) for coupling updates
    
    # Boundary-restricted
    boundary_ops = n_boundary^2
    
    speedup = full_ops / boundary_ops
    
    println("Full network:")
    println("  Neurons: $n_total")
    println("  Operations per step: $full_ops")
    
    println("\nBoundary-restricted:")
    println("  Boundary neurons: $n_boundary")
    println("  Operations per step: $boundary_ops")
    
    println("\nSpeedup: $(round(speedup, digits=1))x")
    println("=" ^ 70)
    
    return speedup
end

