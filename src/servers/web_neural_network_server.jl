# Web-based Neural Network Monitor Server
# Serves real-time neural network visualization via HTTP

using HTTP
using Sockets
using JSON
using Random
using LinearAlgebra
using Statistics

# Network code (inline to avoid GLMakie dependency from NEURAL_NETWORK_UI.jl)

# Neuron structure (with individual time dimension - Sakana AI continuous thought)
mutable struct Neuron
    activation::Float64
    phase::Float64
    frequency::Float64
    position::Vector{Float64}
    morphology::Float64
    local_time::Float64  # Individual time dimension τᵢ(t) - evolves independently
    time_dilation::Float64  # dτᵢ/dt - how fast neuron's time flows relative to global time
    contribution::Float64  # Contribution score 𝒞ᵢ(t) - measures impact on goal pursuit via counterfactual impact
end

# Neural network structure
mutable struct NeuralNetwork
    neurons::Vector{Neuron}
    connections::Matrix{Float64}
    n_neurons::Int
    time::Float64
    boundary_indices::Vector{Int}  # Indices of neurons on topological boundaries
    boundary_cache_time::Float64  # When boundaries were last computed
end

# ============================================================================
# GLOBAL SIGNALS (Indirect Propagation - Like Hunger Signals)
# ============================================================================

mutable struct GlobalSignal
    signal_type::Symbol  # :hunger, :goal_drive, :satisfaction, :curiosity
    intensity::Float64
    decay_rate::Float64
    target_goal::Symbol
end

mutable struct GlobalSignalSystem
    signals::Vector{GlobalSignal}
    signal_history::Vector{Dict}
end

function create_global_signal_system()
    return GlobalSignalSystem([], [])
end

# ============================================================================
# GOAL SYSTEM
# ============================================================================

mutable struct Goal
    id::Symbol
    description::String
    terminal::Bool
    dependencies::Vector{Symbol}
    achieved::Bool
    value::Float64
    progress::Float64
    state::Symbol  # :active, :paused, :achieved, :abandoned
    created_at::Float64
    last_active::Float64
    switch_count::Int  # How many times this goal was paused/resumed
    abandonment_risk::Float64  # Measurable decay rate based on time_since_active, switch_count, progress_rate, goal_value (0-1)
end

mutable struct GoalHierarchy
    goals::Dict{Symbol, Goal}
    active_goal::Symbol
    terminal_goals::Vector{Symbol}
    goal_history::Vector{Dict}  # Track goal state changes
    dropped_goals::Vector{Symbol}  # Goals that were abandoned
end

# Helper function to create a Goal with default persistence tracking
function create_goal(id::Symbol, description::String, terminal::Bool, dependencies::Vector{Symbol}, current_time::Float64 = 0.0)
    return Goal(id, description, terminal, dependencies, false, 0.0, 0.0, :active, current_time, current_time, 0, 0.0)  # abandonment_risk initialized to 0.0
end

# ============================================================================
# CONCEPT EMBEDDINGS
# ============================================================================

mutable struct ConceptEmbedding
    embeddings::Dict{String, Vector{Float64}}  # word -> embedding vector
    embedding_dim::Int
    concept_memory::Dict{String, Dict}  # concept -> associated memories
    word_to_goal_associations::Dict{String, Dict{Symbol, Float64}}  # word -> goal_type -> strength (learned via Hebbian)
    action_verbs::Dict{String, Symbol}  # "make" -> :design, "learn" -> :learn, etc.
end

function create_concept_embedding(dim::Int = 64)
    # Initialize action verb mappings (seed knowledge - network will learn more through experience)
    action_verbs = Dict{String, Symbol}(
        "make" => :design, "create" => :design, "build" => :design, "design" => :design,
        "learn" => :learn, "understand" => :learn, "study" => :learn, "research" => :learn,
        "integrate" => :integrate, "combine" => :integrate, "assemble" => :integrate
    )
    
    return ConceptEmbedding(
        Dict{String, Vector{Float64}}(), 
        dim, 
        Dict{String, Dict}(),
        Dict{String, Dict{Symbol, Float64}}(),  # word_to_goal_associations - starts empty, learns from experience
        action_verbs
    )
end

# ============================================================================
# PATTERN MODEL (Kurzweil-Style)
# ============================================================================

mutable struct PatternModel
    patterns::Dict{String, Dict}  # pattern_id -> pattern data
    predictions::Dict{Int, Float64}  # neuron_id -> predicted activation
    invariants::Dict{String, Float64}  # topological invariants
    last_update::Float64
end

function create_pattern_model()
    return PatternModel(Dict(), Dict(), Dict(), 0.0)
end

# ============================================================================
# INTERNET QUERY SYSTEM
# ============================================================================

mutable struct InternetQuerySystem
    cache::Dict{String, Dict}  # Cache query results
    query_history::Vector{Dict}  # Track queries
    rate_limit::Float64  # Queries per minute
    last_query_time::Float64
end

function create_internet_query_system(rate_limit::Float64 = 10.0)
    return InternetQuerySystem(Dict(), [], rate_limit, 0.0)
end

# ============================================================================
# THOUGHT PROCESS LOGGING
# ============================================================================

mutable struct ThoughtLog
    timestamp::Float64
    neuron_id::Int
    thought_type::Symbol  # :activation, :synchronization, :wave_propagation, :goal_pursuit, :learning, :understanding, :confusion, :stagnation, :uncertainty
    message::String
    data::Dict  # Additional data (activation, phase, goal, etc.)
end

mutable struct ThoughtStream
    thoughts::Vector{ThoughtLog}
    max_thoughts::Int
end

function create_thought_stream(max_thoughts::Int = 1000)
    return ThoughtStream(ThoughtLog[], max_thoughts)
end

# ============================================================================
# METACOGNITIVE SIGNALS
# ============================================================================

mutable struct MetacognitiveSignals
    confusion_signals::Dict{Symbol, Float64}  # Goal -> confusion signal C_G(t)
    stagnation_signals::Dict{Symbol, Float64}  # Goal -> stagnation signal S_G(t)
    last_progress_time::Dict{Symbol, Float64}  # Goal -> time of last progress
    time_window::Float64  # T: time window for aggregation (e.g., 100 timesteps)
    decay_rate::Float64  # α: exponential decay rate (typically 0.01-0.05)
end

function create_metacognitive_signals(time_window::Float64 = 100.0, decay_rate::Float64 = 0.02)
    return MetacognitiveSignals(Dict{Symbol, Float64}(), Dict{Symbol, Float64}(), Dict{Symbol, Float64}(), time_window, decay_rate)
end

# ============================================================================
# HYPERBOLIC GEOMETRY (Poincaré Disk Model)
# ============================================================================

# Hyperbolic distance in Poincaré disk model
# CRITICAL: This is the foundation for proper hyperbolic geometry
# Theory: d(u,v) = arccosh(1 + 2*||u-v||²/((1-||u||²)(1-||v||²)))
function hyperbolic_distance(u::Vector{Float64}, v::Vector{Float64})::Float64
    # Ensure positions are within unit disk (Poincaré disk constraint)
    u_norm = min(norm(u), 0.999)  # Prevent division by zero
    v_norm = min(norm(v), 0.999)
    
    # Compute hyperbolic distance
    diff_sq = norm(u - v)^2
    denom = (1 - u_norm^2) * (1 - v_norm^2)
    
    if denom < 1e-10
        return 10.0  # Large distance if too close to boundary
    end
    
    arg = 1 + 2 * diff_sq / denom
    if arg < 1.0
        arg = 1.0  # Ensure arccosh is valid
    end
    
    return acosh(arg)
end

# Hyperbolic Laplacian (simplified for boundary-constrained computation)
# CRITICAL: Only computed on boundaries for efficiency
function hyperbolic_laplacian(network::NeuralNetwork, neuron_idx::Int, boundary_indices::Vector{Int})::Float64
    neuron = network.neurons[neuron_idx]
    laplacian = 0.0
    
    # Only compute from boundary neighbors
    boundary_neighbors = [j for j in boundary_indices if network.connections[neuron_idx, j] > 0.01]
    
    if isempty(boundary_neighbors)
        return 0.0
    end
    
    for j in boundary_neighbors
        neighbor = network.neurons[j]
        # Hyperbolic distance
        d_ij = hyperbolic_distance(neuron.position, neighbor.position)
        
        # Laplacian contribution (simplified - full implementation would use hyperbolic metric tensor)
        # Weight by connection strength and distance
        weight = network.connections[neuron_idx, j] * exp(-d_ij)
        laplacian += weight * (neighbor.activation - neuron.activation)
    end
    
    return laplacian / length(boundary_neighbors)
end

# Create network
function create_network(n_neurons::Int = 1000)
    neurons = Vector{Neuron}()
    for i in 1:n_neurons
        r = sqrt(rand()) * 0.9
        theta = rand() * 2π
        position = [r * cos(theta), r * sin(theta)]
        # Each neuron starts with its own time dimension (Sakana AI continuous thought)
        local_time = rand() * 0.1  # Small random initial offset
        time_dilation = 0.8 + 0.4 * rand()  # Initial dilation between 0.8-1.2
        contribution = 0.0  # Initial contribution score
        neuron = Neuron(rand() * 0.5, rand() * 2π, 1.0 + 0.2 * randn(), position, rand() * 0.5 + 0.5, local_time, time_dilation, contribution)
        push!(neurons, neuron)
    end
    connections = zeros(n_neurons, n_neurons)
    for i in 1:n_neurons
        for j in 1:n_neurons
            if i != j
                # Use hyperbolic distance instead of Euclidean
                hyp_dist = hyperbolic_distance(neurons[i].position, neurons[j].position)
                # Connect neurons within reasonable hyperbolic distance
                if hyp_dist < 2.0  # Hyperbolic distance threshold
                    connections[i, j] = 0.1 * exp(-hyp_dist / 0.5)
                end
            end
        end
    end
    return NeuralNetwork(neurons, connections, n_neurons, 0.0, Int[], -1.0)
end

# Topological boundary detection using persistent homology
# This is the KEY efficiency mechanism: only compute on boundaries
function compute_topological_boundaries!(network::NeuralNetwork, persistence_threshold::Float64 = 0.1)
    # Check if we need to recompute boundaries (cache for efficiency)
    if network.boundary_cache_time >= 0 && (network.time - network.boundary_cache_time) < 0.5
        return network.boundary_indices  # Use cached boundaries
    end
    
    n = network.n_neurons
    
    # Extract neuron activations and positions for topology computation
    activations = [n.activation for n in network.neurons]
    positions = [n.position for n in network.neurons]
    
    # Method 1: Use persistent homology (Ripserer.jl) if available
    # Try to use Ripserer.jl for proper persistent homology
    try
        # Check if Ripserer is available (would need to be added to Project.toml)
        # using Ripserer  # Uncomment when Ripserer.jl is added
        
        # Compute barcode B(X) of neuron activations
        # Convert positions to matrix format
        pos_matrix = hcat(positions...)
        
        # Compute persistent homology barcode
        # barcode = ripserer(pos_matrix, maxdim=1)
        
        # Extract persistent features (boundaries are persistent features)
        # H_0: connected components, H_1: loops, H_2: voids
        # persistent_features = extract_persistent_features(barcode, persistence_threshold)
        # boundary_indices = persistent_features
        
        # For now, fall through to Method 2 since Ripserer integration requires package installation
    catch
        # Fall back to simplified method if Ripserer not available
    end
    
    # Method 2: Simplified boundary detection using hyperbolic distance and activation variance
    # Boundary neurons are those with high local variation in activation (using hyperbolic neighbors)
    boundary_indices = Int[]
    
    for i in 1:n
        # Compute local activation variance using hyperbolic neighbors
        neighbor_activations = Float64[]
        neighbor_distances = Float64[]
        
        for j in 1:n
            if network.connections[i, j] > 0.01  # Only consider connected neighbors
                # Use hyperbolic distance
                hyp_dist = hyperbolic_distance(network.neurons[i].position, network.neurons[j].position)
                if hyp_dist < 1.5  # Only consider nearby neighbors in hyperbolic space
                    push!(neighbor_activations, network.neurons[j].activation)
                    push!(neighbor_distances, hyp_dist)
                end
            end
        end
        
        if length(neighbor_activations) > 2
            local_mean = mean(neighbor_activations)
            local_std = std(neighbor_activations)
            
            # Boundary if activation differs significantly from neighbors (in hyperbolic space)
            activation_diff = abs(network.neurons[i].activation - local_mean)
            threshold = persistence_threshold * (1 + local_std)
            
            if activation_diff > threshold
                push!(boundary_indices, i)
            end
        elseif network.neurons[i].activation > 0.7  # High activation neurons are likely boundaries
            push!(boundary_indices, i)
        end
    end
    
    # Ensure we have at least some boundary neurons (fallback)
    if isempty(boundary_indices)
        # Select top 10% by activation as boundaries
        sorted_indices = sortperm(activations, rev=true)
        boundary_indices = sorted_indices[1:max(10, div(n, 10))]
    end
    
    network.boundary_indices = boundary_indices
    network.boundary_cache_time = network.time
    
    return boundary_indices
end

# Helper function to extract persistent features from barcode (for Ripserer integration)
# This would be used when Ripserer.jl is properly integrated
# Theory: Extract H_0 (connected components), H_1 (loops), H_2 (voids) that persist beyond threshold
function extract_persistent_features(barcode, persistence_threshold::Float64)
    # TODO: Implement when Ripserer.jl is integrated
    # Extract features that persist beyond threshold
    # Return indices of boundary neurons
    # Example pseudocode:
    #   boundary_indices = Int[]
    #   for feature in barcode
    #       if feature.persistence > persistence_threshold
    #           push!(boundary_indices, feature.vertices...)
    #       end
    #   end
    #   return unique(boundary_indices)
    return Int[]
end

# Update individual time dimensions (Sakana AI continuous thought)
# CRITICAL: Only updates boundary neurons for efficiency (O(|boundary|) instead of O(N))
function update_time_dimensions!(network::NeuralNetwork, neuron_goals::Dict{Int, Symbol}, hierarchy, dt::Float64)
    # Compute topological boundaries
    boundary_indices = compute_topological_boundaries!(network)
    
    # ONLY update time dimensions for boundary neurons
    for i in boundary_indices
        neuron = network.neurons[i]
        
        # Time dilation depends on:
        # 1. Activation level (high activation → faster time)
        activation_factor = 0.5 + 0.5 * neuron.activation
        
        # 2. Input intensity (strong inputs → accelerated time) - only from boundary neighbors
        boundary_neighbors = [j for j in boundary_indices if network.connections[i, j] > 0.01]
        input_sum = isempty(boundary_neighbors) ? 0.0 : sum(network.connections[i, j] * network.neurons[j].activation for j in boundary_neighbors)
        input_factor = 0.8 + 0.4 * tanh(input_sum)
        
        # 3. Goal urgency (neurons working on urgent goals → faster processing)
        goal_factor = 1.0
        if haskey(neuron_goals, i)
            goal_id = neuron_goals[i]
            if haskey(hierarchy.goals, goal_id)
                goal = hierarchy.goals[goal_id]
                if goal.id == hierarchy.active_goal && !goal.achieved
                    urgency = 0.3 + 0.7 * (1.0 - goal.progress)
                    goal_factor = 0.8 + 0.4 * urgency
                end
            end
        end
        
        # Combined time dilation
        neuron.time_dilation = activation_factor * input_factor * goal_factor
        
        # Update local time based on dilation
        neuron.local_time += dt * neuron.time_dilation
    end
    
    # Interior neurons: freeze time (no computation)
    # Only iterate over non-boundary neurons (boundary-constrained)
    interior_indices = setdiff(1:network.n_neurons, boundary_indices)
    for i in interior_indices
        network.neurons[i].time_dilation = 0.0
        # Local time doesn't advance for interior neurons
    end
end

# Kuramoto synchronization (using local time dimensions)
# CRITICAL: Only synchronizes boundary neurons (O(|boundary|^2) instead of O(N^2))
function kuramoto_update!(network::NeuralNetwork, dt::Float64)
    lambda = 0.2
    boundary_indices = compute_topological_boundaries!(network)
    
    # ONLY synchronize boundary neurons
    for i in boundary_indices
        neuron = network.neurons[i]
        # Use local time for phase updates (Sakana AI continuous thought)
        local_dt = dt * neuron.time_dilation
        
        dphi_dt = neuron.frequency
        
        # Only couple with boundary neighbors (using hyperbolic distance)
        for j in boundary_indices
            if i != j && network.connections[i, j] > 0.01
                # CRITICAL: Use hyperbolic distance instead of Euclidean
                hyp_dist = hyperbolic_distance(neuron.position, network.neurons[j].position)
                distance_weight = exp(-hyp_dist / lambda)
                phase_diff = network.neurons[j].phase - neuron.phase
                coupling = network.connections[i, j] * sin(phase_diff) * distance_weight
                dphi_dt += coupling
            end
        end
        
        neuron.phase += local_dt * dphi_dt
        neuron.phase = mod(neuron.phase, 2π)
    end
    
    # Interior neurons: phases don't update (no computation)
end

# Wave propagation with morphological reflection and state-phase coupling
# CRITICAL: Waves only propagate along boundaries (O(|boundary|) instead of O(N^2))
function wave_propagation!(network::NeuralNetwork, dt::Float64)
    boundary_indices = compute_topological_boundaries!(network)
    
    # Store previous activations for interference calculation
    prev_activations = [n.activation for n in network.neurons]
    
    # ONLY propagate waves along boundaries
    for i in boundary_indices
        neuron = network.neurons[i]
        
        # State-Phase Coupling: s_i(t) = A_i * sin(phi_i(t) + theta_i)
        # A_i = amplitude (from wave equation), theta_i = phase offset (from morphology)
        amplitude = 0.5 + 0.3 * neuron.morphology  # Amplitude depends on morphology
        phase_offset = neuron.morphology * π / 4  # Phase offset from morphology
        neuron.activation = amplitude * sin(neuron.phase + phase_offset)
        
        # Wave propagation with hyperbolic distance
        incident_wave = 0.0
        for j in boundary_indices
            if i != j && network.connections[i, j] > 0.01
                # Use hyperbolic distance for wave propagation
                hyp_dist = hyperbolic_distance(neuron.position, network.neurons[j].position)
                wave_speed = 1.0 + 0.5 * neuron.time_dilation  # Wave speed depends on local time
                time_delay = hyp_dist / wave_speed
                
                # Wave contribution (simplified time-delayed coupling)
                incident_wave += 0.1 * network.connections[i, j] * 
                    prev_activations[j] * exp(-hyp_dist / 0.3)
            end
        end
        
        # Morphological Wave Reflection
        # Reflection coefficient depends on morphology (boundary shape)
        reflection_coeff = 0.3 + 0.4 * neuron.morphology  # R(p) from morphology
        refraction_coeff = 1.0 - reflection_coeff  # T(p) = 1 - R(p)
        
        # Reflected wave (from previous activation)
        reflected_wave = reflection_coeff * prev_activations[i] * 0.9  # Decay on reflection
        
        # Transmitted wave (incident wave passes through)
        transmitted_wave = refraction_coeff * incident_wave
        
        # Interference: w_combined = w_incident + w_reflected
        combined_wave = transmitted_wave + reflected_wave
        
        # Peak Multiplication: When peaks align → s_combined = s_1 * s_2 (non-linear amplification)
        # Cancellation: When trough meets peak → s_combined = s_1 - s_2 (built-in inhibition)
        if neuron.activation > 0.5 && combined_wave > 0.5
            # Peaks align → multiplication (amplification)
            neuron.activation = neuron.activation * (1.0 + 0.3 * combined_wave)
        elseif neuron.activation < 0.3 && combined_wave > 0.7
            # Trough meets peak → cancellation (inhibition)
            neuron.activation = max(0.0, neuron.activation - 0.2 * combined_wave)
        else
            # Normal interference
            neuron.activation += combined_wave
        end
        
        # Add hyperbolic Laplacian term (boundary-constrained)
        laplacian_term = hyperbolic_laplacian(network, i, boundary_indices)
        neuron.activation += 0.05 * laplacian_term * dt
        
        # Decay
        neuron.activation *= 0.95
        neuron.activation = clamp(neuron.activation, 0.0, 1.0)
    end
    
    # Interior neurons: no wave propagation (activation decays)
    # Only iterate over non-boundary neurons (boundary-constrained)
    interior_indices = setdiff(1:network.n_neurons, boundary_indices)
    for i in interior_indices
        network.neurons[i].activation *= 0.9  # Decay for interior neurons
        network.neurons[i].activation = clamp(network.neurons[i].activation, 0.0, 1.0)
    end
end

# Hebbian learning with synchronization-dependent learning rate and phase-locked optimization
# CRITICAL: Only learns boundary connections (O(|boundary|^2) instead of O(N^2))
function hebbian_update!(network::NeuralNetwork, learning_rate::Float64 = 0.01)
    boundary_indices = compute_topological_boundaries!(network)
    
    # Compute synchronization order parameter for adaptive learning rate
    r, psi = order_parameter(network, boundary_indices)
    
    # Synchronization-Dependent Learning Rate: η(t) = η₀ · (1 + β · r(t))
    # Theory: When synchronized, learning accelerates because phase alignment makes updates more reliable
    beta = 0.3  # Synchronization boost factor
    adaptive_rate = learning_rate * (1.0 + beta * r)
    
    # Phase-Locked Optimization: When neurons are phase-locked (r > 0.8), simplify update
    # Theory: If φ_i = Ωt + ψ_i (phase-locked), then ΔK_ij = η · s_i · s_j · cos(ψ_i - ψ_j)
    # This creates stable connection patterns based on phase offsets
    is_phase_locked = r > 0.8
    
    # ONLY update connections between boundary neurons
    for i in boundary_indices
        for j in boundary_indices
            if i != j && network.connections[i, j] > 0.01
                activation_term = network.neurons[i].activation * network.neurons[j].activation
                
                if is_phase_locked
                    # Phase-Locked Learning: Simplified update using phase offsets
                    # Extract phase offset: ψ_i = φ_i - Ωt (relative to collective phase)
                    psi_i = network.neurons[i].phase - psi
                    psi_j = network.neurons[j].phase - psi
                    phase_offset_diff = psi_i - psi_j
                    
                    # Simplified: ΔK_ij = η · s_i · s_j · cos(ψ_i - ψ_j)
                    hebbian_term = adaptive_rate * activation_term * cos(phase_offset_diff)
                else
                    # Standard Kuramoto-Coupled Hebbian: ΔK_ij = η · [s_i · s_j · cos(φ_i - φ_j) + α · r_ij]
                    phase_term = cos(network.neurons[i].phase - network.neurons[j].phase)
                    
                    # Local synchronization measure
                    r_ij = abs(exp(im * (network.neurons[i].phase - network.neurons[j].phase)))
                    alpha = 0.1  # Synchronization weight
                    
                    hebbian_term = adaptive_rate * (activation_term * phase_term + alpha * r_ij)
                end
                
                network.connections[i, j] += hebbian_term
                network.connections[i, j] = clamp(network.connections[i, j], 0.0, 1.0)
            end
        end
    end
    
    # Interior connections: no learning (frozen)
end

# Energy-Based Deformation on Boundaries
# Theory: E(s, a) = ∫_∂M ||∇s||² dμ, gradient descent only on boundary points
# CRITICAL: Only computed on boundaries for efficiency O(|boundary|) instead of O(N)
function compute_energy_gradient!(network::NeuralNetwork, boundary_indices::Vector{Int}, dt::Float64)
    # Energy function: E = ∫_∂M ||∇s||² dμ
    # Gradient: ∇_a E only computed on boundary points
    
    for i in boundary_indices
        neuron = network.neurons[i]
        
        # Compute gradient of activation along boundary
        # Simplified: ∇s ≈ difference from boundary neighbors
        boundary_neighbors = [j for j in boundary_indices if network.connections[i, j] > 0.01]
        
        if !isempty(boundary_neighbors)
            # Compute activation gradient (simplified)
            neighbor_activations = [network.neurons[j].activation for j in boundary_neighbors]
            avg_neighbor_activation = mean(neighbor_activations)
            gradient_magnitude = abs(neuron.activation - avg_neighbor_activation)
            
            # Energy-based update: minimize energy by aligning activations
            # This creates smooth activation patterns along boundaries
            energy_minimization_rate = 0.01
            neuron.activation += energy_minimization_rate * (avg_neighbor_activation - neuron.activation) * dt
            neuron.activation = clamp(neuron.activation, 0.0, 1.0)
        end
    end
    
    # Interior neurons: no energy-based updates (frozen)
end

# Boundary-Guided Action Selection
# Theory: Actions restricted to boundary-respecting space
# CRITICAL: Only select actions from boundary goals, reducing action space exponentially
function select_boundary_action(hierarchy, boundary_indices::Vector{Int}, neuron_goals::Dict{Int, Symbol})
    # Extract boundary goals (goals with boundary neurons)
    boundary_goals = Set{Symbol}()
    for i in boundary_indices
        if haskey(neuron_goals, i)
            push!(boundary_goals, neuron_goals[i])
        end
    end
    
    # Only consider actions/goals that are on boundaries
    # Traditional: Consider all |G| goals → O(|G|)
    # Topological: Consider only |G_boundary| boundary goals → O(|G_boundary|)
    # Speedup: |G| / |G_boundary| (typically 5-10x reduction)
    
    if isempty(boundary_goals)
        return hierarchy.active_goal  # Fallback to current goal
    end
    
    # Select best boundary goal (simplified - could use value estimation)
    # In practice, this would use estimate_goal_value() for each boundary goal
    best_goal = hierarchy.active_goal
    best_value = 0.0
    
    for goal_id in boundary_goals
        if haskey(hierarchy.goals, goal_id)
            goal = hierarchy.goals[goal_id]
            # Simple heuristic: prefer active or high-progress goals
            value = goal.progress + (goal.id == hierarchy.active_goal ? 0.3 : 0.0)
            if value > best_value
                best_value = value
                best_goal = goal_id
            end
        end
    end
    
    return best_goal
end

# Compute critical coupling strength K_c
# Theory: K_c = 2/(π·g(0)) where g(ω) is frequency distribution
# CRITICAL: Only computed from boundary neurons for efficiency
function compute_critical_coupling_strength(network::NeuralNetwork, boundary_indices::Vector{Int})::Float64
    if isempty(boundary_indices)
        return 0.0
    end
    
    # Get frequencies of boundary neurons
    frequencies = [network.neurons[i].frequency for i in boundary_indices]
    
    # Estimate frequency distribution g(ω) at ω=0 (mean frequency)
    mean_freq = mean(frequencies)
    std_freq = std(frequencies)
    
    # Gaussian approximation: g(0) ≈ 1/(σ√(2π)) * exp(-(0-μ)²/(2σ²))
    if std_freq > 1e-10
        g_0 = 1.0 / (std_freq * sqrt(2π)) * exp(-mean_freq^2 / (2 * std_freq^2))
    else
        g_0 = 1.0  # Fallback if no variance
    end
    
    # Critical coupling strength
    K_c = 2.0 / (π * g_0)
    
    return K_c
end

# Order parameter (synchronization measure)
# CRITICAL: Can compute from boundary neurons only for efficiency
function order_parameter(network::NeuralNetwork, boundary_indices::Vector{Int} = Int[])
    if isempty(boundary_indices)
        boundary_indices = compute_topological_boundaries!(network)
    end
    
    n_boundary = length(boundary_indices)
    if n_boundary == 0
        return 0.0, 0.0
    end
    
    # Compute synchronization only from boundary neurons
    z = sum(exp(im * network.neurons[i].phase) for i in boundary_indices) / n_boundary
    return abs(z), angle(z)
end

# Compute neuron contribution scores (self-monitoring mechanism)
# Contributing neurons trigger external verification systems (global signals)
# Theory: Contribution measured by counterfactual impact - what happens to goal pursuit when neuron state changes
function compute_neuron_contribution!(network::NeuralNetwork, neuron_goals::Dict{Int, Symbol}, hierarchy, boundary_indices::Vector{Int})
    active_goal = hierarchy.active_goal
    
    # Compute synchronization of goal neurons
    goal_boundary_neurons = [i for i in boundary_indices if haskey(neuron_goals, i) && neuron_goals[i] == active_goal]
    goal_sync = 0.0
    if !isempty(goal_boundary_neurons)
        goal_phases = [network.neurons[i].phase for i in goal_boundary_neurons]
        goal_sync = abs(sum(exp(im * phase) for phase in goal_phases) / length(goal_boundary_neurons))
    end
    
    # Compute contribution for each boundary neuron
    # Theory: 𝒞ᵢ(t) = β₁·Δ_goal_progress + β₂·removal_impact + β₃·synchronization_contribution
    # Simplified implementation: Use activation, sync, and goal assignment as proxies for counterfactual impact
    for i in boundary_indices
        neuron = network.neurons[i]
        
        # Contribution components (proxies for counterfactual impact):
        # 1. Activation level (proxy for Δ_goal_progress: high activation → more contribution)
        activation_component = neuron.activation
        
        # 2. Synchronization with goal neurons (proxy for synchronization_contribution)
        sync_component = 0.0
        if haskey(neuron_goals, i) && neuron_goals[i] == active_goal
            # Compute local synchronization with other goal neurons
            goal_neighbor_phases = [network.neurons[j].phase for j in goal_boundary_neurons if j != i && network.connections[i, j] > 0.01]
            if !isempty(goal_neighbor_phases)
                local_sync = abs(sum(exp(im * (phase - neuron.phase)) for phase in goal_neighbor_phases) / length(goal_neighbor_phases))
                sync_component = local_sync
            else
                sync_component = goal_sync  # Use global goal sync if no neighbors
            end
        end
        
        # 3. Goal assignment strength (proxy for removal_impact: working on active goal → more contribution)
        goal_component = haskey(neuron_goals, i) && neuron_goals[i] == active_goal ? 1.0 : 0.0
        
        # 4. Boundary membership (boundary neuron → contributes)
        boundary_component = 1.0  # Already a boundary neuron
        
        # Combined contribution score (simplified - full implementation would compute actual counterfactuals)
        neuron.contribution = 0.3 * activation_component + 
                            0.3 * sync_component + 
                            0.2 * goal_component + 
                            0.2 * boundary_component
    end
    
    # Interior neurons have zero contribution (not contributing)
    # Only iterate over non-boundary neurons (boundary-constrained)
    interior_indices = setdiff(1:network.n_neurons, boundary_indices)
    for i in interior_indices
        network.neurons[i].contribution = 0.0
    end
    
    return boundary_indices
end

# Self-verification: Contributing neurons trigger external signals (don't communicate directly)
function trigger_contribution_signals!(network::NeuralNetwork, neuron_goals::Dict{Int, Symbol}, hierarchy, global_signals::GlobalSignalSystem, boundary_indices::Vector{Int}, contribution_threshold::Float64 = 0.7)
    active_goal = hierarchy.active_goal
    
    # Find contributing neurons
    contributing_neurons = [i for i in boundary_indices if network.neurons[i].contribution > contribution_threshold]
    
    if !isempty(contributing_neurons)
        # Compute aggregate contribution
        avg_contribution = mean([network.neurons[i].contribution for i in contributing_neurons])
        
        # Contributing neurons trigger external verification signals (not direct communication)
        # This is the KEY: neurons don't talk to each other—they trigger external systems
        if avg_contribution > contribution_threshold
            # Trigger goal drive signal (external system)
            urgency = 0.3 + 0.7 * avg_contribution
            trigger_goal_drive!(global_signals, active_goal, urgency)
            
            # If synchronization is high, emit satisfaction signal (verification)
            goal_boundary_neurons = [i for i in boundary_indices if haskey(neuron_goals, i) && neuron_goals[i] == active_goal]
            if !isempty(goal_boundary_neurons)
                goal_phases = [network.neurons[i].phase for i in goal_boundary_neurons]
                goal_sync = abs(sum(exp(im * phase) for phase in goal_phases) / length(goal_boundary_neurons))
                
                if goal_sync > 0.7
                    # High synchronization → satisfaction signal (self-verification)
                    emit_signal!(global_signals, :satisfaction, goal_sync, active_goal)
                elseif goal_sync < 0.3
                    # Low synchronization → urgency signal (self-correction)
                    emit_signal!(global_signals, :goal_drive, 1.0 - goal_sync, active_goal)
                end
            end
        end
    end
end

# ============================================================================
# EMBEDDING-BASED LANGUAGE UNDERSTANDING & MEMORY
# ============================================================================

# Get boundary-relevant concepts (concepts associated with boundary goals)
# CRITICAL: Memory retrieval only searches boundary-relevant concepts for efficiency
function get_boundary_relevant_concepts(hierarchy, neuron_goals::Dict{Int, Symbol}, boundary_indices::Vector{Int})::Set{String}
    boundary_concepts = Set{String}()
    
    # Get goals that have boundary neurons
    boundary_goal_ids = Set{Symbol}()
    for i in boundary_indices
        if haskey(neuron_goals, i)
            push!(boundary_goal_ids, neuron_goals[i])
        end
    end
    
    # Extract concepts from boundary goal descriptions
    for goal_id in boundary_goal_ids
        if haskey(hierarchy.goals, goal_id)
            goal = hierarchy.goals[goal_id]
            # Extract words from goal description
            words = split(lowercase(goal.description), r"[^a-z]+")
            words = filter(w -> length(w) > 2, words)
            for word in words
                push!(boundary_concepts, word)
            end
        end
    end
    
    return boundary_concepts
end

# Generate orthogonal embedding for a concept (prevents forgetting via interference)
# CRITICAL: Orthogonalization can be boundary-constrained for efficiency
# Define the 4-parameter version first (base implementation)
function get_or_create_embedding(ce::ConceptEmbedding, concept::String, current_time::Float64, boundary_concepts::Union{Set{String}, Nothing})::Vector{Float64}
    # CRITICAL DEBUG: This should never fail - if it does, we have a scoping issue
    # Force an error if method dispatch somehow fails
    if !(typeof(boundary_concepts) <: Union{Set{String}, Nothing})
        error("TYPE ERROR: boundary_concepts type $(typeof(boundary_concepts)) doesn't match Union{Set{String}, Nothing}")
    end
    concept_lower = lowercase(concept)
    if haskey(ce.embeddings, concept_lower)
        return ce.embeddings[concept_lower]
    end
    
    # Create new embedding orthogonal to existing ones (prevents forgetting)
    new_embedding = randn(ce.embedding_dim)
    new_embedding = new_embedding / norm(new_embedding)  # Normalize
    
    # Orthogonalize against existing embeddings (Gram-Schmidt)
    # CRITICAL: If boundary_concepts provided, only orthogonalize against boundary-relevant concepts
    embeddings_to_check = if boundary_concepts !== nothing
        # Only check boundary-relevant concepts (computational efficiency)
        [(c, e) for (c, e) in ce.embeddings if c in boundary_concepts]
    else
        # Check all concepts (fallback for backward compatibility)
        collect(ce.embeddings)
    end
    
    for (existing_concept, existing_emb) in embeddings_to_check
        overlap = dot(new_embedding, existing_emb)
        new_embedding = new_embedding - overlap * existing_emb
    end
    
    # Renormalize after orthogonalization
    if norm(new_embedding) > 1e-10
        new_embedding = new_embedding / norm(new_embedding)
    else
        # If too small, regenerate
        new_embedding = randn(ce.embedding_dim)
        new_embedding = new_embedding / norm(new_embedding)
    end
    
    ce.embeddings[concept_lower] = new_embedding
    ce.concept_memory[concept_lower] = Dict("created_at" => current_time, "usage_count" => 0)
    return new_embedding
end

# Only one method definition - no overloading to avoid Julia dispatch issues
# Always call with 4 parameters: get_or_create_embedding(ce, concept, current_time, boundary_concepts)
# Pass nothing for boundary_concepts if you don't have boundary information

# Parse prompt using embedding similarity (not just keyword matching)
# CRITICAL: Can use boundary-constrained concept search for efficiency
function parse_prompt_with_embeddings(ce::ConceptEmbedding, prompt::String, current_time::Float64 = 0.0, boundary_concepts::Union{Set{String}, Nothing} = nothing)
    prompt_lower = lowercase(prompt)
    words = split(prompt_lower, r"[^a-z]+")
    words = filter(w -> length(w) > 2, words)  # Filter short words
    
    # CRITICAL: Boundary-Constrained Embedding Creation
    # Only create/retrieve embeddings for boundary-relevant words (O(|boundary_words|) instead of O(|all_words|))
    words_to_process = if boundary_concepts !== nothing && !isempty(boundary_concepts)
        # Only process boundary-relevant words (computational efficiency)
        [w for w in words if lowercase(w) in boundary_concepts]
    else
        # If network is blank (no boundary concepts), process all words (necessary for initial parsing)
        words
    end
    
    prompt_embedding = zeros(ce.embedding_dim)
    for word in words_to_process
        # CRITICAL FIX: Convert SubString{String} to String (split() returns SubString, but method expects String)
        word_str = String(word)
        
        # Call function - always pass boundary_concepts explicitly (even if nothing)
        # CRITICAL: Explicitly type boundary_concepts to ensure Julia dispatches correctly
        boundary_concepts_typed::Union{Set{String}, Nothing} = boundary_concepts
        
        emb = get_or_create_embedding(ce, word_str, current_time, boundary_concepts_typed)
        prompt_embedding += emb
    end
    if length(words_to_process) > 0
        prompt_embedding = prompt_embedding / length(words_to_process)
    end
    
    # OPTION 4: Learn word→action mappings through experience (Boundary-Constrained)
    # Extract action verb (e.g., "make", "design", "learn") and object (e.g., "toaster")
    action_type = :design  # Default
    design_object = "object"  # Default
    
    # Boundary-Constrained: Only check learned associations for boundary-relevant words
    words_to_check = if boundary_concepts !== nothing && !isempty(boundary_concepts)
        # Only check boundary-relevant words (computational efficiency)
        [w for w in words if lowercase(w) in boundary_concepts]
    else
        # Check all words if no boundary info (fallback)
        words
    end
    
    # Find action verb using learned associations (boundary-constrained search)
    for word in words_to_check
        if haskey(ce.action_verbs, word)
            action_type = ce.action_verbs[word]
            break
        elseif haskey(ce.word_to_goal_associations, word)
            # Use learned associations - find strongest goal type (boundary-constrained)
            associations = ce.word_to_goal_associations[word]
            if !isempty(associations)
                action_type = argmax(associations)  # Most associated goal type
            end
        end
    end
    
    # Fallback: Check non-boundary words only if no boundary match found
    if action_type == :design  # Still default, check remaining words
        remaining_words = setdiff(words, words_to_check)
        for word in remaining_words
            if haskey(ce.action_verbs, word)
                action_type = ce.action_verbs[word]
                break
            end
        end
    end
    
    # Find design object (noun) - Boundary-Constrained: Check boundary words first
    for word in words_to_check
        word_str = String(word)  # Convert SubString to String
        if !haskey(ce.action_verbs, word_str) && length(word_str) > 3
            design_object = word_str
            break
        end
    end
    
    # Fallback: If no boundary match, check remaining words (only if network is blank)
    if design_object == "object"
        remaining_words = setdiff(words, words_to_check)
        for word in remaining_words
            word_str = String(word)  # Convert SubString to String
            if !haskey(ce.action_verbs, word_str) && length(word_str) > 3
                design_object = word_str
                break
            end
        end
    end
    
    # Final fallback: use first significant word
    if design_object == "object" && !isempty(words)
        design_object = String(words[1])  # Convert SubString to String
    end
    
    best_match = String(design_object)  # Ensure best_match is String, not SubString
    
    # Extract requirements from prompt words - Boundary-Constrained: Only extract from boundary-relevant words
    requirements = String[]
    # Extract significant words that aren't the design type itself (only from boundary words)
    for word in words_to_process
        word_str = String(word)  # Convert SubString to String
        if word_str != best_match && length(word_str) > 3  # Skip short words and design type
            # Network learns this word as a requirement concept (boundary-constrained)
            push!(requirements, word_str)
        end
    end
    
    # Extract numbers from prompt (e.g., "3 buttons")
    number_pattern = r"(\d+)\s+(\w+)"
    num_matches = eachmatch(number_pattern, prompt_lower)
    for m in num_matches
        count = m.captures[1]
        item = m.captures[2]
        push!(requirements, "$item:$count")
    end
    
    # CRITICAL: Boundary-Constrained Memory Updates
    # Only update memory usage for boundary-relevant words (O(|boundary_words|) instead of O(|all_words|))
    for word in words_to_process
        if haskey(ce.concept_memory, word)
            ce.concept_memory[word]["usage_count"] += 1
        end
    end
    
    # Ensure all return values are proper types (not SubString)
    return String(best_match), requirements, prompt_embedding, action_type
end

# Hebbian learning: Strengthen word→goal associations when goals succeed
# CRITICAL: Boundary-Constrained Learning - only update associations for boundary-relevant words
# Theory: "Neurons that fire together, wire together" - if word leads to successful goal, strengthen association
# Efficiency: Only compute for words that are boundary-relevant (O(|boundary_words|) instead of O(|all_words|))
function strengthen_word_goal_association!(ce::ConceptEmbedding, words::Vector{String}, goal_type::Symbol, 
                                             boundary_concepts::Union{Set{String}, Nothing} = nothing, 
                                             success_strength::Float64 = 0.1)
    # Boundary-Constrained: Only learn associations for boundary-relevant words
    words_to_update = if boundary_concepts !== nothing && !isempty(boundary_concepts)
        # Only update words that are boundary-relevant (computational efficiency)
        [w for w in words if lowercase(w) in boundary_concepts]
    else
        # If no boundary info, update all words (fallback - less efficient)
        words
    end
    
    for word in words_to_update
        word_lower = lowercase(word)
        if !haskey(ce.word_to_goal_associations, word_lower)
            ce.word_to_goal_associations[word_lower] = Dict{Symbol, Float64}()
        end
        
        # Hebbian update: Δstrength = η · success · co_occurrence (boundary-constrained)
        if !haskey(ce.word_to_goal_associations[word_lower], goal_type)
            ce.word_to_goal_associations[word_lower][goal_type] = 0.0
        end
        
        # Strengthen association (Hebbian: fire together → wire together)
        ce.word_to_goal_associations[word_lower][goal_type] += success_strength
        ce.word_to_goal_associations[word_lower][goal_type] = min(1.0, ce.word_to_goal_associations[word_lower][goal_type])
    end
end

# ============================================================================
# Emit a global signal (like hunger triggering eating)
function emit_signal!(gss::GlobalSignalSystem, signal_type::Symbol, intensity::Float64, target_goal::Symbol = :none)
    push!(gss.signals, GlobalSignal(signal_type, intensity, 0.95, target_goal))
end

# Update signals and apply to network (indirect propagation)
# CRITICAL: Signals only affect boundary neurons (O(|boundary|) instead of O(N))
function update_global_signals!(gss::GlobalSignalSystem, network::NeuralNetwork, hierarchy, dt::Float64)
    total_signal_strength = 0.0
    
    # Get boundary neurons (signals only propagate to boundaries)
    boundary_indices = compute_topological_boundaries!(network)
    
    # Update signal intensities
    for signal in gss.signals
        signal.intensity *= signal.decay_rate
        total_signal_strength += signal.intensity
        
        # Apply signal to BOUNDARY neurons only (boundary-constrained)
        # Signals propagate externally (like cytokines) but only affect boundary neurons
        if signal.target_goal != :none && haskey(hierarchy.goals, signal.target_goal)
            # Signal affects boundary neurons working on this goal more strongly
            for i in boundary_indices
                network.neurons[i].activation += 0.1 * signal.intensity * dt
            end
        else
            # Global signal affects all boundary neurons (like cytokines affect all neurons)
            for i in boundary_indices
                network.neurons[i].activation += 0.05 * signal.intensity * dt
            end
        end
    end
    
    # Remove decayed signals
    filter!(s -> s.intensity > 0.01, gss.signals)
    
    # Record signal state
    push!(gss.signal_history, Dict(
        "time" => network.time,
        "signal_count" => length(gss.signals),
        "total_intensity" => total_signal_strength
    ))
    
    # Keep only recent history
    if length(gss.signal_history) > 1000
        gss.signal_history = gss.signal_history[end-999:end]
    end
end

# Trigger goal-driven signal (like hunger for food, but for goals)
function trigger_goal_drive!(gss::GlobalSignalSystem, goal_id::Symbol, urgency::Float64)
    emit_signal!(gss, :goal_drive, urgency, goal_id)
end

# ============================================================================
# KURZWEIL-STYLE PATTERN DETECTION (Boundary-Constrained)
# ============================================================================

# Kurzweil-style pattern detection (boundary-constrained)
# CRITICAL: Only detects patterns in boundary neurons
function kurzweil_update!(pattern_model::PatternModel, network::NeuralNetwork, 
                         boundary_indices::Vector{Int}, current_time::Float64)
    # 1. Pattern Detection: Identify recurring wave patterns ONLY in boundary neurons
    boundary_activations = [network.neurons[i].activation for i in boundary_indices]
    boundary_phases = [network.neurons[i].phase for i in boundary_indices]
    
    if length(boundary_indices) < 3
        return  # Need at least 3 neurons for pattern detection
    end
    
    # Detect synchronization patterns
    sync_level = abs(sum(exp(im * phase) for phase in boundary_phases) / length(boundary_phases))
    
    # Detect activation patterns (clusters of high/low activation)
    activation_mean = mean(boundary_activations)
    activation_std = std(boundary_activations)
    
    # Pattern: High synchronization + moderate activation variance
    pattern_key = "sync_$(round(sync_level, digits=2))_act_$(round(activation_mean, digits=2))"
    
    if !haskey(pattern_model.patterns, pattern_key)
        pattern_model.patterns[pattern_key] = Dict(
            "sync_level" => sync_level,
            "activation_mean" => activation_mean,
            "activation_std" => activation_std,
            "frequency" => 1,
            "first_seen" => current_time,
            "last_seen" => current_time
        )
    else
        pattern_model.patterns[pattern_key]["frequency"] += 1
        pattern_model.patterns[pattern_key]["last_seen"] = current_time
    end
    
    # 2. Abstraction: Extract invariant features (simplified - would use persistent homology in full implementation)
    # Topological invariants: persistent features that survive across scales
    if sync_level > 0.7
        pattern_model.invariants["high_sync"] = sync_level
    end
    if activation_std < 0.1
        pattern_model.invariants["low_variance"] = activation_std
    end
    
    # 3. Prediction: Predict future boundary activations
    for i in boundary_indices
        neuron = network.neurons[i]
        # Simple prediction: activation will tend toward mean if synchronized
        if sync_level > 0.6
            predicted = activation_mean + 0.1 * (neuron.activation - activation_mean)
        else
            predicted = neuron.activation * 0.95  # Decay prediction
        end
        pattern_model.predictions[i] = predicted
    end
    
    # 4. Feedback: Update patterns based on prediction error (simplified)
    # In full implementation, would compare predictions to actual future activations
    pattern_model.last_update = current_time
end

# ============================================================================
# GOAL DRIFT DETECTION (Boundary-Constrained)
# ============================================================================

# Detect policy divergence using Bayesian policy comparison (boundary-constrained)
# Theory: policy_divergence(G_i, G_j) = D_KL(P(a|G_i, s_boundary) || P(a|G_j, s_boundary))
# CRITICAL: Only evaluates boundary goals for computational efficiency
function detect_policy_divergence(hierarchy, boundary_indices::Vector{Int}, 
                                  neuron_goals::Dict{Int, Symbol}, switching_cost::Float64 = 0.2)::Tuple{Bool, Symbol}
    current_goal_id = hierarchy.active_goal
    
    if !haskey(hierarchy.goals, current_goal_id)
        return false, current_goal_id
    end
    
    current_goal = hierarchy.goals[current_goal_id]
    
    # Get boundary goals only
    boundary_goal_ids = Set{Symbol}()
    for i in boundary_indices
        if haskey(neuron_goals, i)
            push!(boundary_goal_ids, neuron_goals[i])
        end
    end
    
    # Evaluate value of current goal (boundary-constrained)
    current_value = estimate_goal_value(current_goal, hierarchy, boundary_indices, neuron_goals)
    
    # Find best alternative boundary goal
    best_alternative = current_goal_id
    best_value = current_value
    
    for goal_id in boundary_goal_ids
        if goal_id == current_goal_id || !haskey(hierarchy.goals, goal_id)
            continue
        end
        
        goal = hierarchy.goals[goal_id]
        
        # Check if dependencies satisfied
        deps_ok = all(haskey(hierarchy.goals, dep) && hierarchy.goals[dep].achieved for dep in goal.dependencies)
        
        if deps_ok && !goal.achieved
            # Estimate value (boundary-constrained)
            value = estimate_goal_value(goal, hierarchy, boundary_indices, neuron_goals)
            
            # Account for switching cost
            net_value = value - switching_cost
            
            if net_value > best_value
                best_value = net_value
                best_alternative = goal_id
            end
        end
    end
    
    # Policy divergence detected if alternative is significantly better
    # Theory: value_change > switching_cost triggers goal switch
    divergence_detected = best_alternative != current_goal_id && (best_value - current_value) > switching_cost
    
    return divergence_detected, best_alternative
end

# Compute switching cost explicitly (boundary-constrained)
# Theory: switching_cost(G_i, G_j) = α·context_distance + β·policy_divergence + γ·time_since_switch
function compute_switching_cost(hierarchy, goal_i::Symbol, goal_j::Symbol, 
                                boundary_indices::Vector{Int}, neuron_goals::Dict{Int, Symbol},
                                last_switch_time::Float64, current_time::Float64)::Float64
    # Only compute for boundary goals
    goal_i_boundary = any(i -> haskey(neuron_goals, i) && neuron_goals[i] == goal_i, boundary_indices)
    goal_j_boundary = any(i -> haskey(neuron_goals, i) && neuron_goals[i] == goal_j, boundary_indices)
    
    if !goal_i_boundary || !goal_j_boundary
        return Inf  # Cannot switch to/from non-boundary goals
    end
    
    # Context distance (simplified: based on goal embeddings/descriptions)
    # Full implementation would use goal context vectors
    context_distance = 0.5  # Placeholder - would compute from goal contexts
    
    # Policy divergence (simplified: based on value difference)
    # Full implementation would compute KL divergence of action distributions
    if haskey(hierarchy.goals, goal_i) && haskey(hierarchy.goals, goal_j)
        value_i = estimate_goal_value(hierarchy.goals[goal_i], hierarchy, boundary_indices, neuron_goals)
        value_j = estimate_goal_value(hierarchy.goals[goal_j], hierarchy, boundary_indices, neuron_goals)
        policy_divergence = abs(value_i - value_j)
    else
        policy_divergence = 0.5
    end
    
    # Time since switch
    time_since_switch = current_time - last_switch_time
    
    # Weighted combination
    alpha = 0.3
    beta = 0.4
    gamma = 0.3
    
    switching_cost = alpha * context_distance + beta * policy_divergence + gamma * time_since_switch / 100.0
    
    return switching_cost
end

# Compute abandonment risk (boundary-constrained)
# Theory: abandonment_risk(G, t) = f(time_since_active, switch_count, progress_rate, goal_value)
# Decay model: abandonment_probability = 1 - exp(-λ·time_since_active·(1 + α·switch_count))
function compute_abandonment_risk(goal::Goal, hierarchy, boundary_indices::Vector{Int}, 
                                  neuron_goals::Dict{Int, Symbol}, current_time::Float64)::Float64
    # Only compute for boundary goals
    goal_boundary_neurons = [i for i in boundary_indices if haskey(neuron_goals, i) && neuron_goals[i] == goal.id]
    if isempty(goal_boundary_neurons)
        return goal.abandonment_risk  # Return cached risk for non-boundary goals
    end
    
    # Measurable factors (all computed only for boundary goals):
    time_since_active = current_time - goal.last_active
    switch_count = goal.switch_count
    
    # Progress rate (from boundary neurons only)
    # Simplified: use goal.progress as proxy
    progress_rate = goal.progress > 0 ? goal.progress / max(time_since_active, 1.0) : 0.0
    
    # Goal value (computed only from boundary goals)
    goal_value = estimate_goal_value(goal, hierarchy, boundary_indices, neuron_goals)
    
    # Decay model parameters (learned from data)
    lambda = 0.01  # Base decay rate
    alpha = 0.1    # Switch count penalty
    
    # Abandonment probability (decay model)
    abandonment_probability = 1.0 - exp(-lambda * time_since_active * (1.0 + alpha * switch_count))
    
    # Adjust based on progress rate and goal value (protective factors)
    if progress_rate > 0.01
        abandonment_probability *= (1.0 - min(progress_rate, 0.5))  # Progress protects
    end
    if goal_value > 0.5
        abandonment_probability *= (1.0 - min(goal_value, 0.5))  # High value protects
    end
    
    return max(0.0, min(1.0, abandonment_probability))
end

# ============================================================================
# AUTONOMOUS GOAL GENERATION (Boundary-Constrained)
# ============================================================================

# Generate goals from information gaps (boundary-constrained)
# CRITICAL: Only generates goals for boundary information gaps
function generate_goals_from_gaps(gaps::Vector{Dict}, discovered_info::Dict, current_time::Float64)::Vector{Goal}
    new_goals = Goal[]
    
    # Extract components and principles from discovered information
    components = get(discovered_info, "components", [])
    principles = get(discovered_info, "principles", [])
    features = get(discovered_info, "features", [])
    
    # Generate goals for each significant component/principle
    all_concepts = vcat(components, principles, features)
    
    for concept in all_concepts
        concept_lower = lowercase(concept)
        
        # Skip if too generic
        if concept_lower in ["mechanism", "feature", "component", "principle"]
            continue
        end
        
        # Generate goal description
        goal_desc = "Learn $concept principles"
        
        # Create goal
        goal_id = Symbol(replace(concept_lower, " " => "_"))
        new_goal = create_goal(goal_id, goal_desc, false, [], current_time)
        push!(new_goals, new_goal)
    end
    
    return new_goals
end

# Add autonomously generated goals to hierarchy
function add_autonomous_goals!(hierarchy, new_goals::Vector{Goal}, parent_goal_id::Symbol)
    for goal in new_goals
        # Check if goal already exists
        if !haskey(hierarchy.goals, goal.id)
            # Set parent as dependency
            goal.dependencies = [parent_goal_id]
            hierarchy.goals[goal.id] = goal
            
            # Log goal addition
            push!(hierarchy.goal_history, Dict(
                "time" => goal.created_at,
                "event" => "goal_added",
                "goal_id" => goal.id,
                "description" => goal.description,
                "source" => "autonomous_generation",
                "parent" => parent_goal_id
            ))
        end
    end
end

# ============================================================================
# INTERNET ACCESS AND INFORMATION GATHERING (Boundary-Constrained)
# ============================================================================

# Query internet for information (boundary-constrained)
# CRITICAL: Only queries for boundary goals
function query_internet(query_system::InternetQuerySystem, query::String, context::Dict = Dict(); 
                       boundary_goal_id::Union{Symbol, Nothing} = nothing)::Dict
    current_time = get(context, "time", 0.0)
    
    # Rate limiting
    if current_time - query_system.last_query_time < 60.0 / query_system.rate_limit
        return Dict("error" => "Rate limit exceeded", "cached" => true)
    end
    
    # Check cache first
    query_lower = lowercase(query)
    if haskey(query_system.cache, query_lower)
        return Dict("result" => query_system.cache[query_lower], "cached" => true)
    end
    
    # TODO: Integrate with actual search API (e.g., DuckDuckGo, Wikipedia API)
    # For now, simulate with knowledge base
    result = simulate_internet_search(query, boundary_goal_id)
    
    # Cache result
    query_system.cache[query_lower] = result
    query_system.last_query_time = current_time
    
    # Log query
    push!(query_system.query_history, Dict(
        "query" => query,
        "goal_id" => boundary_goal_id,
        "time" => current_time,
        "result_length" => length(get(result, "summary", ""))
    ))
    
    return Dict("result" => result, "cached" => false)
end

# Simulate internet search (placeholder until real API integration)
function simulate_internet_search(query::String, goal_id::Union{Symbol, Nothing})
    query_lower = lowercase(query)
    
    # Knowledge base for common queries (general, not hardcoded to specific objects)
    knowledge_base = Dict(
        "heating element" => Dict(
            "summary" => "Heating elements convert electrical energy to heat via resistance. Common materials: nichrome wire, tungsten. Principles: Joule heating, resistance increases with temperature.",
            "materials" => ["nichrome", "tungsten", "ceramic"],
            "principles" => ["Joule heating", "electrical resistance", "thermal conductivity"]
        ),
        "safety" => Dict(
            "summary" => "Safety features prevent fires and injuries. Common mechanisms: auto-shutoff timers, thermal fuses, cool-touch exteriors, and automatic pop-up mechanisms.",
            "features" => ["auto-shutoff", "thermal fuse", "cool-touch", "automatic pop-up"]
        ),
        "shirt" => Dict(
            "summary" => "A shirt is a garment with sleeves, collar, and buttons. Components include: fabric (cotton, polyester), pattern pieces, buttons, sleeves, collar, and cuffs.",
            "components" => ["fabric", "pattern", "buttons", "sleeves", "collar", "cuffs"],
            "materials" => ["cotton", "polyester", "linen", "silk"]
        )
    )
    
    # Try to match query to knowledge base
    for (key, info) in knowledge_base
        if occursin(key, query_lower)
            return info
        end
    end
    
    # Default response
    return Dict(
        "summary" => "Information about: $query. This is a simulated response. Real implementation would query actual internet sources.",
        "components" => [],
        "principles" => []
    )
end

# Generate query from information gap (boundary-constrained)
function generate_query_from_gap(gap::Dict)::String
    goal_desc = get(gap, "goal_description", "")
    missing = get(gap, "missing_concepts", String[])
    
    if !isempty(missing)
        # Generate query from missing concepts
        return "What is $(join(missing, " and "))? How does it work?"
    else
        # Generate query from goal description
        return "What is $goal_desc? How does it work?"
    end
end

# Integrate discovered information into concept embeddings (boundary-constrained)
function integrate_information!(ce::ConceptEmbedding, info::Dict, gap::Dict, current_time::Float64)
    # Extract concepts from information
    summary = get(info, "summary", "")
    components = get(info, "components", [])
    principles = get(info, "principles", [])
    
    # Update embeddings for discovered concepts
    all_concepts = vcat(components, principles)
    
    for concept in all_concepts
        concept_lower = lowercase(concept)
        # Get or create embedding (will be orthogonalized)
        emb = get_or_create_embedding(ce, concept_lower, current_time, nothing)
        
        # Update memory
        if !haskey(ce.concept_memory, concept_lower)
            ce.concept_memory[concept_lower] = Dict("created_at" => current_time, "usage_count" => 0)
        end
        ce.concept_memory[concept_lower]["usage_count"] += 1
        ce.concept_memory[concept_lower]["last_accessed"] = current_time
        ce.concept_memory[concept_lower]["source"] = "internet_query"
    end
    
    return all_concepts
end

# ============================================================================
# WEB SERVER
# ============================================================================

# REMOVED: create_toaster_goals - use generate_goals_from_prompt instead (general, no hardcoded examples)

# Create empty goal hierarchy - network starts blank
function create_empty_goal_hierarchy(current_time::Float64 = 0.0)
    return GoalHierarchy(Dict{Symbol, Goal}(), :none, Symbol[], [], Symbol[])
end

# Map neurons to goals (each neuron cluster works on a goal)
# CRITICAL: Semantic Integration - Use learned word→goal associations to guide assignments
function assign_neurons_to_goals(network::NeuralNetwork, hierarchy, concept_embedding::ConceptEmbedding = ConceptEmbedding(Dict{String, Vector{Float64}}(), 64, Dict{String, Dict}(), Dict{String, Dict{Symbol, Float64}}(), Dict{String, Symbol}()))
    if isempty(hierarchy.goals)
        return Dict{Int, Symbol}()  # No goals, no assignments
    end
    
    # Extract goal descriptions and compute semantic similarity
    goal_embeddings = Dict{Symbol, Vector{Float64}}()
    for (goal_id, goal) in hierarchy.goals
        # Extract words from goal description
        goal_words = filter(w -> length(w) > 2, split(lowercase(goal.description), r"[^a-z]+"))
        goal_emb = zeros(concept_embedding.embedding_dim)
        for word in goal_words
            if haskey(concept_embedding.embeddings, word)
                goal_emb += concept_embedding.embeddings[word]
            end
        end
        if length(goal_words) > 0
            goal_emb = goal_emb / length(goal_words)
        end
        goal_embeddings[goal_id] = goal_emb
    end
    
    # Assign neurons based on semantic similarity (if embeddings exist) or evenly distribute
    neuron_goals = Dict{Int, Symbol}()
    goal_list = collect(keys(hierarchy.goals))
    
    if !isempty(goal_embeddings) && any(norm(emb) > 0.01 for emb in values(goal_embeddings))
        # Semantic assignment: Match neurons to goals based on learned associations
        # Use neuron positions as "semantic coordinates" - neurons closer to goal embedding space get assigned
        for i in 1:network.n_neurons
            neuron_pos = network.neurons[i].position
            # Map neuron position to embedding space (simple projection)
            neuron_emb = [neuron_pos[1], neuron_pos[2], zeros(concept_embedding.embedding_dim - 2)...]
            if length(neuron_emb) > concept_embedding.embedding_dim
                neuron_emb = neuron_emb[1:concept_embedding.embedding_dim]
            elseif length(neuron_emb) < concept_embedding.embedding_dim
                neuron_emb = [neuron_emb; zeros(concept_embedding.embedding_dim - length(neuron_emb))]
            end
            
            # Find goal with highest semantic similarity
            best_goal = goal_list[1]
            best_sim = -1.0
            for goal_id in goal_list
                if haskey(goal_embeddings, goal_id)
                    sim = dot(neuron_emb, goal_embeddings[goal_id])
                    if sim > best_sim
                        best_sim = sim
                        best_goal = goal_id
                    end
                end
            end
            neuron_goals[i] = best_goal
        end
    else
        # Fallback: Even distribution (when no embeddings exist yet)
        n_per_goal = div(network.n_neurons, length(hierarchy.goals))
        for (idx, goal_id) in enumerate(goal_list)
            start_idx = (idx - 1) * n_per_goal + 1
            end_idx = idx == length(goal_list) ? network.n_neurons : idx * n_per_goal
            for i in start_idx:end_idx
                neuron_goals[i] = goal_id
            end
        end
    end
    
    return neuron_goals
end

# ============================================================================
# INFORMATION GAP DETECTION (Boundary-Constrained)
# ============================================================================

# Detect information gaps for boundary goals
# CRITICAL: Only detects gaps for boundary goals (computational efficiency)
function detect_information_gaps(ce::ConceptEmbedding, hierarchy, 
                                neuron_goals::Dict{Int, Symbol}, boundary_indices::Vector{Int})::Vector{Dict}
    gaps = Dict[]
    
    # Get boundary goals
    boundary_goal_ids = Set{Symbol}()
    for i in boundary_indices
        if haskey(neuron_goals, i)
            push!(boundary_goal_ids, neuron_goals[i])
        end
    end
    
    # Check each boundary goal for information gaps
    for goal_id in boundary_goal_ids
        if !haskey(hierarchy.goals, goal_id)
            continue
        end
        
        goal = hierarchy.goals[goal_id]
        
        # Extract concepts from goal description
        words = split(lowercase(goal.description), r"[^a-z]+")
        words = filter(w -> length(w) > 2, words)
        
        # Check which concepts are missing or have low knowledge
        missing_concepts = String[]
        low_knowledge_concepts = String[]
        
        for word in words
            concept_lower = lowercase(word)
            
            # Check if concept exists in embeddings
            if !haskey(ce.embeddings, concept_lower)
                push!(missing_concepts, word)
            else
                # Check knowledge quality (usage count, recency)
                if haskey(ce.concept_memory, concept_lower)
                    memory = ce.concept_memory[concept_lower]
                    usage_count = get(memory, "usage_count", 0)
                    
                    # Low knowledge if rarely used
                    if usage_count < 3
                        push!(low_knowledge_concepts, word)
                    end
                else
                    push!(missing_concepts, word)
                end
            end
        end
        
        # Calculate gap entropy: H(required) - H(current)
        # Simplified: gap_size = number of missing/low-knowledge concepts
        gap_size = length(missing_concepts) + 0.5 * length(low_knowledge_concepts)
        
        if gap_size > 0.5  # Threshold for significant gap
            push!(gaps, Dict(
                "goal_id" => goal_id,
                "goal_description" => goal.description,
                "missing_concepts" => missing_concepts,
                "low_knowledge_concepts" => low_knowledge_concepts,
                "gap_size" => gap_size,
                "priority" => goal.terminal ? gap_size * 2.0 : gap_size  # Terminal goals have higher priority
            ))
        end
    end
    
    # Sort by priority (highest first)
    sort!(gaps, by=x -> x["priority"], rev=true)
    return gaps
end

# Estimate goal value (boundary-constrained)
# CRITICAL: Only evaluates boundary goals for computational efficiency
function estimate_goal_value(goal::Goal, hierarchy, boundary_indices::Vector{Int}, 
                            neuron_goals::Dict{Int, Symbol})::Float64
    # Check if goal has boundary neurons (only evaluate boundary goals)
    goal_has_boundary = any(i -> haskey(neuron_goals, i) && neuron_goals[i] == goal.id, boundary_indices)
    
    if !goal_has_boundary
        return 0.0  # Non-boundary goals have zero value (not actively pursued)
    end
    
    # Direct reward (progress)
    direct_reward = goal.progress
    
    # Alignment with terminal goal (measured via boundary goal pursuit)
    # Alignment reward (boundary-constrained)
    # Theory: R_alignment(G, t) = 1.0 if G = G_T, else path_length(G, G_T)^(-1)
    alignment = 0.0
    if !isempty(hierarchy.terminal_goals)
        terminal_id = hierarchy.terminal_goals[1]
        if goal.id == terminal_id
            alignment = 1.0  # Terminal goal
        else
            # Compute dependency path length
            path_len = compute_dependency_path_length(goal, terminal_id, hierarchy, boundary_indices, neuron_goals)
            if path_len < Inf
                alignment = 1.0 / (1.0 + path_len)  # Inverse path length (closer = higher alignment)
            else
                alignment = 0.0  # No path to terminal goal
            end
        end
    else
        alignment = goal.terminal ? 1.0 : 0.5  # Fallback if no terminal goals defined
    end
    
    # Information gain (boundary-constrained)
    # Theory: info_gain(G, t) = H(knowledge|G, t-Δt) - H(knowledge|G, t)
    # Simplified: Use concept entropy reduction as proxy
    # Full implementation would compute actual concept entropy from concept embeddings
    if isempty(goal.dependencies)
        info_gain = 0.5  # No dependencies = moderate info gain
    else
        # Only count dependencies that are boundary goals
        boundary_deps = [dep for dep in goal.dependencies 
                        if haskey(hierarchy.goals, dep) && 
                           any(i -> haskey(neuron_goals, i) && neuron_goals[i] == dep, boundary_indices)]
        deps_satisfied = sum(hierarchy.goals[dep].achieved for dep in boundary_deps)
        # Proxy for entropy reduction: more dependencies satisfied → lower entropy → higher info gain
        info_gain = isempty(boundary_deps) ? 0.5 : deps_satisfied / length(boundary_deps)
    end
    
    # Combined value (only for boundary goals)
    # Theory: R(G, t) = w₁·R_progress + w₂·R_info + w₃·R_alignment + w₄·R_efficiency
    # Simplified implementation:
    progress_reward = direct_reward  # R_progress: rate of progress
    info_reward = info_gain  # R_info: information gain (concept entropy reduction)
    alignment_reward = alignment  # R_alignment: alignment with terminal goal (uses dependency path length)
    # Efficiency reward (simplified - would need time tracking)
    efficiency_reward = 0.0  # R_efficiency: progress per unit time (not implemented yet)
    
    value = 0.4 * progress_reward + 0.3 * info_reward + 0.3 * alignment_reward + 0.0 * efficiency_reward  # Efficiency weight = 0 for now
    
    return value
end

# Compute dependency path length (boundary-constrained)
# Theory: path_length(G, G_T) = min_{path P: G→G_T} |P| (shortest path through dependency graph)
function compute_dependency_path_length(goal::Goal, terminal_goal_id::Symbol, hierarchy, 
                                       boundary_indices::Vector{Int}, neuron_goals::Dict{Int, Symbol})::Float64
    # Only compute for boundary goals
    goal_boundary = any(i -> haskey(neuron_goals, i) && neuron_goals[i] == goal.id, boundary_indices)
    terminal_boundary = any(i -> haskey(neuron_goals, i) && neuron_goals[i] == terminal_goal_id, boundary_indices)
    
    if !goal_boundary || !terminal_boundary
        return Inf  # Non-boundary goals have infinite path length
    end
    
    if goal.id == terminal_goal_id
        return 0.0  # Same goal
    end
    
    # BFS to find shortest path (only through boundary goals)
    visited = Set{Symbol}()
    queue = [(goal.id, 0)]
    
    while !isempty(queue)
        current_id, depth = popfirst!(queue)
        
        if current_id == terminal_goal_id
            return Float64(depth)
        end
        
        if current_id in visited
            continue
        end
        push!(visited, current_id)
        
        if !haskey(hierarchy.goals, current_id)
            continue
        end
        
        current_goal = hierarchy.goals[current_id]
        
        # Only traverse through boundary goals
        for dep in current_goal.dependencies
            dep_boundary = any(i -> haskey(neuron_goals, i) && neuron_goals[i] == dep, boundary_indices)
            if dep_boundary && !(dep in visited)
                push!(queue, (dep, depth + 1))
            end
        end
    end
    
    return Inf  # No path found
end

function log_thought!(stream::ThoughtStream, neuron_id::Int, thought_type::Symbol, message::String, data::Dict = Dict(), current_time::Float64 = 0.0)
    thought = ThoughtLog(current_time, neuron_id, thought_type, message, data)
    push!(stream.thoughts, thought)
    
    # Keep only recent thoughts
    if length(stream.thoughts) > stream.max_thoughts
        stream.thoughts = stream.thoughts[end-stream.max_thoughts+1:end]
    end
end

# ============================================================================
# METACOGNITIVE THOUGHT AGGREGATION (Boundary-Constrained)
# ============================================================================

# Aggregate metacognitive thoughts into confusion/stagnation signals
# Theory: C_G(t) = Σ w(τ) · I(thought_τ = confusion ∧ goal_τ = G)
# CRITICAL: Only aggregates thoughts from boundary neurons
function aggregate_metacognitive_thoughts!(signals::MetacognitiveSignals, thought_stream::ThoughtStream, current_time::Float64)
    # Reset signals (will recompute from thoughts)
    signals.confusion_signals = Dict{Symbol, Float64}()
    signals.stagnation_signals = Dict{Symbol, Float64}()
    
    # Get thoughts within time window
    window_start = current_time - signals.time_window
    recent_thoughts = [t for t in thought_stream.thoughts if t.timestamp >= window_start]
    
    # Aggregate thoughts per goal
    for thought in recent_thoughts
        # Only consider metacognitive thoughts from boundary neurons
        if thought.thought_type in [:confusion, :stagnation, :uncertainty]
            goal_id = get(thought.data, "goal", :none)
            if goal_id != :none && goal_id isa Symbol
                # Compute weight: w(τ) = exp(-α · (t - τ))
                time_diff = current_time - thought.timestamp
                weight = exp(-signals.decay_rate * time_diff)
                
                # Extract thought weight from data if available
                thought_weight = get(thought.data, "confusion_weight", get(thought.data, "stagnation_weight", 1.0))
                weighted_contribution = weight * thought_weight
                
                if thought.thought_type == :confusion || thought.thought_type == :uncertainty
                    # Add to confusion signal
                    signals.confusion_signals[goal_id] = get(signals.confusion_signals, goal_id, 0.0) + weighted_contribution
                elseif thought.thought_type == :stagnation
                    # Add to stagnation signal
                    signals.stagnation_signals[goal_id] = get(signals.stagnation_signals, goal_id, 0.0) + weighted_contribution
                end
            end
        end
    end
end

# Get confusion signal for a goal
function get_confusion_signal(signals::MetacognitiveSignals, goal_id::Symbol)::Float64
    return get(signals.confusion_signals, goal_id, 0.0)
end

# Get stagnation signal for a goal
function get_stagnation_signal(signals::MetacognitiveSignals, goal_id::Symbol)::Float64
    return get(signals.stagnation_signals, goal_id, 0.0)
end

mutable struct ServerState
    network::NeuralNetwork
    hierarchy::GoalHierarchy
    neuron_goals::Dict{Int, Symbol}
    current_design::Dict
    design_requirements::Vector{String}  # Store requirements from prompt
    running::Bool
    update_count::Int
    step::Int
    concept_embedding::ConceptEmbedding  # Language understanding system
    global_signals::GlobalSignalSystem  # Indirect propagation system
    thought_stream::ThoughtStream  # Real-time thought process logging
    internet_query_system::InternetQuerySystem  # Internet access for information gathering
    pattern_model::PatternModel  # Kurzweil-style pattern detection
    metacognitive_signals::MetacognitiveSignals  # Confusion/stagnation signal aggregation
    last_prompt_words::Vector{String}  # Store words from last prompt for Hebbian learning
end

function create_server_state()
    Random.seed!(42)
    network = create_network(1000)
    hierarchy = create_empty_goal_hierarchy(network.time)  # Start blank - network learns from prompts
    neuron_goals = Dict{Int, Symbol}()  # No neurons assigned until goals are created
    concept_emb = create_concept_embedding(64)
    signal_system = create_global_signal_system()
    thought_stream = create_thought_stream(1000)
    internet_system = create_internet_query_system(10.0)  # 10 queries per minute
    pattern_model = create_pattern_model()  # Kurzweil pattern detection
    metacog_signals = create_metacognitive_signals(100.0, 0.02)  # 100 timestep window, 0.02 decay rate
    return ServerState(network, hierarchy, neuron_goals, Dict(), String[], true, 0, 0, concept_emb, signal_system, thought_stream, internet_system, pattern_model, metacog_signals, String[])
end

state = create_server_state()

# HTML Dashboard
const HTML_DASHBOARD = """
<!DOCTYPE html>
<html>
<head>
    <title>Neural Network Monitor</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #0a0e27;
            color: #e0e0e0;
            padding: 20px;
        }
        .container {
            max-width: 1800px;
            margin: 0 auto;
        }
        h1 {
            color: #4CAF50;
            margin-bottom: 20px;
            text-align: center;
        }
        .grid {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        .panel {
            background: #1a1e3f;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        .panel h2 {
            color: #4CAF50;
            margin-bottom: 15px;
            font-size: 18px;
        }
        #networkCanvas {
            width: 100%;
            height: 600px;
            background: #0f1329;
            border-radius: 5px;
            border: 2px solid #2a2e4f;
        }
        .metrics {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
        }
        .metric-card {
            background: #25294f;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #4CAF50;
        }
        .metric-label {
            color: #888;
            font-size: 12px;
            margin-bottom: 5px;
        }
        .metric-value {
            color: #4CAF50;
            font-size: 24px;
            font-weight: bold;
        }
        .plot-container {
            height: 200px;
            margin-top: 10px;
        }
        .status {
            text-align: center;
            padding: 10px;
            background: #1a1e3f;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .status.running { border-left: 4px solid #4CAF50; }
        .status.stopped { border-left: 4px solid #f44336; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 Neural Network Monitor</h1>
        
        <div class="status running" id="status">
            <strong>Status:</strong> <span id="statusText">Running</span> | 
            <strong>Updates:</strong> <span id="updateCount">0</span> | 
            <strong>Time:</strong> <span id="currentTime">0.00</span>s |
            <strong>Current Goal:</strong> <span id="currentGoal">learn_heating</span>
        </div>
        
        <div class="panel" style="margin-bottom: 20px;">
            <h2>🎯 Goals</h2>
            <div id="goalsDisplay"></div>
        </div>
        
        <div class="panel" style="margin-bottom: 20px;">
            <h2>🎨 Current Design</h2>
            <div id="designDisplay">Designing... Neurons are synchronizing...</div>
        </div>
        
        <div class="panel" style="margin-bottom: 20px;">
            <h2>🧠 Real-Time Thought Stream</h2>
            <div style="background: #0a0e27; border: 1px solid #333; border-radius: 5px; height: 300px; overflow-y: auto; padding: 10px; font-family: 'Courier New', monospace; font-size: 11px;" id="thoughtStream">
                <div style="color: #888; font-style: italic;">Waiting for neuron thoughts...</div>
            </div>
            <div style="margin-top: 5px; font-size: 10px; color: #666;">
                <span id="thoughtCount">0</span> thoughts logged | 
                <span style="color: #4CAF50;">🟢 Synchronization</span> | 
                <span style="color: #4ecdc4;">🔵 Wave Propagation</span> | 
                <span style="color: #ffe66d;">🟡 Goal Pursuit</span> | 
                <span style="color: #95e1d3;">🟢 Learning</span>
            </div>
        </div>
        
        <div class="panel" style="margin-bottom: 20px;">
            <h2>💬 Command Interface</h2>
            <div style="display: flex; gap: 10px; margin-bottom: 10px;">
                <input type="text" id="commandInput" placeholder="Enter design prompt (e.g., 'make a toaster', 'design a shirt with 3 buttons') or command ('status', 'reset', etc.)" 
                       style="flex: 1; padding: 10px; background: #25294f; border: 1px solid #4CAF50; border-radius: 5px; color: #e0e0e0; font-size: 14px;">
                <button onclick="sendCommand()" style="padding: 10px 20px; background: #4CAF50; border: none; border-radius: 5px; color: white; cursor: pointer; font-weight: bold;">
                    Send
                </button>
            </div>
            <div id="commandOutput" style="background: #25294f; padding: 10px; border-radius: 5px; min-height: 100px; max-height: 200px; overflow-y: auto; font-family: monospace; font-size: 12px;">
                <div style="color: #888;">Ready for design prompts! Try: 'make a toaster', 'design a shirt with 3 buttons on the sleeve', 'create a car', etc.</div>
            </div>
        </div>
        
        <div class="grid">
            <div class="panel">
                <h2>Neural Network (Hyperbolic Space - Poincaré Disk)</h2>
                <canvas id="networkCanvas"></canvas>
            </div>
            
            <div class="panel">
                <h2>Real-Time Metrics</h2>
                <div class="metrics">
                    <div class="metric-card">
                        <div class="metric-label">Synchronization</div>
                        <div class="metric-value" id="syncValue">0.000</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Avg Activation</div>
                        <div class="metric-value" id="actValue">0.000</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Connections</div>
                        <div class="metric-value" id="connValue">0.000</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Neurons</div>
                        <div class="metric-value" id="neuronCount">Loading...</div>
                    </div>
                </div>
                
                <h2 style="margin-top: 20px;">Synchronization Over Time</h2>
                <div id="syncPlot" class="plot-container"></div>
                
                <h2 style="margin-top: 20px;">Activation Over Time</h2>
                <div id="actPlot" class="plot-container"></div>
            </div>
        </div>
    </div>
    
    <script>
        const canvas = document.getElementById('networkCanvas');
        const ctx = canvas.getContext('2d');
        canvas.width = canvas.offsetWidth;
        canvas.height = canvas.offsetHeight;
        
        // Data storage
        let timeData = [];
        let syncData = [];
        let actData = [];
        let maxPoints = 100;
        
        // Plotly traces
        const syncTrace = {
            x: [],
            y: [],
            type: 'scatter',
            mode: 'lines',
            line: { color: '#4CAF50', width: 2 },
            name: 'Synchronization'
        };
        
        const actTrace = {
            x: [],
            y: [],
            type: 'scatter',
            mode: 'lines',
            line: { color: '#2196F3', width: 2 },
            name: 'Activation'
        };
        
        Plotly.newPlot('syncPlot', [syncTrace], {
            margin: { l: 40, r: 20, t: 20, b: 40 },
            paper_bgcolor: '#1a1e3f',
            plot_bgcolor: '#25294f',
            font: { color: '#e0e0e0' },
            xaxis: { title: 'Time' },
            yaxis: { title: 'r(t)', range: [0, 1] }
        });
        
        Plotly.newPlot('actPlot', [actTrace], {
            margin: { l: 40, r: 20, t: 20, b: 40 },
            paper_bgcolor: '#1a1e3f',
            plot_bgcolor: '#25294f',
            font: { color: '#e0e0e0' },
            xaxis: { title: 'Time' },
            yaxis: { title: 'Activation', range: [0, 1] }
        });
        
        function drawNetwork(data) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            const centerX = canvas.width / 2;
            const centerY = canvas.height / 2;
            const radius = Math.min(canvas.width, canvas.height) / 2 - 20;
            
            // Draw Poincaré disk boundary
            ctx.strokeStyle = '#4CAF50';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
            ctx.stroke();
            
            // Draw connections
            ctx.strokeStyle = 'rgba(100, 100, 100, 0.3)';
            ctx.lineWidth = 1;
            for (let conn of data.connections) {
                if (conn.strength > 0.05) {
                    const x1 = centerX + conn.from[0] * radius;
                    const y1 = centerY + conn.from[1] * radius;
                    const x2 = centerX + conn.to[0] * radius;
                    const y2 = centerY + conn.to[1] * radius;
                    
                    ctx.beginPath();
                    ctx.moveTo(x1, y1);
                    ctx.lineTo(x2, y2);
                    ctx.stroke();
                }
            }
            
            // Draw neurons with goal-based coloring
            for (let i = 0; i < data.neurons.length; i++) {
                const neuron = data.neurons[i];
                const x = centerX + neuron.position[0] * radius;
                const y = centerY + neuron.position[1] * radius;
                
                // Color based on goal assignment (dynamic - no hardcoded goals)
                const goalId = data.neuron_goals[i.toString()];
                // Generate color hash from goal ID for consistent coloring
                let hash = 0;
                for (let j = 0; j < goalId.length; j++) {
                    hash = goalId.charCodeAt(j) + ((hash << 5) - hash);
                }
                const hue = Math.abs(hash % 360);
                const color = "hsl(" + hue + ", 70%, 60%)";
                
                // Brightness based on activation
                const brightness = Math.floor(neuron.activation * 200) + 55;
                ctx.fillStyle = color;
                ctx.globalAlpha = neuron.activation * 0.8 + 0.2;
                
                // Draw neuron
                ctx.beginPath();
                ctx.arc(x, y, 8, 0, 2 * Math.PI);
                ctx.fill();
                ctx.globalAlpha = 1.0;
                
                // Draw phase arrow
                const arrowLength = 15 * neuron.activation;
                const arrowX = x + arrowLength * Math.cos(neuron.phase);
                const arrowY = y + arrowLength * Math.sin(neuron.phase);
                
                ctx.strokeStyle = '#ff4444';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(x, y);
                ctx.lineTo(arrowX, arrowY);
                ctx.stroke();
            }
        }
        
        function updateGoals(data) {
            const goalsDiv = document.getElementById('goalsDisplay');
            if (!data.goals || data.goals.length === 0) {
                goalsDiv.innerHTML = '<div style="color: #888; padding: 20px; text-align: center;">No goals yet. Send a command to create goals.</div>';
                document.getElementById('currentGoal').textContent = 'None';
                return;
            }
            let html = '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 10px;">';
            for (let goal of data.goals) {
                const progress = (goal.progress * 100).toFixed(1);
                const activeClass = goal.active ? 'border-left: 4px solid #4CAF50;' : '';
                const stateClass = goal.state === 'paused' ? 'opacity: 0.6;' : '';
                const achievedIcon = goal.achieved ? '✅' : goal.active ? '→' : goal.state === 'paused' ? '⏸' : '⏳';
                const stateLabel = goal.state === 'paused' ? ' (Paused)' : goal.state === 'abandoned' ? ' (Dropped)' : '';
                const abandonmentRisk = goal.abandonment_risk > 0.5 ? ' 🔴' : goal.abandonment_risk > 0.2 ? ' 🟡' : '';
                
                html += '<div class="metric-card" style="' + activeClass + stateClass + '">';
                html += '<div style="font-weight: bold; margin-bottom: 5px;">' + achievedIcon + ' ' + goal.id + stateLabel + abandonmentRisk + '</div>';
                html += '<div style="font-size: 12px; color: #888; margin-bottom: 5px;">' + goal.description + '</div>';
                html += '<div style="background: #1a1e3f; border-radius: 5px; height: 20px; overflow: hidden;">';
                html += '<div style="background: #4CAF50; height: 100%; width: ' + progress + '%; transition: width 0.3s;"></div>';
                html += '</div>';
                html += '<div style="font-size: 11px; margin-top: 5px;">' + progress + '%';
                if (goal.switch_count > 0) {
                    html += ' | Switches: ' + goal.switch_count;
                }
                if (goal.time_since_active > 0 && !goal.active) {
                    html += ' | Idle: ' + goal.time_since_active.toFixed(1) + 's';
                }
                html += '</div>';
                html += '</div>';
            }
            
            // Show dropped goals
            if (data.dropped_goals && data.dropped_goals.length > 0) {
                html += '<div style="grid-column: 1 / -1; margin-top: 10px; padding: 10px; background: #2a1e1e; border-radius: 5px; border-left: 4px solid #f44336;">';
                html += '<div style="font-weight: bold; color: #f44336; margin-bottom: 5px;">⚠️ Dropped Goals</div>';
                for (let dropped of data.dropped_goals) {
                    html += '<div style="font-size: 12px; color: #888;">❌ ' + dropped.id + '</div>';
                }
                html += '</div>';
            }
            
            html += '</div>';
            goalsDiv.innerHTML = html;
            
            document.getElementById('currentGoal').textContent = data.current_goal;
        }
        
        function updateDesign(data) {
            const designDiv = document.getElementById('designDisplay');
            if (Object.keys(data.current_design).length === 0) {
                designDiv.innerHTML = '<div style="color: #888;">Designing... Neurons are synchronizing...</div>';
            } else {
                let html = '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px;">';
                for (let [key, value] of Object.entries(data.current_design)) {
                    html += '<div class="metric-card">';
                    html += '<div class="metric-label">' + key.replace(/_/g, ' ') + '</div>';
                    html += '<div class="metric-value" style="font-size: 18px;">' + value + '</div>';
                    html += '</div>';
                }
                html += '</div>';
                designDiv.innerHTML = html;
            }
        }
        
        let lastThoughtCount = 0;
        
        function updateThoughtStream(data) {
            const thoughtStreamDiv = document.getElementById('thoughtStream');
            const thoughtCountSpan = document.getElementById('thoughtCount');
            
            if (!data.thought_stream || data.thought_stream.length === 0) {
                if (lastThoughtCount === 0) {
                    thoughtStreamDiv.innerHTML = '<div style="color: #888; font-style: italic;">Waiting for neuron thoughts...</div>';
                }
                return;
            }
            
            // Update thought count
            thoughtCountSpan.textContent = data.thought_stream.length;
            
            // Only update if new thoughts arrived
            if (data.thought_stream.length > lastThoughtCount) {
                let html = '';
                
                // Display thoughts in reverse chronological order (newest first)
                for (let i = data.thought_stream.length - 1; i >= 0; i--) {
                    const thought = data.thought_stream[i];
                    const timeStr = thought.timestamp.toFixed(2) + 's';
                    
                    // Color coding by thought type
                    let color = '#888';
                    let icon = '•';
                    if (thought.thought_type === 'synchronization') {
                        color = '#4CAF50';
                        icon = '🟢';
                    } else if (thought.thought_type === 'wave_propagation') {
                        color = '#4ecdc4';
                        icon = '🔵';
                    } else if (thought.thought_type === 'goal_pursuit') {
                        color = '#ffe66d';
                        icon = '🟡';
                    } else if (thought.thought_type === 'learning') {
                        color = '#95e1d3';
                        icon = '🟢';
                    } else if (thought.thought_type === 'understanding') {
                        color = '#4CAF50';
                        icon = '✅';
                    } else if (thought.thought_type === 'confusion') {
                        color = '#f44336';
                        icon = '❓';
                    } else if (thought.thought_type === 'stagnation') {
                        color = '#ff9800';
                        icon = '⏸';
                    } else if (thought.thought_type === 'uncertainty') {
                        color = '#9c27b0';
                        icon = '💭';
                    }
                    
                    const neuronLabel = thought.neuron_id === 0 ? 'NETWORK' : 'N' + thought.neuron_id;
                    
                    html += '<div style="margin-bottom: 8px; padding: 5px; background: #15192e; border-left: 3px solid ' + color + '; border-radius: 3px;">';
                    html += '<div style="display: flex; justify-content: space-between; margin-bottom: 3px;">';
                    html += '<span style="color: ' + color + '; font-weight: bold;">' + icon + ' [' + timeStr + '] ' + neuronLabel + '</span>';
                    html += '<span style="color: #666; font-size: 10px;">' + thought.thought_type + '</span>';
                    html += '</div>';
                    html += '<div style="color: #e0e0e0; line-height: 1.4;">' + thought.message + '</div>';
                    html += '</div>';
                }
                
                thoughtStreamDiv.innerHTML = html;
                
                // Auto-scroll to top (newest thoughts)
                thoughtStreamDiv.scrollTop = 0;
                
                lastThoughtCount = data.thought_stream.length;
            }
        }
        
        function updateMetrics(data) {
            document.getElementById('syncValue').textContent = data.synchronization.toFixed(3);
            document.getElementById('actValue').textContent = data.avg_activation.toFixed(3);
            document.getElementById('connValue').textContent = data.avg_connection.toFixed(3);
            document.getElementById('neuronCount').textContent = data.n_neurons;
            document.getElementById('currentTime').textContent = data.time.toFixed(2);
            document.getElementById('updateCount').textContent = data.update_count;
            
            updateGoals(data);
            updateDesign(data);
            updateThoughtStream(data);
            
            // Update plots
            timeData.push(data.time);
            syncData.push(data.synchronization);
            actData.push(data.avg_activation);
            
            if (timeData.length > maxPoints) {
                timeData.shift();
                syncData.shift();
                actData.shift();
            }
            
            // Update Plotly plots
            Plotly.update('syncPlot', {
                x: [timeData],
                y: [syncData]
            }, {
                xaxis: { range: [Math.max(0, data.time - 10), data.time + 1] }
            });
            
            Plotly.update('actPlot', {
                x: [timeData],
                y: [actData]
            }, {
                xaxis: { range: [Math.max(0, data.time - 10), data.time + 1] }
            });
        }
        
        // Fetch data every 100ms
        let fetchCount = 0;
        setInterval(async () => {
            try {
                fetchCount++;
                const response = await fetch('/api/network?t=' + Date.now()); // Cache bust
                const data = await response.json();
                
                // Debug logging
                if (fetchCount % 10 === 0) {
                    console.log('Update #' + fetchCount + ', Time: ' + data.time + ', Sync: ' + data.synchronization);
                }
                
                drawNetwork(data);
                updateMetrics(data);
            } catch (error) {
                console.error('Error fetching data:', error);
            }
        }, 100);
        
        // Handle window resize
        window.addEventListener('resize', () => {
            canvas.width = canvas.offsetWidth;
            canvas.height = canvas.offsetHeight;
        });
        
        // Command interface
        function addCommandOutput(text, type = 'info') {
            const output = document.getElementById('commandOutput');
            const color = type === 'error' ? '#f44336' : type === 'success' ? '#4CAF50' : '#e0e0e0';
            const time = new Date().toLocaleTimeString();
            output.innerHTML += '<div style="color: ' + color + '; margin: 5px 0;">[' + time + '] ' + text + '</div>';
            output.scrollTop = output.scrollHeight;
        }
        
        function sendCommand() {
            const input = document.getElementById('commandInput');
            const command = input.value.trim().toLowerCase();
            input.value = '';
            
            if (!command) return;
            
            addCommandOutput('> ' + command, 'info');
            
            fetch('/api/command', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ command: command })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    addCommandOutput(data.message, 'success');
                    // Force immediate UI refresh after command to show new goals
                    setTimeout(() => {
                        fetch('/api/network')
                            .then(response => response.json())
                            .then(networkData => {
                                updateMetrics(networkData);
                                updateGoals(networkData);
                                updateDesign(networkData);
                            })
                            .catch(error => console.error('Error refreshing UI:', error));
                    }, 200);
                } else {
                    addCommandOutput('Error: ' + data.message, 'error');
                }
            })
            .catch(error => {
                addCommandOutput('Error: ' + error.message, 'error');
            });
        }
        
        // Allow Enter key to send command
        document.getElementById('commandInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                sendCommand();
            }
        });
    </script>
</body>
</html>
"""

# Toaster design progress
function update_goal_progress!()
    # Check if we have any goals
    if isempty(state.hierarchy.goals) || state.hierarchy.active_goal == :none
        return  # No goals to update
    end
    # Check if current goal is achieved based on neuron synchronization
    current_goal = state.hierarchy.goals[state.hierarchy.active_goal]
    
    # Get BOUNDARY neurons working on current goal (only boundary neurons matter)
    boundary_indices = compute_topological_boundaries!(state.network)
    goal_boundary_neurons = [i for i in boundary_indices if haskey(state.neuron_goals, i) && state.neuron_goals[i] == state.hierarchy.active_goal]
    
    if !isempty(goal_boundary_neurons)
        # Boundary-Constrained Reward Computation
        # Theory: Reward R(G) computed only from boundary neurons pursuing goal G
        # This is O(|boundary|) instead of O(N)
        
        # Progress based on synchronization of boundary goal neurons
        goal_phases = [state.network.neurons[i].phase for i in goal_boundary_neurons]
        goal_sync = abs(sum(exp(im * phase) for phase in goal_phases) / length(goal_boundary_neurons))
        
        # Activation-based reward (only from boundary neurons)
        goal_activations = [state.network.neurons[i].activation for i in goal_boundary_neurons]
        avg_activation = mean(goal_activations)
        
        # Combined reward: R(G) = α·sync + β·activation (boundary-constrained)
        reward = 0.6 * goal_sync + 0.4 * avg_activation
        
        # Track progress rate for stagnation detection
        old_progress = current_goal.progress
        current_goal.progress = min(1.0, current_goal.progress + 0.01 * reward)
        progress_rate = current_goal.progress - old_progress
        
        # Update last progress time if progress was made
        if progress_rate > 0.001
            current_goal.last_active = state.network.time
        end
        
        # Generate Stagnation Thoughts (boundary-constrained)
        # Theory: Generated when progress rate is low for extended time
        time_since_progress = state.network.time - current_goal.last_active
        if progress_rate < 0.001 && time_since_progress > 5.0  # Stuck for 5+ timesteps
            stagnation_weight = time_since_progress / 50.0  # Increases with time stuck
            # Sample a boundary neuron to generate stagnation thought
            if !isempty(goal_boundary_neurons)
                sample_neuron = rand(goal_boundary_neurons)
                log_thought!(state.thought_stream, sample_neuron, :stagnation,
                    "Boundary neuron $sample_neuron detecting stagnation on goal '$(current_goal.id)' (progress_rate=$(round(progress_rate, digits=4)), time_stuck=$(round(time_since_progress, digits=1)) steps)",
                    Dict("goal" => string(current_goal.id), "progress_rate" => progress_rate, 
                         "time_since_progress" => time_since_progress, "stagnation_weight" => stagnation_weight,
                         "current_progress" => current_goal.progress, "boundary" => true),
                    state.network.time)
            end
        end
        
        # Check if goal achieved
        if current_goal.progress >= 0.8 && !current_goal.achieved
            current_goal.achieved = true
            
            # OPTION 4: Hebbian learning - strengthen word→goal associations when goal succeeds (Boundary-Constrained)
            if !isempty(state.last_prompt_words)
                goal_type = :design  # Infer from goal structure
                goal_desc_lower = lowercase(current_goal.description)
                if occursin("learn", goal_desc_lower) || occursin("understand", goal_desc_lower) || occursin("research", goal_desc_lower)
                    goal_type = :learn
                elseif occursin("integrate", goal_desc_lower) || occursin("combine", goal_desc_lower)
                    goal_type = :integrate
                end
                
                # Boundary-Constrained Learning: Only strengthen associations for boundary-relevant words
                boundary_indices = compute_topological_boundaries!(state.network)
                boundary_concepts_set = get_boundary_relevant_concepts(state.hierarchy, state.neuron_goals, boundary_indices)
                boundary_concepts = isempty(boundary_concepts_set) ? nothing : boundary_concepts_set
                
                strengthen_word_goal_association!(state.concept_embedding, state.last_prompt_words, goal_type, boundary_concepts, 0.1)
            end
            
            # Switch to next goal
            old_goal = state.hierarchy.active_goal
            for (id, goal) in state.hierarchy.goals
                if !goal.achieved
                    deps_ok = all(state.hierarchy.goals[dep].achieved for dep in goal.dependencies)
                    if deps_ok
                        state.hierarchy.active_goal = id
                        
                        # Log goal switch
                        if old_goal != id
                            log_thought!(state.thought_stream, 0, :goal_pursuit,
                                "Goal switch: '$(old_goal)' → '$(id)' (dependencies satisfied)",
                                Dict("old_goal" => string(old_goal), "new_goal" => string(id), "switch_type" => "dependency_satisfied"),
                                state.network.time)
                        end
                        break
                    end
                end
            end
            
            # Generate design component when goal achieved (dynamic based on current goal)
            if state.hierarchy.active_goal != :none && haskey(state.hierarchy.goals, state.hierarchy.active_goal)
                generate_design_component!(state.hierarchy.active_goal, state.design_requirements, state.current_design)
            end
            
            # Log design component generation
            if !isempty(state.current_design)
                log_thought!(state.thought_stream, 0, :goal_pursuit,
                    "Design component generated for goal '$(state.hierarchy.active_goal)'",
                    Dict("goal" => string(state.hierarchy.active_goal), "design" => state.current_design),
                    state.network.time)
            end
        end
    end
end

# API endpoint for network data
function network_data_handler(req)
    # Update network
    dt = 0.05
    
    # Update individual time dimensions first (Sakana AI continuous thought)
    update_time_dimensions!(state.network, state.neuron_goals, state.hierarchy, dt)
    
    kuramoto_update!(state.network, dt)
    wave_propagation!(state.network, dt)
    hebbian_update!(state.network, 0.01)
    
    # Log synchronization thoughts (sample BOUNDARY neurons only)
    if state.step % 5 == 0  # Sample every 5 steps
        boundary_indices = compute_topological_boundaries!(state.network)
        r, psi = order_parameter(state.network, boundary_indices)  # Compute from boundaries only
        if r > 0.5 && !isempty(boundary_indices)
            # Sample only from boundary neurons
            n_sample = min(5, length(boundary_indices))
            sample_neurons = rand(boundary_indices, n_sample)
            for i in sample_neurons
                goal_id = get(state.neuron_goals, i, :none)
                log_thought!(state.thought_stream, i, :synchronization, 
                    "Boundary neuron $i synchronizing (phase=$(round(state.network.neurons[i].phase, digits=2)), activation=$(round(state.network.neurons[i].activation, digits=2)))",
                    Dict("phase" => state.network.neurons[i].phase, "activation" => state.network.neurons[i].activation, 
                         "goal" => string(goal_id), "sync_level" => r, "boundary" => true),
                    state.network.time)
            end
        end
    end
    
    # Metacognitive Thought Generation (Boundary-Constrained)
    # CRITICAL: Only boundary neurons generate metacognitive thoughts about their understanding
    if state.step % 10 == 0  # Generate metacognitive thoughts every 10 steps
        boundary_indices = compute_topological_boundaries!(state.network)
        r, psi = order_parameter(state.network, boundary_indices)
        
        # Sample boundary neurons for metacognitive thought generation
        n_sample = min(10, length(boundary_indices))
        sample_neurons = rand(boundary_indices, n_sample)
        
        for i in sample_neurons
            neuron = state.network.neurons[i]
            goal_id = get(state.neuron_goals, i, :none)
            
            if goal_id != :none && haskey(state.hierarchy.goals, goal_id)
                goal = state.hierarchy.goals[goal_id]
                
                # Get goal-specific synchronization (only boundary neurons on this goal)
                goal_boundary_neurons = [j for j in boundary_indices if haskey(state.neuron_goals, j) && state.neuron_goals[j] == goal_id]
                goal_sync = 0.0
                if !isempty(goal_boundary_neurons)
                    goal_phases = [state.network.neurons[j].phase for j in goal_boundary_neurons]
                    goal_sync = abs(sum(exp(im * phase) for phase in goal_phases) / length(goal_boundary_neurons))
                end
                
                # Extract concepts from goal description to check understanding
                words = split(lowercase(goal.description), r"[^a-z]+")
                words = filter(w -> length(w) > 2, words)
                
                # Check concept understanding confidence
                low_confidence_concepts = String[]
                for word in words
                    concept_lower = lowercase(word)
                    if haskey(state.concept_embedding.concept_memory, concept_lower)
                        memory = state.concept_embedding.concept_memory[concept_lower]
                        usage_count = get(memory, "usage_count", 0)
                        if usage_count < 3  # Low usage = low confidence
                            push!(low_confidence_concepts, word)
                        end
                    else
                        push!(low_confidence_concepts, word)  # Missing concept
                    end
                end
                
                # Generate Confusion Thoughts
                # Theory: Generated when low usage_count OR low sync despite high activation
                confusion_weight = (1.0 - goal_sync) * neuron.activation
                if confusion_weight > 0.5 || !isempty(low_confidence_concepts)
                    concept_str = isempty(low_confidence_concepts) ? "general" : join(low_confidence_concepts[1:min(3, length(low_confidence_concepts))], ", ")
                    log_thought!(state.thought_stream, i, :confusion,
                        "Boundary neuron $i confused about concept(s): $concept_str (sync=$(round(goal_sync, digits=2)), activation=$(round(neuron.activation, digits=2)), confusion_weight=$(round(confusion_weight, digits=2)))",
                        Dict("goal" => string(goal_id), "sync" => goal_sync, "activation" => neuron.activation, 
                             "confusion_weight" => confusion_weight, "low_confidence_concepts" => low_confidence_concepts, "boundary" => true),
                        state.network.time)
                end
                
                # Generate Understanding Thoughts (high confidence)
                # Theory: Generated when high usage_count AND high sync
                if goal_sync > 0.7 && neuron.activation > 0.6 && isempty(low_confidence_concepts)
                    understanding_weight = length(words) * goal_sync  # More concepts understood = higher weight
                    log_thought!(state.thought_stream, i, :understanding,
                        "Boundary neuron $i understands goal '$(goal.id)' well (sync=$(round(goal_sync, digits=2)), activation=$(round(neuron.activation, digits=2)))",
                        Dict("goal" => string(goal_id), "sync" => goal_sync, "activation" => neuron.activation,
                             "understanding_weight" => understanding_weight, "boundary" => true),
                        state.network.time)
                end
                
                # Generate Uncertainty Thoughts
                # Theory: Generated when synchronization is inconsistent (high variance)
                if state.step > 20  # Need history to compute variance
                    # Simplified: uncertainty if sync is inconsistent (low sync despite high activation)
                    if goal_sync < 0.4 && neuron.activation > 0.7
                        log_thought!(state.thought_stream, i, :uncertainty,
                            "Boundary neuron $i uncertain about approach (low sync=$(round(goal_sync, digits=2)) despite high activation=$(round(neuron.activation, digits=2)))",
                            Dict("goal" => string(goal_id), "sync" => goal_sync, "activation" => neuron.activation,
                                 "uncertainty_reason" => "low_sync_high_activation", "boundary" => true),
                            state.network.time)
                    end
                end
            end
        end
    end
    
    # Self-monitoring: Compute neuron contribution and trigger external verification
    boundary_indices = compute_topological_boundaries!(state.network)
    compute_neuron_contribution!(state.network, state.neuron_goals, state.hierarchy, boundary_indices)
    
    # Contributing neurons trigger external signals (self-verification mechanism)
    # Neurons don't communicate directly—they trigger external systems
    trigger_contribution_signals!(state.network, state.neuron_goals, state.hierarchy, state.global_signals, boundary_indices)
    
    # Update global signals (indirect propagation - like hunger signals)
    # This allows goals to affect all neurons indirectly, not just direct connections
    update_global_signals!(state.global_signals, state.network, state.hierarchy, dt)
    
    # Energy-Based Deformation on Boundaries (boundary-constrained optimization)
    # Theory: Minimize energy E(s, a) = ∫_∂M ||∇s||² dμ only on boundary points
    compute_energy_gradient!(state.network, boundary_indices, dt)
    
    # Log global signal thoughts
    if !isempty(state.global_signals.signals) && state.step % 10 == 0
        for signal in state.global_signals.signals
            if signal.intensity > 0.1
                log_thought!(state.thought_stream, 0, :goal_pursuit,
                    "Global $(signal.signal_type) signal active (intensity=$(round(signal.intensity, digits=2)), target=$(signal.target_goal))",
                    Dict("signal_type" => string(signal.signal_type), "intensity" => signal.intensity, 
                         "target_goal" => string(signal.target_goal)),
                    state.network.time)
            end
        end
    end
    
    # Aggregate metacognitive thoughts into confusion/stagnation signals (every 5 steps)
    # Theory: C_G(t) = Σ w(τ) · I(thought_τ = confusion ∧ goal_τ = G)
    if state.step % 5 == 0
        aggregate_metacognitive_thoughts!(state.metacognitive_signals, state.thought_stream, state.network.time)
        
        # Metacognitive Direction Change Trigger (thought-based)
        # Theory: change_direction if C_G(t) > θ_confusion AND S_G(t) > θ_stagnation AND t - t_last_progress > T_min
        current_goal_id = state.hierarchy.active_goal
        if haskey(state.hierarchy.goals, current_goal_id)
            current_goal = state.hierarchy.goals[current_goal_id]
            
            confusion_signal = get_confusion_signal(state.metacognitive_signals, current_goal_id)
            stagnation_signal = get_stagnation_signal(state.metacognitive_signals, current_goal_id)
            time_since_progress = state.network.time - current_goal.last_active
            
            # Thresholds from theory
            confusion_threshold = 20.0  # θ_confusion
            stagnation_threshold = 10.0  # θ_stagnation
            min_stagnation_time = 50.0  # T_min (50 timesteps)
            
            # Check if direction change should be triggered
            if confusion_signal > confusion_threshold && stagnation_signal > stagnation_threshold && time_since_progress > min_stagnation_time
                # Generate stagnation thought about direction change
                log_thought!(state.thought_stream, 0, :stagnation,
                    "Changing direction due to accumulated confusion (confusion=$(round(confusion_signal, digits=1)), stagnation=$(round(stagnation_signal, digits=1)), time_stuck=$(round(time_since_progress, digits=1)))",
                    Dict("goal" => string(current_goal_id), "confusion_signal" => confusion_signal, 
                         "stagnation_signal" => stagnation_signal, "time_since_progress" => time_since_progress,
                         "trigger" => "metacognitive_direction_change"),
                    state.network.time)
                
                # Pause current goal
                current_goal.state = :paused
                current_goal.last_active = state.network.time
                
                # Find alternative goal (related goal, or different approach)
                # Strategy: Look for goals that share concepts but haven't been tried, or explore unrelated goal
                alternative_goal = :none
                best_value = -1.0
                
                # First, try related goals (share dependencies or concepts)
                for (id, goal) in state.hierarchy.goals
                    if id != current_goal_id && !goal.achieved && goal.state != :paused
                        # Check if dependencies are satisfied
                        deps_ok = all(state.hierarchy.goals[dep].achieved for dep in goal.dependencies)
                        if deps_ok
                            # Estimate value (simplified - could use estimate_goal_value)
                            value = goal.progress + (goal.terminal ? 0.3 : 0.0)
                            if value > best_value
                                best_value = value
                                alternative_goal = id
                            end
                        end
                    end
                end
                
                # If no related goal found, try paused goals (resume exploration)
                if alternative_goal == :none
                    for (id, goal) in state.hierarchy.goals
                        if goal.state == :paused && !goal.achieved
                            alternative_goal = id
                            break
                        end
                    end
                end
                
                # If still no alternative, create exploration goal or stay paused
                if alternative_goal != :none
                    # Activate alternative goal
                    new_goal = state.hierarchy.goals[alternative_goal]
                    new_goal.state = :active
                    new_goal.last_active = state.network.time
                    new_goal.switch_count += 1
                    state.hierarchy.active_goal = alternative_goal
                    
                    # Log direction change
                    log_thought!(state.thought_stream, 0, :goal_pursuit,
                        "Direction change triggered by metacognitive thoughts: '$(current_goal_id)' → '$(alternative_goal)' (confusion=$(round(confusion_signal, digits=1)), stagnation=$(round(stagnation_signal, digits=1)))",
                        Dict("old_goal" => string(current_goal_id), "new_goal" => string(alternative_goal),
                             "confusion_signal" => confusion_signal, "stagnation_signal" => stagnation_signal,
                             "switch_type" => "metacognitive_direction_change"),
                        state.network.time)
                else
                    # No alternative found - log that we're exploring
                    log_thought!(state.thought_stream, 0, :uncertainty,
                        "No clear alternative goal found after confusion. Current goal paused, exploring...",
                        Dict("goal" => string(current_goal_id), "confusion_signal" => confusion_signal,
                             "stagnation_signal" => stagnation_signal),
                        state.network.time)
                end
            end
        end
    end
    
    # Kurzweil pattern detection (every 5 steps, boundary-constrained)
    if state.step % 5 == 0
        kurzweil_update!(state.pattern_model, state.network, boundary_indices, state.network.time)
    end
    
    # Policy divergence detection (every 15 steps, boundary-constrained)
    if state.step % 15 == 0
        divergence_detected, best_goal = detect_policy_divergence(state.hierarchy, boundary_indices, state.neuron_goals, 0.2)
        
        if divergence_detected && best_goal != state.hierarchy.active_goal && state.hierarchy.active_goal != :none && haskey(state.hierarchy.goals, state.hierarchy.active_goal)
            # Pause current goal
            old_goal = state.hierarchy.goals[state.hierarchy.active_goal]
            old_goal.state = :paused
            old_goal.last_active = state.network.time
            
            # Activate best goal
            new_goal = state.hierarchy.goals[best_goal]
            new_goal.state = :active
            new_goal.last_active = state.network.time
            new_goal.switch_count += 1
            
            state.hierarchy.active_goal = best_goal
            
            # Log policy divergence (goal switch)
            log_thought!(state.thought_stream, 0, :goal_pursuit,
                "Policy divergence detected: '$(old_goal.id)' → '$(new_goal.id)' (value improvement)",
                Dict("old_goal" => string(old_goal.id), "new_goal" => string(new_goal.id), 
                     "old_value" => estimate_goal_value(old_goal, state.hierarchy, boundary_indices, state.neuron_goals),
                     "new_value" => estimate_goal_value(new_goal, state.hierarchy, boundary_indices, state.neuron_goals),
                     "type" => "policy_divergence"),
                state.network.time)
        end
    end
    
    # Information gap detection and autonomous goal generation (every 10 steps, boundary-constrained)
    if state.step % 10 == 0
        # Detect information gaps for boundary goals
        gaps = detect_information_gaps(state.concept_embedding, state.hierarchy, 
                                      state.neuron_goals, boundary_indices)
        
        # Process top gap (highest priority)
        if !isempty(gaps)
            top_gap = gaps[1]
            gap_goal_id = top_gap["goal_id"]
            
            # Generate query from gap
            query = generate_query_from_gap(top_gap)
            
            # Query internet (boundary-constrained)
            context = Dict("time" => state.network.time)
            query_result = query_internet(state.internet_query_system, query, context; 
                                         boundary_goal_id=gap_goal_id)
            
            if !haskey(query_result, "error")
                info = query_result["result"]
                
                # Integrate information into embeddings
                discovered_concepts = integrate_information!(state.concept_embedding, info, top_gap, state.network.time)
                
                # Generate sub-goals from discovered information
                new_goals = generate_goals_from_gaps([top_gap], info, state.network.time)
                
                # Add goals to hierarchy
                add_autonomous_goals!(state.hierarchy, new_goals, gap_goal_id)
                
                # Log information discovery
                log_thought!(state.thought_stream, 0, :goal_pursuit,
                    "Discovered information for gap: $(top_gap["goal_description"]) - Found $(length(discovered_concepts)) concepts",
                    Dict("gap" => top_gap, "concepts" => discovered_concepts, "query" => query, "type" => "information_discovery"),
                    state.network.time)
            end
        end
    end
    
    # Trigger goal-driven signals based on active goal progress
    if state.step % 20 == 0  # Check goal urgency periodically
        if state.hierarchy.active_goal == :none || !haskey(state.hierarchy.goals, state.hierarchy.active_goal)
            return  # No active goal
        end
        active_goal = state.hierarchy.goals[state.hierarchy.active_goal]
        if !active_goal.achieved
            # Urgency increases as goal becomes more important
            urgency = 0.3 + 0.7 * (1.0 - active_goal.progress)
            trigger_goal_drive!(state.global_signals, state.hierarchy.active_goal, urgency)
            
            # Log goal pursuit thoughts (only boundary neurons)
            boundary_indices = compute_topological_boundaries!(state.network)
            goal_boundary_neurons = [i for i in boundary_indices if haskey(state.neuron_goals, i) && state.neuron_goals[i] == state.hierarchy.active_goal]
            if !isempty(goal_boundary_neurons)
                sample_goal_neurons = rand(goal_boundary_neurons, min(3, length(goal_boundary_neurons)))
                for i in sample_goal_neurons
                    log_thought!(state.thought_stream, i, :goal_pursuit,
                        "Boundary neuron $i pursuing goal '$(active_goal.id)' (progress=$(round(active_goal.progress*100, digits=1))%, urgency=$(round(urgency, digits=2)))",
                        Dict("goal" => string(active_goal.id), "progress" => active_goal.progress, "urgency" => urgency, "boundary" => true),
                        state.network.time)
                end
            end
        end
    end
    
    state.network.time += dt
    state.update_count += 1
    state.step += 1
    
    # Log wave propagation thoughts (sample high activation BOUNDARY neurons)
    if state.step % 8 == 0
        boundary_indices = compute_topological_boundaries!(state.network)
        high_activation_boundary = [i for i in boundary_indices if state.network.neurons[i].activation > 0.7]
        if !isempty(high_activation_boundary)
            sample_high = rand(high_activation_boundary, min(3, length(high_activation_boundary)))
            for i in sample_high
                goal_id = get(state.neuron_goals, i, :none)
                log_thought!(state.thought_stream, i, :wave_propagation,
                    "Boundary neuron $i high activation (activation=$(round(state.network.neurons[i].activation, digits=2)), propagating waves along boundary)",
                    Dict("activation" => state.network.neurons[i].activation, "phase" => state.network.neurons[i].phase,
                         "goal" => string(goal_id), "boundary" => true),
                    state.network.time)
            end
        end
    end
    
    # Update toaster design progress
    if state.step % 10 == 0  # Every 10 updates
        update_goal_progress!()
        
        # Log goal achievement thoughts
        for (id, goal) in state.hierarchy.goals
            if goal.achieved && state.step % 50 == 0  # Log achievement occasionally
                log_thought!(state.thought_stream, 0, :goal_pursuit,
                    "Goal '$(goal.id)' achieved! (progress=$(round(goal.progress*100, digits=1))%)",
                    Dict("goal" => string(id), "progress" => goal.progress, "achieved" => true),
                    state.network.time)
            end
        end
    end
    
    # Compute metrics (from boundary neurons only)
    boundary_indices = compute_topological_boundaries!(state.network)
    r, psi = order_parameter(state.network, boundary_indices)
    # Only compute averages from boundary neurons (boundary-constrained)
    boundary_activations = [state.network.neurons[i].activation for i in boundary_indices]
    avg_activation = isempty(boundary_activations) ? 0.0 : mean(boundary_activations)
    # Only compute connection average from boundary connections
    boundary_connections = [state.network.connections[i, j] for i in boundary_indices for j in boundary_indices if i != j]
    avg_connection = isempty(boundary_connections) ? 0.0 : mean(boundary_connections)
    
    # Prepare data (include contribution scores for self-monitoring)
    neurons_data = [
        Dict(
            "position" => n.position,
            "activation" => n.activation,
            "phase" => n.phase,
            "contribution" => n.contribution,  # Self-monitoring: contribution score (counterfactual impact)
            "is_boundary" => i in boundary_indices,
            "is_contributing" => n.contribution > 0.7  # Contributing neurons trigger external signals
        ) for (i, n) in enumerate(state.network.neurons)
    ]
    
    # Only show boundary connections (boundary-constrained visualization)
    connections_data = []
    for i in boundary_indices
        for j in boundary_indices
            if i != j && state.network.connections[i, j] > 0.05
                push!(connections_data, Dict(
                    "from" => state.network.neurons[i].position,
                    "to" => state.network.neurons[j].position,
                    "strength" => state.network.connections[i, j]
                ))
            end
        end
    end
    
    # Goal progress data with persistence tracking
    goals_data = []
    for (id, goal) in state.hierarchy.goals
        push!(goals_data, Dict(
            "id" => string(id),
            "description" => goal.description,
            "progress" => goal.progress,
            "achieved" => goal.achieved,
            "active" => id == state.hierarchy.active_goal,
            "state" => string(goal.state),
            "switch_count" => goal.switch_count,
            "abandonment_risk" => goal.abandonment_risk,
            "time_since_active" => state.network.time - goal.last_active
        ))
    end
    
    # Include dropped goals in response
    dropped_goals_data = []
    for id in state.hierarchy.dropped_goals
        push!(dropped_goals_data, Dict("id" => string(id), "status" => "dropped"))
    end
    
    # Neuron goal assignments
    neuron_goal_map = Dict(string(i) => string(g) for (i, g) in state.neuron_goals)
    
    # Thought stream data (last 50 thoughts for real-time display)
    recent_thoughts = []
    thought_count = length(state.thought_stream.thoughts)
    start_idx = max(1, thought_count - 49)  # Last 50 thoughts
    for i in start_idx:thought_count
        thought = state.thought_stream.thoughts[i]
        push!(recent_thoughts, Dict(
            "timestamp" => thought.timestamp,
            "neuron_id" => thought.neuron_id,
            "thought_type" => string(thought.thought_type),
            "message" => thought.message,
            "data" => thought.data
        ))
    end
    
    response_data = Dict(
        "neurons" => neurons_data,
        "connections" => connections_data,
        "synchronization" => r,
        "avg_activation" => avg_activation,
        "avg_connection" => avg_connection,
        "n_neurons" => state.network.n_neurons,
        "time" => state.network.time,
        "update_count" => state.update_count,
        "goals" => goals_data,
        "dropped_goals" => dropped_goals_data,
        "current_goal" => state.hierarchy.active_goal == :none ? "None" : string(state.hierarchy.active_goal),
        "neuron_goals" => neuron_goal_map,
        "current_design" => state.current_design,
        "goal_history" => state.hierarchy.goal_history,
        "thought_stream" => recent_thoughts
    )
    
    return HTTP.Response(200, ["Content-Type" => "application/json"], body=JSON.json(response_data))
end

# Natural language prompt parser - NOW USES EMBEDDINGS!
# This replaces simple keyword matching with embedding-based understanding
# CRITICAL: Uses boundary-constrained concept search for efficiency
# NOTE: This function is kept for backward compatibility but parse_prompt_with_embeddings is called directly now
function parse_design_prompt(prompt::String)
    # Get boundary-relevant concepts for efficient memory retrieval
    boundary_indices = compute_topological_boundaries!(state.network)
    boundary_concepts = get_boundary_relevant_concepts(state.hierarchy, state.neuron_goals, boundary_indices)
    
    # Use embedding-based parser from concept embedding system (boundary-constrained)
    design_type, requirements, prompt_emb, action_type = parse_prompt_with_embeddings(
        state.concept_embedding, prompt, state.network.time, 
        isempty(boundary_concepts) ? nothing : boundary_concepts
    )
    return design_type, requirements
end

# Generate goals from design prompt - GENERAL FUNCTION LEARNING (no hardcoded examples)
# Theory: Network learns to decompose goals through experience, not memorization
function generate_goals_from_prompt(design_type::String, requirements::Vector{String}, current_time::Float64 = 0.0, action_type::Symbol = :design)
    goals = Dict{Symbol, Goal}()
    
    # Generate goal ID from design_type (sanitize for Symbol)
    design_id_str = replace(lowercase(design_type), " " => "_", "-" => "_", "." => "")
    design_id = Symbol("design_$(design_id_str)")
    
    # Main design goal
    design_desc = action_type == :design ? "Design $design_type" : 
                  action_type == :learn ? "Learn about $design_type" :
                  action_type == :integrate ? "Integrate $design_type" : "Work on $design_type"
    goals[design_id] = create_goal(design_id, design_desc, true, Symbol[], current_time)
    
    # Generate learning goal (always needed to understand the object)
    learn_id = Symbol("learn_$(design_id_str)")
    learn_desc = "Learn about $design_type principles and components"
    goals[learn_id] = create_goal(learn_id, learn_desc, false, Symbol[], current_time)
    
    # Generate sub-goals from requirements (dynamic, not hardcoded)
    sub_goal_ids = Symbol[]
    for (idx, req) in enumerate(requirements)
        req_clean = replace(lowercase(req), " " => "_", "-" => "_", "." => "")
        req_id = Symbol("design_$(req_clean)_$(idx)")
        req_desc = "Design $req for $design_type"
        goals[req_id] = create_goal(req_id, req_desc, false, Symbol[learn_id], current_time)
        push!(sub_goal_ids, req_id)
    end
    
    # If no specific requirements, create generic structure/features goals
    if isempty(requirements)
        struct_id = Symbol("design_structure_$(design_id_str)")
        goals[struct_id] = create_goal(struct_id, "Design structure and form for $design_type", false, Symbol[learn_id], current_time)
        push!(sub_goal_ids, struct_id)
        
        features_id = Symbol("design_features_$(design_id_str)")
        goals[features_id] = create_goal(features_id, "Design features and details for $design_type", false, Symbol[learn_id], current_time)
        push!(sub_goal_ids, features_id)
    end
    
    # Integration goal depends on all sub-goals
    integrate_id = Symbol("integrate_$(design_id_str)")
    goals[integrate_id] = create_goal(integrate_id, "Integrate all components for $design_type", false, sub_goal_ids, current_time)
    
    # Main design goal depends on integration
    goals[design_id].dependencies = [integrate_id]
    
    # Start with learning goal (foundational knowledge first)
    return GoalHierarchy(goals, learn_id, Symbol[design_id], [], Symbol[])
end

# Generate design components based on goals and requirements
function generate_design_component!(goal_id::Symbol, requirements::Vector{String}, design::Dict)
    if goal_id == :learn_heating || goal_id == :learn_materials
        design[:power] = rand(800:1500)
        design[:material] = rand(["stainless_steel", "plastic", "ceramic", "fabric", "cotton"])
    elseif goal_id == :design_safety
        design[:auto_shutoff] = true
        design[:shutoff_time] = rand(30:300)
    elseif goal_id == :design_mechanics || goal_id == :design_structure
        design[:spring_force] = rand(5:20)
        design[:lever_ratio] = rand(3:8)
    elseif goal_id == :design_details
        # Parse requirements
        for req in requirements
            if startswith(req, "buttons:")
                num = split(req, ":")[2]
                design[:buttons] = num == "some" ? rand(2:5) : parse(Int, num)
                design[:button_location] = "sleeves"
            end
            if occursin("sleeves", req)
                design[:sleeves] = true
                design[:sleeve_length] = rand(["short", "long", "three_quarter"])
            end
            if occursin("pockets", req)
                design[:pockets] = true
                design[:pocket_count] = rand(1:4)
            end
        end
    elseif goal_id == :integrate_system || goal_id == :integrate_design
        design[:slots] = rand(2:4)
        design[:browning_levels] = rand(1:7)
        design[:color] = rand(["black", "white", "red", "blue", "gray"])
    end
end

# Command handler
function command_handler(req)
    try
        body = JSON.parse(String(HTTP.payload(req)))
        command = get(body, "command", "")
        command_lower = lowercase(strip(command))
        
        result = Dict("success" => false, "message" => "Unknown command")
        
        # Check if it's a design prompt (not a system command)
        design_keywords = ["design", "make", "create", "build", "toaster", "shirt", "car", "phone", "fashion"]
        is_design_prompt = any(kw -> occursin(kw, command_lower), design_keywords)
        
        if is_design_prompt && !(command_lower in ["reset", "restart", "status", "help", "speed up", "slow down"])
            try
                # Log prompt reception
                log_thought!(state.thought_stream, 0, :goal_pursuit,
                    "Received design prompt: \"$command\"",
                    Dict("prompt" => command, "type" => "user_input"),
                    state.network.time)
                
                # Parse design prompt using embedding-based understanding
                # Network starts blank - learn concepts from prompt itself, not from pre-existing goals
                # Only use boundary concepts if we have existing goals (for efficiency), otherwise parse fresh
                # CRITICAL: Explicitly type boundary_concepts to ensure Julia dispatches correctly
                boundary_concepts::Union{Set{String}, Nothing} = nothing
                if !isempty(state.hierarchy.goals)
                    boundary_indices = compute_topological_boundaries!(state.network)
                    boundary_concepts_set = get_boundary_relevant_concepts(state.hierarchy, state.neuron_goals, boundary_indices)
                    boundary_concepts = isempty(boundary_concepts_set) ? nothing : boundary_concepts_set
                end
                
                # Parse prompt - network learns English concepts as it processes (Option 4)
                design_type, requirements, prompt_emb, action_type = parse_prompt_with_embeddings(
                    state.concept_embedding, command, state.network.time, boundary_concepts
                )
                
                # Extract words from prompt for learning (Option 4)
                prompt_words = filter(w -> length(w) > 2, split(lowercase(command), r"[^a-z]+"))
                state.last_prompt_words = prompt_words  # Store for Hebbian learning when goals succeed
                
                # Log understanding
                log_thought!(state.thought_stream, 0, :goal_pursuit,
                    "Understood prompt: design_type='$design_type', action='$action_type', requirements=[$(join(requirements, ", "))]",
                    Dict("design_type" => design_type, "action_type" => string(action_type), "requirements" => requirements, "type" => "understanding"),
                    state.network.time)
                
                # Generate goals from prompt using learned action_type
                state.hierarchy = generate_goals_from_prompt(design_type, requirements, state.network.time, action_type)
                state.design_requirements = requirements  # Store for later use
                state.network = create_network(1000)
                state.neuron_goals = assign_neurons_to_goals(state.network, state.hierarchy, state.concept_embedding)
                state.current_design = Dict()
                state.update_count = 0
                state.step = 0
                
                # Initialize network: compute boundaries and update once to get things started
                compute_topological_boundaries!(state.network)
                update_time_dimensions!(state.network, state.neuron_goals, state.hierarchy, 0.05)
                kuramoto_update!(state.network, 0.05)
                wave_propagation!(state.network, 0.05)
                
                # Log goal generation
                goal_list = [string(id) for id in keys(state.hierarchy.goals)]
                log_thought!(state.thought_stream, 0, :goal_pursuit,
                    "Generated $(length(goal_list)) goals: [$(join(goal_list, ", "))]. Active goal: '$(state.hierarchy.active_goal)'",
                    Dict("goals" => goal_list, "active_goal" => string(state.hierarchy.active_goal), "type" => "goal_generation"),
                    state.network.time)
                
                req_str = isempty(requirements) ? "" : " (Requirements: $(join(requirements, ", ")))"
                result = Dict("success" => true, "message" => "Starting design: $design_type$req_str. Neurons are synchronizing to work on this task!")
            catch e
                result = Dict("success" => false, "message" => "Error parsing prompt: $(string(e))")
            end
            
        elseif command_lower == "speed up" || command_lower == "faster"
            # Increase update rate
            state.network.time += 1.0
            result = Dict("success" => true, "message" => "Sped up simulation")
            
        elseif command == "slow down" || command == "slower"
            # Decrease update rate (already handled by dt)
            result = Dict("success" => true, "message" => "Simulation speed is fixed at optimal rate")
            
        elseif startswith(command, "add neurons") || startswith(command, "add")
            # Add neurons
            new_count = state.network.n_neurons + 10
            old_network = state.network
            state.network = create_network(new_count)
            state.network.time = old_network.time
            state.neuron_goals = assign_neurons_to_goals(state.network, state.hierarchy)
            result = Dict("success" => true, "message" => "Added neurons. Total: $(new_count)")
            
        elseif command_lower == "reset" || command_lower == "restart"
            # Reset network (keeps current design task)
            state.network = create_network(1000)
            state.neuron_goals = assign_neurons_to_goals(state.network, state.hierarchy)
            state.current_design = Dict()
            state.update_count = 0
            state.step = 0
            result = Dict("success" => true, "message" => "Network reset. Restarting current design task.")
            
        elseif command_lower == "status"
            achieved = sum(g.achieved for (_, g) in state.hierarchy.goals)
            total = length(state.hierarchy.goals)
            boundary_indices = compute_topological_boundaries!(state.network)
            r, _ = order_parameter(state.network, boundary_indices)
            result = Dict("success" => true, "message" => "Status: $(achieved)/$(total) goals achieved, Sync: $(round(r, digits=3)), Time: $(round(state.network.time, digits=2))s")
            
        # Removed hardcoded "make a toaster" command - all design prompts go through general handler above
            
        elseif command == "help"
            result = Dict("success" => true, "message" => "Commands: 'make a toaster', 'speed up', 'add neurons', 'reset', 'status', 'help'")
            
        else
            result = Dict("success" => false, "message" => "Unknown command. Try: 'make a toaster', 'speed up', 'add neurons', 'reset', 'status', 'help'")
        end
        
        return HTTP.Response(200, ["Content-Type" => "application/json"], body=JSON.json(result))
    catch e
        return HTTP.Response(200, ["Content-Type" => "application/json"], 
            body=JSON.json(Dict("success" => false, "message" => "Error: $(string(e))")))
    end
end

# Main server
function run_web_server(port=8080)
    println("=" ^ 70)
    println("Neural Network Web Monitor Server")
    println("=" ^ 70)
    println("\nStarting server on http://localhost:$port")
    println("Open this URL in your browser to view the neural network")
    println("\nPress Ctrl+C to stop the server\n")
    
    router = HTTP.Router()
    HTTP.register!(router, "GET", "/", req -> HTTP.Response(200, body=HTML_DASHBOARD))
    HTTP.register!(router, "GET", "/api/network", network_data_handler)
    HTTP.register!(router, "POST", "/api/command", command_handler)
    
    server = HTTP.serve!(router, "127.0.0.1", port)
    
    try
        wait(server)
    catch e
        if isa(e, InterruptException)
            println("\n\nServer stopped.")
        else
            rethrow(e)
        end
    end
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    Random.seed!(42)
    run_web_server(8080)
end

