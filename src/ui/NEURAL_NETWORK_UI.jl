# Real Neural Network with Live Visualization
# Shows actual neurons, connections, activations, waves, synchronization

using GLMakie
using Observables
using LinearAlgebra
using Random
using Printf

# ============================================================================
# 1. NEURAL NETWORK STRUCTURE
# ============================================================================

mutable struct Neuron
    activation::Float64      # Current activation level
    phase::Float64           # Wave phase [0, 2π]
    frequency::Float64       # Natural frequency (for Kuramoto)
    position::Vector{Float64}  # Position in hyperbolic space (2D Poincaré disk)
    morphology::Float64      # Boundary shape parameter
end

mutable struct NeuralNetwork
    neurons::Vector{Neuron}
    connections::Matrix{Float64}  # Connection strengths (learned via Hebbian)
    n_neurons::Int
    time::Float64
end

function create_network(n_neurons::Int = 50)
    neurons = Vector{Neuron}()
    
    # Initialize neurons in Poincaré disk (within unit circle)
    for i in 1:n_neurons
        # Random position in disk (keep within radius 0.9)
        r = sqrt(rand()) * 0.9
        theta = rand() * 2π
        position = [r * cos(theta), r * sin(theta)]
        
        neuron = Neuron(
            rand() * 0.5,                    # Initial activation
            rand() * 2π,                     # Initial phase
            1.0 + 0.2 * randn(),            # Natural frequency
            position,                        # Position
            rand() * 0.5 + 0.5              # Morphology parameter
        )
        push!(neurons, neuron)
    end
    
    # Initialize sparse connections (only nearby neurons)
    connections = zeros(n_neurons, n_neurons)
    for i in 1:n_neurons
        for j in 1:n_neurons
            if i != j
                dist = norm(neurons[i].position - neurons[j].position)
                if dist < 0.3  # Only connect nearby neurons
                    connections[i, j] = 0.1 * exp(-dist / 0.1)
                end
            end
        end
    end
    
    return NeuralNetwork(neurons, connections, n_neurons, 0.0)
end

# ============================================================================
# 2. KURAMOTO SYNCHRONIZATION
# ============================================================================

function kuramoto_update!(network::NeuralNetwork, dt::Float64)
    n = network.n_neurons
    lambda = 0.2  # Distance decay
    
    for i in 1:n
        dphi_dt = network.neurons[i].frequency
        
        # Coupling term
        for j in 1:n
            if network.connections[i, j] > 0
                dist = norm(network.neurons[i].position - network.neurons[j].position)
                distance_weight = exp(-dist / lambda)
                
                phase_diff = network.neurons[j].phase - network.neurons[i].phase
                coupling = network.connections[i, j] * sin(phase_diff) * distance_weight
                dphi_dt += coupling
            end
        end
        
        # Update phase
        network.neurons[i].phase += dt * dphi_dt
        network.neurons[i].phase = mod(network.neurons[i].phase, 2π)
    end
end

function order_parameter(network::NeuralNetwork)
    n = network.n_neurons
    z = sum(exp(im * neuron.phase) for neuron in network.neurons) / n
    return abs(z), angle(z)
end

# ============================================================================
# 3. WAVE PROPAGATION
# ============================================================================

function wave_propagation!(network::NeuralNetwork, dt::Float64)
    n = network.n_neurons
    c = 1.0  # Wave speed
    
    # Update activations based on phases (wave equation simplified)
    for i in 1:n
        # Activation couples to phase
        network.neurons[i].activation = 0.5 * (1 + sin(network.neurons[i].phase))
        
        # Add wave propagation from neighbors
        for j in 1:n
            if network.connections[i, j] > 0
                dist = norm(network.neurons[i].position - network.neurons[j].position)
                delay = dist / c
                # Simplified: activation propagates with delay
                network.neurons[i].activation += 0.1 * network.connections[i, j] * 
                    network.neurons[j].activation * exp(-dist / 0.2)
            end
        end
        
        # Damping
        network.neurons[i].activation *= 0.95
        network.neurons[i].activation = clamp(network.neurons[i].activation, 0.0, 1.0)
    end
end

# ============================================================================
# 4. HEBBIAN LEARNING
# ============================================================================

function hebbian_update!(network::NeuralNetwork, learning_rate::Float64 = 0.01)
    n = network.n_neurons
    
    for i in 1:n
        for j in 1:n
            if i != j && network.connections[i, j] > 0
                # Hebbian rule: Δw = η * s_i * s_j * cos(φ_i - φ_j)
                hebbian_term = learning_rate * 
                    network.neurons[i].activation * 
                    network.neurons[j].activation * 
                    cos(network.neurons[i].phase - network.neurons[j].phase)
                
                network.connections[i, j] += hebbian_term
                network.connections[i, j] = max(0.0, network.connections[i, j])
                network.connections[i, j] = min(1.0, network.connections[i, j])
            end
        end
    end
end

# ============================================================================
# 5. TOPOLOGICAL BOUNDARIES
# ============================================================================

function compute_boundaries(network::NeuralNetwork, threshold::Float64 = 0.3)
    # Simple boundary detection: neurons with high activation variance
    boundary_indices = Int[]
    
    activations = [n.activation for n in network.neurons]
    mean_activation = mean(activations)
    
    for i in 1:network.n_neurons
        # Boundary if activation differs significantly from neighbors
        neighbor_activations = Float64[]
        for j in 1:network.n_neurons
            if network.connections[i, j] > 0
                push!(neighbor_activations, network.neurons[j].activation)
            end
        end
        
        if !isempty(neighbor_activations)
            local_mean = mean(neighbor_activations)
            if abs(network.neurons[i].activation - local_mean) > threshold
                push!(boundary_indices, i)
            end
        end
    end
    
    return boundary_indices
end

# ============================================================================
# 6. VISUALIZATION UI
# ============================================================================

function create_neural_network_ui()
    # Create network
    network = create_network(50)
    
    # Create figure
    fig = Figure(resolution=(1600, 1000), title="Neural Network Monitor")
    
    # Layout
    g1 = fig[1, 1] = GridLayout()  # Main network view
    g2 = fig[1, 2] = GridLayout()  # Metrics
    g3 = fig[2, 1:2] = GridLayout()  # Time series
    
    # Main network visualization
    ax1 = Axis(g1[1, 1], 
        title="Neural Network (Hyperbolic Space)",
        aspect=DataAspect(),
        xlabel="X (Poincaré disk)",
        ylabel="Y (Poincaré disk)"
    )
    xlims!(ax1, -1.1, 1.1)
    ylims!(ax1, -1.1, 1.1)
    
    # Draw unit circle (boundary of Poincaré disk)
    circle = Circle(Point2f(0, 0), 1.0f0)
    poly!(ax1, circle, color=:black, strokewidth=2, strokecolor=:gray)
    
    # Observable data
    neuron_positions = Observable([Point2f(n.position...) for n in network.neurons])
    neuron_activations = Observable([n.activation for n in network.neurons])
    neuron_phases = Observable([n.phase for n in network.neurons])
    connections_visible = Observable([(i, j) for i in 1:network.n_neurons for j in 1:network.n_neurons 
                                     if network.connections[i, j] > 0.05])
    
    # Draw connections
    connection_lines = Observable(Vector{Point2f}[])
    function update_connections()
        lines = Point2f[]
        for (i, j) in connections_visible[]
            if network.connections[i, j] > 0.05
                strength = network.connections[i, j]
                push!(lines, Point2f(network.neurons[i].position...))
                push!(lines, Point2f(network.neurons[j].position...))
            end
        end
        connection_lines[] = lines
    end
    update_connections()
    
    linesegs!(ax1, connection_lines, color=:gray, linewidth=1)
    
    # Draw neurons
    scatter!(ax1, neuron_positions, 
        color=neuron_activations,
        colormap=:viridis,
        markersize=15,
        strokewidth=1,
        strokecolor=:white
    )
    
    # Phase visualization (as arrows)
    phase_arrows = Observable(Vector{Point2f}[])
    function update_phase_arrows()
        arrows = Point2f[]
        for (i, neuron) in enumerate(network.neurons)
            angle = neuron.phase
            length = 0.1 * neuron.activation
            end_point = neuron.position + [length * cos(angle), length * sin(angle)]
            push!(arrows, Point2f(neuron.position...))
            push!(arrows, Point2f(end_point...))
        end
        phase_arrows[] = arrows
    end
    update_phase_arrows()
    
    arrows!(ax1, [Point2f(n.position...) for n in network.neurons],
        [Point2f(cos(n.phase), sin(n.phase)) * 0.1 * n.activation for n in network.neurons],
        color=:red, arrowsize=0.05, lengthscale=1.0)
    
    # Metrics panel
    ax2 = Axis(g2[1, 1], title="Synchronization Order Parameter")
    ax3 = Axis(g2[2, 1], title="Average Activation")
    ax4 = Axis(g2[3, 1], title="Connection Strength")
    
    # Time series data
    time_data = Observable([0.0])
    sync_data = Observable([0.0])
    activation_data = Observable([0.0])
    connection_data = Observable([0.0])
    
    # Initial values
    r, _ = order_parameter(network)
    push!(sync_data[], r)
    push!(activation_data[], mean([n.activation for n in network.neurons]))
    push!(connection_data[], mean(network.connections))
    
    # Plot time series
    lines!(ax2, time_data, sync_data, color=:blue, linewidth=2, label="r(t)")
    lines!(ax3, time_data, activation_data, color=:green, linewidth=2, label="Activation")
    lines!(ax4, time_data, connection_data, color=:orange, linewidth=2, label="Connections")
    
    xlims!(ax2, 0, 10)
    xlims!(ax3, 0, 10)
    xlims!(ax4, 0, 10)
    ylims!(ax2, 0, 1)
    ylims!(ax3, 0, 1)
    
    # Text displays
    text_display = Observable("Time: 0.0\nSynchronization: 0.0\nActivation: 0.0\nConnections: 0.0")
    text!(g2[4, 1], text_display, fontsize=12, align=(:left, :top))
    
    # Simulation loop
    dt = 0.05
    max_time = 100.0
    
    function update_frame()
        # Update network
        kuramoto_update!(network, dt)
        wave_propagation!(network, dt)
        hebbian_update!(network, 0.01)
        
        network.time += dt
        
        # Update observables
        neuron_positions[] = [Point2f(n.position...) for n in network.neurons]
        neuron_activations[] = [n.activation for n in network.neurons]
        neuron_phases[] = [n.phase for n in network.neurons]
        
        update_connections()
        update_phase_arrows()
        
        # Update metrics
        r, psi = order_parameter(network)
        avg_activation = mean([n.activation for n in network.neurons])
        avg_connection = mean(network.connections)
        
        push!(time_data[], network.time)
        push!(sync_data[], r)
        push!(activation_data[], avg_activation)
        push!(connection_data[], avg_connection)
        
        # Keep only last 200 points
        if length(time_data[]) > 200
            time_data[] = time_data[][end-199:end]
            sync_data[] = sync_data[][end-199:end]
            activation_data[] = activation_data[][end-199:end]
            connection_data[] = connection_data[][end-199:end]
        end
        
        # Update text
        text_display[] = @sprintf("""
        Time: %.2f
        Synchronization: %.3f
        Avg Activation: %.3f
        Avg Connection: %.3f
        Neurons: %d
        Active Connections: %d
        """, 
        network.time, r, avg_activation, avg_connection,
        network.n_neurons, 
        sum(network.connections .> 0.05))
        
        # Update axis limits
        if network.time > 10
            xlims!(ax2, network.time - 10, network.time)
            xlims!(ax3, network.time - 10, network.time)
            xlims!(ax4, network.time - 10, network.time)
        end
        
        # Check boundaries
        boundaries = compute_boundaries(network)
        if !isempty(boundaries)
            # Highlight boundary neurons
            # (Could add visual indicator here)
        end
    end
    
    # Run simulation
    println("Starting neural network visualization...")
    println("Close window to stop")
    
    # Use record to create animation
    record(update_frame, fig, "neural_network_live.mp4", 1:1000) do frame
        update_frame()
        sleep(0.05)
    end
    
    return fig, network
end

# ============================================================================
# 7. INTERACTIVE VERSION (Real-time)
# ============================================================================

function run_interactive_ui()
    println("=" ^ 70)
    println("Neural Network Interactive Monitor")
    println("=" ^ 70)
    println("\nCreating network...")
    
    network = create_network(50)
    
    println("Network created: $(network.n_neurons) neurons")
    println("\nStarting visualization...")
    println("Close window to stop simulation")
    
    # Create figure
    fig = Figure(resolution=(1600, 1000), title="Neural Network Live Monitor")
    
    # Main view
    ax = Axis(fig[1, 1], 
        title="Neural Network (Hyperbolic Space - Poincaré Disk)",
        aspect=DataAspect()
    )
    xlims!(ax, -1.1, 1.1)
    ylims!(ax, -1.1, 1.1)
    
    # Draw unit circle
    circle = Circle(Point2f(0, 0), 1.0f0)
    poly!(ax, circle, color=(:black, 0.1), strokewidth=2, strokecolor=:gray)
    
    # Metrics
    ax_sync = Axis(fig[1, 2], title="Synchronization r(t)", height=200)
    ax_act = Axis(fig[2, 2], title="Avg Activation", height=200)
    ax_conn = Axis(fig[3, 2], title="Avg Connection Strength", height=200)
    
    # Data storage
    times = Float64[]
    syncs = Float64[]
    acts = Float64[]
    conns = Float64[]
    
    # Initial plot
    pos = [Point2f(n.position...) for n in network.neurons]
    acts_current = [n.activation for n in network.neurons]
    sc = scatter!(ax, pos, color=acts_current, colormap=:viridis, 
                 markersize=20, strokewidth=1, strokecolor=:white)
    
    # Connections
    conn_lines = Point2f[]
    for i in 1:network.n_neurons
        for j in 1:network.n_neurons
            if network.connections[i, j] > 0.05
                push!(conn_lines, Point2f(network.neurons[i].position...))
                push!(conn_lines, Point2f(network.neurons[j].position...))
            end
        end
    end
    if !isempty(conn_lines)
        linesegs!(ax, conn_lines, color=(:gray, 0.3), linewidth=0.5)
    end
    
    # Time series
    line_sync = lines!(ax_sync, times, syncs, color=:blue, linewidth=2)[1]
    line_act = lines!(ax_act, times, acts, color=:green, linewidth=2)[1]
    line_conn = lines!(ax_conn, times, conns, color=:orange, linewidth=2)[1]
    
    xlims!(ax_sync, 0, 10)
    xlims!(ax_act, 0, 10)
    xlims!(ax_conn, 0, 10)
    ylims!(ax_sync, 0, 1)
    ylims!(ax_act, 0, 1)
    
    # Text info
    info_text = "Time: 0.0\nSync: 0.0\nActivation: 0.0\nConnections: 0.0"
    text!(fig[4, 2], info_text, fontsize=14, align=(:left, :top))
    
    display(fig)
    
    # Simulation loop
    dt = 0.05
    frame_count = 0
    
    println("\nSimulation running... (Close window to stop)")
    
    while isopen(fig.scene)
        # Update network
        kuramoto_update!(network, dt)
        wave_propagation!(network, dt)
        hebbian_update!(network, 0.01)
        
        network.time += dt
        frame_count += 1
        
        # Update every 5 frames
        if frame_count % 5 == 0
            # Update scatter plot
            acts_current = [n.activation for n in network.neurons]
            sc.color[] = acts_current
            
            # Update metrics
            r, _ = order_parameter(network)
            avg_act = mean([n.activation for n in network.neurons])
            avg_conn = mean(network.connections)
            
            push!(times, network.time)
            push!(syncs, r)
            push!(acts, avg_act)
            push!(conns, avg_conn)
            
            # Keep last 200 points
            if length(times) > 200
                times = times[end-199:end]
                syncs = syncs[end-199:end]
                acts = acts[end-199:end]
                conns = conns[end-199:end]
            end
            
            # Update plots
            line_sync.input_args[1][] = (times, syncs)
            line_act.input_args[1][] = (times, acts)
            line_conn.input_args[1][] = (times, conns)
            
            # Update axis limits
            if network.time > 10
                xlims!(ax_sync, network.time - 10, network.time)
                xlims!(ax_act, network.time - 10, network.time)
                xlims!(ax_conn, network.time - 10, network.time)
            end
            
            # Update text
            info_text = @sprintf("""
            Time: %.2f
            Synchronization: %.3f
            Avg Activation: %.3f
            Avg Connection: %.3f
            Neurons: %d
            Active Connections: %d
            """, 
            network.time, r, avg_act, avg_conn,
            network.n_neurons, 
            sum(network.connections .> 0.05))
            
            # Force redraw
            notify(sc.color)
        end
        
        sleep(0.01)
    end
    
    println("\nSimulation stopped.")
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    Random.seed!(42)
    run_interactive_ui()
end

