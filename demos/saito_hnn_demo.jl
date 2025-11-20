using SaitoHNN
using LinearAlgebra
using Plots

# Set up the visualization
function plot_poincare_disk(points; labels=nothing, title="Poincaré Disk")
    # Create the unit circle
    θ = range(0, 2π, length=100)
    circle_x = cos.(θ)
    circle_y = sin.(θ)
    
    # Plot the boundary
    plot(circle_x, circle_y, color=:black, label="", aspect_ratio=:equal, legend=:bottomright)
    
    # Plot the points
    if labels === nothing
        scatter!([p[1] for p in points], [p[2] for p in points], 
                color=:blue, label="Points", legend=:bottomright)
    else
        for (i, (point, label)) in enumerate(zip(points, labels))
            scatter!([point[1]], [point[2]], 
                    color=i, label=label, marker=(10, 3, :circle), 
                    legend=:bottomright)
        end
    end
    
    # Set plot limits and title
    xlims!(-1.1, 1.1)
    ylims!(-1.1, 1.1)
    title!(title)
    
    return plot!()
end

# Demo 1: Basic hyperbolic operations
function demo_hyperbolic_operations()
    println("=== Demo 1: Basic Hyperbolic Operations ===")
    
    # Create some points in the Poincaré disk
    origin = HyperbolicPoint([0.0, 0.0])
    p1 = HyperbolicPoint([0.3, 0.0])
    p2 = HyperbolicPoint([0.0, 0.4])
    
    # Calculate distances
    d1 = hyperbolic_distance(origin, p1)
    d2 = hyperbolic_distance(origin, p2)
    d12 = hyperbolic_distance(p1, p2)
    
    println("Distance from origin to p1: ", d1)
    println("Distance from origin to p2: ", d2)
    println("Distance between p1 and p2: ", d12)
    
    # Visualize
    points = [origin.coords, p1.coords, p2.coords]
    labels = ["Origin", "Point 1", "Point 2"]
    p = plot_poincare_disk(points, labels=labels, 
                          title="Hyperbolic Points in Poincaré Disk")
    
    return p
end

# Demo 2: Hyperbolic neural network layer
function demo_hyperbolic_layer()
    println("\n=== Demo 2: Hyperbolic Neural Network Layer ===")
    
    # Create a hyperbolic layer (2D input, 3D output)
    layer = HyperbolicLayer(2, 3)
    
    # Create some input points
    inputs = [
        HyperbolicPoint([0.1, 0.2]),
        HyperbolicPoint([-0.3, 0.1]),
        HyperbolicPoint([0.2, -0.2])
    ]
    
    # Process each input through the layer
    println("Processing inputs through the hyperbolic layer:")
    for (i, input) in enumerate(inputs)
        output = forward(layer, input)
        println("Input ", i, " → Output norm: ", norm(output.coords))
        @assert check_geometric_constraint(output) "Output violates geometric constraint!"
    end
    
    # Visualize the transformation
    input_points = [p.coords for p in inputs]
    output_points = [forward(layer, p).coords for p in inputs]
    
    p1 = plot_poincare_disk(input_points, labels=["Point 1", "Point 2", "Point 3"], 
                           title="Input Points")
    p2 = plot_poincare_disk(output_points, labels=["Output 1", "Output 2", "Output 3"], 
                           title="Output Points after Hyperbolic Layer")
    
    return plot(p1, p2, layout=(1,2), size=(1000, 500))
end

# Demo 3: Hyperbolic Hebbian Learning
function demo_hebbian_learning()
    println("\n=== Demo 3: Hyperbolic Hebbian Learning ===")
    
    # Create a simple network with one input and one output neuron
    layer = HyperbolicLayer(2, 1)
    
    # Training data (input-output pairs)
    training_data = [
        (HyperbolicPoint([0.1, 0.2]), HyperbolicPoint([0.3])),
        (HyperbolicPoint([-0.1, 0.3]), HyperbolicPoint([0.4])),
        (HyperbolicPoint([0.2, -0.1]), HyperbolicPoint([0.25]))
    ]
    
    # Training loop
    learning_rate = 0.1
    epochs = 10
    
    println("Initial weights: ", [w.value for w in layer.weights])
    
    for epoch in 1:epochs
        total_error = 0.0
        
        for (input, target) in training_data
            # Forward pass
            output = forward(layer, input)
            
            # Calculate error (in tangent space)
            error = hyperbolic_distance(output, target)
            total_error += error
            
            # Update weights using hyperbolic Hebbian rule
            for i in 1:length(layer.weights)
                layer.weights[i] = hyperbolic_hebbian_update(
                    layer.weights[i], input, output, learning_rate)
            end
        end
        
        println("Epoch ", epoch, " - Average error: ", total_error / length(training_data))
    end
    
    println("Final weights: ", [w.value for w in layer.weights])
    
    # Test the trained network
    println("\nTesting the trained network:")
    for (i, (input, target)) in enumerate(training_data)
        output = forward(layer, input)
        error = hyperbolic_distance(output, target)
        println("Test ", i, " - Error: ", error)
    end
    
    return nothing
end

# Run all demos
println("Starting SAITO-HNN Demo"
println("======================")

# Run Demo 1
p1 = demo_hyperbolic_operations()
display(p1)

# Run Demo 2
p2 = demo_hyperbolic_layer()
display(p2)

# Run Demo 3
demo_hebbian_learning()

println("\nDemo completed successfully!")
