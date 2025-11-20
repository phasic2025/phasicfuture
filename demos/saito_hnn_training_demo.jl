#!/usr/bin/env julia
# SAITO-Constrained HNN Training Demo
# This script demonstrates training a SAITO-constrained Hyperbolic Neural Network
# on a simple synthetic dataset.

using Pkg
Pkg.activate("..")

using Random
using Flux
using Plots
using Statistics
using BSON
using ProgressMeter
using LinearAlgebra

# Import our SAITO modules
include("../src/geometry/SaitoHyperbolic.jl")
include("../src/network/SaitoNetwork.jl")
include("../src/models/SaitoHNN.jl")

using .SaitoHyperbolic
using .NetworkBlockchain
using .SaitoHNN

# Set random seed for reproducibility
Random.seed!(42)

function generate_hyperbolic_data(n_samples=1000, n_features=10, n_classes=3)
    """Generate synthetic hyperbolic data."""
    # Generate random points in hyperbolic space
    X = randn(Float32, n_features, n_samples)
    
    # Normalize to have norm < 1 (Poincaré ball constraint)
    X = X ./ (2f0 .* maximum(abs, X, dims=1))
    
    # Create simple class separation
    y = zeros(Float32, n_classes, n_samples)
    for i in 1:n_samples
        class = mod1(i, n_classes)
        y[class, i] = 1.0f0
    end
    
    return X, y
end

function create_data_loaders(X, y; batch_size=32, train_ratio=0.8)
    """Split data into training and validation sets and create data loaders."""
    n_samples = size(X, 2)
    n_train = Int(floor(n_samples * train_ratio))
    
    # Shuffle data
    idx = shuffle(1:n_samples)
    train_idx = idx[1:n_train]
    val_idx = idx[n_train+1:end]
    
    # Create data loaders
    train_loader = Flux.DataLoader((X[:, train_idx], y[:, train_idx]), 
                                 batchsize=batch_size, shuffle=true)
    val_loader = Flux.DataLoader((X[:, val_idx], y[:, val_idx]), 
                               batchsize=batch_size, shuffle=false)
    
    return train_loader, val_loader
end

function train_model(model, train_loader, val_loader, optimizer, n_epochs=10)
    """Train the model and return training history."""
    train_losses = Float32[]
    val_losses = Float32[]
    
    # Training loop
    @showprogress for epoch in 1:n_epochs
        # Training
        train_loss = 0.0f0
        num_batches = 0
        
        for (x, y) in train_loader
            # Move data to device (CPU for this demo)
            x, y = cpu(x), cpu(y)
            
            # Compute loss and gradients
            loss, grads = Flux.withgradient(model) do m
                y_pred = m(x)
                SaitoHNN.hyperbolic_mse_loss(y_pred, y)
            end
            
            # Update parameters
            Flux.update!(optimizer, model, grads[1])
            
            # Track metrics
            train_loss += loss
            num_batches += 1
        end
        
        # Average training loss for this epoch
        push!(train_losses, train_loss / num_batches)
        
        # Validation
        val_loss = 0.0f0
        num_batches = 0
        
        for (x, y) in val_loader
            x, y = cpu(x), cpu(y)
            y_pred = model(x)
            val_loss += SaitoHNN.hyperbolic_mse_loss(y_pred, y)
            num_batches += 1
        end
        
        push!(val_losses, val_loss / num_batches)
        
        @info "Epoch $epoch: Train Loss = $(train_losses[end]), Val Loss = $(val_losses[end])"
    end
    
    return train_losses, val_losses
end

function plot_training_history(train_losses, val_losses; save_path=nothing)
    """Plot training and validation loss over epochs."""
    p = plot(1:length(train_losses), train_losses, 
             label="Training Loss", 
             xlabel="Epoch", 
             ylabel="Loss", 
             title="Training Progress",
             legend=:topright)
    
    plot!(p, 1:length(val_losses), val_losses, label="Validation Loss")
    
    if save_path !== nothing
        savefig(p, save_path)
        @info "Training plot saved to $save_path"
    end
    
    return p
end

function main()
    # Generate synthetic data
    @info "Generating synthetic data..."
    X, y = generate_hyperbolic_data(1000, 10, 3)
    train_loader, val_loader = create_data_loaders(X, y, batch_size=32)
    
    # Create model
    @info "Creating SAITO-constrained HNN..."
    model = SaitoHNN.SaitoHNN([10, 20, 10, 3], [relu, relu, identity])
    
    # Set up optimizer
    optimizer = Flux.setup(Adam(0.01), model)
    
    # Train model
    @info "Starting training..."
    train_losses, val_losses = train_model(model, train_loader, val_loader, optimizer, 20)
    
    # Plot training history
    plot_training_history(train_losses, val_losses, 
                         save_path="saito_hnn_training.png")
    
    # Save the trained model
    model_path = "saito_hnn_model.bson"
    SaitoHNN.save_model(model, model_path)
    @info "Trained model saved to $model_path"
    
    # Demonstrate loading the model
    loaded_model = SaitoHNN.load_model(model_path, model)
    @info "Model loaded successfully. Testing inference..."
    
    # Test inference
    test_sample = X[:, 1:1]  # Get first sample
    prediction = loaded_model(test_sample)
    @info "Test sample prediction: $(argmax(prediction, dims=1)) (true class: $(argmax(y[:,1:1], dims=1)))"
    
    # In a real application, we would connect to the network here:
    # SaitoHNN.connect_to_network!(model, 8000)
    
    return model
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
