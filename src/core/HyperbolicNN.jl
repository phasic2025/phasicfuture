"""
HyperbolicNN module implements the core neural network operations
for the SAITO-Constrained Hyperbolic Neural Network.
"""
module HyperbolicNN

using LinearAlgebra
using ..HyperbolicSpace

"""
    HyperbolicLayer
A single layer in the Hyperbolic Neural Network.
"""
mutable struct HyperbolicLayer
    weights::Matrix{Float64}  # Weight matrix
    bias::Vector{Float64}     # Bias vector
    c::Float64               # Curvature parameter
    
    function HyperbolicLayer(input_dim::Int, output_dim::Int, c::Float64=HyperbolicSpace.C_TARGET)
        # Initialize weights using Xavier/Glorot initialization
        # Scaled by 1/sqrt(c) for hyperbolic space
        scale = 1.0 / sqrt(c * input_dim)
        weights = randn(Float64, output_dim, input_dim) * scale
        bias = zeros(Float64, output_dim)
        new(weights, bias, c)
    end
end

"""
    HyperbolicNN
A complete Hyperbolic Neural Network composed of multiple HyperbolicLayers.
"""
struct HyperbolicNN
    layers::Vector{HypebolicLayer}
    c::Float64  # Global curvature parameter
    
    function HyperbolicNN(layer_dims::Vector{Int}; c::Float64=HyperbolicSpace.C_TARGET)
        @assert length(layer_dims) >= 2 "At least input and output dimensions required"
        layers = [HyperbolicLayer(layer_dims[i], layer_dims[i+1], c) for i in 1:length(layer_dims)-1]
        new(layers, c)
    end
end

"""
    forward(layer::HyperbolicLayer, x)
Forward pass through a single hyperbolic layer.
"""
function forward(layer::HyperbolicLayer, x::AbstractVector{Float64})
    # Apply linear transformation in the tangent space at origin
    z = layer.weights * x .+ layer.bias
    
    # Project back to hyperbolic space using exponential map
    return HyperbolicSpace.exp_map(zeros(size(x)), z)
end

"""
    forward(model::HyperbolicNN, x)
Forward pass through the entire network.
"""
function forward(model::HyperbolicNN, x::AbstractVector{Float64})
    h = copy(x)
    for layer in model.layers
        h = forward(layer, h)
    end
    return h
end

"""
    hyperbolic_hebbian_update!(layer::HyperbolicLayer, x, y, learning_rate)
Update the weights using the Hyperbolic Hebbian learning rule.
"""
function hyperbolic_hebbian_update!(layer::HyperbolicLayer, 
                                   x::AbstractVector{Float64}, 
                                   y::AbstractVector{Float64}, 
                                   learning_rate::Float64)
    # Move to tangent space at origin
    x_tangent = HyperbolicSpace.log_map(zeros(size(x)), x)
    y_tangent = HyperbolicSpace.log_map(zeros(size(y)), y)
    
    # Compute the outer product in tangent space
    delta_W = y_tangent * x_tangent'
    
    # Update weights with momentum and learning rate
    layer.weights .+= learning_rate .* delta_W
    
    # Project weights back to the manifold if needed
    # (This is a simplified version - in practice, you might need more sophisticated projection)
    for i in 1:size(layer.weights, 1)
        norm_w = norm(layer.weights[i,:])
        if norm_w > 1.0
            layer.weights[i,:] ./= (norm_w + HyperbolicSpace.EPSILON)
        end
    end
    
    return layer
end

"""
    train!(model::HyperbolicNN, x, y, learning_rate=0.01)
Train the model on a single example using the Hyperbolic Hebbian rule.
"""
function train!(model::HyperbolicNN, x::AbstractVector{Float64}, 
               y::AbstractVector{Float64}, learning_rate::Float64=0.01)
    # Forward pass
    activations = Vector{Vector{Float64}}(undef, length(model.layers) + 1)
    activations[1] = copy(x)
    
    for (i, layer) in enumerate(model.layers)
        activations[i+1] = forward(layer, activations[i])
    end
    
    # Backward pass with Hebbian updates
    for i in length(model.layers):-1:1
        if i == length(model.layers)
            # Output layer uses target for update
            hyperbolic_hebbian_update!(model.layers[i], activations[i], y, learning_rate)
        else
            # Hidden layers use next layer's activations
            hyperbolic_hebbian_update!(model.layers[i], activations[i], activations[i+2], learning_rate)
        end
    end
    
    return activations[end]
end

export HyperbolicLayer, HyperbolicNN, forward, train!, hyperbolic_hebbian_update!

end # module HyperbolicNN
