"""
HyperbolicNN.jl

Implements hyperbolic neural network layers and training routines for the TFTS architecture.
"""
module HyperbolicNN

using ..HyperbolicGeometry
using ..SAITOCore
using ..Tokenomics
using ..Topology
using Flux
using Zygote
using LinearAlgebra
using Statistics
using Random

# Re-exports
export HyperbolicLayer, HyperbolicMLP, train_hnn!, predict, 
       hyperbolic_manifold_regularizer, HyperbolicAttention

"""
    HyperbolicLayer(in_dim::Int, out_dim::Int; 
                  curvature=11.7, 
                  use_bias=true, 
                  init_scale=1.0)

A hyperbolic neural network layer that operates in the Poincaré ball model.
"""
struct HyperbolicLayer
    weight::Matrix{Float64}
    bias::Vector{Float64}
    curvature::Float64
    use_bias::Bool
    
    function HyperbolicLayer(in_dim::Int, out_dim::Int; 
                           curvature=11.7, 
                           use_bias=true, 
                           init_scale=1.0)
        # Initialize weights using a normal distribution scaled by 1/sqrt(in_dim)
        scale = init_scale / sqrt(in_dim)
        weight = randn(Float64, (out_dim, in_dim)) .* scale
        
        # Initialize bias if used
        bias = use_bias ? randn(Float64, out_dim) .* scale : zeros(Float64, out_dim)
        
        new(weight, bias, curvature, use_bias)
    end
end

Flux.@functor HyperbolicLayer

""
    (layer::HyperbolicLayer)(x::HyperbolicPoint)

Forward pass of the hyperbolic layer.
"""
function (layer::HyperbolicLayer)(x::HyperbolicPoint)
    # Project to tangent space at origin
    x_tangent = log_map(zero(x), x)
    
    # Linear transformation in the tangent space
    y_tangent = layer.weight * x_tangent.coords
    
    if layer.use_bias
        y_tangent .+= layer.bias
    end
    
    # Project back to hyperbolic space
    y = exp_map(zero(x), HyperbolicVector(y_tangent))
    
    return y
end

"""
    HyperbolicMLP(layer_dims::Vector{Int}; 
                 curvature=11.7, 
                 use_bias=true, 
                 init_scale=1.0)

A multi-layer perceptron with hyperbolic hidden layers.
"""
struct HyperbolicMLP
    layers::Vector{HyperbolicLayer}
    curvature::Float64
    
    function HyperbolicMLP(layer_dims::Vector{Int}; 
                          curvature=11.7, 
                          use_bias=true, 
                          init_scale=1.0)
        @assert length(layer_dims) >= 2 "At least input and output dimensions required"
        
        layers = HyperbolicLayer[]
        for i in 1:length(layer_dims)-1
            push!(layers, HyperbolicLayer(
                layer_dims[i], 
                layer_dims[i+1], 
                curvature=curvature, 
                use_bias=use_bias, 
                init_scale=init_scale
            ))
        end
        
        new(layers, curvature)
    end
end

Flux.@functor HyperbolicMLP

function (model::HyperbolicMLP)(x::HyperbolicPoint)
    for layer in model.layers[1:end-1]
        x = layer(x)
        # Apply hyperbolic ReLU (project to tangent space, apply ReLU, project back)
        x_tangent = log_map(zero(x), x)
        x_tangent = HyperbolicVector(relu.(x_tangent.coords))
        x = exp_map(zero(x), x_tangent)
    end
    
    # Final layer (no activation)
    return model.layers[end](x)
end

"""
    HyperbolicAttention(dim::Int, num_heads::Int; 
                       curvature=11.7, 
                       dropout=0.1)

Hyperbolic multi-head attention layer.
"""
struct HyperbolicAttention
    q_proj::HyperbolicLayer
    k_proj::HyperbolicLayer
    v_proj::HyperbolicLayer
    out_proj::HyperbolicLayer
    num_heads::Int
    head_dim::Int
    dropout::Float64
    curvature::Float64
    
    function HyperbolicAttention(dim::Int, num_heads::Int; 
                               curvature=11.7, 
                               dropout=0.1)
        @assert dim % num_heads == 0 "dim must be divisible by num_heads"
        head_dim = div(dim, num_heads)
        
        new(
            HyperbolicLayer(dim, dim, curvature=curvature),  # Q projection
            HyperbolicLayer(dim, dim, curvature=curvature),  # K projection
            HyperbolicLayer(dim, dim, curvature=curvature),  # V projection
            HyperbolicLayer(dim, dim, curvature=curvature),  # Output projection
            num_heads,
            head_dim,
            dropout,
            curvature
        )
    end
end

Flux.@functor HyperbolicAttention

function (m::HyperbolicAttention)(q::HyperbolicPoint, k::HyperbolicPoint, v::HyperbolicPoint)
    batch_size = 1  # For simplicity, extend for batches if needed
    
    # Project queries, keys, and values
    q_proj = m.q_proj(q)
    k_proj = m.k_proj(k)
    v_proj = m.v_proj(v)
    
    # Reshape for multi-head attention
    q_heads = reshape(q_proj, (m.head_dim, m.num_heads, batch_size))
    k_heads = reshape(k_proj, (m.head_dim, m.num_heads, batch_size))
    v_heads = reshape(v_proj, (m.head_dim, m.num_heads, batch_size))
    
    # Scaled dot-product attention in the tangent space
    # (This is a simplified version - in practice, we'd need to handle the hyperbolic geometry carefully)
    scale = 1.0 / sqrt(m.head_dim)
    scores = scale * sum(q_heads .* k_heads, dims=1)
    attn_weights = softmax(scores, dims=1)
    
    # Apply attention to values
    attn_output = sum(attn_weights .* v_heads, dims=2)
    
    # Concatenate heads and project back
    attn_output = reshape(attn_output, (m.head_dim * m.num_heads, batch_size))
    output = m.out_proj(HyperbolicPoint(attn_output))
    
    return output
end

"""
    hyperbolic_manifold_regularizer(model::Union{HyperbolicMLP,HyperbolicLayer}, 
                                  x::HyperbolicPoint; 
                                  λ=0.01)

Regularization term to keep points on the hyperbolic manifold.
"""
function hyperbolic_manifold_regularizer(model::Union{HyperbolicMLP,HyperbolicLayer}, 
                                        x::HyperbolicPoint; 
                                        λ=0.01)
    # Project the output back to the hyperbolic space and measure the distance
    y = model(x)
    y_proj = exp_map(zero(x), log_map(zero(x), y))
    
    # Calculate the distance between the output and its projection
    dist = hyperbolic_distance(y, y_proj)
    
    return λ * dist
end

"""
    train_hnn!(model::Union{HyperbolicMLP,HyperbolicAttention}, 
              data::AbstractArray, 
              opt::Flux.Optimiser, 
              loss_fn::Function;
              epochs::Int=10, 
              batch_size::Int=32, 
              λ_reg=0.01)

Train a hyperbolic neural network.
"""
function train_hnn!(model::Union{HyperbolicMLP,HyperbolicAttention}, 
                   data::AbstractArray, 
                   opt::Flux.Optimiser, 
                   loss_fn::Function;
                   epochs::Int=10, 
                   batch_size::Int=32, 
                   λ_reg=0.01)
    
    # Convert data to hyperbolic points
    hyperbolic_data = [HyperbolicPoint(x) for x in eachcol(data)]
    
    # Training loop
    for epoch in 1:epochs
        # Shuffle data
        idxs = shuffle(1:length(hyperbolic_data))
        
        for i in 1:batch_size:length(hyperbolic_data)-batch_size+1
            batch = hyperbolic_data[idxs[i:i+batch_size-1]]
            
            # Compute loss and gradients
            loss, grads = Flux.withgradient(model) do m
                # Compute prediction loss
                preds = [m(x) for x in batch[1:end-1]]
                targets = batch[2:end]
                
                # Compute manifold regularization
                reg = sum(hyperbolic_manifold_regularizer(m, x) for x in batch)
                
                # Total loss
                loss_fn(preds, targets) + λ_reg * reg
            end
            
            # Update parameters
            Flux.update!(opt, Flux.params(model), grads[1])
        end
        
        @info "Epoch $epoch: Loss = $loss"
    end
    
    return model
end

"""
    predict(model::Union{HyperbolicMLP,HyperbolicAttention}, 
           x::HyperbolicPoint)

Make a prediction using the hyperbolic neural network.
"""
predict(model::Union{HyperbolicMLP,HyperbolicAttention}, x::HyperbolicPoint) = model(x)

"""
    hyperbolic_contrastive_loss(z1::HyperbolicPoint, 
                              z2::HyperbolicPoint, 
                              temperature=0.1)

Contrastive loss in hyperbolic space.
"""
function hyperbolic_contrastive_loss(z1::HyperbolicPoint, 
                                   z2::HyperbolicPoint, 
                                   temperature=0.1)
    # Negative squared hyperbolic distance
    sim = -hyperbolic_distance(z1, z2)^2 / temperature
    return -log(exp(sim) / (exp(sim) + 1.0))  # Simplified contrastive loss
end

end # module HyperbolicNN
