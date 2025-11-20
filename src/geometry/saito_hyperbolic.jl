"""
SAITO-Constrained Hyperbolic Geometry Module

This module implements numerically stable hyperbolic operations specifically optimized
for the SAITO-constrained HNN with fixed curvature c_target ≈ 11.7.
"""
module SaitoHyperbolic

using LinearAlgebra
using StaticArrays
using ForwardDiff
using ..Hyperbolic  # Reuse base hyperbolic operations

# SAITO-specific constants
const C_TARGET = 11.7  # Fixed curvature from SAITO/α
const EPSILON = 1e-8   # Numerical stability constant
const D_MAX = 1.0      # Maximum allowed distance (dynamic constraint)
const MAX_DIM = 20     # Maximum embedding dimension (D ≤ 20)

"""
    stabilized_distance(u::AbstractVector, v::AbstractVector, c=C_TARGET)

Compute the stabilized hyperbolic distance between points u and v with curvature c.
Implements the stabilized version of the distance function with numerical safeguards.
"""
function stabilized_distance(u::AbstractVector{T}, v::AbstractVector{T}, c::T=C_TARGET) where {T<:AbstractFloat}
    # Stabilize the difference vector
    diff = u .- v
    norm_diff_sq = max(0, norm(diff)^2)  # Ensure non-negative
    
    # Stabilize the norms
    norm_u_sq = max(0, norm(u)^2)
    norm_v_sq = max(0, norm(v)^2)
    
    # Calculate the argument with stabilization
    denominator = (1 - c * norm_u_sq) * (1 - c * norm_v_sq)
    denominator = max(denominator, EPSILON)  # Avoid division by zero
    
    gamma = 1 + (2 * c * norm_diff_sq) / denominator
    gamma = max(1.0 + EPSILON, gamma)  # Ensure gamma > 1 for acosh
    
    # Calculate final distance with curvature scaling
    distance = acosh(gamma) / sqrt(c)
    return min(distance, D_MAX)  # Enforce dynamic constraint
end

"""
    stabilized_exp_map(x::AbstractVector, v::AbstractVector, c=C_TARGET)

Stabilized exponential map from tangent space at x with curvature c.
"""
function stabilized_exp_map(x::AbstractVector{T}, v::AbstractVector{T}, c::T=C_TARGET) where {T<:AbstractFloat}
    norm_v = norm(v)
    if norm_v < EPSILON
        return copy(x)
    end
    
    sqrt_c = sqrt(c)
    lambda_x = 2 / max(1 - c * norm(x)^2, EPSILON)  # Stabilized conformal factor
    v_norm = norm_v * sqrt_c
    
    # Use Taylor expansion near zero for numerical stability
    if v_norm < 1e-4
        term1 = x .* (1 + v_norm^2/2)
        term2 = v .* (1 + v_norm^2/6) / sqrt_c
    else
        term1 = cosh(v_norm) * x
        term2 = (sinh(v_norm) / v_norm) * v / sqrt_c
    end
    
    result = term1 .+ term2
    result .*= tanh(lambda_x * norm_v / 2) / (norm(result) + EPSILON)
    
    # Project back to the Poincaré ball if needed
    norm_result = norm(result)
    if norm_result >= 1.0 - EPSILON
        result .*= (1.0 - EPSILON) / norm_result
    end
    
    return result
end

"""
    stabilized_log_map(x::AbstractVector, y::AbstractVector, c=C_TARGET)

Stabilized logarithmic map from x to y with curvature c.
"""
function stabilized_log_map(x::AbstractVector{T}, y::AbstractVector{T}, c::T=C_TARGET) where {T<:AbstractFloat}
    # Möbius subtraction with stabilization
    diff = (y .- x) ./ (1 .- c .* dot(x, y) .+ EPSILON)
    diff_norm = norm(diff)
    
    if diff_norm < EPSILON
        return zero(x)
    end
    
    # Calculate distance and direction with stabilization
    d = stabilized_distance(x, y, c)
    direction = diff ./ (diff_norm + EPSILON)
    
    # Scale by the stabilized distance
    return direction .* (2 / (sqrt(c) * (1 - c * norm(x)^2 + EPSILON))) .* atanh(sqrt(c) * d)
end

"""
    stabilized_mobius_add(x::AbstractVector, y::AbstractVector, c=C_TARGET)

Stabilized Möbius addition in the Poincaré ball model.
"""
function stabilized_mobius_add(x::AbstractVector{T}, y::AbstractVector{T}, c::T=C_TARGET) where {T<:AbstractFloat}
    x_dot_y = dot(x, y)
    x_norm_sq = norm(x)^2
    y_norm_sq = norm(y)^2
    
    denominator = 1 + 2 * c * x_dot_y + c^2 * x_norm_sq * y_norm_sq
    denominator = max(denominator, EPSILON)  # Prevent division by zero
    
    term1 = (1 + 2 * c * x_dot_y + c * y_norm_sq) .* x
    term2 = (1 - c * x_norm_sq) .* y
    
    result = (term1 .+ term2) ./ denominator
    
    # Ensure result stays within the Poincaré ball
    norm_result = norm(result)
    if norm_result >= 1.0 - EPSILON
        result .*= (1.0 - EPSILON) / norm_result
    end
    
    return result
end

"""
    hyperbolic_hebbian_update(W_old::AbstractMatrix, A_i::AbstractVector, A_j::AbstractVector, 
                            R::Float64, η::Float64, c=C_TARGET)

Stabilized Hyperbolic Hebbian Update rule with economic constraints.
"""
function hyperbolic_hebbian_update(W_old::AbstractMatrix{T}, 
                                 A_i::AbstractVector{T}, 
                                 A_j::AbstractVector{T},
                                 R::T,  # Utility function value
                                 η::T;  # Learning rate
                                 c::T=C_TARGET) where {T<:AbstractFloat}
    # Project to tangent space at W_old
    log_W_Ai = stabilized_log_map(W_old, A_i, c)
    log_W_Aj = stabilized_log_map(W_old, A_j, c)
    
    # Calculate correlation in tangent space
    ΔW_tangent = η * R * (log_W_Ai * log_W_Aj')
    
    # Project back to manifold using exponential map
    W_new = similar(W_old)
    for i in 1:size(W_old, 2)
        W_new[:, i] = stabilized_exp_map(W_old[:, i], ΔW_tangent[:, i], c)
    end
    
    return W_new
end

export stabilized_distance, stabilized_exp_map, stabilized_log_map, 
       stabilized_mobius_add, hyperbolic_hebbian_update,
       C_TARGET, EPSILON, D_MAX, MAX_DIM

end # module SaitoHyperbolic
