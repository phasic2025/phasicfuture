"""
HyperbolicSpace module implements the core hyperbolic geometry operations
with numerical stability for the SAITO-Constrained HNN.
"""
module HyperbolicSpace

using LinearAlgebra
using StaticArrays: SVector, SMatrix

# SAITO constant - fixed curvature (c_target ≈ 11.7)
const C_TARGET = 11.7
const EPSILON = 1e-8  # Numerical stability constant

"""
    HyperbolicPoint
Abstract type representing a point in hyperbolic space.
"""
abstract type AbstractHyperbolicPoint end

"""
    distance(u, v)
Compute the hyperbolic distance between points u and v in the Poincaré ball model.
"""
function distance(u::AbstractVector{T}, v::AbstractVector{T}) where {T<:Real}
    # Add numerical stability terms to prevent NaNs
    norm_u_sq = max(EPSILON, norm(u)^2)
    norm_v_sq = max(EPSILON, norm(v)^2)
    
    # Calculate the Möbius addition term with numerical stability
    mobius_term = norm(u - v)^2 / ((1 - norm_u_sq) * (1 - norm_v_sq) + EPSILON)
    
    # Calculate the distance with numerical stability
    dist = acosh(1 + 2 * mobius_term) / sqrt(C_TARGET)
    
    # Ensure the result is finite and positive
    isfinite(dist) ? max(EPSILON, dist) : zero(T)
end

"""
    exp_map(x, v)
Exponential map at point x in the direction of tangent vector v.
"""
function exp_map(x::AbstractVector{T}, v::AbstractVector{T}) where {T<:Real}
    norm_x = norm(x)
    norm_v = norm(v)
    
    # Handle the zero vector case
    if norm_v < EPSILON
        return copy(x)
    end
    
    # Calculate the exponential map with numerical stability
    λ = 2 / (1 - norm_x^2 + EPSILON)
    c = sqrt(C_TARGET)
    v_hat = v / norm_v
    
    # Use the exponential map formula for the Poincaré ball
    term = tanh(c * λ * norm_v / 2)
    
    # Möbius addition in the Poincaré ball
    numerator = (1 + 2 * term * dot(x, v_hat) / (c * λ) + term^2) * x +
                (1 - norm_x^2) * term * v_hat / (c * λ)
    denominator = 1 + 2 * term * dot(x, v_hat) / (c * λ) + term^2 * norm_x^2
    
    return numerator / (denominator + EPSILON)
end

"""
    log_map(x, y)
Logarithmic map (inverse of exp_map) from point x to point y.
"""
function log_map(x::AbstractVector{T}, y::AbstractVector{T}) where {T<:Real}
    norm_x = norm(x)
    norm_y = norm(y)
    
    # Handle the case when x and y are the same point
    if norm(x - y) < EPSILON
        return zero(x)
    end
    
    # Calculate the logarithmic map with numerical stability
    λ = 2 / (1 - norm_x^2 + EPSILON)
    c = sqrt(C_TARGET)
    
    # Möbius subtraction
    y_minus_x = (1 - norm_x^2) * (y - x) / (1 - 2 * dot(x, y) + norm_x^2 * norm_y^2 + EPSILON)
    
    # Scale by the inverse of the metric tensor
    return (2 / (c * λ)) * atanh(c * norm(y_minus_x)) * y_minus_x / (norm(y_minus_x) + EPSILON)
end

"""
    mobius_add(u, v)
Möbius addition in the Poincaré ball model.
"""
function mobius_add(u::AbstractVector{T}, v::AbstractVector{T}) where {T<:Real}
    norm_u_sq = norm(u)^2
    norm_v_sq = norm(v)^2
    dot_uv = dot(u, v)
    
    denominator = 1 + 2 * C_TARGET * dot_uv + C_TARGET^2 * norm_u_sq * norm_v_sq + EPSILON
    
    return ((1 + 2 * C_TARGET * dot_uv + C_TARGET * norm_v_sq) * u +
            (1 - C_TARGET * norm_u_sq) * v) / denominator
end

"""
    mobius_mul(r, x)
Möbius scalar multiplication in the Poincaré ball model.
"""
function mobius_mul(r::T, x::AbstractVector{T}) where {T<:Real}
    norm_x = norm(x)
    if norm_x < EPSILON
        return zero(x)
    end
    
    return tanh(r * atanh(sqrt(C_TARGET) * norm_x)) * x / (sqrt(C_TARGET) * norm_x + EPSILON)
end

export distance, exp_map, log_map, mobius_add, mobius_mul, C_TARGET, EPSILON

end # module HyperbolicSpace
