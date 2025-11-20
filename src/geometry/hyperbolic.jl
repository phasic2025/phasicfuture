"""
Hyperbolic Geometry Module

This module implements core hyperbolic geometry operations needed for the SAITO-Constrained HNN,
including exponential and logarithmic maps, distance calculations, and gradient operations.
"""
module Hyperbolic

using LinearAlgebra
using StaticArrays
using ForwardDiff

# Constants
const DEFAULT_CURVATURE = 11.7  # SAITO target curvature
const EPSILON = 1e-8  # Small constant for numerical stability

"""
    HyperbolicEmbedding(embedding, curvature=DEFAULT_CURVATURE)

A structure representing points in hyperbolic space with a given curvature.
"""
struct HyperbolicEmbedding{T<:AbstractFloat}
    embedding::Matrix{T}  # Each column is a point in the Poincaré ball
    curvature::T
    
    function HyperbolicEmbedding(embedding::AbstractMatrix{T}, 
                               curvature::T=DEFAULT_CURVATURE) where {T<:AbstractFloat}
        @assert curvature > 0 "Curvature must be positive"
        @assert all(x -> 0 ≤ x < 1, eachcol(norm.(eachcol(embedding)))) "Points must be within the unit ball"
        new{T}(embedding, curvature)
    end
end

"""
    exp_map(x, v, c=DEFAULT_CURVATURE)

Compute the exponential map at point `x` with tangent vector `v` and curvature `c`.
"""
function exp_map(x::AbstractVector{T}, v::AbstractVector{T}, 
                c::T=DEFAULT_CURVATURE) where {T<:AbstractFloat}
    norm_v = norm(v)
    norm_v < EPSILON && return copy(x)
    
    sqrt_c = sqrt(c)
    lambda_x = 2 / (1 - c * norm(x)^2 + EPSILON)
    v_norm = norm_v * sqrt_c
    
    term1 = cosh(v_norm) * x
    term2 = (sinh(v_norm) / (v_norm + EPSILON)) * v
    
    return (term1 + term2) / (cosh(v_norm) + sinh(v_norm) * sqrt_c * norm(x) + EPSILON)
end

"""
    log_map(x, y, c=DEFAULT_CURVATURE)

Compute the logarithmic map from point `x` to point `y` with curvature `c`.
"""
function log_map(x::AbstractVector{T}, y::AbstractVector{T}, 
                c::T=DEFAULT_CURVATURE) where {T<:AbstractFloat}
    sqrt_c = sqrt(c)
    lambda_x = 2 / (1 - c * norm(x)^2 + EPSILON)
    
    # Handle the case when x and y are very close
    if norm(x - y) < EPSILON
        return zero(x)
    end
    
    # Handle antipodal points
    if norm(x + y) < EPSILON
        return (π / (2 * sqrt_c)) * (y / norm(y))
    end
    
    # General case
    alpha = -c * dot(x, y)
    alpha = max(1.0 + alpha, 1.0 + EPSILON)  # Ensure numerical stability
    
    # Calculate the distance
    d = acosh(alpha) / sqrt_c
    
    # Calculate the direction
    direction = y - (alpha * x)
    direction_norm = norm(direction)
    
    if direction_norm < EPSILON
        return zero(x)
    end
    
    # Scale by the distance
    return (d / (direction_norm + EPSILON)) * direction
end

"""
    distance(x, y, c=DEFAULT_CURVATURE)

Compute the hyperbolic distance between points `x` and `y` with curvature `c`.
"""
function distance(x::AbstractVector{T}, y::AbstractVector{T}, 
                 c::T=DEFAULT_CURVATURE) where {T<:AbstractFloat}
    norm_x = norm(x)
    norm_y = norm(y)
    
    # Handle points at the origin
    if norm_x < EPSILON
        return 2 * atanh(sqrt_c * norm_y) / sqrt_c
    elseif norm_y < EPSILON
        return 2 * atanh(sqrt_c * norm_x) / sqrt_c
    end
    
    # General case using the hyperbolic law of cosines
    alpha = (1 - c * norm_x^2) * (1 - c * norm_y^2)
    alpha = max(alpha, EPSILON^2)  # Ensure positive
    
    beta = 1 - 2 * c * norm(x - y)^2 / alpha
    beta = clamp(beta, 1.0 + EPSILON, Inf)  # Ensure valid input for acosh
    
    return acosh(beta) / sqrt(c)
end

"""
    mobius_add(x, y, c=DEFAULT_CURVATURE)

Möbius addition in the Poincaré ball model with curvature `c`.
"""
function mobius_add(x::AbstractVector{T}, y::AbstractVector{T}, 
                   c::T=DEFAULT_CURVATURE) where {T<:AbstractFloat}
    xy = dot(x, y)
    x2 = norm(x)^2
    y2 = norm(y)^2
    
    numerator = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denominator = 1 + 2 * c * xy + c^2 * x2 * y2
    
    return numerator / (denominator + EPSILON)
end

"""
    parallel_transport(x, y, v, c=DEFAULT_CURVATURE)

Parallel transport of vector `v` from the tangent space at `x` to `y`.
"""
function parallel_transport(x::AbstractVector{T}, y::AbstractVector{T}, 
                          v::AbstractVector{T}, c::T=DEFAULT_CURVATURE) where {T<:AbstractFloat}
    # Implementation of parallel transport in the Poincaré ball model
    sqrt_c = sqrt(c)
    lambda_x = 2 / (1 - c * norm(x)^2 + EPSILON)
    lambda_y = 2 / (1 - c * norm(y)^2 + EPSILON)
    
    # Handle the case when x and y are the same point
    if norm(x - y) < EPSILON
        return copy(v)
    end
    
    # Calculate the direction and distance
    d = distance(x, y, c)
    u = log_map(x, y, c)
    u_norm = norm(u)
    
    if u_norm < EPSILON
        return copy(v)
    end
    
    u = u / u_norm
    
    # Project v onto u and the orthogonal complement
    v_parallel = dot(v, u) * u
    v_ortho = v - v_parallel
    
    # Transport both components
    v_parallel_transported = (lambda_x / (lambda_y + EPSILON)) * v_parallel
    
    # For the orthogonal part, we need to compute the angle
    if norm(v_ortho) < EPSILON
        return v_parallel_transported
    end
    
    # The orthogonal part is preserved in magnitude but changes direction
    # according to the parallel transport along the geodesic
    v_ortho_transported = v_ortho  # In the Poincaré ball, this is a simplification
    
    return v_parallel_transported + v_ortho_transported
end

"""
    project_to_manifold(x, c=DEFAULT_CURVATURE)

Project a point onto the Poincaré ball with curvature `c`.
"""
function project_to_manifold(x::AbstractVector{T}, 
                           c::T=DEFAULT_CURVATURE) where {T<:AbstractFloat}
    norm_x = norm(x)
    max_norm = 1.0 / sqrt(c) - EPSILON
    
    if norm_x >= max_norm
        return (max_norm / (norm_x + EPSILON)) * x
    end
    
    return copy(x)
end

"""
    init_embedding(n_points::Int, dim::Int, c=DEFAULT_CURVATURE; 
                  rng=Random.GLOBAL_RNG, init_scale=0.01)

Initialize points in hyperbolic space with curvature `c`.
"""
function init_embedding(n_points::Int, dim::Int, 
                       c::T=DEFAULT_CURVATURE;
                       rng=Random.GLOBAL_RNG, 
                       init_scale=0.01) where {T<:AbstractFloat}
    # Sample from a normal distribution and scale
    embedding = init_scale * randn(rng, T, dim, n_points)
    
    # Project to the Poincaré ball
    embedding = mapslices(x -> project_to_manifold(x, c), embedding; dims=1)
    
    return HyperbolicEmbedding(embedding, c)
end

# Export all functions
export HyperbolicEmbedding, exp_map, log_map, distance, mobius_add,
       parallel_transport, project_to_manifold, init_embedding

end # module
