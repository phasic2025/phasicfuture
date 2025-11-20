"""
HyperbolicGeometry.jl

Implements core hyperbolic geometry operations with numerical stability for the SAITO-Constrained HNN.
Maintains fixed curvature c_target = 11.7 as per the SAITO constraint.
"""
module HyperbolicGeometry

using LinearAlgebra
using StaticArrays
using DoubleFloats
using ForwardDiff
using Optim

# Constants
export HyperbolicPoint, HyperbolicVector, c_target, ϵ, MAX_ITER

# Core Operations
export hyperbolic_distance, mobius_add, exp_map, log_map, parallel_transport,
       geodesic, figure_eight_knot, parallel_transport_along_geodesic

# Numerical Stability Constants
const c_target = 11.7  # Fixed curvature from SAITO/α
const ϵ = 1e-8         # Small constant for numerical stability
const MAX_ITER = 100   # Maximum iterations for numerical methods

"""
    HyperbolicPoint <: AbstractVector{Float64}

Represents a point in the Poincaré ball model of hyperbolic space.
"""
struct HyperbolicPoint{N} <: AbstractVector{Float64}
    coords::SVector{N,Float64}
    
    function HyperbolicPoint(coords::AbstractVector{<:Real})
        n = length(coords)
        new{n}(SVector{n,Float64}(coords))
    end
end

Base.size(::HyperbolicPoint{N}) where N = (N,)
Base.getindex(p::HyperbolicPoint, i::Int) = p.coords[i]
Base.zero(::Type{HyperbolicPoint{N}}) where N = HyperbolicPoint(zeros(N))

"""
    HyperbolicVector <: AbstractVector{Float64}

Represents a vector in the tangent space at a point in hyperbolic space.
"""
struct HyperbolicVector{N} <: AbstractVector{Float64}
    coords::SVector{N,Float64}
    
    function HyperbolicVector(coords::AbstractVector{<:Real})
        n = length(coords)
        new{n}(SVector{n,Float64}(coords))
    end
end

Base.size(::HyperbolicVector{N}) where N = (N,)
Base.getindex(v::HyperbolicVector, i::Int) = v.coords[i]
Base.zero(::Type{HyperbolicVector{N}}) where N = HyperbolicVector(zeros(N))

# Core Hyperbolic Operations

"""
    safe_sqrt(x)

Numerically stable square root that handles values near zero.
"""
@inline function safe_sqrt(x::Real)
    x = max(0.0, x)
    return sqrt(x + ϵ)
end

"""
    safe_acosh(x)

Numerically stable inverse hyperbolic cosine.
"""
@inline function safe_acosh(x::Real)
    x < 1 + ϵ && return 0.0
    return acosh(x)
end

"""
    safe_artanh(x)

Numerically stable inverse hyperbolic tangent.
"""
@inline function safe_artanh(x::Real)
    abs_x = abs(x)
    if abs_x >= 1.0 - ϵ
        return sign(x) * 19.07  # Approximate limit as x -> 1
    end
    return atanh(x)
end

"""
    mobius_add(u::HyperbolicPoint, v::HyperbolicPoint)

Möbius addition in the Poincaré ball model.
"""
function mobius_add(u::HyperbolicPoint{N}, v::HyperbolicPoint{N}) where N
    u_norm_sq = sum(abs2, u)
    v_norm_sq = sum(abs2, v)
    uv_dot = dot(u, v)
    
    denominator = 1 + 2 * c_target * uv_dot + c_target^2 * u_norm_sq * v_norm_sq
    
    # Numerically stable computation of numerator
    term1 = (1 + 2 * c_target * uv_dot + c_target * v_norm_sq) * u.coords
    term2 = (1 - c_target * u_norm_sq) * v.coords
    
    result = (term1 + term2) / (denominator + ϵ)
    
    # Project back to the Poincaré ball if needed (shouldn't happen with proper constraints)
    norm_result = norm(result)
    if norm_result >= 1.0 - ϵ
        result .*= (1.0 - 1e-8) / (norm_result + ϵ)
    end
    
    return HyperbolicPoint(result)
end

"""
    hyperbolic_distance(u::HyperbolicPoint, v::HyperbolicPoint)

Compute the hyperbolic distance between two points in the Poincaré ball.
"""
function hyperbolic_distance(u::HyperbolicPoint{N}, v::HyperbolicPoint{N}) where N
    diff = mobius_add(-u, v)
    diff_norm = norm(diff)
    u_norm_sq = sum(abs2, u)
    v_norm_sq = sum(abs2, v)
    
    # Numerically stable computation of the argument to arcosh
    numerator = (1 - 2 * c_target * dot(u, v) + c_target^2 * u_norm_sq * v_norm_sq)
    denominator = (1 - c_target * u_norm_sq) * (1 - c_target * v_norm_sq)
    
    # Handle edge cases
    if denominator < ϵ
        return Inf  # Points are at the boundary
    end
    
    arg = 1 + 2 * c_target * diff_norm^2 / (denominator + ϵ)
    return (2 / sqrt(c_target)) * safe_acosh(arg)
end

"""
    exp_map(p::HyperbolicPoint, v::HyperbolicVector)

Exponential map from the tangent space at point p to the manifold.
"""
function exp_map(p::HyperbolicPoint{N}, v::HyperbolicVector{N}) where N
    v_norm = norm(v)
    if v_norm < 1e-10
        return copy(p)
    end
    
    # Convert to tangent vector at origin
    λ = 2 / (1 - c_target * sum(abs2, p))
    v0 = v.coords / (λ + ϵ)
    
    # Apply exponential map in the tangent space at origin
    v0_norm = norm(v0)
    if v0_norm < 1e-10
        return copy(p)
    end
    
    exp_term = tanh(sqrt(c_target) * v0_norm / 2)
    exp_v0 = (exp_term / (sqrt(c_target) * v0_norm + ϵ)) .* v0
    
    # Parallel transport back to p
    return mobius_add(p, HyperbolicPoint(exp_v0))
end

"""
    log_map(p::HyperbolicPoint, q::HyperbolicPoint)

Logarithmic map from point q to the tangent space at point p.
"""
function log_map(p::HyperbolicPoint{N}, q::HyperbolicPoint{N}) where N
    if p ≈ q
        return zero(HyperbolicVector{N})
    end
    
    # Move p to origin
    neg_p = HyperbolicPoint(-p.coords)
    q_at_origin = mobius_add(neg_p, q)
    
    q_norm = norm(q_at_origin)
    if q_norm < 1e-10
        return zero(HyperbolicVector{N})
    end
    
    # Compute log map at origin
    λ = 2 / (1 - c_target * sum(abs2, p))
    scale = (2 / (sqrt(c_target) * λ + ϵ)) * safe_artanh(sqrt(c_target) * q_norm) / (q_norm + ϵ)
    
    log_q = scale .* q_at_origin.coords
    
    # Parallel transport back to p's tangent space
    return HyperbolicVector(log_q)
end

"""
    parallel_transport(u::HyperbolicVector, p::HyperbolicPoint, q::HyperbolicPoint)

Parallel transport vector u from the tangent space at p to the tangent space at q
along the geodesic connecting them.
"""
function parallel_transport(u::HyperbolicVector{N}, p::HyperbolicPoint{N}, q::HyperbolicPoint{N}) where N
    if p ≈ q
        return u
    end
    
    # Move p to origin
    neg_p = HyperbolicPoint(-p.coords)
    q_at_origin = mobius_add(neg_p, q)
    
    # Compute parallel transport in the Poincaré ball model
    λ = 2 / (1 - c_target * sum(abs2, p))
    q_norm = norm(q_at_origin)
    
    if q_norm < 1e-10
        return u  # No transport needed if q is at origin
    end
    
    # Compute the parallel transport matrix
    u_parallel = similar(u.coords)
    for i in 1:N
        u_parallel[i] = u[i] * (1 - q_norm^2) / (1 - 2*dot(q_at_origin, u.coords) + q_norm^2 * sum(abs2, u))
    end
    
    return HyperbolicVector(u_parallel)
end

"""
    geodesic(p::HyperbolicPoint, q::HyperbolicPoint, t::Real)

Compute the point at parameter t along the geodesic from p to q.
For t=0, returns p; for t=1, returns q.
"""
function geodesic(p::HyperbolicPoint{N}, q::HyperbolicPoint{N}, t::Real) where N
    t = clamp(t, 0.0, 1.0)
    
    if t ≈ 0.0
        return copy(p)
    elseif t ≈ 1.0
        return copy(q)
    end
    
    log_pq = log_map(p, q)
    return exp_map(p, HyperbolicVector(t .* log_pq.coords))
end

"""
    figure_eight_knot(t; R=1.0, r=0.3, c=0.4)

Generate points on a figure-eight knot in 3D Euclidean space.
Parameters:
- t: Parameter along the knot (0 to 2π)
- R: Major radius
- r: Minor radius
- c: Parameter controlling the "pinch" of the figure-eight
"""
function figure_eight_knot(t; R=1.0, r=0.3, c=0.4)
    x = (R + r*cos(2*t)) * cos(t)
    y = (R + r*cos(2*t)) * sin(t)
    z = c * r * sin(4*t)
    return [x, y, z]
end

"""
    project_to_hyperboloid(p::Vector{Float64})

Project a point from R³ to the hyperboloid model of hyperbolic space.
"""
function project_to_hyperboloid(p::Vector{Float64})
    x, y, z = p
    t = sqrt(1 + x^2 + y^2 + z^2)
    return [t, x, y, z]
end

"""
    hyperboloid_to_poincare(p::Vector{Float64})

Convert from hyperboloid model to Poincaré ball model.
"""
function hyperboloid_to_poincare(p::Vector{Float64})
    t, x, y, z = p
    return [x, y, z] ./ (1 + t)
end

"""
    parallel_transport_along_geodesic(u::HyperbolicVector, p::HyperbolicPoint, q::HyperbolicPoint, t::Real)

Parallel transport vector u from point p along the geodesic toward q by fraction t.
"""
function parallel_transport_along_geodesic(u::HyperbolicVector{N}, p::HyperbolicPoint{N}, q::HyperbolicPoint{N}, t::Real) where N
    intermediate_point = geodesic(p, q, t)
    return parallel_transport(u, p, intermediate_point)
end

"""
    curvature_tensor(p::HyperbolicPoint, u::HyperbolicVector, v::HyperbolicVector, w::HyperbolicVector)

Compute the curvature tensor R(u,v)w at point p.
"""
function curvature_tensor(p::HyperbolicPoint{N}, u::HyperbolicVector{N}, v::HyperbolicVector{N}, w::HyperbolicVector{N}) where N
    # In hyperbolic space with constant curvature -c_target, the curvature tensor is:
    # R(u,v)w = -c_target * (dot(v,w)*u - dot(u,w)*v)
    return -c_target * (dot(v, w) * u.coords - dot(u, w) * v.coords)
end

"""
    exponential_map_derivative(p::HyperbolicPoint, v::HyperbolicVector, w::HyperbolicVector)

Compute the derivative of the exponential map at p in direction v applied to w.
This is useful for Jacobi field computations.
"""
function exponential_map_derivative(p::HyperbolicPoint{N}, v::HyperbolicVector{N}, w::HyperbolicVector{N}) where N
    v_norm = norm(v)
    if v_norm < 1e-10
        return w
    end
    
    # Project w onto v and its orthogonal complement
    w_parallel = (dot(v, w) / (v_norm^2 + ϵ)) * v
    w_perp = w - w_parallel
    
    # Compute the derivative components
    parallel_scale = 1.0
    perp_scale = (sinh(sqrt(c_target) * v_norm) / (sqrt(c_target) * v_norm + ϵ))
    
    return HyperbolicVector(parallel_scale * w_parallel.coords + perp_scale * w_perp.coords)
end
end

"""
    parallel_transport(p::HyperbolicPoint, q::HyperbolicPoint, v::HyperbolicVector)

Parallel transport of vector v from the tangent space at p to the tangent space at q.
"""
function parallel_transport(p::HyperbolicPoint{N}, q::HyperbolicPoint{N}, v::HyperbolicVector{N}) where N
    if p ≈ q
        return copy(v)
    end
    
    # Move p to origin
    neg_p = HyperbolicPoint(-p.coords)
    q_prime = mobius_add(neg_p, q)
    v_prime = v.coords .* (1 / (1 - c_target * sum(abs2, p)))
    
    # Transport to origin
    q_prime_norm = norm(q_prime)
    if q_prime_norm < 1e-10
        return zero(HyperbolicVector{N})
    end
    
    # Apply parallel transport in the Poincaré ball model
    λ_p = 2 / (1 - c_target * sum(abs2, p))
    λ_q = 2 / (1 - c_target * sum(abs2, q))
    
    v_parallel = (λ_p / (λ_q + ϵ)) .* v_prime
    
    # Move back to q's tangent space
    return HyperbolicVector(v_parallel)
end

end # module
