# Logarithmic and Exponential Maps in Hyperbolic Space

## 1. Mathematical Foundations

### 1.1 Exponential Map (Expᵤ)
Projects a vector from the tangent space at point u onto the manifold:

$$\text{Exp}_u(\mathbf{v}) = \mathbf{u} \oplus_c \left( \tanh\left( \frac{\sqrt{c} \lambda_u^c \|\mathbf{v}\|}{2} \right) \frac{\mathbf{v}}{\sqrt{c} \|\mathbf{v}\|} \right)$$

### 1.2 Logarithmic Map (Logᵤ)
Projects a point from the manifold to the tangent space at point u:

$$\text{Log}_u(\mathbf{v}) = \frac{2}{\sqrt{c} \lambda_u^c} \text{artanh}\left( \sqrt{c} \| -\mathbf{u} \oplus_c \mathbf{v} \| \right) \frac{-\mathbf{u} \oplus_c \mathbf{v}}{\| -\mathbf{u} \oplus_c \mathbf{v} \|}$$

Where:
- $\lambda_u^c = \frac{2}{1 - c\|u\|^2}$ is the conformal factor
- $\oplus_c$ denotes Möbius addition
- $c = c_{target} \approx 11.7$ (SAITO constant)

## 2. Core Implementations

### 2.1 Möbius Addition
```julia
function mobius_add(u, v, c=11.7; ϵ=1e-8)
    u_norm_sq = max(ϵ, sum(abs2, u))
    v_norm_sq = max(ϵ, sum(abs2, v))
    uv_dot = max(ϵ, dot(u, v))
    
    denominator = 1 + 2*c*uv_dot + c^2 * u_norm_sq * v_norm_sq
    
    return ((1 + 2*c*uv_dot + c*v_norm_sq) .* u .+ (1 - c*u_norm_sq) .* v) ./ denominator
end
```

### 2.2 Exponential Map
```julia
function exp_map(u, v, c=11.7; ϵ=1e-8)
    v_norm = max(ϵ, norm(v))
    λ = 2 / max(ϵ, 1 - c * sum(abs2, u))
    
    # Direction vector (unit vector in direction of v)
    direction = v ./ v_norm
    
    # Scale factor
    scale = tanh((√c * λ * v_norm) / 2) / (√c)
    
    # Möbius addition of u and scaled direction
    return mobius_add(u, direction .* scale, c)
end
```

### 2.3 Logarithmic Map
```julia
function log_map(u, v, c=11.7; ϵ=1e-8)
    # Möbius subtraction: -u ⊕ v
    mobius_sub = mobius_add(-u, v, c)
    mobius_sub_norm = max(ϵ, norm(mobius_sub))
    
    λ = 2 / max(ϵ, 1 - c * sum(abs2, u))
    
    # Scale factor
    scale = (2 / (√c * λ)) * atanh(√c * mobius_sub_norm) / mobius_sub_norm
    
    return mobius_sub .* scale
end
```

## 3. Numerical Stability

### 3.1 Safe Division
```julia
function safe_divide(a, b, ϵ=1e-8)
    return a / max(ϵ, b)
end
```

### 3.2 Stable Norm Calculation
```julia
function safe_norm(x, ϵ=1e-8)
    return max(ϵ, norm(x))
end
```

## 4. Applications in SAITO-HNN

### 4.1 Weight Updates in Hyperbolic Space
```julia
function hyperbolic_hebbian_update(W_old, ∇R, η, c=11.7)
    # Project gradient to tangent space at W_old
    grad_tangent = log_map(W_old, ∇R, c)
    
    # Scale by learning rate
    update = η .* grad_tangent
    
    # Project back to manifold
    W_new = exp_map(W_old, update, c)
    
    return W_new
end
```

### 4.2 Parallel Transport
```julia
function parallel_transport(u, v, w, c=11.7)
    # Transport vector w from u to v
    λ_u = 2 / (1 - c * sum(abs2, u))
    λ_v = 2 / (1 - c * sum(abs2, v))
    
    # Project w to tangent space at u
    w_tangent = log_map(u, w, c)
    
    # Transport to tangent space at v
    return (λ_u / λ_v) .* w_tangent
end
```

## 5. Testing and Validation

### 5.1 Unit Tests
```julia
@testset "Exponential and Logarithmic Maps" begin
    c = 11.7
    ϵ = 1e-6
    
    # Test exp and log are inverses
    u = rand(10) .* 0.1
    v = rand(10) .* 0.1
    
    # Project to tangent space and back
    v_tangent = log_map(u, v, c)
    v_reconstructed = exp_map(u, v_tangent, c)
    
    @test isapprox(v, v_reconstructed, atol=ϵ)
end
```

### 5.2 Numerical Stability Tests
```julia
@testset "Numerical Stability at Boundary" begin
    c = 11.7
    ϵ = 1e-8
    
    # Points near the boundary
    u = fill((1/sqrt(c)) - 1e-6, 10)
    v = fill((1/sqrt(c)) - 1e-5, 10)
    
    # Should not throw or return NaN/Inf
    @test all(isfinite.(exp_map(u, v, c)))
    @test all(isfinite.(log_map(u, v, c)))
end
```

## 6. Performance Optimizations

### 6.1 Batch Processing
```julia
function batch_exp_map(us, vs, c=11.7)
    # Vectorized implementation for multiple points
    λs = @. 2 / (1 - c * sum(abs2, us, dims=1))
    v_norms = max.(ϵ, sqrt.(sum(abs2, vs, dims=1)))
    
    # Normalize directions
    directions = vs ./ v_norms
    
    # Compute scales
    scales = @. tanh((√c * λs * v_norms) / 2) / (√c)
    
    # Apply Möbius addition
    return [mobius_add(us[:,i], directions[:,i] .* scales[i], c) for i in 1:size(us,2)]
end
```

### 6.2 GPU Acceleration
```julia
using CUDA

function exp_map_gpu(u::CuArray, v::CuArray, c=11.7f0; ϵ=1f-8)
    v_norm = CUDA.max(ϵ, CUDA.norm(v))
    λ = 2f0 / CUDA.max(ϵ, 1f0 - c * CUDA.sum(abs2, u))
    
    direction = v / v_norm
    scale = tanh((√c * λ * v_norm) / 2f0) / √c
    
    return mobius_add_gpu(u, direction .* scale, c)
end
```

## 7. References
1. Ganea, O., Bécigneul, G., & Hofmann, T. (2018). Hyperbolic Neural Networks. NeurIPS 2018.
2. Nickel, M., & Kiela, D. (2017). Poincaré Embeddings for Learning Hierarchical Representations. arXiv:1705.08039
3. Skopek, O., Ganea, O., & Bécigneul, G. (2019). Mixed-curvature Variational Autoencoders. arXiv:1911.08411
