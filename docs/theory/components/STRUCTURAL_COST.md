# Structural Cost Calculation in SAITO-HNN

## 1. Overview

The structural cost ($\mathcal{R}_{Topo}$) ensures that the network maintains its topological integrity during learning. It consists of two main components:

1. **Betti Number Change Cost**: Measures changes in the topological invariants
2. **Local Curvature Variance**: Ensures geometric consistency

## 2. Betti Number Calculation

### 2.1 Simplicial Complex Construction
```julia
using Distances

function build_rips_complex(points, max_radius; dim_max=2)
    n = length(points)
    complex = []
    
    # Add 0-simplices (points)
    for i in 1:n
        push!(complex, Set([i]))
    end
    
    # Add higher-dimensional simplices
    for d in 1:dim_max
        # Find all candidate (d+1)-simplices
        candidates = Set()
        for simplex in filter(s -> length(s) == d, complex)
            for point in 1:n
                if !(point in simplex)
                    # Check if all edges are within max_radius
                    if all(p -> euclidean(points[p], points[point]) ≤ max_radius, simplex)
                        new_simplex = union(simplex, Set([point]))
                        push!(candidates, new_simplex)
                    end
                end
            end
        end
        
        # Add valid simplices to the complex
        for s in candidates
            push!(complex, s)
        end
    end
    
    return complex
end
```

### 2.2 Betti Number Calculation
```julia
function calculate_betti(complex, dim_max=2)
    # Initialize boundary matrices
    boundaries = []
    
    for d in 0:dim_max
        # Get d-simplices and (d+1)-simplices
        d_simplices = sort(collect(filter(s -> length(s) == d+1, complex)), by=collect)
        d1_simplices = sort(collect(filter(s -> length(s) == d+2, complex)), by=collect)
        
        # Build boundary matrix
        if isempty(d1_simplices)
            push!(boundaries, zeros(Int, 0, length(d_simplices)))
        else
            boundary = zeros(Int, length(d_simplices), length(d1_simplices))
            
            for (j, simplex) in enumerate(d1_simplices)
                for (sign, face) in enumerate(subsets(collect(simplex), d+1))
                    idx = findfirst(==(Set(face)), d_simplices)
                    if !isnothing(idx)
                        boundary[idx, j] = (-1)^(sign-1)
                    end
                end
            end
            push!(boundaries, boundary)
        end
    end
    
    # Compute Betti numbers
    betti = Int[]
    for d in 0:dim_max
        if d == 0
            rank_d = 0
        else
            rank_d = rank(boundaries[d])
        end
        
        if d+1 > dim_max || isempty(boundaries[d+1])
            rank_d1 = 0
        else
            rank_d1 = rank(boundaries[d+1])
        end
        
        # Betti_d = dim(ker ∂_d) - rank(∂_{d+1})
        push!(betti, size(boundaries[d+1], 1) - rank_d - rank_d1)
    end
    
    return betti
end
```

## 3. Local Curvature Variance

### 3.1 Curvature Calculation
```julia
function local_curvature(points, center_idx, radius, c=11.7)
    center = points[center_idx]
    
    # Find neighbors within radius
    neighbors = [i for (i,p) in enumerate(points) 
                if i != center_idx && norm(p - center) ≤ radius]
    
    if length(neighbors) < 2
        return 0.0  # Not enough points to compute curvature
    end
    
    # Project to tangent space at center using log map
    tangent_vectors = [log_map(center, points[i], c) for i in neighbors]
    
    # Compute angles between tangent vectors
    angles = Float64[]
    for i in 1:length(neighbors)
        for j in i+1:length(neighbors)
            cos_θ = dot(tangent_vectors[i], tangent_vectors[j]) / 
                   (norm(tangent_vectors[i]) * norm(tangent_vectors[j]))
            push!(angles, acos(clamp(cos_θ, -1.0, 1.0)))
        end
    end
    
    # Curvature is related to angle deficit
    expected_angle = 2π / length(neighbors)
    angle_deficit = [abs(θ - expected_angle) for θ in angles]
    
    return var(angle_deficit)
end
```

## 4. Structural Cost Calculation

### 4.1 Main Structural Cost Function
```julia
function structural_cost(graph_before, graph_after, points_before, points_after, c=11.7; 
                        λ_betti=1.0, λ_curv=0.5, radius_scale=0.1)
    
    # 1. Calculate Betti numbers before and after
    betti_before = calculate_betti(graph_before)
    betti_after = calculate_betti(graph_after)
    
    # 2. Calculate Betti cost
    betti_diff = abs.(betti_after .- betti_before)
    cost_betti = λ_betti * sum(betti_diff)
    
    # 3. Calculate local curvature variance
    radius = radius_scale * maximum([norm(p) for p in points_before])
    
    curv_var_before = [local_curvature(points_before, i, radius, c) 
                      for i in 1:length(points_before)]
    curv_var_after = [local_curvature(points_after, i, radius, c) 
                     for i in 1:length(points_after)]
    
    # 4. Calculate curvature cost
    cost_curv = λ_curv * mean(abs.(curv_var_after .- curv_var_before))
    
    # 5. Total structural cost
    return cost_betti + cost_curv, (betti_diff, mean(curv_var_before), mean(curv_var_after))
end
```

## 5. Integration with Learning

### 5.1 Structural Cost in Hyperbolic Hebbian Update
```julia
function hyperbolic_hebbian_update_with_structural_cost(W_old, ∇R, points_before, 
                                                       c=11.7, η=0.01, max_radius=0.5)
    
    # 1. Calculate proposed update without structural cost
    W_proposed = hyperbolic_hebbian_update(W_old, ∇R, η, c)
    
    # 2. Build Rips complexes before and after update
    graph_before = build_rips_complex(points_before, max_radius)
    
    # Update points (in practice, this would involve the new weights)
    points_after = [exp_map(p, ∇R .* η, c) for p in points_before]
    graph_after = build_rips_complex(points_after, max_radius)
    
    # 3. Calculate structural cost
    struct_cost, _ = structural_cost(graph_before, graph_after, 
                                    points_before, points_after, c)
    
    # 4. Adjust learning rate based on structural cost
    η_eff = η / (1 + struct_cost)
    
    # 5. Apply final update with adjusted learning rate
    return hyperbolic_hebbian_update(W_old, ∇R, η_eff, c), struct_cost
end
```

## 6. Testing and Validation

### 6.1 Unit Tests
```julia
@testset "Structural Cost Calculation" begin
    # Create a simple graph (2D points in a circle)
    θ = range(0, 2π, length=10)[1:end-1]
    points_before = [ [cos(ϕ), sin(ϕ)] for ϕ in θ ]
    
    # Slightly perturbed version
    points_after = [p .+ 0.1*randn(2) for p in points_before]
    
    # Build complexes
    graph_before = build_rips_complex(points_before, 1.5)
    graph_after = build_rips_complex(points_after, 1.5)
    
    # Calculate structural cost
    cost, _ = structural_cost(graph_before, graph_after, 
                             points_before, points_after)
    
    @test cost ≥ 0.0
    @test isfinite(cost)
end
```

### 6.2 Performance Optimization

For large graphs, consider these optimizations:

1. **Approximate Betti Numbers**: Use persistent homology with a sparse approximation
2. **Parallel Processing**: Calculate local curvatures in parallel
3. **Incremental Updates**: Only recompute affected parts of the graph

## 7. References
1. Edelsbrunner, H., & Harer, J. (2010). Computational Topology: An Introduction. American Mathematical Society.
2. Chazal, F., & Michel, B. (2017). An Introduction to Topological Data Analysis. arXiv:1710.04019
3. SAITO Protocol (2025). Hyperbolic Topological Constraints for Stable Learning.
