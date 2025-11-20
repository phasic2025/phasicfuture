# Hyperbolic Hebbian Learning in SAITO-Constrained HNN

## Introduction

The Hyperbolic Hebbian Learning rule is the core adaptive mechanism in the SAITO-Constrained HNN. It enables the network to learn and adapt while respecting the geometric and economic constraints of the system.

## Mathematical Foundation

### Basic Hebbian Learning
Traditional Hebbian learning follows the principle:

```math
\Delta w_{ij} = \eta \cdot a_i \cdot a_j
```
where:
- $w_{ij}$ is the weight between neurons $i$ and $j$
- $\eta$ is the learning rate
- $a_i$ and $a_j$ are the activations of the connected neurons

### Hyperbolic Adaptation
In the SAITO-Constrained HNN, we modify this for hyperbolic space:

```math
\Delta \mathbf{W} = \eta \cdot R \cdot \text{Exp}_{\mathbf{W}_{\text{old}}}(\Delta \mathbf{W}_{\text{tangent}})
```

## The Complete Update Rule

The full Hyperbolic Hebbian Update consists of three main steps:

1. **Projection to Tangent Space**
   ```math
   \mathbf{v}_{\text{tangent}} = \text{Log}_{\mathbf{u}}(\mathbf{v})
   ```

2. **Utility-Weighted Update**
   ```math
   \Delta \mathbf{W}_{\text{tangent}} = R \cdot (\mathbf{a}_i \otimes \mathbf{a}_j)
   ```
   where $R$ is the total utility of the connection.

3. **Projection Back to Manifold**
   ```math
   \mathbf{W}_{\text{new}} = \text{Exp}_{\mathbf{W}_{\text{old}}}(\eta \cdot \Delta \mathbf{W}_{\text{tangent}})
   ```

## Implementation Considerations

### Numerical Stability
Key considerations for stable implementation:
1. **Gradient Clipping**: Prevent exploding gradients
2. **Curvature Scaling**: Properly handle the $c_{target} \approx 11.7$ curvature
3. **Precision Handling**: Use appropriate numerical precision for hyperbolic functions

### Economic Constraints
Each update must respect:
1. **Dynamic Constraint**: $d_c(i,j) \leq d_{\text{max}}$
2. **Topological Constraint**: $\mathcal{R}_{\text{Topo}} \leq \text{threshold}$

## Example Implementation

```julia
function hyperbolic_hebbian_update(
    W_old::AbstractMatrix,
    a_i::AbstractVector,
    a_j::AbstractVector,
    R::Float64,
    η::Float64,
    c_target::Float64,
    d_max::Float64
)::Tuple{AbstractMatrix, Bool}
    # 1. Check dynamic constraint
    d = hyperbolic_distance(W_old, a_i, a_j, c_target)
    if d > d_max
        return (W_old, false)  # Update rejected
    end
    
    # 2. Project to tangent space
    v_tangent = log_map(a_i, a_j, c_target)
    
    # 3. Calculate utility-weighted update
    ΔW_tangent = R .* (a_i * a_j')
    
    # 4. Project back to manifold
    W_new = exp_map(W_old, η .* ΔW_tangent, c_target)
    
    # 5. Verify topological constraint
    if !check_topological_constraint(W_old, W_new, c_target)
        return (W_old, false)  # Update rejected
    end
    
    return (W_new, true)  # Update accepted
end
```

## Performance Considerations

1. **Computational Complexity**: $O(d^2)$ per update where $d$ is the embedding dimension
2. **Memory Requirements**: Need to store tangent space projections
3. **Parallelization**: Can be parallelized across network connections

## Next Steps

- [ ] Implement core geometric operations (`log_map`, `exp_map`)
- [ ] Add numerical stability checks
- [ ] Integrate with economic constraints
- [ ] Optimize for GPU acceleration
