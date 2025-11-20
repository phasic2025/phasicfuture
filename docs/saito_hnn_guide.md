# SAITO-Constrained Hyperbolic Neural Network (SAITO-HNN)

## Overview

The SAITO-HNN is a novel implementation of a Hyperbolic Neural Network that adheres to the principles of the SAITO framework, which enforces physical and economic constraints on the network's operations. This implementation provides a robust foundation for working with hierarchical data in hyperbolic space while maintaining numerical stability and computational efficiency.

## Core Concepts

### 1. Hyperbolic Geometry

SAITO-HNN operates in the Poincaré ball model of hyperbolic space, which provides a natural framework for representing hierarchical data. Key properties include:

- **Exponential growth of space**: The area/volume grows exponentially with radius, allowing efficient representation of hierarchical structures.
- **Constant negative curvature**: The space has a fixed negative curvature (`c_target ≈ 11.7`), which is a fundamental parameter of the model.

### 2. Key Constraints

SAITO-HNN enforces three fundamental constraints:

1. **Geometric Law (R_Phys)**: Maintains the fixed curvature of the hyperbolic space.
2. **Dynamic Law (R_Dyn)**: Enforces a maximum distance (speed limit) for information propagation.
3. **Topological Law (R_Topo)**: Preserves the structural integrity of the knowledge graph.

## Installation

1. Ensure you have Julia 1.6 or later installed.
2. Add the SAITO-HNN module to your project:

```julia
# In the Julia REPL
using Pkg
Pkg.activate(".")
Pkg.add(url="https://github.com/yourusername/Phase1-v2PhasicExperimentTopologyMapping.git")
```

## Basic Usage

### Creating Hyperbolic Points and Vectors

```julia
using SaitoHNN

# Create points in hyperbolic space
origin = HyperbolicPoint([0.0, 0.0])
p1 = HyperbolicPoint([0.3, 0.0])
p2 = HyperbolicPoint([0.0, 0.4])

# Create a vector in the tangent space
v = HyperbolicVector([0.1, 0.2])
```

### Computing Distances and Transformations

```julia
# Calculate hyperbolic distance
d = hyperbolic_distance(p1, p2)

# Map between hyperbolic space and tangent space
tangent_vec = log_map(origin, p1)
reconstructed_point = exp_map(origin, tangent_vec)
```

### Using the Hyperbolic Layer

```julia
# Create a hyperbolic layer (2D input, 3D output)
layer = HyperbolicLayer(2, 3)

# Forward pass
input = HyperbolicPoint([0.1, 0.2])
output = forward(layer, input)
```

### Training with Hyperbolic Hebbian Learning

```julia
# Create a simple network
layer = HyperbolicLayer(2, 1)

# Training data (input-output pairs)
training_data = [
    (HyperbolicPoint([0.1, 0.2]), HyperbolicPoint([0.3])),
    (HyperbolicPoint([-0.1, 0.3]), HyperbolicPoint([0.4])),
    (HyperbolicPoint([0.2, -0.1]), HyperbolicPoint([0.25]))
]

# Training loop
learning_rate = 0.1
epochs = 10

for epoch in 1:epochs
    for (input, target) in training_data
        output = forward(layer, input)
        
        # Update weights using hyperbolic Hebbian rule
        for i in 1:length(layer.weights)
            layer.weights[i] = hyperbolic_hebbian_update(
                layer.weights[i], input, output, learning_rate)
        end
    end
end
```

## Advanced Features

### Constraint Checking

```julia
# Check geometric constraint
is_valid = check_geometric_constraint(p1)

# Check dynamic constraint (distance limit)
is_within_limit = check_dynamic_constraint(p1, p2, 1.0)

# Check topological constraint
is_valid_topology, cost = check_topological_constraint([p1, p2])
```

### Custom Utility Functions

You can implement custom utility functions for the Hebbian learning rule by modifying the `hyperbolic_hebbian_update` function to include your specific utility calculation.

## Performance Considerations

1. **Numerical Stability**: The implementation includes safeguards against numerical instability, especially near the boundary of the Poincaré disk.
2. **Dimensionality**: The current implementation is optimized for moderate dimensions. For very high-dimensional spaces, consider additional optimizations.
3. **Batch Processing**: For large datasets, consider implementing batch processing for the forward and backward passes.

## Examples

See the `demos/` directory for complete examples of using SAITO-HNN, including visualization of hyperbolic space and training workflows.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss potential improvements or report bugs.

## License

[Specify your license here]

## References

1. [Hyperbolic Neural Networks, Nickel & Kiela, 2017]
2. [Learning Continuous Hierarchies in the Lorentz Model of Hyperbolic Geometry, Nickel & Kiela, 2018]
3. [SAITO Framework Documentation]
