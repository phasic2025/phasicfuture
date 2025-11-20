# SAITO-Constrained HNN Demo Guide

This guide will walk you through running and understanding the SAITO-constrained Hyperbolic Neural Network (HNN) demo.

## Prerequisites

- Julia 1.8 or later
- Required Julia packages (will be installed automatically):
  - Flux.jl
  - Plots.jl
  - BSON.jl
  - ProgressMeter.jl

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Phase1-v2PhasicExperimentTopologyMapping
   ```

2. Start Julia and enter package mode by pressing `]` and then run:
   ```julia
   activate .
   instantiate
   ```

## Running the Demo

1. Run the training demo:
   ```bash
   julia --project=. demos/saito_hnn_training_demo.jl
   ```

2. The script will:
   - Generate synthetic hyperbolic data
   - Train a SAITO-constrained HNN
   - Save the training plot as `saito_hnn_training.png`
   - Save the trained model as `saito_hnn_model.bson`

## Understanding the Demo

### Data Generation
- The demo creates synthetic data in hyperbolic space
- 1000 samples with 10 features and 3 classes
- Data is normalized to stay within the Poincaré ball

### Model Architecture
- Input layer: 10 dimensions
- Hidden layers: 20 and 10 units with ReLU activation
- Output layer: 3 units (one per class)
- All layers maintain hyperbolic geometry constraints

### Training Process
- Uses hyperbolic MSE loss
- Adam optimizer with learning rate 0.01
- 20 epochs of training
- Batch size of 32
- Training progress is displayed with a progress bar

### Outputs
1. **Training Plot**: Shows training and validation loss over epochs
2. **Saved Model**: The trained model is saved as `saito_hnn_model.bson`
3. **Console Output**: Displays training progress and final test accuracy

## Advanced Usage

### Connecting to the SAITO Network
To connect to the SAITO network (requires running network nodes):

```julia
# In a Julia REPL or script
using .SaitoHNN
model = SaitoHNN.SaitoHNN([10, 20, 10, 3], [relu, relu, identity])
SaitoHNN.connect_to_network!(model, 8000)  # Port number
```

### Loading a Saved Model
```julia
using .SaitoHNN
model = SaitoHNN.SaitoHNN([10, 20, 10, 3])  # Same architecture as saved model
loaded_model = SaitoHNN.load_model("saito_hnn_model.bson", model)
```

## Troubleshooting

### Common Issues
1. **Package Installation**: If package installation fails, try updating your package manager:
   ```julia
   ] up
   ] instantiate
   ```

2. **Plotting Backend**: If plotting fails, try setting a different backend:
   ```julia
   using Plots
   gr()  # or plotly(), pyplot(), etc.
   ```

3. **Memory Issues**: For large models, you may need to increase Julia's memory allocation:
   ```bash
   julia --project=. --heap-size=4G demos/saito_hnn_training_demo.jl
   ```

## Next Steps

1. Try different model architectures
2. Experiment with different learning rates and optimizers
3. Test on real hyperbolic data
4. Connect multiple nodes to form a network

## License

This project is licensed under the MIT License - see the LICENSE file for details.
