# Topological Wave-Based Hyperbolic Neural Network

## Overview

This project implements an advanced AI system that combines hyperbolic neural networks with wave-based computation and topological data analysis. The system is designed for efficient learning and information processing using principles inspired by neural synchronization and wave propagation.

## Key Features

- **Hyperbolic Neural Networks**: Neurons embedded in hyperbolic space (Poincaré disk) for natural hierarchical structure
- **Kuramoto Phase Synchronization**: Oscillatory neurons synchronize phases via distance-dependent coupling
- **Wave Propagation**: Waves propagate between neurons, reflecting off morphological boundaries
- **Topological Boundaries**: Persistent homology identifies boundaries that naturally restrict action space
- **Goal-Adapted RL**: Hierarchical reinforcement learning that adapts goals based on information gain
- **Computational Efficiency**: Topological boundaries reduce computation from exponential to polynomial

## Project Structure

```
├── src/
│   ├── core/           # Core neural network implementations
│   ├── ui/             # User interface components
│   ├── servers/        # Web server implementations
│   ├── demos/          # Demo applications
│   └── utils/          # Utility functions and tests
├── docs/
│   ├── theory/         # Theoretical framework and research
│   ├── guides/         # User guides and setup instructions
│   └── api/            # API documentation
├── scripts/            # Setup and utility scripts
└── tests/              # Test files (if applicable)
```

## Getting Started

### Prerequisites

- Julia 1.6 or later
- Required Julia packages (see Project.toml)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Phase1-v2PhasicExperimentTopologyMapping
   ```

2. Install dependencies:
   ```bash
   julia --project=@. -e 'using Pkg; Pkg.instantiate()'
   ```

3. Run the setup script:
   ```bash
   bash scripts/setup_environment.sh
   ```

### Quick Start

See [docs/guides/QUICK_START.md](docs/guides/QUICK_START.md) for detailed usage instructions.

## Documentation

- **Theory**: See [docs/theory/](docs/theory/) for the theoretical framework
- **Guides**: See [docs/guides/](docs/guides/) for setup and usage instructions
- **API**: See [docs/api/](docs/api/) for API documentation

## Examples

### Running the Neural Network UI

```bash
julia --project=@. src/run_neural_network_ui.jl
```

### Starting the Web Server

```bash
julia --project=@. src/servers/web_neural_network_server.jl
```

### Running Demos

```bash
julia --project=@. src/demos/TOASTER_DESIGN_DEMO.jl
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
