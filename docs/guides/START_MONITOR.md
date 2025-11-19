# Neural Network Web Monitor - Quick Start

## Server is Running! 🚀

**Open in your browser:**
```
http://localhost:8080
```

## What You'll See

### Main View
- **Neural Network Visualization**: 
  - Neurons in hyperbolic space (Poincaré disk)
  - Color-coded by activation level
  - Red arrows showing wave phases
  - Gray lines showing connections (strength > 0.05)

### Real-Time Metrics
- **Synchronization**: Order parameter r(t) [0, 1]
- **Average Activation**: Mean neuron activation
- **Connection Strength**: Average connection weight
- **Neuron Count**: Total neurons in network

### Live Plots
- **Synchronization Over Time**: Shows how neurons synchronize
- **Activation Over Time**: Shows activation dynamics

## What's Happening

1. **Kuramoto Synchronization**: Neurons synchronize their phases
2. **Wave Propagation**: Activations propagate through network
3. **Hebbian Learning**: Connections strengthen when neurons fire together
4. **Topological Boundaries**: Boundary detection (not yet visualized)

## Controls

- **Auto-updates**: Every 100ms
- **Stop**: Press Ctrl+C in terminal

## Troubleshooting

If the page doesn't load:
1. Check server is running: `curl http://localhost:8080/api/network`
2. Check port 8080 is available
3. Try different port: Edit `web_neural_network_server.jl` line with `port=8080`

## Network Details

- **Neurons**: 50
- **Update Rate**: ~20 Hz (every 50ms)
- **Space**: Hyperbolic (Poincaré disk)
- **Learning**: Hebbian (connections adapt)

---

**Status**: Server running on http://localhost:8080

