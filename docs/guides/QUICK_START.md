# Quick Start: Neural Network Web Monitor

## 🚀 Start the Server

**Option 1: Direct Julia**
```bash
cd /home/o2/Documents/Phase1-v2PhasicExperimentTopologyMapping
julia web_neural_network_server.jl
```

**Option 2: Using launcher script**
```bash
cd /home/o2/Documents/Phase1-v2PhasicExperimentTopologyMapping
./run_web_server.sh
```

## 🌐 Open in Browser

Once the server starts, open:
```
http://localhost:8080
```

## 📊 What You'll Monitor

### Real-Time Visualization
- **Neurons**: 50 neurons in hyperbolic space (Poincaré disk)
- **Activations**: Color-coded by activation level (green = high, blue = low)
- **Phases**: Red arrows showing wave phases
- **Connections**: Gray lines showing connection strengths

### Live Metrics
- **Synchronization**: Order parameter r(t) - measures phase synchronization
- **Average Activation**: Mean neuron activation level
- **Connection Strength**: Average connection weight (learned via Hebbian)
- **Time**: Simulation time

### Time Series Plots
- **Synchronization Over Time**: Watch neurons synchronize
- **Activation Over Time**: Watch activation dynamics

## 🔬 What's Happening

1. **Kuramoto Synchronization**: Neurons synchronize their phases
2. **Wave Propagation**: Activations propagate through network
3. **Hebbian Learning**: Connections strengthen when neurons fire together
4. **Topological Boundaries**: Boundary detection (in progress)

## ⚙️ Technical Details

- **Update Rate**: ~20 Hz (every 50ms)
- **Network Size**: 50 neurons
- **Space**: Hyperbolic (Poincaré disk model)
- **Learning**: Hebbian rule with phase-dependent coupling

## 🛑 Stop Server

Press `Ctrl+C` in the terminal

## 🔧 Troubleshooting

**Port already in use?**
- Edit `web_neural_network_server.jl` and change `port=8080` to another port

**Dependencies missing?**
- Run: `julia -e 'using Pkg; Pkg.add("HTTP"); Pkg.add("JSON")'`

**Server won't start?**
- Check Julia version: `julia --version`
- Check if port 8080 is available: `netstat -an | grep 8080`

---

**Ready to monitor!** 🧠✨

