# Julia Environment Setup Guide

## ⚠️ Important: This is Julia, Not Python!

Julia doesn't use `venv` (that's Python). Julia has its own environment system.

---

## Quick Setup (Recommended)

### Step 1: Setup Environment
```bash
cd /home/o2/Documents/Phase1-v2PhasicExperimentTopologyMapping
./setup_environment.sh
```

This will:
- Activate the Julia project environment
- Install all dependencies (HTTP, JSON, etc.)

### Step 2: Run Server
```bash
./run_server.sh
```

Or directly:
```bash
julia --project=. web_neural_network_server.jl
```

---

## Manual Setup

### Option 1: Using Project.toml (Recommended)

The project already has a `Project.toml` file. Just activate it:

```bash
# Activate environment and install dependencies
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Run server
julia --project=. web_neural_network_server.jl
```

### Option 2: In Julia REPL

```julia
# Start Julia
julia

# Activate project
using Pkg
Pkg.activate(".")

# Install dependencies (if needed)
Pkg.instantiate()

# Run server
include("web_neural_network_server.jl")
```

---

## What's Different from Python venv?

| Python | Julia |
|--------|-------|
| `python -m venv venv` | `julia --project=.` |
| `source venv/bin/activate` | `julia --project=.` |
| `pip install package` | `using Pkg; Pkg.add("package")` |
| `requirements.txt` | `Project.toml` |

**Key Difference**: Julia environments are activated by passing `--project=.` flag, not by sourcing a script.

---

## Verify Environment

Check if environment is active:
```bash
julia --project=. -e 'using Pkg; println(Pkg.project().path)'
```

Should show: `/home/o2/Documents/Phase1-v2PhasicExperimentTopologyMapping/Project.toml`

---

## Dependencies

The project needs:
- `HTTP` - Web server
- `JSON` - JSON encoding/decoding
- `LinearAlgebra` - Built-in
- `Random` - Built-in
- `Sockets` - Built-in

All are specified in `Project.toml` and will be installed automatically.

---

## Troubleshooting

**"Package not found" error?**
```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

**"Project.toml not found"?**
- Make sure you're in the project directory
- Run `./setup_environment.sh` to create it

**Port already in use?**
- Edit `web_neural_network_server.jl` and change port number
- Or kill the process: `lsof -ti:8080 | xargs kill`

---

## Quick Reference

```bash
# Setup (one time)
./setup_environment.sh

# Run server
./run_server.sh

# Or directly
julia --project=. web_neural_network_server.jl
```

Then open: **http://localhost:8080**

---

**Ready to go!** 🚀

