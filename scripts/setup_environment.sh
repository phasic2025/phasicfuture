#!/bin/bash
# Setup Julia environment for Neural Network project

cd "$(dirname "$0")"

echo "============================================================"
echo "Setting up Julia Environment"
echo "============================================================"
echo ""

# Activate the project environment
echo "Activating Julia project environment..."
julia --project=. -e 'using Pkg; Pkg.instantiate()'

echo ""
echo "✅ Environment setup complete!"
echo ""
echo "To use this environment, run:"
echo "  julia --project=. web_neural_network_server.jl"
echo ""
echo "Or activate it in Julia REPL:"
echo "  julia --project=."
echo "  julia> using Pkg"
echo "  julia> Pkg.activate(\".\")"
echo ""

