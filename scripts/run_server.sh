#!/bin/bash
# Run server in Julia environment

cd "$(dirname "$0")"

echo "============================================================"
echo "Starting Neural Network Web Monitor"
echo "============================================================"
echo ""
echo "Server will start on: http://localhost:8080"
echo "Open this URL in your browser!"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Run with project environment activated
julia --project=. web_neural_network_server.jl

