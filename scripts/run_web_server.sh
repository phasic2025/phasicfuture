#!/bin/bash
# Quick launcher for Neural Network Web Monitor

cd "$(dirname "$0")"

echo "============================================================"
echo "Neural Network Web Monitor"
echo "============================================================"
echo ""
echo "Starting server on http://localhost:8080"
echo "Open this URL in your browser to view the neural network"
echo ""
echo "Press Ctrl+C to stop"
echo ""

julia web_neural_network_server.jl

