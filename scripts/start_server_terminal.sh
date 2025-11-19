#!/bin/bash
# Start server in visible terminal

cd "$(dirname "$0")"

echo "============================================================"
echo "Starting Neural Network Web Server"
echo "============================================================"
echo ""
echo "Server will start on: http://localhost:8080"
echo "Open this URL in your browser!"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

julia --project=. web_neural_network_server.jl

