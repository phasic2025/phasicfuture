#!/bin/bash
# Fix port 8080 conflict

echo "Checking port 8080..."
PID=$(lsof -ti :8080 2>/dev/null)

if [ -z "$PID" ]; then
    echo "✅ Port 8080 is free"
else
    echo "Killing process $PID on port 8080..."
    kill -9 $PID 2>/dev/null
    sleep 1
    echo "✅ Port 8080 freed"
fi

echo ""
echo "Now you can run:"
echo "  julia --project=. web_neural_network_server.jl"

