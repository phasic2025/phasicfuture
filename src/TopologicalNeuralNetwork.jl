module TopologicalNeuralNetwork

# Core modules
include("core/KURAMOTO_INTEGRATION.jl")
include("core/IMPLEMENTATION_SKETCH.jl")

# UI modules
include("ui/MONITOR_UI.jl")
include("ui/NEURAL_NETWORK_UI.jl")

# Server modules
include("servers/web_neural_network_server.jl")

# Demo modules
include("demos/TOASTER_DESIGN_DEMO.jl")

# Utility modules
include("utils/install_packages.jl")
include("utils/test_embedding_error.jl")

# Export main functions
export run_neural_network_ui, start_web_monitor, run_toaster_demo

end # module
