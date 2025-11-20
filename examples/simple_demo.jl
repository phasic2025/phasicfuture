using .SAITOHNN  # Using local module
using Sockets
using Random

# Set random seed for reproducibility
Random.seed!(42)

function run_demo()
    println("=== SAITO-Constrained HNN Demo ===\n")
    
    # 1. Initialize a simple network
    input_dim = 3
    hidden_dims = [4, 2]  # Two hidden layers
    output_dim = 1
    
    println("Creating a new node with a Hyperbolic Neural Network...")
    node = Network.Node(8080, input_dim, hidden_dims, output_dim)
    
    # 2. Start the node's network services
    println("Starting node on port 8080...")
    Network.start_node(node)
    
    # 3. Add some peers (in a real scenario, these would be other nodes)
    println("\nAdding some peers...")
    Network.add_peer!(node, "peer1", ip"127.0.0.1", 8081)
    Network.add_peer!(node, "peer2", ip"127.0.0.1", 8082)
    
    # 4. Create a simple transaction
    println("\nCreating a transaction...")
    transaction = Blockchain.Transaction(
        node.id,
        "receiver_id",
        1.0,  # amount
        "private_key_placeholder",
        Dict("data" => "Test transaction")
    )
    
    # 5. Initialize blockchain
    println("Initializing blockchain...")
    blockchain = Blockchain.Blockchain()
    
    # 6. Add transaction to pending transactions
    println("Adding transaction to pending transactions...")
    Blockchain.add_transaction!(blockchain, transaction)
    
    # 7. Mine pending transactions
    println("Mining pending transactions...")
    Blockchain.mine_pending_transactions!(blockchain, node.id)
    
    # 8. Create some sample data
    println("\nCreating sample training data...")
    X = randn(10, input_dim)
    y = randn(10, output_dim)
    
    # 9. Train the model with a few examples
    println("Training the model...")
    learning_rate = 0.01
    for i in 1:5
        for j in 1:size(X, 1)
            input = X[j,:]
            target = y[j,:]
            
            # Enforce physical law on input
            input = EconomicLayer.enforce_physical_law(input)
            
            # Train the model
            prediction = HyperbolicNN.forward(node.model, input)
            HyperbolicNN.hyperbolic_hebbian_update!(node.model.layers[1], input, target, learning_rate)
            
            println("Epoch ", i, " - Sample ", j, " - Prediction: ", prediction[1])
        end
    end
    
    # 10. Demonstrate model update propagation
    println("\nPreparing model update for propagation...")
    model_update = Dict(
        "layer_weights" => [layer.weights for layer in node.model.layers],
        "layer_biases" => [layer.bias for layer in node.model.layers],
        "sender_id" => node.id
    )
    
    # 11. Broadcast the model update (commented out to prevent actual network calls in demo)
    # Network.propagate_model_update(node, model_update)
    
    println("\n=== Demo Complete ===")
    println("A simple SAITO-Constrained HNN node has been created and trained.")
    println("In a real deployment, it would now be communicating with peers.")
end

# Run the demo
run_demo()
