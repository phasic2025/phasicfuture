using Test
using Sockets
using Random
using ..SaitoNetwork

@testset "SaitoNetwork Tests" begin
    # Set random seed for reproducibility
    Random.seed!(42)
    
    @testset "Node Initialization" begin
        # Test node creation
        node = SaitoNetwork.SaitoNode(8001)
        
        @test length(node.id) == 64  # SHA-256 hex string
        @test node.port == 8001
        @test isempty(node.peers)  # No bootstrap nodes provided
        @test node.wallet["balance"] == 1000.0
        @test isempty(node.model_weights)
        @test isempty(node.blockchain)
    end
    
    @testset "Peer Management" begin
        node = SaitoNetwork.SaitoNode(8002)
        
        # Test adding a peer
        peer_added = SaitoNetwork.add_peer!(node, ip"127.0.0.1", 8003)
        @test peer_added
        @test length(node.peers) == 1
        
        # Test adding duplicate peer
        peer_added_again = SaitoNetwork.add_peer!(node, ip"127.0.0.1", 8003)
        @test !peer_added_again
        @test length(node.peers) == 1  # Shouldn't add duplicate
        
        # Test adding self as peer
        peer_added_self = SaitoNetwork.add_peer!(node, node.address, node.port)
        @test !peer_added_self
        @test length(node.peers) == 1  # Shouldn't add self
    end
    
    @testset "Message Creation and Validation" begin
        node = SaitoNetwork.SaitoNode(8004)
        
        # Test message creation
        data = Dict("test" => "data", "value" => 42)
        message = SaitoNetwork.create_message(node, :test_message, data)
        
        @test message.type == :test_message
        @test message.sender_id == node.id
        @test message.data == data
        @test length(message.signature) == 64  # SHA-256 hex string
        
        # Test message validation (simplified)
        # In a real implementation, we'd test cryptographic signature verification
        @test message.signature == bytes2hex(sha256(string(
            node.id, 
            :test_message, 
            message.timestamp, 
            data, 
            message.nonce
        )))
    end
    
    @testset "Message Processing" begin
        # This is a simplified test since we can't easily test network operations
        node = SaitoNetwork.SaitoNode(8005)
        
        # Create a test message
        test_data = Dict("test" => "value")
        message = SaitoNetwork.create_message(node, :test_message, test_data)
        
        # Process the message
        # In a real implementation, we'd test the actual processing logic
        @test message.type == :test_message
        @test message.data == test_data
    end
    
    @testset "Economic Validation" begin
        node = SaitoNetwork.SaitoNode(8006)
        
        # Test transaction validation
        valid_tx = Dict(
            "from" => "sender123",
            "to" => "recipient456",
            "amount" => 10.0,
            "fee" => 0.1,
            "timestamp" => time(),
            "signature" => "test_signature"
        )
        
        @test SaitoNetwork.validate_transaction(node, valid_tx)
        
        # Test invalid transaction (missing field)
        invalid_tx = copy(valid_tx)
        delete!(invalid_tx, "amount")
        @test !SaitoNetwork.validate_transaction(node, invalid_tx)
        
        # Test expired transaction
        old_tx = copy(valid_tx)
        old_tx["timestamp"] = time() - 4000  # More than 1 hour old
        @test !SaitoNetwork.validate_transaction(node, old_tx)
    end
end

# Run the tests
@testset "SaitoNetwork Tests" begin
    include("saito_network_tests.jl")
end
