using Test
using Random
using Sockets
using ..NetworkBlockchain

@testset "NetworkBlockchain Integration Tests" begin
    # Set random seed for reproducibility
    Random.seed!(42)
    
    @testset "Node Initialization" begin
        # Test node creation
        node = NetworkBlockchain.SaitoNode(9000)
        
        @test node.network_node.port == 9000
        @test !node.is_mining
        @test !node.is_validating
        @test isempty(node.model_weights)
        @test length(node.blockchain.chain) == 1  # Genesis block
    end
    
    @testset "Transaction Submission" begin
        # Create two nodes
        node1 = NetworkBlockchain.SaitoNode(9001)
        node2 = NetworkBlockchain.SaitoNode(9002)
        
        # Start nodes
        NetworkBlockchain.start_node(node1)
        NetworkBlockchain.start_node(node2)
        
        # Add node2 as a peer to node1
        SaitoNetwork.add_peer!(node1.network_node, ip"127.0.0.1", 9002)
        
        # Submit a transaction
        success, msg = NetworkBlockchain.submit_transaction(node1, "alice", "bob", 10.0, 0.1)
        @test success
        @test msg == "Transaction submitted"
        
        # Check that the transaction was added to pending transactions
        @test length(node1.blockchain.pending_transactions) == 1
        @test node1.blockchain.pending_transactions[1].from == "alice"
        @test node1.blockchain.pending_transactions[1].to == "bob"
        @test node1.blockchain.pending_transactions[1].amount == 10.0
    end
    
    @testset "Block Mining" begin
        # Create a mining node
        miner = NetworkBlockchain.SaitoNode(9003)
        miner.is_mining = true
        
        # Add some transactions
        NetworkBlockchain.submit_transaction(miner, "alice", "bob", 5.0, 0.05)
        NetworkBlockchain.submit_transaction(miner, "bob", "charlie", 2.0, 0.02)
        
        # Mine a block
        NetworkBlockchain.mine_pending_transactions!(miner)
        
        # Check that a new block was created
        @test length(miner.blockchain.chain) == 2
        @test miner.blockchain.chain[2].index == 2
        @test length(miner.blockchain.chain[2].transactions) == 2
        @test isempty(miner.blockchain.pending_transactions)  # Should be cleared after mining
    end
    
    @testset "Model Updates" begin
        # Create a node for model updates
        node = NetworkBlockchain.SaitoNode(9004)
        
        # Test model weight updates
        test_weights = Dict("layer1" => [1.0, 2.0, 3.0], "layer2" => [4.0, 5.0])
        success, msg = NetworkBlockchain.update_model_weights!(node, test_weights, 1.0)
        
        @test success
        @test msg == "Model update submitted"
        @test node.model_weights == test_weights
    end
    
    @testset "Chain Validation" do
        # Create a node and add some blocks
        node = NetworkBlockchain.SaitoNode(9005)
        node.is_mining = true
        
        # Add some transactions and mine blocks
        NetworkBlockchain.submit_transaction(node, "alice", "bob", 10.0, 0.1)
        NetworkBlockchain.mine_pending_transactions!(node)
        
        NetworkBlockchain.submit_transaction(node, "bob", "charlie", 5.0, 0.05)
        NetworkBlockchain.mine_pending_transactions!(node)
        
        # Chain should be valid
        @test SaitoBlockchain.validate_chain(node.blockchain)
        
        # Tamper with the chain
        node.blockchain.chain[2].transactions[1].amount = 1000.0
        
        # Should detect tampering
        @test !SaitoBlockchain.validate_chain(node.blockchain)
    end
end

# Run the tests
@testset "NetworkBlockchain Integration Tests" begin
    include("network_blockchain_tests.jl")
end
