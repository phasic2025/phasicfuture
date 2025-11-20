using Test
using Random
using ..SaitoBlockchain

@testset "SaitoBlockchain Tests" begin
    # Set random seed for reproducibility
    Random.seed!(42)
    
    @testset "Blockchain Initialization" begin
        # Test blockchain creation
        blockchain = SaitoBlockchain.Blockchain()
        
        @test length(blockchain.chain) == 1  # Should have genesis block
        @test isempty(blockchain.pending_transactions)
        @test blockchain.chain[1].index == 1
        @test blockchain.chain[1].previous_hash == "0"
    end
    
    @testset "Block Mining" begin
        blockchain = SaitoBlockchain.Blockchain()
        
        # Add some test transactions
        privkey = "test_private_key"
        tx1 = SaitoBlockchain.Transaction("alice", "bob", 10.0, 0.1, privkey)
        tx2 = SaitoBlockchain.Transaction("bob", "charlie", 5.0, 0.05, privkey)
        
        # Add transactions to pending pool
        push!(blockchain.pending_transactions, tx1, tx2)
        
        # Mine a new block
        validator = "validator_node"
        new_block = SaitoBlockchain.mine_block!(blockchain, validator, privkey)
        
        @test new_block !== nothing
        @test length(blockchain.chain) == 2
        @test new_block.index == 2
        @test new_block.validator == validator
        @test length(new_block.transactions) == 2
        @test isempty(blockchain.pending_transactions)  # Should be cleared after mining
        
        # Test chain validation
        @test SaitoBlockchain.validate_chain(blockchain)
    end
    
    @testset "Transaction Validation" do
        blockchain = SaitoBlockchain.Blockchain()
        privkey = "test_private_key"
        
        # Test valid transaction
        valid_tx = SaitoBlockchain.Transaction("alice", "bob", 10.0, 0.1, privkey)
        @test SaitoBlockchain.add_transaction!(blockchain, valid_tx)
        
        # Test invalid transaction (zero amount)
        invalid_tx1 = SaitoBlockchain.Transaction("alice", "bob", 0.0, 0.1, privkey)
        @test !SaitoBlockchain.add_transaction!(blockchain, invalid_tx1)
        
        # Test invalid transaction (negative fee)
        invalid_tx2 = SaitoBlockchain.Transaction("alice", "bob", 10.0, -0.1, privkey)
        @test !SaitoBlockchain.add_transaction!(blockchain, invalid_tx2)
    end
    
    @testset "Staking and Validation" do
        blockchain = SaitoBlockchain.Blockchain()
        
        # Test staking
        @test SaitoBlockchain.stake_tokens!(blockchain, "node1", 100.0)
        @test SaitoBlockchain.stake_tokens!(blockchain, "node2", 200.0)
        
        # Test minimum stake
        @test !SaitoBlockchain.stake_tokens!(blockchain, "node3", 50.0)  # Below minimum
        
        # Test validator selection (probabilistic, but with fixed seed should be deterministic)
        validators = String[]
        for _ in 1:100
            push!(validators, SaitoBlockchain.select_validator(blockchain))
        end
        
        # Should select both validators, but node2 more often (2/3 of the time)
        count_node1 = count(==("node1"), validators)
        count_node2 = count(==("node2"), validators)
        
        @test count_node1 > 20  # Roughly 1/3 of 100
        @test count_node2 > 60  # Roughly 2/3 of 100
    end
    
    @testset "Chain Validation" do
        blockchain = SaitoBlockchain.Blockchain()
        privkey = "test_private_key"
        
        # Add some blocks
        for i in 1:3
            tx = SaitoBlockchain.Transaction("miner$i", "recipient$i", Float64(i), 0.1, privkey)
            push!(blockchain.pending_transactions, tx)
            SaitoBlockchain.mine_block!(blockchain, "validator$i", privkey)
        end
        
        # Should be valid
        @test SaitoBlockchain.validate_chain(blockchain)
        
        # Tamper with the chain
        blockchain.chain[2].transactions[1].amount = 1000.0
        
        # Should detect tampering
        @test !SaitoBlockchain.validate_chain(blockchain)
    end
end

# Run the tests
@testset "SaitoBlockchain Tests" begin
    include("saito_blockchain_tests.jl")
end
