"""
NetworkBlockchain Module

Integrates the P2P network with the blockchain to create a complete
SAITO-constrained HNN network with economic incentives.
"""
module NetworkBlockchain

using ..SaitoNetwork
using ..SaitoBlockchain
using ..SaitoHyperbolic
using JSON3
using Sockets
using SHA

"""
    SaitoNode

Enhanced node that combines P2P networking with blockchain functionality.
"""
mutable struct SaitoNode
    # Network components
    network_node::SaitoNetwork.SaitoNode
    
    # Blockchain components
    blockchain::SaitoBlockchain.Blockchain
    
    # Node state
    is_mining::Bool
    is_validating::Bool
    private_key::String
    
    # Model state
    model_weights::Dict{String,Any}
    
    function SaitoNode(port::Int=8000; bootstrap_nodes=SaitoNetwork.BOOTSTRAP_NODES)
        # Initialize network node
        network_node = SaitoNetwork.SaitoNode(port; bootstrap_nodes)
        
        # Initialize blockchain
        blockchain = SaitoBlockchain.Blockchain()
        
        # Generate a private key (in production, use proper cryptographic key generation)
        private_key = "privkey_$(randstring(16))"
        
        # Initialize with empty model weights
        model_weights = Dict{String,Any}()
        
        new(network_node, blockchain, false, false, private_key, model_weights)
    end
end

"""
    start_node(node::SaitoNode)

Start the SAITO node with both network and blockchain functionality.
"""
function start_node(node::SaitoNode)
    # Start the network layer
    SaitoNetwork.start_node(node.network_node)
    
    @info "SAITO node $(node.network_node.id) started on $(node.network_node.address):$(node.network_node.port)"
    
    # Start mining/validation if configured
    if node.is_mining
        @async begin
            while true
                try
                    mine_pending_transactions!(node)
                catch e
                    @error "Error in mining loop" exception=(e, catch_backtrace())
                end
                sleep(1)  # Prevent tight loop
            end
        end
    end
    
    # Start block validation if configured
    if node.is_validating
        @async begin
            while true
                try
                    validate_pending_blocks!(node)
                catch e
                    @error "Error in validation loop" exception=(e, catch_backtrace())
                end
                sleep(1)  # Prevent tight loop
            end
        end
    end
end

"""
    mine_pending_transactions!(node::SaitoNode)

Mine pending transactions into a new block if this node is the current validator.
"""
function mine_pending_transactions!(node::SaitoNode)
    # Check if we should be the next validator
    current_validator = SaitoBlockchain.select_validator(node.blockchain)
    
    if current_validator == node.network_node.id && !isempty(node.blockchain.pending_transactions)
        @info "Node $(node.network_node.id) is the current validator. Mining new block..."
        
        # Mine the block
        new_block = SaitoBlockchain.mine_block!(
            node.blockchain,
            node.network_node.id,
            node.private_key
        )
        
        if new_block !== nothing
            # Process rewards
            SaitoBlockchain.process_rewards!(node.blockchain, new_block)
            
            # Broadcast the new block to the network
            block_data = Dict(
                "index" => new_block.index,
                "timestamp" => new_block.timestamp,
                "transactions" => [t for t in new_block.transactions],
                "previous_hash" => new_block.previous_hash,
                "nonce" => new_block.nonce,
                "hash" => new_block.hash,
                "validator" => new_block.validator,
                "signature" => new_block.signature
            )
            
            message = SaitoNetwork.create_message(
                node.network_node,
                :new_block,
                block_data
            )
            
            SaitoNetwork.broadcast_message(node.network_node, message)
        end
    end
end

"""
    validate_pending_blocks!(node::SaitoNode)

Validate and process incoming blocks from the network.
"""
function validate_pending_blocks!(node::SaitoNode)
    # In a real implementation, this would process blocks from a queue
    # For now, we'll just validate our local chain
    if !SaitoBlockchain.validate_chain(node.blockchain)
        @error "Local blockchain validation failed!"
        # In a real implementation, we would request the correct chain from peers
    end
end

"""
    submit_transaction(node::SaitoNode, from::String, to::String, amount::Float64, fee::Float64)

Submit a new transaction to the network.
"""
function submit_transaction(node::SaitoNode, from::String, to::String, amount::Float64, fee::Float64)
    # Create and sign the transaction
    tx = SaitoBlockchain.Transaction(from, to, amount, fee, node.private_key)
    
    # Add to pending transactions
    if SaitoBlockchain.add_transaction!(node.blockchain, tx)
        # Broadcast to network
        tx_data = Dict(
            "from" => tx.from,
            "to" => tx.to,
            "amount" => tx.amount,
            "fee" => tx.fee,
            "timestamp" => tx.timestamp,
            "signature" => tx.signature,
            "nonce" => tx.nonce
        )
        
        message = SaitoNetwork.create_message(node.network_node, :new_transaction, tx_data)
        SaitoNetwork.broadcast_message(node.network_node, message)
        
        return true, "Transaction submitted"
    else
        return false, "Invalid transaction"
    end
end

"""
    update_model_weights!(node::SaitoNode, weights::Dict{String,Any}, reward::Float64)

Submit updated model weights to the network.
"""
function update_model_weights!(node::SaitoNode, weights::Dict{String,Any}, reward::Float64)
    # In a real implementation, this would:
    # 1. Create a special transaction with the model update
    # 2. Include a proof of work/stake
    # 3. Pay the required fee
    
    # For now, we'll just update the local weights and broadcast
    node.model_weights = weights
    
    # Create a model update message
    update_data = Dict(
        "weights" => weights,
        "reward" => reward,
        "timestamp" => time(),
        "node_id" => node.network_node.id
    )
    
    message = SaitoNetwork.create_message(node.network_node, :model_update, update_data)
    SaitoNetwork.broadcast_message(node.network_node, message)
    
    return true, "Model update submitted"
end

"""
    get_balance(node::SaitoNode, address::String)::Float64

Get the balance of a given address.
"""
get_balance(node::SaitoNode, address::String) = 
    SaitoBlockchain.get_balance(node.blockchain, address)

"""
    get_last_block(node::SaitoNode)

Get the last block in the blockchain.
"""
get_last_block(node::SaitoNode) = SaitoBlockchain.get_last_block(node.blockchain)

"""
    get_peers(node::SaitoNode)

Get the list of connected peers.
"""
get_peers(node::SaitoNode) = node.network_node.peers

export SaitoNode, start_node, submit_transaction, update_model_weights!,
       get_balance, get_last_block, get_peers

end # module NetworkBlockchain
