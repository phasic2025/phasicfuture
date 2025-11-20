module SAITOHNN

# Core functionality
export HyperbolicNN, train!, predict, HyperbolicEmbedding

# Geometry
export HyperbolicSpace, PoincareBall, LorentzModel, exp_map, log_map, mobius_add, mobius_mul, distance

# Economics
export compute_utility, compute_structural_cost, validate_transaction

# Network
export Node, Network, propagate, update_weights

# Blockchain
export Block, Transaction, validate_block, mine_block, add_transaction

# Core modules
include("core/HyperbolicNN.jl")

# Geometry modules
include("geometry/HyperbolicSpace.jl")
include("geometry/PoincareBall.jl")
include("geometry/LorentzModel.jl")

# Economics module
include("economics/EconomicLayer.jl")

# Network module
include("network/Network.jl")

# Blockchain module
include("blockchain/Blockchain.jl")

end # module
