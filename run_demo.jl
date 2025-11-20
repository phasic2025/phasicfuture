# First load core dependencies
include("src/geometry/SaitoHyperbolic.jl")

# Then load SaitoNetwork first since it's needed by SaitoBlockchain
include("src/network/SaitoNetwork.jl")
using .SaitoNetwork

# Then load SaitoBlockchain which depends on SaitoNetwork
include("src/blockchain/SaitoBlockchain.jl")
using .SaitoBlockchain

# Then load NetworkBlockchain which depends on both
include("src/network/NetworkBlockchain.jl")
using .NetworkBlockchain

# Then load SaitoHNN which depends on NetworkBlockchain
include("src/models/SaitoHNN.jl")
using .SaitoHNN

# Finally load and run the demo
include("demos/saito_hnn_training_demo.jl")
main()
