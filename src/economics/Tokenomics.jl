"""
Tokenomics.jl

Implements the economic layer of the TFTS architecture, including:
- T_Geo (Utility/Work Unit) and T_Inv (Bonding/Governance Unit) tokens
- Transaction fee mechanisms
- Staking and bonding mechanisms
- Global Topological Tax system
"""
module Tokenomics

using ..HyperbolicGeometry
using ..SAITOCore
using DataStructures
using Random
using Printf

# Token types
export TGeo, TInv, Wallet, TokenSystem, GlobalTaxSystem

# Core operations
export create_wallet, transfer, stake, unstake, calculate_rewards, apply_tax

# Token types
struct TGeo  # Utility/Work Unit
    amount::Float64
    TGeo(amount::Real) = new(Float64(amount))
end

struct TInv  # Bonding/Governance Unit
    amount::Float64
    TInv(amount::Real) = new(Float64(amount))
end

Base.:(+)(a::TGeo, b::TGeo) = TGeo(a.amount + b.amount)
Base.:(-)(a::TGeo, b::TGeo) = TGeo(a.amount - b.amount)
Base.isless(a::TGeo, b::TGeo) = isless(a.amount, b.amount)

Base.:(+)(a::TInv, b::TInv) = TInv(a.amount + b.amount)
Base.:(-)(a::TInv, b::TInv) = TInv(a.amount - b.amount)
Base.isless(a::TInv, b::TInv) = isless(a.amount, b.amount)

"""
    Wallet

Represents a user's wallet containing TGeo and TInv tokens.
"""
mutable struct Wallet
    id::String
    tgeo::TGeo
    tinv::TInv
    staked_tgeo::TGeo
    staked_tinv::TInv
    last_update::Float64  # Timestamp of last update
    
    function Wallet(id::String)
        new(id, TGeo(0.0), TInv(0.0), TGeo(0.0), TInv(0.0), time())
    end
end

"""
    TokenSystem

Manages the token economy including wallets, staking, and transactions.
"""
mutable struct TokenSystem
    wallets::Dict{String, Wallet}
    total_supply_tgeo::TGeo
    total_supply_tinv::TInv
    staked_tgeo::TGeo
    staked_tinv::TInv
    inflation_rate::Float64  # Annual inflation rate for TGeo
    
    function TokenSystem(;inflation_rate=0.05)
        new(Dict{String, Wallet}(), 
            TGeo(0.0), TInv(0.0), 
            TGeo(0.0), TInv(0.0),
            inflation_rate)
    end
end

"""
    GlobalTaxSystem

Manages the global topological tax based on system chaos.
"""
mutable struct GlobalTaxSystem
    base_tax_rate::Float64
    chaos_factor::Float64  # 0 (no chaos) to 1 (maximum chaos)
    tax_history::Vector{Tuple{Float64, Float64}}  # (time, tax_rate)
    
    function GlobalTaxSystem(;base_tax_rate=0.01)
        new(base_tax_rate, 0.0, [])
    end
end

"""
    create_wallet(ts::TokenSystem, id::String; initial_tgeo=0.0, initial_tinv=0.0)

Create a new wallet with initial token balances.
"""
function create_wallet(ts::TokenSystem, id::String; initial_tgeo=0.0, initial_tinv=0.0)
    if haskey(ts.wallets, id)
        error("Wallet with id $id already exists")
    end
    wallet = Wallet(id)
    wallet.tgeo = TGeo(initial_tgeo)
    wallet.tinv = TInv(initial_tinv)
    ts.wallets[id] = wallet
    ts.total_supply_tgeo += TGeo(initial_tgeo)
    ts.total_supply_tinv += TInv(initial_tinv)
    return wallet
end

"""
    transfer(ts::TokenSystem, from_id::String, to_id::String, 
            amount::Union{TGeo,TInv}; fee_ratio=0.01)

Transfer tokens between wallets with a fee.
"""
function transfer(ts::TokenSystem, from_id::String, to_id::String, 
                 amount::Union{TGeo,TInv}; fee_ratio=0.01)
    # Verify wallets exist
    haskey(ts.wallets, from_id) || error("Sender wallet not found")
    haskey(ts.wallets, to_id) || error("Recipient wallet not found")
    
    from = ts.wallets[from_id]
    to = ts.wallets[to_id]
    
    # Calculate fee
    fee = typeof(amount) === TGeo ? 
          TGeo(amount.amount * fee_ratio) : 
          TInv(amount.amount * fee_ratio)
    
    total_amount = typeof(amount) === TGeo ? 
                  TGeo(amount.amount + fee.amount) : 
                  TInv(amount.amount + fee.amount)
    
    # Check balance
    if typeof(amount) === TGeo
        from.tgeo.amount < total_amount.amount && error("Insufficient TGeo balance")
        from.tgeo -= total_amount
        to.tgeo += amount
    else
        from.tinv.amount < total_amount.amount && error("Insufficient TInv balance")
        from.tinv -= total_amount
        to.tinv += amount
    end
    
    # Update last update time
    current_time = time()
    from.last_update = current_time
    to.last_update = current_time
    
    return fee
end

"""
    stake(ts::TokenSystem, wallet_id::String, tgeo::TGeo, tinv::TInv)

Stake tokens to participate in the network and earn rewards.
"""
function stake(ts::TokenSystem, wallet_id::String, tgeo::TGeo, tinv::TInv)
    wallet = get(ts.wallets, wallet_id, nothing)
    wallet === nothing && error("Wallet not found")
    
    if tgeo.amount > 0
        wallet.tgeo.amount < tgeo.amount && error("Insufficient TGeo balance")
        wallet.tgeo -= tgeo
        wallet.staked_tgeo += tgeo
        ts.staked_tgeo += tgeo
    end
    
    if tinv.amount > 0
        wallet.tinv.amount < tinv.amount && error("Insufficient TInv balance")
        wallet.tinv -= tinv
        wallet.staked_tinv += tinv
        ts.staked_tinv += tinv
    end
    
    wallet.last_update = time()
end

"""
    unstake(ts::TokenSystem, wallet_id::String, tgeo::TGeo, tinv::TInv)

Unstake tokens from the network.
"""
function unstake(ts::TokenSystem, wallet_id::String, tgeo::TGeo, tinv::TInv)
    wallet = get(ts.wallets, wallet_id, nothing)
    wallet === nothing && error("Wallet not found")
    
    if tgeo.amount > 0
        wallet.staked_tgeo.amount < tgeo.amount && error("Insufficient staked TGeo")
        wallet.staked_tgeo -= tgeo
        wallet.tgeo += tgeo
        ts.staked_tgeo -= tgeo
    end
    
    if tinv.amount > 0
        wallet.staked_tinv.amount < tinv.amount && error("Insufficient staked TInv")
        wallet.staked_tinv -= tinv
        wallet.tinv += tinv
        ts.staked_tinv -= tinv
    end
    
    wallet.last_update = time()
end

"""
    calculate_rewards(ts::TokenSystem, wallet_id::String)

Calculate staking rewards for a wallet.
"""
function calculate_rewards(ts::TokenSystem, wallet_id::String)
    wallet = get(ts.wallets, wallet_id, nothing)
    wallet === nothing && error("Wallet not found")
    
    current_time = time()
    time_elapsed = (current_time - wallet.last_update) / (365.25 * 24 * 3600)  # in years
    
    # Calculate TGeo rewards (inflation-based)
    tgeo_rewards = if wallet.staked_tgeo.amount > 0
        inflation = (1 + ts.inflation_rate)^time_elapsed - 1
        user_share = wallet.staked_tgeo.amount / (ts.staked_tgeo.amount + 1e-10)
        TGeo(ts.total_supply_tgeo.amount * inflation * user_share)
    else
        TGeo(0.0)
    end
    
    # Update wallet and total supply
    if tgeo_rewards.amount > 0
        wallet.tgeo += tgeo_rewards
        ts.total_supply_tgeo += tgeo_rewards
    end
    
    wallet.last_update = current_time
    
    return (tgeo=tgeo_rewards, tinv=TInv(0.0))  # TInv rewards would be calculated differently
end

"""
    update_chaos_factor(gts::GlobalTaxSystem, chaos_metric::Float64)

Update the chaos factor and calculate the current tax rate.
"""
function update_chaos_factor(gts::GlobalTaxSystem, chaos_metric::Float64)
    # Chaos metric should be between 0 and 1
    chaos_metric = clamp(chaos_metric, 0.0, 1.0)
    gts.chaos_factor = chaos_metric
    
    # Calculate tax rate (quadratic increase with chaos)
    tax_rate = gts.base_tax_rate * (1.0 + chaos_metric^2)
    
    # Record in history
    push!(gts.tax_history, (time(), tax_rate))
    
    # Keep only the last 1000 entries
    if length(gts.tax_history) > 1000
        gts.tax_history = gts.tax_history[end-999:end]
    end
    
    return tax_rate
end

"""
    apply_tax(ts::TokenSystem, gts::GlobalTaxSystem, wallet_id::String, amount::TGeo)

Apply the global topological tax to a transaction amount.
"""
function apply_tax(ts::TokenSystem, gts::GlobalTaxSystem, wallet_id::String, amount::TGeo)
    wallet = get(ts.wallets, wallet_id, nothing)
    wallet === nothing && error("Wallet not found")
    
    # Calculate tax based on current chaos factor
    tax_rate = gts.base_tax_rate * (1.0 + gts.chaos_factor^2)
    tax = TGeo(amount.amount * tax_rate)
    
    # Deduct tax from the wallet
    if wallet.tgeo.amount < tax.amount
        error("Insufficient balance to pay transaction tax")
    end
    
    wallet.tgeo -= tax
    
    # Burn a portion of the tax (deflationary pressure)
    burn_amount = TGeo(tax.amount * 0.5)  # Burn 50% of taxes
    ts.total_supply_tgeo -= burn_amount
    
    # The rest could be distributed to validators or other mechanisms
    distributed_amount = tax - burn_amount
    
    return (net_amount=TGeo(amount.amount - tax.amount), 
            tax_paid=tax, 
            burned=burn_amount,
            distributed=distributed_amount)
end

end # module Tokenomics
