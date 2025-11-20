"""
Topology.jl

Manages the network topology and evolution in the TFTS architecture, including:
- Network growth and preferential attachment
- Topological constraints and validation
- Node and edge management in hyperbolic space
- Network health metrics
"""
module Topology

using ..HyperbolicGeometry
using ..SAITOCore
using ..Tokenomics
using DataStructures
using Random
using LinearAlgebra
using StatsBase

# Re-exports
export NetworkTopology, add_node!, remove_node!, add_edge!, remove_edge!, 
       evolve_network!, calculate_network_health, find_optimal_connections

"""
    NetworkTopology

Represents the network topology with hyperbolic embeddings and economic constraints.
"""
mutable struct NetworkTopology
    nodes::Dict{String, SAITONode}
    edges::Dict{Tuple{String,String}, SAITOEdge}
    embedding_dim::Int
    token_system::Tokenomics.TokenSystem
    tax_system::Tokenomics.GlobalTaxSystem
    
    # Network parameters
    target_avg_degree::Float64
    temperature::Float64  # Controls randomness in network evolution
    last_health_check::Float64
    
    function NetworkTopology(dim::Int; 
                           target_avg_degree=6.0, 
                           temperature=0.1,
                           initial_nodes=0)
        ts = Tokenomics.TokenSystem()
        gts = Tokenomics.GlobalTaxSystem()
        
        new(Dict{String, SAITONode}(),
            Dict{Tuple{String,String}, SAITOEdge}(),
            dim,
            ts,
            gts,
            target_avg_degree,
            temperature,
            time())
    end
end

"""
    add_node!(nt::NetworkTopology, node_id::String; initial_embedding=nothing)

Add a new node to the network with an optional initial embedding.
"""
function add_node!(nt::NetworkTopology, node_id::String; initial_embedding=nothing)
    if haskey(nt.nodes, node_id)
        @warn "Node $node_id already exists"
        return nothing
    end
    
    # Create wallet for the node
    Tokenomics.create_wallet(nt.token_system, node_id, 
                           initial_tgeo=100.0,  # Initial stake
                           initial_tinv=10.0)   # Governance tokens
    
    # Create node with either provided or random embedding
    if initial_embedding !== nothing
        node = SAITONode(node_id, HyperbolicPoint(initial_embedding))
    else
        # Sample from hyperbolic space with uniform angular distribution
        # and radial distribution favoring the center
        θ = 2π * rand()
        r = atanh(rand())  # Concentrated near the center
        x = r * cos(θ)
        y = r * sin(θ)
        
        if nt.embedding_dim == 3
            ϕ = acos(2*rand() - 1)  # For 3D, add another angle
            x *= sin(ϕ)
            y *= sin(ϕ)
            z = r * cos(ϕ)
            node = SAITONode(node_id, HyperbolicPoint([x, y, z]))
        else
            node = SAITONode(node_id, HyperbolicPoint([x, y]))
        end
    end
    
    nt.nodes[node_id] = node
    return node
end

"""
    remove_node!(nt::NetworkTopology, node_id::String)

Remove a node and all its connections from the network.
"""
function remove_node!(nt::NetworkTopology, node_id::String)
    if !haskey(nt.nodes, node_id)
        @warn "Node $node_id not found"
        return nothing
    end
    
    # Remove all edges connected to this node
    neighbors = copy(nt.nodes[node_id].neighbors)
    for neighbor_id in neighbors
        remove_edge!(nt, node_id, neighbor_id)
    end
    
    # Remove the node
    delete!(nt.nodes, node_id)
    
    # Note: In a real implementation, you might want to handle the node's wallet and tokens
    
    return nothing
end

"""
    add_edge!(nt::NetworkTopology, source_id::String, target_id::String; 
             weight=1.0, distance=nothing)

Add an edge between two nodes with an optional weight and distance.
"""
function add_edge!(nt::NetworkTopology, source_id::String, target_id::String; 
                  weight=1.0, distance=nothing)
    # Check if nodes exist
    haskey(nt.nodes, source_id) || error("Source node $source_id not found")
    haskey(nt.nodes, target_id) || error("Target node $target_id not found")
    
    # Ensure consistent ordering of node IDs for undirected edges
    if source_id > target_id
        source_id, target_id = target_id, source_id
    end
    
    # Calculate hyperbolic distance if not provided
    if distance === nothing
        source = nt.nodes[source_id]
        target = nt.nodes[target_id]
        distance = hyperbolic_distance(source.embedding, target.embedding)
    end
    
    # Create and add the edge
    edge = SAITOEdge(source_id, target_id, weight, distance)
    nt.edges[(source_id, target_id)] = edge
    
    # Update node neighbor lists
    if !(target_id in nt.nodes[source_id].neighbors)
        push!(nt.nodes[source_id].neighbors, target_id)
    end
    
    if !(source_id in nt.nodes[target_id].neighbors) && source_id != target_id
        push!(nt.nodes[target_id].neighbors, source_id)
    end
    
    return edge
end

"""
    remove_edge!(nt::NetworkTopology, source_id::String, target_id::String)

Remove an edge between two nodes.
"""
function remove_edge!(nt::NetworkTopology, source_id::String, target_id::String)
    # Ensure consistent ordering
    if source_id > target_id
        source_id, target_id = target_id, source_id
    end
    
    edge_key = (source_id, target_id)
    if !haskey(nt.edges, edge_key)
        @warn "Edge $edge_key not found"
        return nothing
    end
    
    # Remove the edge
    delete!(nt.edges, edge_key)
    
    # Update neighbor lists
    if haskey(nt.nodes, source_id) && target_id in nt.nodes[source_id].neighbors
        filter!(x -> x != target_id, nt.nodes[source_id].neighbors)
    end
    
    if haskey(nt.nodes, target_id) && source_id in nt.nodes[target_id].neighbors
        filter!(x -> x != source_id, nt.nodes[target_id].neighbors)
    end
    
    return nothing
end

"""
    find_optimal_connections(nt::NetworkTopology, new_node_id::String, k::Int=3)

Find the k best nodes to connect to for a new node using hyperbolic distances.
"""
function find_optimal_connections(nt::NetworkTopology, new_node_id::String, k::Int=3)
    new_node = nt.nodes[new_node_id]
    candidates = []
    
    for (id, node) in nt.nodes
        id == new_node_id && continue  # Skip self
        
        # Calculate hyperbolic distance
        dist = hyperbolic_distance(new_node.embedding, node.embedding)
        
        # Calculate connection score based on distance and node degree
        degree = length(node.neighbors)
        score = exp(-dist) * (degree + 1)  # Prefer closer and higher-degree nodes
        
        push!(candidates, (id, score, dist))
    end
    
    # Sort by score (descending) and take top k
    sort!(candidates, by=x->x[2], rev=true)
    return candidates[1:min(k, length(candidates))]
end

"""
    calculate_network_health(nt::NetworkTopology)

Calculate various health metrics for the network.
"""
function calculate_network_health(nt::NetworkTopology)
    if isempty(nt.nodes)
        return (avg_degree=0.0, 
                clustering=0.0, 
                avg_path_length=0.0,
                efficiency=0.0,
                robustness=0.0)
    end
    
    # Calculate average degree
    degrees = [length(node.neighbors) for (_, node) in nt.nodes]
    avg_degree = mean(degrees)
    
    # Calculate clustering coefficient
    clustering_coeffs = Float64[]
    for (_, node) in nt.nodes
        neighbors = node.neighbors
        k = length(neighbors)
        if k < 2
            push!(clustering_coeffs, 0.0)
            continue
        end
        
        # Count edges between neighbors
        edge_count = 0
        for i in 1:k
            for j in i+1:k
                n1 = neighbors[i]
                n2 = neighbors[j]
                if (n1 < n2 && haskey(nt.edges, (n1, n2))) || 
                   (n2 < n1 && haskey(nt.edges, (n2, n1)))
                    edge_count += 1
                end
            end
        end
        
        push!(clustering_coeffs, 2 * edge_count / (k * (k - 1)))
    end
    
    avg_clustering = mean(clustering_coeffs)
    
    # Calculate average path length (approximate for large networks)
    # For large networks, we might want to sample node pairs
    n_samples = min(100, length(nt.nodes))
    sampled_nodes = sample(collect(keys(nt.nodes)), n_samples, replace=false)
    
    path_lengths = Float64[]
    for i in 1:n_samples
        for j in i+1:n_samples
            # Simple BFS to find shortest path
            dist = bfs_shortest_path(nt, sampled_nodes[i], sampled_nodes[j])
            if dist > 0
                push!(path_lengths, dist)
            end
        end
    end
    
    avg_path_length = isempty(path_lengths) ? 0.0 : mean(path_lengths)
    
    # Network efficiency (inverse of harmonic mean of shortest paths)
    efficiency = isempty(path_lengths) ? 0.0 : 1.0 / mean(1.0 ./ path_lengths)
    
    # Simple robustness metric (fraction of nodes in largest connected component)
    robustness = largest_component_size(nt) / length(nt.nodes)
    
    return (avg_degree=avg_degree, 
            clustering=avg_clustering, 
            avg_path_length=avg_path_length,
            efficiency=efficiency,
            robustness=robustness)
end

# Helper function for BFS shortest path
function bfs_shortest_path(nt::NetworkTopology, source_id::String, target_id::String)
    source_id == target_id && return 0
    
    visited = Set{String}([source_id])
    queue = [(source_id, 0)]
    
    while !isempty(queue)
        current, dist = popfirst!(queue)
        
        for neighbor in nt.nodes[current].neighbors
            neighbor == target_id && return dist + 1
            
            if !(neighbor in visited)
                push!(visited, neighbor)
                push!(queue, (neighbor, dist + 1))
            end
        end
    end
    
    return -1  # No path found
end

# Helper function to find the size of the largest connected component
function largest_component_size(nt::NetworkTopology)
    visited = Set{String}()
    max_size = 0
    
    for node_id in keys(nt.nodes)
        if !(node_id in visited)
            # BFS to find component
            component_size = 0
            queue = [node_id]
            push!(visited, node_id)
            
            while !isempty(queue)
                current = popfirst!(queue)
                component_size += 1
                
                for neighbor in nt.nodes[current].neighbors
                    if !(neighbor in visited)
                        push!(visited, neighbor)
                        push!(queue, neighbor)
                    end
                end
            end
            
            max_size = max(max_size, component_size)
        end
    end
    
    return max_size
end

"""
    evolve_network!(nt::NetworkTopology; 
                   growth_rate=0.1, 
                   rewiring_prob=0.1,
                   max_connections=5)

Evolve the network structure based on hyperbolic geometry and economic incentives.
"""
function evolve_network!(nt::NetworkTopology; 
                        growth_rate=0.1, 
                        rewiring_prob=0.1,
                        max_connections=5)
    current_time = time()
    
    # Add new nodes (network growth)
    n_new_nodes = max(1, floor(Int, length(nt.nodes) * growth_rate))
    for _ in 1:n_new_nodes
        node_id = "node_$(length(nt.nodes) + 1)"
        add_node!(nt, node_id)
        
        # Connect to existing nodes
        if !isempty(nt.nodes) > 1
            candidates = find_optimal_connections(nt, node_id, max_connections)
            
            for (target_id, score, dist) in candidates
                # Add edge with probability based on score and temperature
                p_connect = score / (score + exp(nt.temperature))
                if rand() < p_connect
                    add_edge!(nt, node_id, target_id, distance=dist)
                end
            end
        end
    end
    
    # Rewire existing connections with some probability
    if rand() < rewiring_prob && length(nt.nodes) > 2
        # Select a random edge to rewire
        edge_keys = collect(keys(nt.edges))
        if !isempty(edge_keys)
            src, dst = rand(edge_keys)
            
            # Remove the edge
            remove_edge!(nt, src, dst)
            
            # Find a new connection for the source node
            candidates = []
            for (id, node) in nt.nodes
                id == src && continue
                id in nt.nodes[src].neighbors && continue
                
                dist = hyperbolic_distance(nt.nodes[src].embedding, node.embedding)
                score = exp(-dist) * (length(node.neighbors) + 1)
                push!(candidates, (id, score, dist))
            end
            
            # Connect to the best candidate
            if !isempty(candidates)
                sort!(candidates, by=x->x[2], rev=true)
                new_dst = candidates[1][1]
                add_edge!(nt, src, new_dst, distance=candidates[1][3])
            else
                # If no good candidates, restore the original edge
                add_edge!(nt, src, dst, distance=hyperbolic_distance(
                    nt.nodes[src].embedding, nt.nodes[dst].embedding))
            end
        end
    end
    
    # Update network health and adjust parameters
    health = calculate_network_health(nt)
    
    # Adjust temperature based on network health
    # If clustering is too low, decrease temperature to prefer local connections
    # If path length is too high, increase temperature to allow longer connections
    nt.temperature = clamp(nt.temperature * 
                          (1.0 + 0.1 * (health.avg_path_length - log(length(nt.nodes))/2)) *
                          (1.0 - 0.05 * (health.clustering - 0.5)), 
                          0.01, 1.0)
    
    # Update the global tax based on network health
    # Higher clustering and lower path length indicate lower chaos
    chaos_metric = 1.0 - (health.clustering * 0.5 + 0.5 / (1 + health.avg_path_length))
    Tokenomics.update_chaos_factor(nt.tax_system, chaos_metric)
    
    return health
end

end # module Topology
