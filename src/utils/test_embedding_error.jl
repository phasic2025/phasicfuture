#!/usr/bin/env julia
# Test script to reproduce the MethodError

using LinearAlgebra
using Random

# Copy the exact struct definition
mutable struct ConceptEmbedding
    embeddings::Dict{String, Vector{Float64}}
    embedding_dim::Int
    concept_memory::Dict{String, Dict}
    word_to_goal_associations::Dict{String, Dict{Symbol, Float64}}
    action_verbs::Dict{String, Symbol}
end

function create_concept_embedding(dim::Int = 64)
    action_verbs = Dict{String, Symbol}(
        "make" => :design, "create" => :design, "build" => :design, "design" => :design,
        "learn" => :learn, "understand" => :learn, "study" => :learn, "research" => :learn,
        "integrate" => :integrate, "combine" => :integrate, "assemble" => :integrate
    )
    
    return ConceptEmbedding(
        Dict{String, Vector{Float64}}(), 
        dim, 
        Dict{String, Dict}(),
        Dict{String, Dict{Symbol, Float64}}(),
        action_verbs
    )
end

# Copy the exact function definition
function get_or_create_embedding(ce::ConceptEmbedding, concept::String, current_time::Float64, boundary_concepts::Union{Set{String}, Nothing})::Vector{Float64}
    println("DEBUG: get_or_create_embedding called with:")
    println("  ce type: ", typeof(ce))
    println("  concept: ", concept)
    println("  current_time: ", current_time)
    println("  boundary_concepts type: ", typeof(boundary_concepts))
    println("  boundary_concepts value: ", boundary_concepts)
    
    concept_lower = lowercase(concept)
    if haskey(ce.embeddings, concept_lower)
        return ce.embeddings[concept_lower]
    end
    
    # Create new embedding
    new_embedding = randn(ce.embedding_dim)
    new_embedding = new_embedding / norm(new_embedding)
    
    # Orthogonalize
    embeddings_to_check = if boundary_concepts !== nothing
        [(c, e) for (c, e) in ce.embeddings if c in boundary_concepts]
    else
        collect(ce.embeddings)
    end
    
    for (existing_concept, existing_emb) in embeddings_to_check
        overlap = dot(new_embedding, existing_emb)
        new_embedding = new_embedding - overlap * existing_emb
    end
    
    if norm(new_embedding) > 1e-10
        new_embedding = new_embedding / norm(new_embedding)
    else
        new_embedding = randn(ce.embedding_dim)
        new_embedding = new_embedding / norm(new_embedding)
    end
    
    ce.embeddings[concept_lower] = new_embedding
    ce.concept_memory[concept_lower] = Dict("created_at" => current_time, "usage_count" => 0)
    return new_embedding
end

# Test the exact call pattern from the error
println("=" ^ 70)
println("TEST 1: Direct call with nothing")
println("=" ^ 70)
try
    ce = create_concept_embedding(64)
    emb = get_or_create_embedding(ce, "make", 0.0, nothing)
    println("✓ SUCCESS: Direct call works")
catch e
    println("✗ FAILED: ", e)
    println("Stacktrace:")
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
end

println("\n" * "=" ^ 70)
println("TEST 2: Call with typed variable")
println("=" ^ 70)
try
    ce = create_concept_embedding(64)
    boundary_concepts::Union{Set{String}, Nothing} = nothing
    emb = get_or_create_embedding(ce, "make", 0.0, boundary_concepts)
    println("✓ SUCCESS: Typed variable works")
catch e
    println("✗ FAILED: ", e)
    println("Stacktrace:")
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
end

println("\n" * "=" ^ 70)
println("TEST 3: Check method table")
println("=" ^ 70)
println("Methods for get_or_create_embedding:")
for m in methods(get_or_create_embedding)
    println("  ", m)
end

println("\n" * "=" ^ 70)
println("TEST 4: Check if method exists for signature")
println("=" ^ 70)
ce = create_concept_embedding(64)
println("Checking method dispatch...")
println("  ce type: ", typeof(ce))
println("  \"make\" type: ", typeof("make"))
println("  0.0 type: ", typeof(0.0))
println("  nothing type: ", typeof(nothing))
println("  nothing <: Union{Set{String}, Nothing}? ", typeof(nothing) <: Union{Set{String}, Nothing})

# Try to find the method manually
sig = Tuple{ConceptEmbedding, String, Float64, Union{Set{String}, Nothing}}
println("  Looking for signature: ", sig)
println("  Method exists? ", hasmethod(get_or_create_embedding, sig))

