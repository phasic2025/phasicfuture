using Test
using LinearAlgebra
using StaticArrays
using SparseArrays
using Phase1v2PhasicExperimentTopologyMapping.HyperbolicGeometry
using Phase1v2PhasicExperimentTopologyMapping.SAITOCore

@testset "HyperbolicGeometry Tests" begin
    @testset "HyperbolicPoint and HyperbolicVector" begin
        # Test creation and basic properties
        p1 = HyperbolicPoint([0.1, 0.2])
        p2 = HyperbolicPoint([0.3, 0.4])
        v1 = HyperbolicVector([0.1, 0.0])
        
        @test length(p1) == 2
        @test p1[1] ≈ 0.1
        @test p1[2] ≈ 0.2
        @test norm(p1) ≈ sqrt(0.1^2 + 0.2^2)
    end
    
    @testset "Möbius Addition" begin
        p1 = HyperbolicPoint([0.1, 0.0])
        p2 = HyperbolicPoint([0.2, 0.0])
        
        # Test identity element
        zero_p = HyperbolicPoint(zeros(2))
        @test mobius_add(p1, zero_p) ≈ p1
        
        # Test approximate associativity
        p3 = HyperbolicPoint([0.0, 0.1])
        left = mobius_add(mobius_add(p1, p2), p3)
        right = mobius_add(p1, mobius_add(p2, p3))
        @test hyperbolic_distance(left, right) < 1e-6
    end
    
    @testset "Hyperbolic Distance" begin
        # Test distance from origin
        p = HyperbolicPoint([0.5, 0.0])
        zero_p = HyperbolicPoint(zeros(2))
        d = hyperbolic_distance(zero_p, p)
        @test d ≈ (2/sqrt(c_target)) * atanh(sqrt(c_target)*0.5)
        
        # Test symmetry
        p1 = HyperbolicPoint([0.1, 0.2])
        p2 = HyperbolicPoint([0.3, -0.1])
        @test hyperbolic_distance(p1, p2) ≈ hyperbolic_distance(p2, p1)
    end
    
    @testset "Exponential and Logarithmic Maps" begin
        p = HyperbolicPoint([0.1, 0.2])
        v = HyperbolicVector([0.1, 0.0])
        
        # Test exp ∘ log = id
        q = exp_map(p, v)
        v_back = log_map(p, q)
        @test norm(v.coords - v_back.coords) < 1e-6
        
        # Test log ∘ exp = id
        v_new = exp_map(p, v)
        p_new = log_map(p, v_new)
        @test hyperbolic_distance(p, p_new) < 1e-6
    end
end

@testset "SAITOCore Tests" begin
    @testset "Graph Construction" begin
        graph = SAITOGraph(3)
        
        # Test adding nodes
        add_node!(graph, "A")
        add_node!(graph, "B")
        
        @test haskey(graph.nodes, "A")
        @test haskey(graph.nodes, "B")
        @test length(graph.nodes["A"].embedding) == 3
        
        # Test adding edges
        add_edge!(graph, "A", "B", 0.5)
        @test haskey(graph.edges, ("A", "B"))
        @test graph.edges[("A", "B")].weight ≈ 0.5
        @test "B" in graph.nodes["A"].neighbors
    end
    
    @testset "Geometric Constraints" begin
        graph = SAITOGraph(2)
        add_node!(graph, "A")
        add_node!(graph, "B")
        
        # Set specific embeddings for testing
        graph.nodes["A"].embedding = HyperbolicPoint([0.1, 0.0])
        graph.nodes["B"].embedding = HyperbolicPoint([0.9, 0.0])  # Far away
        
        # Test geometric cost increases with distance
        cost = calculate_geometric_cost(graph.nodes["A"].embedding, graph.nodes["B"].embedding)
        @test cost > 0
    end
    
    @testset "Hebbian Learning" begin
        graph = SAITOGraph(2)
        add_node!(graph, "A")
        add_node!(graph, "B")
        add_edge!(graph, "A", "B")
        
        # Initial distance
        d_before = hyperbolic_distance(graph.nodes["A"].embedding, graph.nodes["B"].embedding)
        
        # Apply learning
        update_node_embedding!(graph, "A")
        
        # Distance should decrease (nodes become more similar)
        d_after = hyperbolic_distance(graph.nodes["A"].embedding, graph.nodes["B"].embedding)
        @test d_after <= d_before || abs(d_after - d_before) < 1e-6
    end
end

@testset "Numerical Stability" begin
    @testset "Edge Cases" begin
        # Test near boundary
        p = HyperbolicPoint([0.999, 0.0])
        q = HyperbolicPoint([-0.999, 0.0])
        
        # Should not throw or return NaN/Inf
        d = hyperbolic_distance(p, q)
        @test isfinite(d)
        @test d > 0
        
        # Test very close points
        p = HyperbolicPoint([0.1, 0.2])
        q = HyperbolicPoint([0.1001, 0.2001])
        d = hyperbolic_distance(p, q)
        @test isfinite(d)
        @test d > 0
    end
end

println("All tests passed!")
