using Test
using LinearAlgebra
using Random
using ..SaitoHyperbolic

@testset "SaitoHyperbolic Tests" begin
    # Set random seed for reproducibility
    Random.seed!(42)
    
    # Test data
    c = SaitoHyperbolic.C_TARGET
    ϵ = SaitoHyperbolic.EPSILON
    
    # Test points in the Poincaré ball
    u = [0.1, 0.2, 0.3]
    v = [0.4, 0.1, 0.2]
    
    @testset "Distance Properties" begin
        # Distance to self is zero
        @test SaitoHyperbolic.stabilized_distance(u, u, c) ≈ 0 atol=1e-6
        
        # Symmetry
        @test SaitoHyperbolic.stabilized_distance(u, v, c) ≈ 
              SaitoHyperbolic.stabilized_distance(v, u, c) atol=1e-6
        
        # Triangle inequality (approximately, since we have a max distance)
        w = [0.2, 0.3, 0.1]
        d_uv = SaitoHyperbolic.stabilized_distance(u, v, c)
        d_vw = SaitoHyperbolic.stabilized_distance(v, w, c)
        d_uw = SaitoHyperbolic.stabilized_distance(u, w, c)
        @test d_uw <= d_uv + d_vw + ϵ
    end
    
    @testset "Exponential and Logarithmic Maps" begin
        # Round-trip test: exp ∘ log ≈ identity
        v_tangent = [0.1, 0.2, 0.1]
        v_exp = SaitoHyperbolic.stabilized_exp_map(u, v_tangent, c)
        v_log = SaitoHyperbolic.stabilized_log_map(u, v_exp, c)
        @test isapprox(v_tangent, v_log, atol=1e-6)
        
        # Test with zero vector
        zero_vec = zero(u)
        @test SaitoHyperbolic.stabilized_exp_map(u, zero_vec, c) ≈ u
        @test SaitoHyperbolic.stabilized_log_map(u, u, c) ≈ zero_vec atol=1e-6
    end
    
    @testset "Möbius Addition" begin
        # Identity element
        zero_vec = zero(u)
        @test SaitoHyperbolic.stabilized_mobius_add(u, zero_vec, c) ≈ u
        
        # Commutativity
        @test SaitoHyperbolic.stabilized_mobius_add(u, v, c) ≈ 
              SaitoHyperbolic.stabilized_mobius_add(v, u, c) atol=1e-6
        
        # Test that result stays in the Poincaré ball
        large_v = 0.99 * ones(3) / sqrt(3)
        result = SaitoHyperbolic.stabilized_mobius_add(u, large_v, c)
        @test norm(result) < 1.0
    end
    
    @testset "Hyperbolic Hebbian Update" begin
        # Test with small random weights and activities
        W = randn(3, 5) .* 0.1
        A_i = randn(3) .* 0.1
        A_j = randn(5) .* 0.1
        R = 0.5  # Utility
        η = 0.01 # Learning rate
        
        W_new = SaitoHyperbolic.hyperbolic_hebbian_update(W, A_i, A_j, R, η, c)
        
        # Check output shape
        @test size(W_new) == size(W)
        
        # Check that weights remain in the Poincaré ball
        @test all(norm(W_new[:,i]) < 1.0 for i in 1:size(W_new,2))
    end
    
    @testset "Numerical Stability" begin
        # Test with very small vectors
        tiny = ones(3) .* 1e-10
        @test SaitoHyperbolic.stabilized_distance(tiny, tiny, c) ≈ 0 atol=1e-6
        
        # Test with vectors near the boundary
        boundary = ones(3) .* (1 - 1e-6) / sqrt(3)
        @test norm(SaitoHyperbolic.stabilized_exp_map(boundary, [0.1, 0.0, 0.0], c)) < 1.0
    end
end

# Run the tests
@testset "SaitoHyperbolic Tests" begin
    include("saito_hyperbolic_tests.jl")
end
