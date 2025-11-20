using Test
using LinearAlgebra
using Random
using StaticArrays
using ForwardDiff

# Include the module to be tested
include("../src/geometry/hyperbolic.jl")
using .Hyperbolic

@testset "Hyperbolic Geometry Tests" begin
    # Set up test parameters
    rng = MersenneTwister(42)
    c = 1.0  # Test with unit curvature for simplicity
    dim = 3
    n_points = 10
    
    @testset "Basic Properties" begin
        # Test that distance from a point to itself is zero
        x = randn(rng, dim)
        x = x / (norm(x) * 2)  # Ensure it's within the unit ball
        @test isapprox(distance(x, x, c), 0.0, atol=1e-6)
        
        # Test that exp and log are inverses
        v = 0.1 * randn(rng, dim)
        y = exp_map(x, v, c)
        v_recovered = log_map(x, y, c)
        @test isapprox(v, v_recovered, rtol=1e-6, atol=1e-6)
    end
    
    @testset "HyperbolicEmbedding" begin
        # Test initialization
        emb = init_embedding(n_points, dim, c, rng=rng)
        @test size(emb.embedding) == (dim, n_points)
        @test emb.curvature == c
        
        # Test all points are within the Poincaré ball
        for i in 1:n_points
            @test norm(emb.embedding[:,i]) < 1.0/sqrt(c)
        end
    end
    
    @testset "Möbius Addition" begin
        x = [0.1, 0.2, 0.3]
        y = [0.4, -0.1, 0.2]
        
        # Test identity element
        zero_vec = zeros(length(x))
        @test isapprox(mobius_add(x, zero_vec, c), x, atol=1e-6)
        
        # Test non-commutativity
        @test !isapprox(mobius_add(x, y, c), mobius_add(y, x, c), atol=1e-6)
    end
    
    @testset "Parallel Transport" begin
        x = [0.1, 0.2, 0.3]
        y = [-0.2, 0.1, 0.4]
        v = [0.5, -0.3, 0.2]
        
        # Transport from x to y and back
        v_transported = parallel_transport(x, y, v, c)
        v_back = parallel_transport(y, x, v_transported, c)
        
        # Should approximately recover the original vector
        @test isapprox(v, v_back, rtol=1e-2, atol=1e-2)
    end
    
    @testset "Projection to Manifold" begin
        # Test points inside the ball remain unchanged
        x = [0.1, 0.2, 0.3]
        @test project_to_manifold(x, c) == x
        
        # Test points outside the ball are projected in
        x_outside = [1.0, 1.0, 1.0]  # Outside the unit ball
        x_projected = project_to_manifold(x_outside, c)
        @test norm(x_projected) < 1.0/sqrt(c)
        @test isapprox(norm(x_projected), 1.0/sqrt(c) - Hyperbolic.EPSILON, atol=1e-6)
    end
    
    @testset "Numerical Stability" begin
        # Test with very small vectors
        x = [1e-8, 1e-8, 1e-8]
        v = [1e-8, -1e-8, 0.0]
        
        # Should not throw any errors
        y = exp_map(x, v, c)
        v_recovered = log_map(x, y, c)
        @test isapprox(v, v_recovered, rtol=1e-2, atol=1e-8)
    end
end

@testset "Gradient Tests" begin
    using ForwardDiff
    
    # Test gradient of distance function
    x = [0.1, 0.2, 0.3]
    y = [-0.1, 0.3, 0.2]
    
    # Define a function for testing gradients
    function test_grad(f, x, y, c)
        # Compute gradient using ForwardDiff
        g = ForwardDiff.gradient(x -> f(x, y, c), x)
        
        # Compute numerical gradient
        h = 1e-6
        g_num = similar(x)
        for i in eachindex(x)
            x_plus = copy(x)
            x_plus[i] += h
            x_minus = copy(x)
            x_minus[i] -= h
            g_num[i] = (f(x_plus, y, c) - f(x_minus, y, c)) / (2h)
        end
        
        # Compare
        return isapprox(g, g_num, rtol=1e-4, atol=1e-6)
    end
    
    # Test gradient of distance squared
    dist_sq(x, y, c) = distance(x, y, c)^2
    @test test_grad(dist_sq, x, y, 1.0)
    
    # Test with different curvature
    @test test_grad(dist_sq, x, y, 2.0)
end
