using Test
using LinearAlgebra
using StaticArrays
using Random
using ForwardDiff

# Include the module to be tested
include("../src/core/HyperbolicGeometry.jl")
using .HyperbolicGeometry

@testset "TFTS Hyperbolic Geometry Tests" begin
    rng = MersenneTwister(42)  # For reproducible tests
    
    # Test points and vectors
    p1 = HyperbolicPoint([0.1, 0.2, 0.3])
    p2 = HyperbolicPoint([-0.1, 0.3, 0.2])
    v1 = HyperbolicVector([0.1, -0.1, 0.05])
    v2 = HyperbolicVector([-0.05, 0.1, -0.1])
    
    @testset "Basic Operations" begin
        # Test basic vector operations
        @test size(p1) == (3,)
        @test p1[1] == 0.1
        @test p1 + p1 ≈ HyperbolicPoint([0.2, 0.4, 0.6])
        @test zero(HyperbolicPoint{3}) == HyperbolicPoint(zeros(3))
    end
    
    @testset "Möbius Addition" begin
        # Test identity
        zero_p = zero(p1)
        @test mobius_add(p1, zero_p) ≈ p1
        
        # Test inverse
        p1_neg = HyperbolicPoint(-p1.coords)
        @test norm(mobius_add(p1, p1_neg).coords) < 1e-8
        
        # Test associativity (approximately)
        p3 = HyperbolicPoint(randn(rng, 3) .* 0.1)
        left = mobius_add(mobius_add(p1, p2), p3)
        right = mobius_add(p1, mobius_add(p2, p3))
        @test left ≈ right atol=1e-6
    end
    
    @testset "Hyperbolic Distance" begin
        # Distance from point to itself is zero
        @test hyperbolic_distance(p1, p1) ≈ 0.0 atol=1e-8
        
        # Symmetry
        @test hyperbolic_distance(p1, p2) ≈ hyperbolic_distance(p2, p1)
        
        # Triangle inequality
        p3 = HyperbolicPoint(randn(rng, 3) .* 0.1)
        d12 = hyperbolic_distance(p1, p2)
        d23 = hyperbolic_distance(p2, p3)
        d13 = hyperbolic_distance(p1, p3)
        @test d13 ≤ d12 + d23 + 1e-8  # Allow for small numerical errors
    end
    
    @testset "Exponential and Logarithmic Maps" begin
        # Test exp and log are inverses
        v = HyperbolicVector(randn(rng, 3) .* 0.1)
        q = exp_map(p1, v)
        v_recovered = log_map(p1, q)
        @test v ≈ v_recovered atol=1e-6
        
        # Test exp_map with zero vector
        zero_v = zero(v)
        @test exp_map(p1, zero_v) ≈ p1
    end
    
    @testset "Parallel Transport" begin
        # Transporting along zero distance should return the same vector
        @test parallel_transport(v1, p1, p1) ≈ v1
        
        # Transporting a zero vector should return zero
        zero_v = zero(v1)
        @test parallel_transport(zero_v, p1, p2) ≈ zero_v
        
        # Test that parallel transport preserves angles
        angle_before = acos(dot(v1.coords, v2.coords) / (norm(v1) * norm(v2)))
        v1_transported = parallel_transport(v1, p1, p2)
        v2_transported = parallel_transport(v2, p1, p2)
        angle_after = acos(dot(v1_transported.coords, v2_transported.coords) / 
                          (norm(v1_transported) * norm(v2_transported)))
        @test angle_before ≈ angle_after atol=1e-6
    end
    
    @testset "Geodesics" begin
        # Start and end points
        @test geodesic(p1, p2, 0.0) ≈ p1
        @test geodesic(p1, p2, 1.0) ≈ p2
        
        # Midpoint should be equidistant
        mid = geodesic(p1, p2, 0.5)
        d1 = hyperbolic_distance(p1, mid)
        d2 = hyperbolic_distance(mid, p2)
        @test abs(d1 - d2) / (d1 + d2) < 0.01  # Within 1% relative error
    end
    
    @testset "Figure-Eight Knot" begin
        # Test periodicity
        t1 = 0.0
        t2 = 2π
        p1 = figure_eight_knot(t1)
        p2 = figure_eight_knot(t2)
        @test p1 ≈ p2 atol=1e-10
        
        # Test parameter range
        @test all(isfinite, figure_eight_knot(π/2))
    end
    
    @testset "Curvature Tensor" begin
        # Test skew-symmetry: R(u,v)w = -R(v,u)w
        w = HyperbolicVector(randn(rng, 3) .* 0.1)
        R1 = curvature_tensor(p1, v1, v2, w)
        R2 = curvature_tensor(p1, v2, v1, w)
        @test R1 ≈ -R2 atol=1e-10
        
        # Test first Bianchi identity: R(u,v)w + R(v,w)u + R(w,u)v ≈ 0
        u, v, w = v1, v2, HyperbolicVector(randn(rng, 3) .* 0.1)
        R_uv_w = curvature_tensor(p1, u, v, w)
        R_vw_u = curvature_tensor(p1, v, w, u)
        R_wu_v = curvature_tensor(p1, w, u, v)
        @test R_uv_w + R_vw_u + R_wu_v ≈ zeros(3) atol=1e-10
    end
    
    @testset "Exponential Map Derivative" begin
        # Test with zero vector (should be identity)
        zero_v = zero(v1)
        w = HyperbolicVector(randn(rng, 3) .* 0.1)
        @test exponential_map_derivative(p1, zero_v, w) ≈ w
        
        # Test linearity in w
        a, b = randn(rng, 2)
        w1 = HyperbolicVector(randn(rng, 3) .* 0.1)
        w2 = HyperbolicVector(randn(rng, 3) .* 0.1)
        left = exponential_map_derivative(p1, v1, a*w1 + b*w2)
        right = a*exponential_map_derivative(p1, v1, w1) + b*exponential_map_derivative(p1, v1, w2)
        @test left ≈ right atol=1e-6
    end
    
    @testset "Model Conversions" begin
        # Test round-trip conversion
        p_euc = randn(rng, 3) .* 0.1
        p_hyp = project_to_hyperboloid(p_euc)
        p_poincare = hyperboloid_to_poincare(p_hyp)
        # Should be approximately equal after normalization
        @test p_poincare ≈ p_euc ./ (1 + sqrt(1 + norm(p_euc)^2)) atol=1e-6
    end
end

@testset "TFTS Hyperbolic Geometry Performance Tests" begin
    # Test performance of key operations
    points = [HyperbolicPoint(randn(3) .* 0.1) for _ in 1:1000]
    vectors = [HyperbolicVector(randn(3) .* 0.1) for _ in 1:1000]
    
    @testset "Bulk Operations" begin
        # Test parallel transport in bulk
        @test length(parallel_transport.(vectors, Ref(points[1]), Ref(points[2]))) == 1000
        
        # Test distance matrix computation
        dists = [hyperbolic_distance(p1, p2) for p1 in points[1:10], p2 in points[1:10]]
        @test size(dists) == (10, 10)
        @test all(≥(0), dists)
    end
end
