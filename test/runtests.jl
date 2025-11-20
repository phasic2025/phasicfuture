using Test
using LinearAlgebra
using StaticArrays
using SaitoHNN

@testset "SaitoHNN Tests" begin
    @testset "HyperbolicPoint and HyperbolicVector" begin
        # Test creation and basic properties
        p1 = HyperbolicPoint([0.1, 0.2])
        p2 = HyperbolicPoint([0.3, 0.4])
        v1 = HyperbolicVector([0.1, 0.2])
        
        @test length(p1) == 2
        @test p1[1] ≈ 0.1
        @test p1[2] ≈ 0.2
        @test p1.coords isa SVector{2, Float64}
        
        # Test conversion and promotion
        w = HyperbolicWeight(0.5)
        @test Float64(w) ≈ 0.5
        @test 2.0 * w ≈ 1.0
    end
    
    @testset "Hyperbolic Distance" begin
        # Test distance from origin
        p0 = HyperbolicPoint([0.0, 0.0])
        p1 = HyperbolicPoint([0.1, 0.0])
        @test hyperbolic_distance(p0, p1) ≈ atanh(√c_target * 0.1) / √c_target
        
        # Test symmetry
        p2 = HyperbolicPoint([0.2, 0.3])
        @test hyperbolic_distance(p1, p2) ≈ hyperbolic_distance(p2, p1)
        
        # Test triangle inequality
        p3 = HyperbolicPoint([-0.1, 0.2])
        d12 = hyperbolic_distance(p1, p2)
        d23 = hyperbolic_distance(p2, p3)
        d13 = hyperbolic_distance(p1, p3)
        @test d13 ≤ d12 + d23 + 1e-10  # Allow for small numerical errors
    end
    
    @testset "Möbius Operations" begin
        p1 = HyperbolicPoint([0.1, 0.0])
        p2 = HyperbolicPoint([0.0, 0.1])
        
        # Test Möbius addition
        p_sum = mobius_add(p1, p2)
        @test p_sum isa HyperbolicPoint{2}
        
        # Test Möbius subtraction (inverse of addition)
        p_diff = mobius_sub(p_sum, p2)
        @test isapprox(p_diff.coords, p1.coords, atol=1e-10)
    end
    
    @testset "Exponential and Logarithmic Maps" begin
        base = HyperbolicPoint([0.1, 0.1])
        point = HyperbolicPoint([0.2, 0.3])
        
        # Test round-trip mapping
        tangent_vec = log_map(base, point)
        @test tangent_vec isa HyperbolicVector{2}
        
        reconstructed_point = exp_map(base, tangent_vec)
        @test isapprox(reconstructed_point.coords, point.coords, atol=1e-10)
    end
    
    @testset "HyperbolicLayer" begin
        # Test layer initialization
        layer = HyperbolicLayer(3, 2)
        @test size(layer.weights) == (2, 3)
        @test length(layer.bias) == 2
        
        # Test forward pass
        input = HyperbolicPoint([0.1, 0.2, 0.3])
        output = forward(layer, input)
        @test output isa HyperbolicPoint{2}
        
        # Check output is still in the Poincaré ball
        @test check_geometric_constraint(output)
    end
    
    @testset "Constraints" begin
        # Test geometric constraint
        p_good = HyperbolicPoint([0.1, 0.1])
        p_bad = HyperbolicPoint([1.0, 0.0])  # Too close to boundary
        @test check_geometric_constraint(p_good)
        @test !check_geometric_constraint(p_bad)
        
        # Test dynamic constraint
        p1 = HyperbolicPoint([0.1, 0.0])
        p2 = HyperbolicPoint([0.2, 0.0])
        d_max = hyperbolic_distance(p1, p2) * 1.1  # Slightly larger than actual distance
        @test check_dynamic_constraint(p1, p2, d_max)
        @test !check_dynamic_constraint(p1, p2, d_max * 0.9)  # Slightly smaller than actual distance
        
        # Test topological constraint (placeholder)
        points = [HyperbolicPoint([0.1, 0.1]), HyperbolicPoint([-0.1, 0.1])]
        is_valid, cost = check_topological_constraint(points)
        @test is_valid
        @test cost == 0.0  # Placeholder implementation
    end
    
    @testset "Hyperbolic Hebbian Update" begin
        weight = HyperbolicWeight(0.1)
        input = HyperbolicPoint([0.1, 0.2])
        output = HyperbolicPoint([0.3, 0.4])
        
        new_weight = hyperbolic_hebbian_update(weight, input, output, 0.1)
        @test new_weight isa HyperbolicWeight
        @test new_weight.value != weight.value  # Weight should change
        
        # Test weight constraints are maintained
        @test abs(new_weight.value) < 1/√c_target - ϵ
    end
end

println("All tests passed!")
