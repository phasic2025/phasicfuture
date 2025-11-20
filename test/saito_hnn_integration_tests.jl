using Test
using Random
using Flux
using BSON
using ..SaitoHNN
using ..NetworkBlockchain

@testset "SaitoHNN Integration Tests" begin
    # Set random seed for reproducibility
    Random.seed!(42)
    
    @testset "Hyperbolic Layer" begin
        # Test HyperbolicLayer construction
        layer = SaitoHNN.HyperbolicLayer(10, 5, relu)
        
        @test size(layer.W) == (5, 10)  # out_dim × in_dim
        @test size(layer.b) == (5,)     # out_dim
        @test layer.activation == relu
        
        # Test forward pass
        x = randn(Float32, 10, 32)  # 10 features, batch size 32
        y = layer(x)
        
        @test size(y) == (5, 32)  # Should preserve batch dimension
        @test all(isfinite.(y))   # Should produce finite outputs
    end
    
    @testset "SaitoHNN Model" begin
        # Test model construction
        model = SaitoHNN.SaitoHNN([10, 20, 5], [relu, identity])
        
        @test length(model.layers) == 2
        @test model.layers[1].in_dim == 10
        @test model.layers[1].out_dim == 20
        @test model.layers[2].in_dim == 20
        @test model.layers[2].out_dim == 5
        
        # Test forward pass
        x = randn(Float32, 10, 32)  # 10 features, batch size 32
        y = model(x)
        
        @test size(y) == (5, 32)  # Should have correct output dimension
        @test all(isfinite.(y))   # Should produce finite outputs
    end
    
    @testset "Hyperbolic Loss" begin
        # Test hyperbolic MSE loss
        y_pred = randn(Float32, 5, 32)
        y_true = randn(Float32, 5, 32)
        
        loss = SaitoHNN.hyperbolic_mse_loss(y_pred, y_true)
        
        @test isfinite(loss)
        @test loss >= 0.0
        
        # Loss should be zero when predictions match targets
        zero_loss = SaitoHNN.hyperbolic_mse_loss(y_true, y_true)
        @test zero_loss ≈ 0.0 atol=1e-6
    end
    
    @testset "Model Serialization" begin
        # Create and save a model
        model1 = SaitoHNN.SaitoHNN([10, 20, 5])
        temp_file = tempname() * ".bson"
        
        # Save and load the model
        SaitoHNN.save_model(model1, temp_file)
        model2 = SaitoHNN.load_model(temp_file, SaitoHNN.SaitoHNN([10, 20, 5]))
        
        # Models should have the same predictions
        x = randn(Float32, 10, 5)
        y1 = model1(x)
        y2 = model2(x)
        
        @test y1 ≈ y2
        
        # Clean up
        rm(temp_file, force=true)
    end
    
    @testset "Network Integration" begin
        # This is a simplified test - in a real scenario, we'd need multiple nodes
        model = SaitoHNN.SaitoHNN([10, 20, 5])
        
        # Connect to network (but don't actually start it in the test)
        model.network_node = nothing  # Skip actual network connection in tests
        
        # Test that the model can be trained (minimal test)
        x = randn(Float32, 10, 32)
        y = randn(Float32, 5, 32)
        
        # Create a simple data loader
        data = [(x, y)]
        
        # Train for one epoch
        optimizer = Flux.setup(Adam(0.01), model)
        loss = SaitoHNN.train_epoch!(model, data, optimizer, cpu)
        
        @test isfinite(loss)
    end
end

# Run the tests
@testset "SaitoHNN Integration Tests" begin
    include("saito_hnn_integration_tests.jl")
end
