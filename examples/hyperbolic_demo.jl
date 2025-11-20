using LinearAlgebra
using Plots
using Random
using .Hyperbolic  # Our hyperbolic geometry module

# Set random seed for reproducibility
rng = MersenneTwister(42)

# Parameters
c = 1.0  # Curvature
dim = 2  # Dimension for visualization
n_points = 50

# Initialize points in hyperbolic space
emb = init_embedding(n_points, dim, c, rng=rng)

# Create a figure to visualize the points in the Poincaré disk
function plot_poincare_disk(points, c=1.0; title="Poincaré Disk")
    # Create unit circle
    θ = range(0, 2π, length=100)
    circle_x = @. 1/sqrt(c) * cos(θ)
    circle_y = @. 1/sqrt(c) * sin(θ)
    
    # Plot the boundary
    p = plot(circle_x, circle_y, linecolor=:black, linewidth=2, 
             legend=false, aspect_ratio=:equal, title=title)
    
    # Plot the points
    scatter!(p, points[1,:], points[2,:], 
             markersize=4, markerstrokewidth=0.5, 
             color=:blue, alpha=0.7)
    
    # Add origin
    scatter!(p, [0], [0], color=:red, markersize=5, label="Origin")
    
    return p
end

# Plot initial points
p1 = plot_poincare_disk(emb.embedding, c, title="Initial Points in Poincaré Disk")
display(p1)

# Function to perform a hyperbolic translation
function hyperbolic_translation(points, v, c)
    n = size(points, 2)
    translated = similar(points)
    for i in 1:n
        translated[:,i] = mobius_add(v, points[:,i], c)
    end
    return translated
end

# Perform a hyperbolic translation
translation_vector = [0.5, 0.0]
translated_points = hyperbolic_translation(emb.embedding, translation_vector, c)

# Plot translated points
p2 = plot_poincare_disk(translated_points, c, title="After Hyperbolic Translation")
display(p2)

# Function to compute hyperbolic distances from origin
function distances_from_origin(points, c)
    n = size(points, 2)
    dists = zeros(n)
    origin = zeros(size(points, 1))
    for i in 1:n
        dists[i] = distance(origin, points[:,i], c)
    end
    return dists
end

# Compare distances
original_dists = distances_from_origin(emb.embedding, c)
translated_dists = distances_from_origin(translated_points, c)

# Plot histogram of distances
p3 = histogram([original_dists, translated_dists], 
               label=["Original" "Translated"], 
               xlabel="Hyperbolic Distance from Origin", 
               ylabel="Count",
               title="Distribution of Distances from Origin")
display(p3)

# Demonstrate parallel transport
function plot_parallel_transport(p1, p2, v, c)
    # Create points along the geodesic from p1 to p2
    t = range(0, 1, length=50)
    geodesic = [exp_map(p1, t_i * log_map(p1, p2, c), c) for t_i in t]
    
    # Transport the vector v along the geodesic
    transported = [parallel_transport(p1, p, v, c) for p in geodesic]
    
    # Plot the geodesic
    p = plot([p[1] for p in geodesic], [p[2] for p in geodesic], 
             label="Geodesic", linewidth=2, aspect_ratio=:equal)
    
    # Plot the original vector at p1
    quiver!([p1[1]], [p1[2]], quiver=([v[1]], [v[2]]), 
            color=:red, label="Original Vector")
    
    # Plot the transported vector at p2
    quiver!([p2[1]], [p2[2]], quiver=([transported[end][1]], [transported[end][2]]), 
            color=:blue, label="Transported Vector")
    
    # Plot intermediate vectors
    for (i, p) in enumerate(geodesic[5:5:end])
        v_t = transported[5i]
        quiver!([p[1]], [p[2]], quiver=([v_t[1]], [v_t[2]]), 
                color=RGBA(0.5, 0, 0.5, 0.5), label="")
    end
    
    # Add points
    scatter!([p1[1]], [p1[2]], color=:red, markersize=5, label="Start Point")
    scatter!([p2[1]], [p2[2]], color=:blue, markersize=5, label="End Point")
    
    title!("Parallel Transport in Hyperbolic Space")
    return p
end

# Example points and vector
p1 = [0.1, 0.1]
p2 = [-0.3, 0.4]
v = [0.2, 0.1]

# Plot parallel transport
p4 = plot_parallel_transport(p1, p2, v, c)
display(p4)

# Save plots
savefig(p1, "poincare_disk.png")
savefig(p2, "translated_points.png")
savefig(p3, "distance_distribution.png")
savefig(p4, "parallel_transport.png")
