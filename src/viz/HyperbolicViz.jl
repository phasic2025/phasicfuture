"""
HyperbolicViz.jl

Visualization tools for SAITO-Constrained HNN, including 2D and 3D hyperbolic embeddings.
"""
module HyperbolicViz

using ..HyperbolicGeometry
using ..SAITOCore
using GLMakie  # For interactive 3D visualization
using CairoMakie  # For static image export
using Colors
using GraphMakie
using NetworkLayout
using LinearAlgebra
using ColorSchemes
using Observables
using Printf

# Re-exports
export plot_hyperbolic_embedding, plot_hyperbolic_graph, animate_learning,
       plot_geodesic, plot_parallel_transport, plot_figure_eight_knot,
       interactive_hyperbolic_space, plot_curvature

"""
    project_to_poincare_disk(point::HyperbolicPoint{2})

Project a 2D hyperbolic point to the Poincaré disk model.
""
function project_to_poincare_disk(point::HyperbolicPoint{2})
    x, y = point.coords
    return Point2f(x, y)
end

""
    project_to_poincare_ball(point::HyperbolicPoint{3})

Project a 3D hyperbolic point to the Poincaré ball model.
""
function project_to_poincare_ball(point::HyperbolicPoint{3})
    x, y, z = point.coords
    return Point3f(x, y, z)
end

""
    plot_hyperbolic_embedding(graph::SAITOGraph; 
                             node_size=10, 
                             edge_width=1.0,
                             title="Hyperbolic Embedding")

Create a 2D visualization of the hyperbolic embedding.
"""
function plot_hyperbolic_embedding(graph::SAITOGraph; 
                                  node_size=10, 
                                  edge_width=1.0,
                                  title="Hyperbolic Embedding")
    # Extract node positions
    node_positions = Point2f[]
    node_colors = Colorant[]
    
    # Create colormap based on distance from origin
    cmap = cgrad(:viridis)
    max_r = 0.0
    
    # First pass: find maximum distance for normalization
    for (id, node) in graph.nodes
        push!(node_positions, project_to_poincare_disk(node.embedding))
        r = norm(node.embedding)
        max_r = max(max_r, r)
    end
    
    # Second pass: assign colors
    for (id, node) in graph.nodes
        r = norm(node.embedding)
        # Map distance to color (closer to edge = more red)
        push!(node_colors, cmap[r / (max_r + 1e-6)])
    end
    
    # Create figure
    fig = Figure(resolution=(800, 800))
    ax = Axis(fig[1, 1], aspect=DataAspect(), title=title)
    
    # Draw Poincaré disk boundary
    circle = Circle(Point2f(0, 0), 1.0)
    poly!(ax, circle, color=(:white, 0.0), strokecolor=:black, strokewidth=2)
    
    # Draw edges
    for ((src_id, dst_id), edge) in graph.edges
        src = graph.nodes[src_id].embedding
        dst = graph.nodes[dst_id].embedding
        
        # Create geodesic curve (approximated by Möbius addition)
        t = range(0, 1, length=50)
        curve = Point2f[]
        
        for ti in t
            # Interpolate in hyperbolic space
            p = exp_map(src, HyperbolicVector(log_map(src, HyperbolicPoint(dst.coords)).coords .* ti))
            push!(curve, project_to_poincare_disk(p))
        end
        
        lines!(ax, curve, color=(:blue, 0.3), linewidth=edge_width)
    end
    
    # Draw nodes
    scatter!(ax, node_positions, color=node_colors, markersize=node_size, strokewidth=1, strokecolor=:black)
    
    # Add node labels
    for (i, (id, node)) in enumerate(graph.nodes)
        pos = project_to_poincare_disk(node.embedding)
        text!(ax, id, position=pos .+ Point2f(0.02, 0.02), textsize=12, color=:black)
    end
    
    # Hide axes for cleaner visualization
    hidedecorations!(ax)
    
    return fig
end

""
    plot_hyperbolic_graph(graph::SAITOGraph; 
                         node_size=10, 
                         edge_width=1.0,
                         title="Hyperbolic Graph")

Create an interactive 3D visualization of the hyperbolic embedding.
"""
function plot_hyperbolic_graph(graph::SAITOGraph; 
                              node_size=10, 
                              edge_width=1.0,
                              title="Hyperbolic Graph")
    # Create figure
    fig = Figure(resolution=(1200, 800))
    ax = LScene(fig[1, 1], show_axis=false)
    
    # Extract node positions and colors
    node_positions = Point3f[]
    node_colors = Colorant[]
    cmap = cgrad(:viridis)
    max_r = 0.0
    
    # First pass: find maximum distance for normalization
    for (id, node) in graph.nodes
        push!(node_positions, project_to_poincare_ball(node.embedding))
        r = norm(node.embedding)
        max_r = max(max_r, r)
    end
    
    # Second pass: assign colors
    for (id, node) in graph.nodes
        r = norm(node.embedding)
        push!(node_colors, cmap[r / (max_r + 1e-6)])
    end
    
    # Draw edges
    for ((src_id, dst_id), edge) in graph.edges
        src = graph.nodes[src_id].embedding
        dst = graph.nodes[dst_id].embedding
        
        # Create geodesic curve (approximated)
        t = range(0, 1, length=20)
        curve = Point3f[]
        
        for ti in t
            p = exp_map(src, HyperbolicVector(log_map(src, HyperbolicPoint(dst.coords)).coords .* ti))
            push!(curve, project_to_poincare_ball(p))
        end
        
        lines!(ax, curve, color=(:blue, 0.3), linewidth=edge_width)
    end
    
    # Draw nodes
    meshscatter!(ax, node_positions, 
                color=node_colors, 
                markersize=node_size/100, 
                transparency=true,
                shading=false)
    
    # Add node labels
    for (i, (id, node)) in enumerate(graph.nodes)
        pos = project_to_poincare_ball(node.embedding)
        text!(ax, id, position=pos .+ Point3f(0.02, 0.02, 0.02), textsize=0.1)
    end
    
    # Add a sphere for the boundary
    mesh!(ax, Sphere(Point3f(0), 1.0), color=(:white, 0.0), 
          transparency=true, shading=false, strokewidth=0.1, strokecolor=:black)
    
    # Add a colorbar
    cbar = Colorbar(fig[1, 2], colormap=cmap, label="Distance from Center")
    
    display(fig)
    return fig
end

""
    animate_learning(graph::SAITOGraph, 
                    n_steps::Int=100; 
                    filename="hyperbolic_learning.gif",
                    node_size=10,
                    edge_width=1.0)

Create an animation of the learning process.
"""
function animate_learning(graph::SAITOGraph, 
                         n_steps::Int=100; 
                         filename="hyperbolic_learning.gif",
                         node_size=10,
                         edge_width=1.0)
    # Create a copy of the graph to avoid modifying the original
    g = deepcopy(graph)
    
    # Set up the figure
    fig = Figure(resolution=(800, 800))
    ax = Axis(fig[1, 1], aspect=DataAspect(), title="Hyperbolic Learning")
    
    # Draw Poincaré disk boundary
    circle = Circle(Point2f(0, 0), 1.0)
    poly!(ax, circle, color=(:white, 0.0), strokecolor=:black, strokewidth=2)
    
    # Initial scatter plot (will be updated)
    pos = [project_to_poincare_disk(node.embedding) for (id, node) in g.nodes]
    scat = scatter!(ax, pos, color=:blue, markersize=node_size)
    
    # Create recording
    record(fig, filename, 1:n_steps; framerate=15) do step
        # Update node positions
        for (id, node) in g.nodes
            update_node_embedding!(g, id, 0.01)
        end
        
        # Update scatter plot
        new_pos = [project_to_poincare_disk(node.embedding) for (id, node) in g.nodes]
        scat[1] = new_pos
        
        # Update title
        ax.title[] = "Step $step/$n_steps"
    end
    
    return g  # Return the final state of the graph
end

"""
    save_figure(fig, filename::String; resolution=(800, 800))

Save a figure to a file.
"""
save_figure(fig, filename::String; resolution=(800, 800)) = save(filename, fig; px_per_unit=2)

"""
    plot_geodesic(p::HyperbolicPoint, q::HyperbolicPoint; n_points=100, color=:blue, linewidth=2.0)

Plot the geodesic between two points in hyperbolic space.
"""
function plot_geodesic(p::HyperbolicPoint{N}, q::HyperbolicPoint{N}; 
                      n_points=100, color=:blue, linewidth=2.0) where N
    ts = range(0, 1, length=n_points)
    points = [geodesic(p, q, t) for t in ts]
    
    if N == 2
        points_2d = [Point2f(p.coords) for p in points]
        lines!(points_2d; color=color, linewidth=linewidth)
    elseif N == 3
        points_3d = [Point3f(p.coords) for p in points]
        lines!(points_3d; color=color, linewidth=linewidth)
    else
        error("Only 2D and 3D points are supported for visualization")
    end
end

"""
    plot_parallel_transport(p::HyperbolicPoint, q::HyperbolicPoint, v::HyperbolicVector; 
                          n_steps=10, color=:red, linewidth=1.5)

Visualize parallel transport of vector v along the geodesic from p to q.
"""
function plot_parallel_transport(p::HyperbolicPoint{N}, q::HyperbolicPoint{N}, 
                               v::HyperbolicVector{N}; n_steps=10, color=:red, 
                               linewidth=1.5) where N
    ts = range(0, 1, length=n_steps)
    
    for i in 1:length(ts)-1
        t1, t2 = ts[i], ts[i+1]
        point1 = geodesic(p, q, t1)
        point2 = geodesic(p, q, t2)
        
        # Transport the vector to the current point
        v_transported = parallel_transport_along_geodesic(v, p, q, t1)
        
        # Scale the vector for visualization
        scale_factor = 0.1
        v_scaled = v_transported.coords .* scale_factor
        
        # Get the point in the Poincaré model
        if N == 2
            p1 = project_to_poincare_disk(point1)
            p2 = project_to_poincare_disk(HyperbolicPoint(point1.coords .+ v_scaled))
            arrows!([p1], [p2 .- p1]; color=color, linewidth=linewidth, arrowsize=0.05)
        elseif N == 3
            p1 = project_to_poincare_ball(point1)
            p2 = project_to_poincare_ball(HyperbolicPoint(point1.coords .+ v_scaled))
            arrows!([p1], [p2 .- p1]; color=color, linewidth=linewidth, arrowsize=0.05)
        end
    end
end

"""
    plot_figure_eight_knot(; n_points=1000, R=1.0, r=0.3, c=0.4, color=:purple, linewidth=2.0)

Plot a figure-eight knot in 3D hyperbolic space.
"""
function plot_figure_eight_knot(; n_points=1000, R=1.0, r=0.3, c=0.4, color=:purple, linewidth=2.0)
    ts = range(0, 2π, length=n_points)
    points = [figure_eight_knot(t; R, r, c) for t in ts]
    
    # Project to hyperboloid model and then to Poincaré ball
    points_hyp = [project_to_hyperboloid(p) for p in points]
    points_poincare = [hyperboloid_to_poincare(p) for p in points_hyp]
    
    # Convert to 3D points
    points_3d = [Point3f(p[1], p[2], p[3]) for p in points_poincare]
    
    # Plot the knot
    lines!(points_3d; color=color, linewidth=linewidth)
end

"""
    interactive_hyperbolic_space(;dim=3)

Create an interactive visualization of hyperbolic space.
"""
function interactive_hyperbolic_space(;dim=3)
    # Set up the scene
    fig = Figure(resolution=(1200, 800))
    ax = LScene(fig[1, 1], show_axis=true)
    
    if dim == 2
        # Create a 2D hyperbolic space visualization
        ax = Axis(fig[1, 1], aspect=DataAspect(), title="2D Hyperbolic Space (Poincaré Disk)")
        
        # Draw the unit circle (boundary of Poincaré disk)
        circle = Circle(Point2f(0, 0), 1.0)
        poly!(ax, circle, color=(:white, 0.1), strokecolor=:black, strokewidth=1)
        
        # Add interactive points
        points = Observable(Point2f[])
        lines = Observable(Point2f[])
        
        # Connect points with geodesics when clicked
        on(events(ax).mousebutton, priority=2) do event
            if event.button == Mouse.left && event.action == Mouse.press
                # Get click position in data coordinates
                pos = mouseposition(ax.scene)
                if norm(pos) < 1.0  # Only add points inside the disk
                    push!(points[], pos)
                    points[] = points[]
                    
                    # If we have at least 2 points, draw a geodesic
                    if length(points[]) >= 2
                        p1 = points[][end-1]
                        p2 = points[][end]
                        
                        # Sample points along the geodesic
                        ts = range(0, 1, length=100)
                        geodesic_points = [geodesic(
                            HyperbolicPoint([p1[1], p1[2]]), 
                            HyperbolicPoint([p2[1], p2[2]]), 
                            t
                        ) for t in ts]
                        
                        # Update the lines
                        lines[] = [Point2f(p.coords[1], p.coords[2]) for p in geodesic_points]
                        notify(lines)
                    end
                end
            end
            return Consume(false)
        end
        
        # Plot the points and lines
        scatter!(ax, points, color=:red, markersize=15)
        lines!(ax, lines, color=:blue, linewidth=2)
        
    else  # 3D
        # Create a 3D visualization of hyperbolic space
        ax = LScene(fig[1, 1], show_axis=true, title="3D Hyperbolic Space (Poincaré Ball)")
        
        # Draw the unit sphere (boundary of Poincaré ball)
        n = 50
        u = range(0, 2π, length=n)
        v = range(0, π, length=n)
        x = [cos(u)*sin(v) for u in u, v in v]
        y = [sin(u)*sin(v) for u in u, v in v]
        z = [cos(v) for u in u, v in v']
        
        surface!(ax, x, y, z, color=(:white, 0.1), transparency=true, 
                colormap=:blues, shading=false)
        
        # Add a figure-eight knot by default
        plot_figure_eight_knot()
    end
    
    display(fig)
    return fig
end

"""
    plot_curvature(points::Vector{HyperbolicPoint{N}}; color=:thermal, linewidth=2.0) where N

Visualize the curvature of a path through hyperbolic space.
"""
function plot_curvature(points::Vector{HyperbolicPoint{N}}; color=:thermal, linewidth=2.0) where N
    # Calculate the geodesic curvature along the path
    curvatures = Float64[]
    for i in 2:length(points)-1
        p_prev = points[i-1]
        p_curr = points[i]
        p_next = points[i+1]
        
        # Approximate the tangent vectors
        v1 = log_map(p_curr, p_prev)
        v2 = log_map(p_curr, p_next)
        
        # Calculate the angle between vectors
        cos_angle = dot(v1, v2) / (norm(v1) * norm(v2) + 1e-10)
        angle = acos(clamp(cos_angle, -1.0, 1.0))
        
        # Approximate curvature
        push!(curvatures, angle / (norm(v1) + norm(v2) + 1e-10))
    end
    
    # Pad the curvatures to match points length
    pushfirst!(curvatures, first(curvatures))
    push!(curvatures, last(curvatures))
    
    # Create a colormap based on curvature
    colors = [get(ColorSchemes[color], c, (0, 1)) for c in normalize(curvatures)]
    
    # Plot the path with color indicating curvature
    if N == 2
        points_2d = [Point2f(p.coords) for p in points]
        linesegments!(points_2d, color=colors, linewidth=linewidth)
    elseif N == 3
        points_3d = [Point3f(p.coords) for p in points]
        linesegments!(points_3d, color=colors, linewidth=linewidth)
    end
    
    # Add a colorbar
    Colorbar(fig[1, 2], limits=(minimum(curvatures), maximum(curvatures)), 
            colormap=color, label="Curvature")
end

end # module
