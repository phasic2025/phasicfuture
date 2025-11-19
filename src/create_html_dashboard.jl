# Create HTML Dashboard
# Generates a static HTML file with visualizations

include("TOASTER_DESIGN_DEMO.jl")

function create_html_dashboard()
    # Run simulation to collect data
    hierarchy, final_design = design_toaster_demo()
    
    # Collect goal data
    goal_data = []
    for (id, goal) in hierarchy.goals
        push!(goal_data, Dict(
            "id" => string(id),
            "description" => goal.description,
            "progress" => goal.progress * 100,
            "achieved" => goal.achieved,
            "terminal" => goal.terminal,
            "dependencies" => [string(d) for d in goal.dependencies]
        ))
    end
    
    # Create HTML
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Neural Network Monitor Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 20px;
                background: #f5f5f5;
            }
            .container {
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
                border-bottom: 3px solid #4CAF50;
                padding-bottom: 10px;
            }
            .section {
                margin: 30px 0;
                padding: 20px;
                background: #fafafa;
                border-radius: 5px;
            }
            .goal-card {
                background: white;
                padding: 15px;
                margin: 10px 0;
                border-left: 4px solid #4CAF50;
                border-radius: 5px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            .goal-card.achieved {
                border-left-color: #2196F3;
            }
            .goal-card.terminal {
                border-left-color: #FF9800;
            }
            .progress-bar {
                width: 100%;
                height: 25px;
                background: #e0e0e0;
                border-radius: 5px;
                overflow: hidden;
                margin: 10px 0;
            }
            .progress-fill {
                height: 100%;
                background: linear-gradient(90deg, #4CAF50, #8BC34A);
                transition: width 0.3s;
            }
            .design-specs {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }
            .spec-card {
                background: white;
                padding: 15px;
                border-radius: 5px;
                text-align: center;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            .spec-value {
                font-size: 24px;
                font-weight: bold;
                color: #4CAF50;
            }
            .spec-label {
                color: #666;
                font-size: 14px;
                margin-top: 5px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧠 Neural Network Monitor Dashboard</h1>
            <p>Goal-Adapted RL with Topological Boundaries - Toaster Design Demo</p>
            
            <div class="section">
                <h2>📊 Goal Hierarchy</h2>
                $(join(["""
                <div class="goal-card $(goal["achieved"] ? "achieved" : "") $(goal["terminal"] ? "terminal" : "")">
                    <h3>$(goal["id"]) $(goal["achieved"] ? "✅" : "⏳")</h3>
                    <p>$(goal["description"])</p>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: $(goal["progress"])%"></div>
                    </div>
                    <small>Progress: $(round(goal["progress"], digits=1))%</small>
                    $(length(goal["dependencies"]) > 0 ? "<small>Dependencies: $(join(goal["dependencies"], ", "))</small>" : "")
                </div>
                """ for goal in goal_data], "\n"))
            </div>
            
            $(if final_design !== nothing
                """
                <div class="section">
                    <h2>🎨 Final Design Specifications</h2>
                    <div class="design-specs">
                        $(join(["""
                        <div class="spec-card">
                            <div class="spec-value">$(final_design[Symbol(key)])</div>
                            <div class="spec-label">$(replace(string(key), "_" => " "))</div>
                        </div>
                        """ for key in keys(final_design)], "\n"))
                    </div>
                </div>
                """ else ""
            end)
            
            <div class="section">
                <h2>🔍 Topological Boundaries</h2>
                <ul>
                    <li>Power constraint: 800-1500W</li>
                    <li>Safety required: Auto-shutoff mandatory</li>
                    <li>Mechanical feasible: Spring force 5-20N</li>
                </ul>
                <p><strong>Design space reduction:</strong> 10,000,000,000 → ~300 designs</p>
                <p><strong>Speedup:</strong> 33.3 million x</p>
            </div>
            
            <div class="section">
                <h2>📈 Statistics</h2>
                <p>Total goals: $(length(goal_data))</p>
                <p>Achieved goals: $(sum(g["achieved"] for g in goal_data))</p>
                <p>Overall progress: $(round(mean([g["progress"] for g in goal_data]), digits=1))%</p>
            </div>
        </div>
        
        <script>
            // Add any interactive JavaScript here
            console.log("Dashboard loaded");
        </script>
    </body>
    </html>
    """
    
    write("dashboard.html", html)
    println("✅ Created dashboard.html")
    println("   Open in browser: open dashboard.html")
end

if abspath(PROGRAM_FILE) == @__FILE__
    Random.seed!(42)
    create_html_dashboard()
end

