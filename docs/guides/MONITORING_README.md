# Neural Network Monitoring UI

Multiple UI options for monitoring the neural network in real-time.

---

## Quick Start

### Option 1: Terminal Monitor (Recommended - Easiest)

**Best for**: Real-time monitoring with live updates

```bash
julia run_monitor.jl
```

**Features**:
- ✅ Real-time updates every 1.5 seconds
- ✅ Goal hierarchy visualization
- ✅ Goal values and progress bars
- ✅ Topological boundaries display
- ✅ Design specifications
- ✅ Automatic goal switching alerts

**What you'll see**:
- Current active goal with progress bar
- Complete goal hierarchy tree
- Goal values sorted by priority
- Topological boundaries and design space reduction
- Best design generated each step
- Overall progress summary

---

### Option 2: HTML Dashboard (Static)

**Best for**: Sharing results, presentations

```bash
julia create_html_dashboard.jl
# Then open dashboard.html in your browser
```

**Features**:
- ✅ Beautiful web-based interface
- ✅ Goal progress bars
- ✅ Final design specifications
- ✅ Statistics summary
- ✅ Shareable (just send the HTML file)

---

### Option 3: Pluto Notebook (Interactive)

**Best for**: Data exploration, analysis

```bash
julia -e 'using Pluto; Pluto.run()'
# Then open MONITOR_NOTEBOOK.jl in Pluto
```

**Features**:
- ✅ Interactive Julia notebook
- ✅ Reactive updates
- ✅ Customizable plots
- ✅ Data exploration

---

## Monitor Components

### 1. Goal Hierarchy
Shows the complete goal structure:
- Terminal goals (orange border)
- Instrumental goals (green border)
- Achieved goals (blue border, ✅)
- Active goal (→ marker)

### 2. Goal Progress
Visual progress bars for each goal:
- Progress percentage
- Dependencies status
- Achievement status

### 3. Goal Values
Real-time value estimation:
- Sorted by priority (🥇🥈🥉)
- Visual bars showing relative values
- Updates as system learns

### 4. Topological Boundaries
Shows active constraints:
- Boundary names
- Design space reduction
- Computational speedup

### 5. Design Specifications
Best design found each step:
- Power, safety, mechanical specs
- Final integrated design

### 6. Overall Progress
Summary statistics:
- Goals achieved / total
- Overall completion percentage
- Progress bar

---

## Example Output

```
================================================================================
STEP 1 | 2025-11-16T19:55:21.228
================================================================================

🎯 CURRENT GOAL: learn_heating
   Description: Understand heating element principles
   Progress: ████████████████████████████████████████ 83.5%
   Status: ⏳ IN PROGRESS

📊 GOAL HIERARCHY:
   Terminal Goals:
      design_toaster: Design a functional toaster
   Instrumental Goals:
   → learn_heating: Understand heating element principles
      Dependencies: none | Progress: 83.5%

💰 GOAL VALUES (sorted by priority):
   → 🥇        learn_heating: ████████████████████████████████ 0.835
     🥈     design_mechanics: ████████████████████████████████ 0.3
     🥉        design_safety: ████████████████████████████████ 0.45

🔍 TOPOLOGICAL BOUNDARIES:
   Active boundaries: 3
   • power_constraint
   • safety_required
   • mechanical_feasible
   Design space reduction: 10000000000 → ~300
   Speedup: 33.3 million x

🎨 BEST DESIGN THIS STEP:
   Power: 856W
   Auto-shutoff: true
   Spring force: 20N

   ✅ GOAL ACHIEVED!

   🔄 GOAL SWITCH DETECTED!
      learn_heating → design_safety
      Reason: Design safety features has higher value (0.45)

📈 OVERALL PROGRESS: 1/5 goals achieved
   ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 20.0%
```

---

## Customization

### Change Update Frequency

In `run_monitor.jl`, modify:
```julia
sleep(1.5)  # Change to desired seconds
```

### Add More Metrics

Edit `run_monitor.jl` to add:
- Goal switching frequency
- Design quality metrics
- Boundary effectiveness
- Learning rate tracking

### Custom Visualizations

For Pluto notebook:
- Add PlotlyJS plots
- Create custom charts
- Export data for analysis

---

## Troubleshooting

### Terminal Monitor Not Clearing Screen

If ANSI codes don't work, comment out:
```julia
# print("\033[2J\033[H")  # Clear screen
```

### HTML Dashboard Not Updating

The HTML dashboard is static. To update:
1. Re-run `create_html_dashboard.jl`
2. Refresh browser

### Pluto Notebook Not Loading

Make sure Pluto is installed:
```julia
using Pkg
Pkg.add("Pluto")
```

---

## Next Steps

1. **Add Real-time Plotting**: Use Makie.jl for live plots
2. **WebSocket Server**: Stream updates to web browser
3. **Database Logging**: Store all metrics for analysis
4. **Alert System**: Notify when goals achieved or drift detected

---

## Files

- `run_monitor.jl` - Terminal-based real-time monitor
- `create_html_dashboard.jl` - Generate static HTML dashboard
- `MONITOR_UI.jl` - Framework for multiple UI types
- `dashboard.html` - Generated HTML dashboard

---

**Status**: ✅ Monitoring UIs ready! Use `julia run_monitor.jl` for real-time monitoring.

