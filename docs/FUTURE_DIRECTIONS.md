# Future Experimental Directions
## Neurogenesis, Morphological Deformation, and Emergent Behavior

---

## Overview

This document outlines **future experimental directions** for extending the current hyperbolic neural network architecture. These are **research questions** to explore after validating the core theory, not modifications to the existing framework.

**Status**: These ideas are **not yet implemented** and represent potential extensions for future investigation.

---

## 1. Dynamic Neurogenesis: Self-Generating Neurons

### 1.1 The Concept

**Current System**: Fixed number of neurons $N$ (e.g., 50 neurons)

**Proposed Extension**: Allow the network to **generate new neurons** during learning, then **cap** the total count to test resource constraints.

### 1.2 Neurogenesis Mechanism

**Trigger Conditions**:
- When synchronization $r(t) < r_{\text{threshold}}$ (network struggling)
- When goal progress stalls: $\frac{d\text{progress}}{dt} < \epsilon$
- When embedding space saturation: $\|\mathbf{e}_{\text{new}} - \mathbf{e}_{\text{existing}}\| < \delta$ (too many similar concepts)

**Neuron Generation**:
$$
\text{Generate neuron } n_{\text{new}} \text{ at position } \mathbf{p}_{\text{new}} \sim \mathcal{U}(\text{boundary region})
$$

Where:
- $\mathbf{p}_{\text{new}}$ = new neuron position in Poincaré disk
- Boundary region = area near existing neurons (to maintain connectivity)
- Initial phase: $\phi_{\text{new}} = \text{average}(\{\phi_i\})$ (synchronize with existing)
- Initial frequency: $\omega_{\text{new}} = \text{average}(\{\omega_i\})$

### 1.3 Capping Mechanism

**Resource Constraint**: Cap total neurons at $N_{\text{max}}$

**Pruning Strategy** (when $N > N_{\text{max}}$):
1. **Low activation neurons**: Remove neurons with $\bar{s}_i < \theta_{\text{activation}}$
2. **Weakly connected neurons**: Remove neurons with $\sum_j w_{ij} < \theta_{\text{connection}}$
3. **Redundant neurons**: Remove neurons with $\min_j \|\mathbf{p}_i - \mathbf{p}_j\| < \theta_{\text{distance}}$ (too close to others)

**Question**: Does dynamic neurogenesis + capping lead to **more efficient** networks that adapt to task complexity?

### 1.4 Expected Behaviors

- **Early learning**: Many neurons generated (exploration)
- **Mature learning**: Fewer neurons, stronger connections (exploitation)
- **Task change**: New neurons generated for new concepts
- **Resource pressure**: Capping forces network to optimize neuron usage

---

## 2. Morphological Deformation: Neuron Shape Dynamics

### 2.1 Biological Inspiration

**Human Brain**: Neurons have **complex morphologies** (dendrites, spines, branches) that change over time:
- **Dendritic spines**: Grow/retract based on activity
- **Axonal branching**: Changes with learning
- **Morphological plasticity**: Shape affects signal propagation

**Key Insight**: Wave reflections depend on **boundary shape** $\partial \mathcal{M}_i$. If morphology changes, wave dynamics change.

### 2.2 Morphological Parameters

**Current System**: Each neuron has a scalar `morphology` parameter (simplified)

**Proposed Extension**: Represent morphology as a **deformable boundary**:

$$
\mathcal{M}_i(t) = \{\mathbf{p}_1(t), \mathbf{p}_2(t), ..., \mathbf{p}_k(t)\}
$$

Where:
- $\mathbf{p}_j(t)$ = boundary point $j$ of neuron $i$ at time $t$
- $k$ = number of boundary points (complexity parameter)

### 2.3 Deformation Dynamics

**Activity-Dependent Deformation**:
$$
\frac{d\mathbf{p}_j}{dt} = \alpha \cdot s_i(t) \cdot \mathbf{n}_j + \beta \cdot \text{curvature}(\mathbf{p}_j)
$$

Where:
- $\mathbf{n}_j$ = normal vector at boundary point $j$
- $\alpha$ = activity-driven growth rate
- $\beta$ = curvature-driven smoothing (prevents extreme deformations)

**Hebbian Morphological Plasticity**:
- High activation → boundary expands (more surface area for wave reflection)
- Low activation → boundary contracts (less surface area)
- Synchronized neurons → boundaries align (cooperative deformation)

### 2.4 Wave Reflection Changes

**Current**: Wave reflects off fixed boundary $\partial \mathcal{M}_i$

**With Deformation**: Wave reflects off **changing boundary** $\partial \mathcal{M}_i(t)$

**Expected Effects**:
1. **Resonance tuning**: Morphology adapts to create resonant frequencies
2. **Directional bias**: Deformations create preferred wave directions
3. **Interference patterns**: Changing boundaries create **dynamic interference**
4. **Emergent oscillations**: Deformation + wave dynamics → new oscillation modes

### 2.5 Research Questions

1. **Do deformations create new oscillation modes?**
   - Hypothesis: Deformed boundaries create **standing wave patterns**
   - Test: Measure frequency spectrum before/after deformation

2. **Do deformations improve learning?**
   - Hypothesis: Adaptive morphology optimizes wave propagation
   - Test: Compare learning speed with fixed vs. deformable morphology

3. **Do deformations create emergent behaviors?**
   - Hypothesis: Complex deformations → complex wave patterns → new behaviors
   - Test: Visualize wave patterns and correlate with task performance

---

## 3. Emergent Behavior from Morphological Dynamics

### 3.1 The Hypothesis

**Core Idea**: If neurons can deform, and waves reflect off deformations, then:
- **Deformation patterns** → **Wave patterns** → **Activation patterns** → **Behavior**

**Emergence Chain**:
$$
\text{Morphology}(t) \rightarrow \text{Wave Reflection}(t) \rightarrow \text{Interference}(t) \rightarrow \text{Activation}(t) \rightarrow \text{Behavior}(t)
$$

### 3.2 Potential Emergent Behaviors

**1. Morphological Clustering**:
- Neurons with similar goals develop similar morphologies
- Creates **functional clusters** via shape similarity
- **Test**: Measure morphology similarity within goal-assigned groups

**2. Wave-Guided Deformation**:
- Waves preferentially propagate along certain paths
- Morphology adapts to "channel" waves
- Creates **information highways** in the network
- **Test**: Track wave paths and correlate with morphology

**3. Oscillatory Modes**:
- Deformed boundaries create **resonant frequencies**
- Network develops preferred oscillation frequencies
- Different tasks → different frequencies
- **Test**: FFT analysis of activation patterns

**4. Morphological Memory**:
- Morphology encodes **long-term memories**
- Shape persists even when activation decays
- **Test**: Freeze learning, measure memory retention

### 3.3 Experimental Protocol

**Phase 1: Baseline**
- Run current system (fixed morphology)
- Measure: synchronization $r(t)$, goal progress, learning speed

**Phase 2: Deformable Morphology**
- Enable morphological deformation
- Measure: same metrics + morphology dynamics
- Compare: Does deformation improve performance?

**Phase 3: Neurogenesis + Deformation**
- Enable both neurogenesis and deformation
- Measure: neuron count, morphology complexity, emergent behaviors
- Test: Does combination create new capabilities?

**Phase 4: Capping + Optimization**
- Cap neurons at $N_{\text{max}}$
- Measure: resource efficiency, performance under constraints
- Test: Does capping force better optimization?

---

## 4. Implementation Considerations

### 4.1 Computational Complexity

**Current System**:
- Wave propagation: $O(N^2)$ (all neuron pairs)
- Morphology: $O(1)$ per neuron (scalar)

**With Deformation**:
- Wave propagation: $O(N^2 \cdot k)$ where $k$ = boundary points per neuron
- Morphology update: $O(N \cdot k)$ per time step
- **Challenge**: Keep $k$ small (e.g., $k = 8-16$ points per neuron)

### 4.2 Hyperbolic Geometry Constraints

**Boundary Points in Poincaré Disk**:
- Must respect hyperbolic distance metric
- Deformation must preserve topology (no self-intersections)
- **Solution**: Constrain deformations to geodesic-preserving transformations

### 4.3 Learning Stability

**Risk**: Deformation + learning could create **instabilities**

**Mitigation**:
- **Smoothing**: Curvature term prevents extreme deformations
- **Rate limiting**: $\alpha, \beta$ small to prevent rapid changes
- **Validation**: Check that deformations don't break wave propagation

---

## 5. Success Metrics

### 5.1 Performance Metrics

- **Learning speed**: Faster convergence with deformation?
- **Task performance**: Better goal achievement?
- **Resource efficiency**: Fewer neurons needed with capping?

### 5.2 Emergence Metrics

- **Oscillation diversity**: More frequency modes?
- **Morphological clustering**: Do similar neurons develop similar shapes?
- **Wave pattern complexity**: More complex interference patterns?

### 5.3 Biological Plausibility

- **Morphological changes**: Match biological timescales?
- **Neurogenesis rate**: Match biological neurogenesis?
- **Deformation patterns**: Resemble dendritic spine dynamics?

---

## 6. Timeline and Priorities

### Phase 1: Validate Current System (Current)
- ✅ Core theory documented
- ✅ Basic implementation working
- ⏳ Test goal-adapted RL
- ⏳ Validate topological boundaries

### Phase 2: Neurogenesis (Future)
- Implement neuron generation mechanism
- Test with/without capping
- Measure resource efficiency

### Phase 3: Morphological Deformation (Future)
- Implement deformable boundaries
- Test wave reflection changes
- Measure emergent behaviors

### Phase 4: Combined System (Future)
- Combine neurogenesis + deformation
- Test for synergistic effects
- Explore emergent behaviors

---

## 7. Open Questions

1. **Does neurogenesis improve learning?**
   - Or does it just add noise?

2. **Do deformations create new capabilities?**
   - Or are they just cosmetic?

3. **What is the optimal neuron count?**
   - Is there a sweet spot for task complexity?

4. **Can morphology encode memories?**
   - Or is it just a byproduct of activity?

5. **Do emergent behaviors generalize?**
   - Or are they task-specific?

---

## 8. References and Inspiration

- **Biological Neurogenesis**: Adult hippocampal neurogenesis in mammals
- **Dendritic Spines**: Activity-dependent spine dynamics
- **Morphological Plasticity**: How neuron shape affects function
- **Wave Dynamics**: How boundaries affect wave propagation in physics
- **Emergent Behavior**: Complex systems literature

---

**Note**: This document represents **experimental directions**, not modifications to the core theory. The current system should be fully validated before exploring these extensions.

