# Systems Thinking Replacements for THEORY.md

## Concrete, Measurable Replacements for Abstract Concepts

**Core Principle**: All measurements and computations follow the **topological boundary efficiency principle**: computation happens **only on boundaries**, not full space. This applies to every replacement below.

**Boundary-Constrained Computation**:
- **Boundary Neurons**: $\partial \mathcal{M} = \{i : i \text{ is on topological boundary}\}$
- **Boundary Goals**: $\mathcal{G}_{\text{boundary}} = \{G : \exists i \in \partial \mathcal{M} : \text{goal}_i = G\}$
- **Boundary Concepts**: $\mathcal{C}_{\text{boundary}} = \{c : c \text{ associated with boundary goals}\}$

**Efficiency**: All measurements scale with boundary size, not full network size:
- **Traditional**: Compute for all $N$ neurons → $O(N)$ or $O(N^2)$
- **Topological**: Compute only for $|\partial \mathcal{M}|$ boundary neurons → $O(|\partial \mathcal{M}|)$ or $O(|\partial \mathcal{M}|^2)$
- **Speedup**: $N / |\partial \mathcal{M}|$ or $N^2 / |\partial \mathcal{M}|^2$ (typically 10-100x reduction)

### 1. Replace "Authority" with Measurable Influence (Section 1.1)

**Current (Line ~12)**:
```
- Natural hierarchy: nodes closer to origin have higher "authority"
```

**Replacement**:
```
- Natural hierarchy: nodes closer to origin have higher **influence** (measured by activation propagation reach)
- **Influence Metric (Boundary-Constrained)**: 
  $$
  I_i = \begin{cases}
  \sum_{j \in \mathcal{N}(i) \cap \partial \mathcal{M}} w_{ij} \cdot \exp(-d_{ij}) & \text{if } i \in \partial \mathcal{M} \\
  0 & \text{otherwise}
  \end{cases}
  $$
  Only compute influence for boundary neurons, only count boundary neighbors.
  
- **Activation Reach (Boundary-Constrained)**: 
  $$
  R_i = |\{j \in \partial \mathcal{M} : \text{activation from } i \text{ reaches } j\}| \quad \text{if } i \in \partial \mathcal{M}
  $$
  Only count boundary neurons reached.
  
- **Synchronization Influence (Boundary-Constrained)**: 
  $$
  S_i = \begin{cases}
  \frac{1}{|\mathcal{N}(i) \cap \partial \mathcal{M}|} \sum_{j \in \mathcal{N}(i) \cap \partial \mathcal{M}} |\sin(\phi_j - \phi_i)| & \text{if } i \in \partial \mathcal{M} \\
  0 & \text{otherwise}
  \end{cases}
  $$
  Only compute synchronization influence for boundary neurons, only with boundary neighbors.

**Efficiency**: 
- **Traditional**: Compute influence for all $N$ neurons → $O(N^2)$
- **Topological**: Compute only for $|\partial \mathcal{M}|$ boundary neurons → $O(|\partial \mathcal{M}|^2)$
- **Speedup**: $N^2 / |\partial \mathcal{M}|^2$ (typically 100x reduction)
```

**Rationale**: "Authority" is abstract. Influence is measurable via connectivity, activation propagation, and synchronization effects. **All computation happens only on boundaries**.

---

### 2. Replace "Importance" with Contribution to Goal Achievement (Section 1.8)

**Current (Line ~574)**:
```
importance = α₁·activation + α₂·sync + α₃·goal + α₄·boundary
```

**Replacement**:
```
**Neuron Contribution to Goal Achievement (Boundary-Constrained)**:

Each neuron $i$ has a **contribution score** $\mathcal{C}_i(t)$ that measures its impact on goal pursuit:

$$
\mathcal{C}_i(t) = \begin{cases}
\beta_1 \cdot \Delta_{\text{goal\_progress}}(i, t) + \beta_2 \cdot \text{removal\_impact}(i, t) + \beta_3 \cdot \text{synchronization\_contribution}(i, t) & \text{if } i \in \partial \mathcal{M} \\
0 & \text{otherwise}
\end{cases}
$$

Where (all computed only for boundary neurons):
- $\Delta_{\text{goal\_progress}}(i, t) = \text{progress}(G, t) - \text{progress}(G, t-\delta t | i \text{ inactive})$ = change in goal progress when boundary neuron $i$ is active vs inactive
- $\text{removal\_impact}(i, t) = \text{goal\_value}(G, t) - \text{goal\_value}(G, t | i \text{ removed})$ = reduction in goal value if boundary neuron $i$ is removed (only affects boundary goal $G$)
- $\text{synchronization\_contribution}(i, t) = r_{\partial \mathcal{M}}(t) - r_{\partial \mathcal{M}}(t | i \text{ desynchronized})$ = change in boundary order parameter when boundary neuron $i$ synchronizes

**Boundary-Constrained Goal Progress**:
- Only measure progress for goals $G \in \mathcal{G}_{\text{boundary}}$ (goals with boundary neurons)
- Progress computed from boundary neuron activations: $\text{progress}(G, t) = \frac{1}{|\partial \mathcal{M}_G|} \sum_{i \in \partial \mathcal{M}_G} s_i(t)$

**Operational Definition**: Contribution is measured by **counterfactual impact** - what happens to goal pursuit when the boundary neuron's state changes. **Only boundary neurons contribute**.
```

**Rationale**: Importance defined in terms of other metrics is circular. Contribution measures actual impact on goal achievement via counterfactuals. **All computation happens only on boundaries**.

---

### 3. Replace "Drift" with Policy Divergence (Sections 3.2, 3.4)

**Current (Line ~839)**:
```
drift(G_i, G_j) = D_KL(P(actions|G_i) || P(actions|G_j))
```

**Replacement**:
```
**Policy Divergence Detection (Boundary-Constrained)**:

Measure how much the action distribution changes when switching goals:

$$
\text{policy\_divergence}(G_i, G_j) = \begin{cases}
D_{KL}(P(a | G_i, s_{\partial \mathcal{M}}) \| P(a | G_j, s_{\partial \mathcal{M}})) & \text{if } G_i, G_j \in \mathcal{G}_{\text{boundary}} \\
0 & \text{otherwise}
\end{cases}
$$

Where:
- $P(a | G, s_{\partial \mathcal{M}})$ is the action distribution under goal $G$ for **boundary state** $s_{\partial \mathcal{M}}$ (only boundary neuron activations)
- Only compute divergence for boundary goals

**Goal Value Change (Boundary-Constrained)**:

Measure how goal values change with new information:

$$
\text{value\_change}(G_i, G_j, \text{info}) = \begin{cases}
|V(G_i | \text{info}) - V(G_i | \text{no info})| - |V(G_j | \text{info}) - V(G_j | \text{no info})| & \text{if } G_i, G_j \in \mathcal{G}_{\text{boundary}} \\
0 & \text{otherwise}
\end{cases}
$$

Where $V(G | \text{info})$ is computed **only from boundary goals** (Section 3.3).

If $\text{value\_change} > \theta_{\text{switch}}$: new boundary goal $G_j$ benefits more from information than current boundary goal $G_i$.

**Context Distance (Boundary-Constrained)** (Section 3.4):

Measure distance between current context and target context:

$$
\text{context\_distance}(\mathbf{c}_t, \mathbf{c}_{\text{target}}) = \|\mathbf{c}_t - \mathbf{c}_{\text{target}}\|_2
$$

Where:
- $\mathbf{c}_t = [\text{recent\_boundary\_goals}, \text{user\_feedback\_vector}, \text{boundary\_progress\_vector}]$ (normalized, only boundary goals)
- $\mathbf{c}_{\text{target}} = [\text{target\_boundary\_goals}, \text{target\_feedback}, \text{target\_boundary\_progress}]$ (normalized, only boundary goals)

**Efficiency**: 
- **Traditional**: Compute divergence for all $|\mathcal{G}|$ goals → $O(|\mathcal{G}|^2)$
- **Topological**: Compute only for $|\mathcal{G}_{\text{boundary}}|$ boundary goals → $O(|\mathcal{G}_{\text{boundary}}|^2)$
- **Speedup**: $|\mathcal{G}|^2 / |\mathcal{G}_{\text{boundary}}|^2$ (typically 25-100x reduction)

**Operational Definition**: Drift is measured by observable changes in policy, goal values, or context vectors - not an abstract concept. **All computation happens only on boundaries**.
```

**Rationale**: "Drift" is a label. Policy divergence, value change, and context distance are measurable behaviors. **All computation happens only on boundaries**.

---

### 4. Replace "Dropout Risk" with Measurable Decay Rate (Section 3.7)

**Current (Line ~1107)**:
```
- Dropout Risk: Probability of being abandoned
```

**Replacement**:
```
**Goal Abandonment Prediction (Boundary-Constrained)**:

Measure factors that predict goal abandonment:

$$
\text{abandonment\_risk}(G, t) = \begin{cases}
f(\text{time\_since\_active}, \text{switch\_count}, \text{progress\_rate}, \text{goal\_value}) & \text{if } G \in \mathcal{G}_{\text{boundary}} \\
\text{cached\_risk} & \text{otherwise}
\end{cases}
$$

Where (all computed only for boundary goals):
- $\text{time\_since\_active} = t - t_{\text{last_active}}$ (measurable, only for boundary goals)
- $\text{switch\_count} = |\{t' : \text{goal switched at } t' \land G \in \mathcal{G}_{\text{boundary}}(t')\}|$ (measurable, only boundary switches)
- $\text{progress\_rate} = \frac{\text{progress}(G, t) - \text{progress}(G, t-\Delta t)}{\Delta t}$ (measurable, progress from boundary neurons only)
- $\text{goal\_value} = V(G, t)$ (measurable, computed only from boundary goals - Section 3.3)

**Boundary-Constrained Progress Rate**:
- Progress computed from boundary neuron activations: $\text{progress}(G, t) = \frac{1}{|\partial \mathcal{M}_G|} \sum_{i \in \partial \mathcal{M}_G} s_i(t)$
- Only measure progress for goals with boundary neurons

**Decay Model**:

$$
\text{abandonment\_probability}(G, t) = \begin{cases}
1 - \exp(-\lambda \cdot \text{time\_since\_active} \cdot (1 + \alpha \cdot \text{switch\_count})) & \text{if } G \in \mathcal{G}_{\text{boundary}} \\
\text{cached\_probability} & \text{otherwise}
\end{cases}
$$

Where:
- $\lambda$ = base decay rate (learned from boundary goal data)
- $\alpha$ = switch count penalty (learned from boundary goal switches)

**Efficiency**: 
- **Traditional**: Compute risk for all $|\mathcal{G}|$ goals → $O(|\mathcal{G}|)$
- **Topological**: Compute only for $|\mathcal{G}_{\text{boundary}}|$ boundary goals → $O(|\mathcal{G}_{\text{boundary}}|)$
- **Speedup**: $|\mathcal{G}| / |\mathcal{G}_{\text{boundary}}|$ (typically 5-10x reduction)

**Operational Definition**: Abandonment risk is computed from observable metrics (time inactive, switches, progress), not an abstract probability. **All computation happens only on boundaries**.
```

**Rationale**: "Dropout risk" without a model is vague. Use measurable factors (time, switches, progress) to predict abandonment. **All computation happens only on boundaries**.

---

### 5. Replace "Abstraction" Language with "Compression" (Section 3.8)

**Current (Line ~1137)**:
```
2. **Abstraction (Boundary-Only)**: Replace verbose descriptions
```

**Replacement**:
```
2. **Concept Extraction (Boundary-Only)**: Extract key concepts from verbose descriptions **only for boundary goals**
   - "Design safety features (auto-shutoff, thermal protection, etc.)" → `["safety", "auto-shutoff"]`
   - Embedding captures full meaning: $\mathbf{e}_G$ encodes all safety concepts
   - **Efficiency**: Only extract concepts for goals with boundary neurons
```

**Rationale**: "Abstraction" is vague. "Concept extraction" describes the operation: extracting key concepts from descriptions.

---

### 6. Replace "Abstraction Levels" with Hierarchical Depth (Section 3.9.1)

**Current (Line ~1272)**:
```
- **Abstraction levels**: High-level goals decompose into lower-level sub-goals
```

**Replacement**:
```
- **Hierarchical Depth (Boundary-Constrained)**: Goals decompose into sub-goals, forming trees of depth $d$
  $$
  d(G) = \begin{cases}
  \max_{\text{path from } G \text{ to leaf}} |\text{path}| & \text{if } G \in \mathcal{G}_{\text{boundary}} \\
  \text{cached} & \text{otherwise}
  \end{cases}
  $$
  Only measure depth for boundary goals.
  
  - **Granularity (Boundary-Constrained)**: 
    $$
    g(G) = \begin{cases}
    \frac{|\text{subgoals}(G) \cap \mathcal{G}_{\text{boundary}}|}{d(G)} & \text{if } G \in \mathcal{G}_{\text{boundary}} \\
    \text{cached} & \text{otherwise}
    \end{cases}
    $$
    Only count boundary sub-goals per level.
  
  - **Measurable**: Count levels, count boundary sub-goals per level

**Efficiency**: 
- **Traditional**: Compute depth for all $|\mathcal{G}|$ goals → $O(|\mathcal{G}|)$
- **Topological**: Compute only for $|\mathcal{G}_{\text{boundary}}|$ boundary goals → $O(|\mathcal{G}_{\text{boundary}}|)$
- **Speedup**: $|\mathcal{G}| / |\mathcal{G}_{\text{boundary}}|$ (typically 5-10x reduction)
```

**Rationale**: "Abstraction levels" is vague. Hierarchical depth and granularity are measurable structural properties. **All computation happens only on boundaries**.

---

### 7. Replace "Threat Score" with Observed Impact (Section 3.9.2)

**Current (Line ~1576)**:
```
threat_score(thought, G) = {high if d < θ_close AND contradicts(G)}
```

**Replacement**:
```
**Threat Impact Measurement (Boundary-Constrained)**:

Measure actual impact of thoughts/information on goal pursuit:

$$
\text{threat\_impact}(\text{thought}, G, t) = \begin{cases}
\beta_1 \cdot \Delta_{\text{goal\_value}} + \beta_2 \cdot \Delta_{\text{confusion}} + \beta_3 \cdot \Delta_{\text{abandonment\_risk}} & \text{if } G \in \mathcal{G}_{\text{boundary}} \\
0 & \text{otherwise}
\end{cases}
$$

Where (all computed only for boundary goals):
- $\Delta_{\text{goal\_value}} = V(G, t | \text{thought}) - V(G, t | \text{no thought})$ = change in boundary goal value when thought is present (computed from boundary neurons)
- $\Delta_{\text{confusion}} = C_G(t | \text{thought}) - C_G(t | \text{no thought})$ = increase in confusion signal from boundary neurons (Section 1.7)
- $\Delta_{\text{abandonment\_risk}} = \text{abandonment\_risk}(G, t | \text{thought}) - \text{abandonment\_risk}(G, t | \text{no thought})$ = increase in abandonment risk for boundary goal

**Boundary-Constrained Confusion Signal**:
- Confusion computed only from boundary neuron thoughts: $C_G(t) = \sum_{\tau=t-T}^{t} w(\tau) \cdot \mathbb{I}(\text{thought}_\tau = \text{confusion} \land \text{neuron}_\tau \in \partial \mathcal{M}_G)$

**Threat Classification**:

$$
\text{threat\_level}(\text{thought}, G) = \begin{cases}
\text{high} & \text{if } G \in \mathcal{G}_{\text{boundary}} \text{ AND } (\text{threat\_impact} > \theta_{\text{high}} \text{ OR } (\Delta_{\text{goal\_value}} < -\delta_{\text{value}} \text{ AND } \Delta_{\text{confusion}} > \delta_{\text{confusion}})) \\
\text{medium} & \text{if } G \in \mathcal{G}_{\text{boundary}} \text{ AND } (\Delta_{\text{confusion}} > \delta_{\text{confusion}} \text{ OR } \Delta_{\text{abandonment\_risk}} > \delta_{\text{risk}}) \\
\text{low} & \text{otherwise}
\end{cases}
$$

Where thresholds $\theta_{\text{high}}, \delta_{\text{value}}, \delta_{\text{confusion}}, \delta_{\text{risk}}$ are learned from observed impacts on boundary goals.

**Efficiency**: 
- **Traditional**: Predict threats for all $|\mathcal{G}|$ goals → $O(|\mathcal{G}|)$
- **Topological**: Predict only for $|\mathcal{G}_{\text{boundary}}|$ boundary goals → $O(|\mathcal{G}_{\text{boundary}}|)$
- **Speedup**: $|\mathcal{G}| / |\mathcal{G}_{\text{boundary}}|$ (typically 5-10x reduction)

**Operational Definition**: Threat is measured by observed impact on goal value, confusion, and abandonment risk - not arbitrary embedding distances. **All computation happens only on boundaries**.
```

**Rationale**: Threat score based on embedding distance is arbitrary. Measure actual impact on goal pursuit (value, confusion, abandonment). **All computation happens only on boundaries**.

---

### 8. Specify "Topological Invariants" Explicitly (Section 3.9.4)

**Current (Line ~1685)**:
```
- Goal structure must preserve certain invariants
```

**Replacement**:
```
**Measurable Topological Invariants (Boundary-Constrained)**:

Goal structure must preserve these measurable properties (computed only for boundary goals):

1. **Dependency Graph Connectivity (Boundary-Constrained)**: 
   $$
   H_0(\mathcal{H}_G) = \begin{cases}
   \text{number of connected components in } \mathcal{H}_G & \text{if } G \in \mathcal{G}_{\text{boundary}} \\
   \text{cached} & \text{otherwise}
   \end{cases}
   $$
   - Invariant: $H_0(\mathcal{H}_G) = H_0(\mathcal{H}_{G, \text{threatened}})$ (same number of components)
   - Only compute for boundary goals

2. **Goal Hierarchy Depth (Boundary-Constrained)**: 
   $$
   d(G) = \begin{cases}
   \max_{\text{path from } G \text{ to leaf}} |\text{path}| & \text{if } G \in \mathcal{G}_{\text{boundary}} \\
   \text{cached} & \text{otherwise}
   \end{cases}
   $$
   - Invariant: $d(G) \leq d(G_{\text{threatened}}) + \epsilon$ (depth doesn't increase dramatically)
   - Only measure depth for boundary goals

3. **Dependency Satisfaction (Boundary-Constrained)**: 
   $$
   D_{\text{sat}} = \begin{cases}
   \frac{|\{(G_i, G_j) \in \mathcal{G}_{\text{boundary}} : G_j \text{ achieved} \land G_i \text{ depends on } G_j\}|}{|\text{dependencies in } \mathcal{G}_{\text{boundary}}|} & \text{if } G \in \mathcal{G}_{\text{boundary}} \\
   \text{cached} & \text{otherwise}
   \end{cases}
   $$
   - Invariant: $D_{\text{sat}}(G) \geq D_{\text{sat}}(G_{\text{threatened}}) - \delta$ (satisfaction doesn't drop below threshold)
   - Only count dependencies between boundary goals

4. **Persistent Homology Features (Boundary-Constrained)**: 
   $$
   H_1(\mathcal{H}_G) = \begin{cases}
   \text{number of cycles in dependency graph} & \text{if } G \in \mathcal{G}_{\text{boundary}} \\
   \text{cached} & \text{otherwise}
   \end{cases}
   $$
   - Invariant: $H_1(\mathcal{H}_G) = H_1(\mathcal{H}_{G, \text{threatened}})$ (same cycles)
   - Only compute cycles for boundary goal graphs

**Goal Structure Persistence (Boundary-Constrained)**:

$$
\text{goal\_persists}(G, \mathcal{T}) = \begin{cases}
\text{true} & \text{if } G \in \mathcal{G}_{\text{boundary}} \text{ AND } H_0(\mathcal{H}_G) = H_0(\mathcal{H}_{G, \text{threatened}}) \text{ AND } d(G) \leq d(G_{\text{threatened}}) + \epsilon \text{ AND } D_{\text{sat}}(G) \geq D_{\text{sat}}(G_{\text{threatened}}) - \delta \\
\text{false} & \text{otherwise}
\end{cases}
$$

**Efficiency**: 
- **Traditional**: Compute invariants for all $|\mathcal{G}|$ goals → $O(|\mathcal{G}|)$
- **Topological**: Compute only for $|\mathcal{G}_{\text{boundary}}|$ boundary goals → $O(|\mathcal{G}_{\text{boundary}}|)$
- **Speedup**: $|\mathcal{G}| / |\mathcal{G}_{\text{boundary}}|$ (typically 5-10x reduction)

**Operational Definition**: Topological invariants are specific measurable properties (connectivity, depth, satisfaction, cycles) - not abstract concepts. **All computation happens only on boundaries**.
```

**Rationale**: "Certain invariants" is vague. Specify measurable invariants: connectivity, depth, dependency satisfaction, persistent homology. **All computation happens only on boundaries**.

---

### 9. Replace "Goal Integrity" with Persistence/Value Stability (Section 3.9.4)

**Current (Line ~1660)**:
```
- **Topological invariants** preserve goal integrity
```

**Replacement**:
```
- **Topological invariants** preserve goal persistence and value stability (boundary-constrained)

**Goal Persistence (Boundary-Constrained)**: Goal survives threats (doesn't get abandoned)
$$
\text{persistence}(G, \mathcal{T}) = \begin{cases}
\mathbb{I}(\text{goal\_state}(G, t+\Delta t | \mathcal{T}) \neq \text{abandoned}) & \text{if } G \in \mathcal{G}_{\text{boundary}} \\
\text{cached} & \text{otherwise}
\end{cases}
$$

Only measure persistence for boundary goals.

**Goal Value Stability (Boundary-Constrained)**: Goal value doesn't drop significantly
$$
\text{value\_stability}(G, \mathcal{T}) = \begin{cases}
\frac{V(G, t+\Delta t | \mathcal{T})}{V(G, t | \text{no } \mathcal{T})} \geq \theta_{\text{stability}} & \text{if } G \in \mathcal{G}_{\text{boundary}} \\
\text{cached} & \text{otherwise}
\end{cases}
$$

Where $V(G, t)$ is computed only from boundary goals (Section 3.3).

**Efficiency**: 
- **Traditional**: Compute persistence/stability for all $|\mathcal{G}|$ goals → $O(|\mathcal{G}|)$
- **Topological**: Compute only for $|\mathcal{G}_{\text{boundary}}|$ boundary goals → $O(|\mathcal{G}_{\text{boundary}}|)$
- **Speedup**: $|\mathcal{G}| / |\mathcal{G}_{\text{boundary}}|$ (typically 5-10x reduction)

**Operational Definition**: "Integrity" is measured by persistence (survival) and value stability (value doesn't drop) - not an abstract property. **All computation happens only on boundaries**.
```

**Rationale**: "Integrity" is vague. Measure persistence (survival) and value stability (value preservation). **All computation happens only on boundaries**.

---

### 10. Replace "Pattern Understanding" with Prediction Accuracy (Section 3.9.1)

**Current (Line ~1462)**:
```
pattern_understanding = dependency_consistency · transfer_success
```

**Replacement**:
```
**Pattern Prediction Accuracy (Boundary-Constrained)**:

Measure how well patterns predict dependencies and transfers:

$$
\text{prediction\_accuracy}(\mathcal{H}) = \begin{cases}
\alpha \cdot \text{dependency\_prediction\_accuracy} + \beta \cdot \text{transfer\_prediction\_accuracy} & \text{if } \mathcal{H} \text{ contains boundary goals} \\
0 & \text{otherwise}
\end{cases}
$$

Where (all computed only for boundary goals):
- $\text{dependency\_prediction\_accuracy} = \frac{|\{(G_i, G_j) \in \mathcal{G}_{\text{boundary}} : \text{pattern\_predicts}(G_i \rightarrow G_j) \land \text{correct}(G_i \rightarrow G_j)\}|}{|\{(G_i, G_j) \in \mathcal{G}_{\text{boundary}} : \text{pattern\_predicts}(G_i \rightarrow G_j)\}|}$ = fraction of predicted boundary dependencies that are correct
- $\text{transfer\_prediction\_accuracy} = \frac{|\{G_{\text{similar}} \in \mathcal{G}_{\text{boundary}} : \text{pattern\_transfers}(\mathcal{H}, G_{\text{similar}}) \land \text{successful}(G_{\text{similar}})\}|}{|\{G_{\text{similar}} \in \mathcal{G}_{\text{boundary}} : \text{pattern\_transfers}(\mathcal{H}, G_{\text{similar}})\}|}$ = fraction of boundary transfers that succeed

**Generalization to Novel Goals (Boundary-Constrained)**:

$$
\text{generalization\_score}(\mathcal{H}) = \mathbb{E}_{G_{\text{novel}} \in \mathcal{G}_{\text{boundary}} \sim \text{unseen}}[\text{success}(\text{decompose}(G_{\text{novel}}))]
$$

Only test generalization on boundary goals.

**Confusion Reduction (Boundary-Constrained)**:

$$
\text{confusion\_reduction}(\mathcal{H}) = C_G(t | \text{no } \mathcal{H}) - C_G(t | \text{with } \mathcal{H})
$$

Where $C_G(t)$ is computed only from boundary neuron thoughts (Section 1.7).

**Pattern Understanding**:

$$
\text{pattern\_understanding}(\mathcal{H}) = \text{prediction\_accuracy}(\mathcal{H}) \cdot \text{generalization\_score}(\mathcal{H}) \cdot \text{confusion\_reduction}(\mathcal{H})
$$

**Efficiency**: 
- **Traditional**: Compute understanding for all decompositions → $O(|\mathcal{H}|^2)$
- **Topological**: Compute only for boundary goal decompositions → $O(|\mathcal{G}_{\text{boundary}}|^2)$
- **Speedup**: $|\mathcal{H}|^2 / |\mathcal{G}_{\text{boundary}}|^2$ (typically 25-100x reduction)

**Operational Definition**: Understanding is measured by prediction accuracy, generalization to novel goals, and confusion reduction - not circular consistency checks. **All computation happens only on boundaries**.
```

**Rationale**: Pattern understanding defined via consistency/transfer is circular. Measure prediction accuracy, generalization, and confusion reduction. **All computation happens only on boundaries**.

---

### 11. Replace "Composition Quality" Compatibility with Co-occurrence/Success (Section 3.9.1)

**Current (Line ~1425)**:
```
compatible(p_i, p_j) via learned patterns
```

**Replacement**:
```
**Primitive Co-occurrence Frequency (Boundary-Constrained)**:

Measure how often primitives appear together in successful decompositions:

$$
\text{co\_occurrence}(p_i, p_j) = \frac{|\{\mathcal{H} : p_i \in \mathcal{H} \land p_j \in \mathcal{H} \land \text{successful}(\mathcal{H}) \land \mathcal{H} \text{ contains boundary goals}\}|}{|\{\mathcal{H} : \text{successful}(\mathcal{H}) \land \mathcal{H} \text{ contains boundary goals}\}|}
$$

Only count decompositions with boundary goals.

**Goal Success Rate When Used Together (Boundary-Constrained)**:

$$
\text{success\_rate}(p_i, p_j) = \frac{|\{G \in \mathcal{G}_{\text{boundary}} : p_i, p_j \in \mathcal{H}_G \land \text{achieved}(G)\}|}{|\{G \in \mathcal{G}_{\text{boundary}} : p_i, p_j \in \mathcal{H}_G\}|}
$$

Only count boundary goals.

**Dependency Satisfaction Rate (Boundary-Constrained)**:

$$
\text{dependency\_satisfaction}(p_i, p_j) = \frac{|\{G \in \mathcal{G}_{\text{boundary}} : p_i \text{ depends on } p_j \land \text{satisfied}(G)\}|}{|\{G \in \mathcal{G}_{\text{boundary}} : p_i \text{ depends on } p_j\}|}
$$

Only count boundary goals.

**Compatibility Score**:

$$
\text{compatible}(p_i, p_j) = \begin{cases}
\text{true} & \text{if } \text{co\_occurrence}(p_i, p_j) > \theta_{\text{co}} \text{ AND } \text{success\_rate}(p_i, p_j) > \theta_{\text{success}} \\
\text{false} & \text{otherwise}
\end{cases}
$$

**Efficiency**: 
- **Traditional**: Compute compatibility for all primitive pairs → $O(|\mathcal{P}|^2)$
- **Topological**: Compute only for primitives used in boundary goals → $O(|\mathcal{P}_{\text{boundary}}|^2)$
- **Speedup**: $|\mathcal{P}|^2 / |\mathcal{P}_{\text{boundary}}|^2$ (typically 25-100x reduction)

**Operational Definition**: Compatibility is measured by co-occurrence frequency and success rates - not learned patterns (which would be circular). **All computation happens only on boundaries**.
```

**Rationale**: Compatibility via learned patterns is circular. Measure co-occurrence frequency and success rates from data. **All computation happens only on boundaries**.

---

### 12. Explicitly Define Goal Value R(G) (Section 3.3)

**Current (Line ~870)**:
```
Where R(G) includes:
- Direct reward from pursuing G via boundary neurons
- Information gain from learning on boundaries
- Alignment with terminal goal GT measured via boundary goal pursuit
```

**Replacement**:
```
**Explicit Goal Value Definition (Boundary-Constrained)**:

$$
R(G, t) = \begin{cases}
w_1 \cdot R_{\text{progress}}(G, t) + w_2 \cdot R_{\text{info}}(G, t) + w_3 \cdot R_{\text{alignment}}(G, t) + w_4 \cdot R_{\text{efficiency}}(G, t) & \text{if } G \in \mathcal{G}_{\text{boundary}} \\
0 & \text{otherwise}
\end{cases}
$$

Where (all computed only for boundary goals):

1. **Progress Reward (Boundary-Constrained)**: 
   $$
   R_{\text{progress}}(G, t) = \frac{\text{progress}(G, t) - \text{progress}(G, t-\Delta t)}{\Delta t}
   $$
   Where $\text{progress}(G, t) = \frac{1}{|\partial \mathcal{M}_G|} \sum_{i \in \partial \mathcal{M}_G} s_i(t)$ (only boundary neuron activations)

2. **Information Gain Reward (Boundary-Constrained)**: 
   $$
   R_{\text{info}}(G, t) = H(\text{knowledge} | G, t-\Delta t) - H(\text{knowledge} | G, t)
   $$
   Where knowledge entropy computed only from boundary-relevant concepts $\mathcal{C}_{\text{boundary}}$

3. **Alignment Reward (Boundary-Constrained)**: 
   $$
   R_{\text{alignment}}(G, t) = \begin{cases}
   1.0 & \text{if } G = G_T \text{ (terminal goal)} \land G \in \mathcal{G}_{\text{boundary}} \\
   \text{dependency\_path\_length}(G, G_T)^{-1} & \text{if } G \text{ enables } G_T \land G, G_T \in \mathcal{G}_{\text{boundary}} \\
   0.0 & \text{otherwise}
   \end{cases}
   $$
   Only measure alignment for boundary goals.

4. **Efficiency Reward (Boundary-Constrained)**: 
   $$
   R_{\text{efficiency}}(G, t) = \frac{\text{progress}(G, t)}{\text{time\_spent}(G) + \text{resources\_used}(G)}
   $$
   Where progress computed from boundary neurons, resources = boundary neuron activations.

**Boundary-Constrained Computation**:

All rewards computed **only for boundary goals**:
- $R_{\text{progress}}$: Only measure progress for goals with boundary neurons ($G \in \mathcal{G}_{\text{boundary}}$)
- $R_{\text{info}}$: Only measure information gain for boundary goals (concepts $\mathcal{C}_{\text{boundary}}$)
- $R_{\text{alignment}}$: Only measure alignment for boundary goals ($G, G_T \in \mathcal{G}_{\text{boundary}}$)
- $R_{\text{efficiency}}$: Only measure efficiency for boundary goals (boundary neuron resources)

**Efficiency**: 
- **Traditional**: Compute value for all $|\mathcal{G}|$ goals → $O(|\mathcal{G}|)$
- **Topological**: Compute only for $|\mathcal{G}_{\text{boundary}}|$ boundary goals → $O(|\mathcal{G}_{\text{boundary}}|)$
- **Speedup**: $|\mathcal{G}| / |\mathcal{G}_{\text{boundary}}|$ (typically 5-10x reduction)

**Operational Definition**: Goal value is explicitly defined as weighted combination of measurable rewards (progress, information, alignment, efficiency) - not vague "includes". **All computation happens only on boundaries**.
```

**Rationale**: "R(G) includes" is vague. Explicitly define R(G) as weighted combination of measurable rewards. **All computation happens only on boundaries**.

---

### 13. Explicitly Define "Information Gain" (Sections 3.3, 3.5, 3.6)

**Current (Line ~936, ~1042)**:
```
- Information gain from learning on boundaries
- info_gain(G_new) - cost(G_new)
```

**Replacement**:
```
**Information Gain (Boundary-Constrained)**:

$$
\text{info\_gain}(G, t) = \begin{cases}
H(\text{knowledge} | G, t-\Delta t) - H(\text{knowledge} | G, t) & \text{if } G \in \mathcal{G}_{\text{boundary}} \\
0 & \text{otherwise}
\end{cases}
$$

Where:
- **Knowledge Entropy (Boundary-Constrained)**: 
  $$
  H(\text{knowledge} | G, t) = -\sum_{c \in \mathcal{C}_{\text{boundary}}} p(c | G, t) \log p(c | G, t)
  $$
  Where $p(c | G, t)$ = frequency of concept $c$ usage in boundary goal $G$ at time $t$
  
- **Concept Usage Frequency**: 
  $$
  p(c | G, t) = \frac{|\{t' \leq t : \text{concept } c \text{ used in } G \land G \in \mathcal{G}_{\text{boundary}}(t')\}|}{|\{t' \leq t : G \in \mathcal{G}_{\text{boundary}}(t')\}|}
  $$

**Operational Definition**: Information gain is measured by reduction in concept entropy (discovery of new concepts) - not vague "learning".

**Efficiency**: 
- **Traditional**: Compute entropy for all concepts → $O(|\mathcal{C}|)$
- **Topological**: Compute only for boundary-relevant concepts → $O(|\mathcal{C}_{\text{boundary}}|)$
- **Speedup**: $|\mathcal{C}| / |\mathcal{C}_{\text{boundary}}|$ (typically 5-10x reduction)
```

**Rationale**: "Information gain" is vague. Measure reduction in concept entropy (discovery of new concepts). **All computation happens only on boundaries**.

---

### 14. Explicitly Define "Switching Cost" (Section 3.2)

**Current (Line ~862)**:
```
switching_cost
```

**Replacement**:
```
**Switching Cost (Boundary-Constrained)**:

$$
\text{switching\_cost}(G_i, G_j) = \begin{cases}
\alpha \cdot \text{context\_distance}(G_i, G_j) + \beta \cdot \text{policy\_divergence}(G_i, G_j) + \gamma \cdot \text{time\_since\_switch} & \text{if } G_i, G_j \in \mathcal{G}_{\text{boundary}} \\
\infty & \text{otherwise}
\end{cases}
$$

Where:
- $\text{context\_distance}(G_i, G_j)$ = distance between goal contexts (from Section 3, replacement #3)
- $\text{policy\_divergence}(G_i, G_j)$ = KL divergence between action distributions (from Section 3, replacement #3)
- $\text{time\_since\_switch} = t - t_{\text{last\_switch}}$ = time since last goal switch

**Operational Definition**: Switching cost is explicitly defined as weighted combination of measurable factors (context distance, policy divergence, time) - not vague parameter.

**Efficiency**: 
- **Traditional**: Compute cost for all goal pairs → $O(|\mathcal{G}|^2)$
- **Topological**: Compute only for boundary goal pairs → $O(|\mathcal{G}_{\text{boundary}}|^2)$
- **Speedup**: $|\mathcal{G}|^2 / |\mathcal{G}_{\text{boundary}}|^2$ (typically 25-100x reduction)
```

**Rationale**: "Switching cost" is vague. Explicitly define as measurable factors. **All computation happens only on boundaries**.

---

### 15. Explicitly Define "Temperature" (Section 3.2)

**Current (Line ~862)**:
```
temperature
```

**Replacement**:
```
**Adaptive Temperature (Boundary-Constrained)**:

$$
T(t) = \begin{cases}
T_0 \cdot (1 - \text{exploration\_rate}(t)) & \text{if } |\mathcal{G}_{\text{boundary}}| > 0 \\
T_0 & \text{otherwise}
\end{cases}
$$

Where:
- **Exploration Rate (Boundary-Constrained)**:
  $$
  \text{exploration\_rate}(t) = \frac{|\{G \in \mathcal{G}_{\text{boundary}} : G \text{ is novel}\}|}{|\mathcal{G}_{\text{boundary}}|}
  $$
  Fraction of boundary goals that are novel (not seen before)
  
- $T_0$ = base temperature (hyperparameter)

**Operational Definition**: Temperature adapts based on exploration rate (fraction of novel goals) - not fixed parameter.

**Efficiency**: 
- **Traditional**: Compute temperature for all goals → $O(|\mathcal{G}|)$
- **Topological**: Compute only for boundary goals → $O(|\mathcal{G}_{\text{boundary}}|)$
- **Speedup**: $|\mathcal{G}| / |\mathcal{G}_{\text{boundary}}|$ (typically 5-10x reduction)
```

**Rationale**: "Temperature" is vague. Explicitly define as adaptive exploration rate. **All computation happens only on boundaries**.

---

### 16. Explicitly Define "Dependency Path Length" (Section 3.3)

**Current (Line ~870)**:
```
dependency_path_length(G, G_T)
```

**Replacement**:
```
**Dependency Path Length (Boundary-Constrained)**:

$$
\text{path\_length}(G, G_T) = \begin{cases}
\min_{\text{path } P: G \rightarrow G_T} |P| & \text{if } G, G_T \in \mathcal{G}_{\text{boundary}} \\
\infty & \text{otherwise}
\end{cases}
$$

Where:
- $P$ = path through dependency graph: $G \rightarrow G_1 \rightarrow G_2 \rightarrow \ldots \rightarrow G_T$
- $|P|$ = number of edges in path
- Path must traverse **only boundary goals**: $G, G_1, G_2, \ldots, G_T \in \mathcal{G}_{\text{boundary}}$

**Operational Definition**: Path length is shortest path through dependency graph (measurable via graph traversal) - not vague distance.

**Efficiency**: 
- **Traditional**: Compute path for all goal pairs → $O(|\mathcal{G}|^2)$
- **Topological**: Compute only for boundary goal pairs → $O(|\mathcal{G}_{\text{boundary}}|^2)$
- **Speedup**: $|\mathcal{G}|^2 / |\mathcal{G}_{\text{boundary}}|^2$ (typically 25-100x reduction)
```

**Rationale**: "Dependency path length" is vague. Explicitly define as shortest path in dependency graph. **All computation happens only on boundaries**.

---

## Summary

All replacements:
1. **Replace abstract labels** with measurable behaviors
2. **Break circular definitions** by grounding in outcomes
3. **Specify operational definitions** for vague concepts
4. **Ground everything** in observable system dynamics
5. **Apply topological boundary efficiency** to all measurements

**Universal Boundary Constraint**: Every measurement, computation, and operation happens **only on boundaries**:
- Neurons: Only boundary neurons $\partial \mathcal{M}$ contribute
- Goals: Only boundary goals $\mathcal{G}_{\text{boundary}}$ are evaluated
- Concepts: Only boundary-relevant concepts $\mathcal{C}_{\text{boundary}}$ are used
- Thoughts: Only boundary neuron thoughts are aggregated
- Signals: Only affect boundary neurons

**Efficiency Gains**: 
- Typical speedup: 5-100x reduction in computation
- Scales with boundary size, not full network size
- Boundary size grows sublinearly: $|\partial \mathcal{M}| \sim O(\sqrt{N})$ typically

**Key Insight**: The topological boundary principle is not just an optimization—it's a fundamental architectural constraint that makes all measurements computationally tractable while preserving biological plausibility.

These replacements make the theory testable, implementable, and computationally efficient.

