### **Applying GAT to All Time Steps Together vs. Per Time Step**

#### **Why Did They Apply GAT to All Time Steps Together?**

The reason GAT was applied once to all time steps combined (instead of separately for each time step) likely stems from:

1. **Computational Efficiency**  
   - Applying the GAT once reduces the number of forward passes, avoiding repeated computations for each time step.  
   - Applying GAT per time step requires looping through each step, which is slower.

2. **Global Feature Sharing**  
   - The attention mechanism can aggregate information across all time steps simultaneously rather than treating each step independently.  
   - This helps the LSTM learn dependencies from a pre-learned spatial representation rather than raw node features.

3. **Flattened Representation for LSTM**  
   - The LSTM processes transformed node features that already encode information from all time steps in a static way.

---

### **How Does This Change the Output Representation?**

#### **1. GAT Applied to All Time Steps Together (Current Implementation)**
- The GAT extracts spatial features from the entire input (all time steps combined).
- The LSTM then captures temporal relationships using this transformed representation.

**Effect:**  
- Attention scores are computed **globally**, meaning information from later time steps could influence earlier ones before the LSTM sees them.  
- The LSTM works with node representations that already encode spatial information from all time steps.

---

#### **2. GAT Applied Per Time Step (Alternative Approach)**
- The GAT layer processes **each time step separately**, preserving the temporal separation.
- The LSTM captures how these spatial relationships evolve over time instead of working on a pre-mixed representation.

**Effect:**  
- Spatial relationships are learned **independently** at each time step, preventing future data from "leaking" into earlier time steps.  
- The model explicitly captures how node relationships change over time.

---

### **Trade-Off: Which Approach is Better?**

| Approach | Pros | Cons |
|----------|------|------|
| **GAT applied to all time steps together** (current) | Faster, more global spatial awareness, may help LSTM learn long-term dependencies | Future time steps may influence past ones before LSTM sees them (leakage), less explicit modeling of spatial evolution |
| **GAT applied per time step** (alternative) | Explicitly captures spatial dynamics per time step, more faithful to ST-GAT approach | Slower computation (applies GAT multiple times), may require more parameters |

---

### **Final Thoughts**
- If the goal is to explicitly model how graph structures evolve over time, **applying GAT per time step is better**.
- If efficiency and a global representation are priorities, the current approach might be acceptable.

Would need to modify the implementation to apply GAT separately per time step if choosing the second approach.

