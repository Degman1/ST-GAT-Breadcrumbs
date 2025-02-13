# POI Selection Strategy for The Spatio-Temporal Graph Attention Network

## Analyzing The ST-GAT Traffic Prediction [Medium Article] Approach

Article: https://medium.com/stanford-cs224w/predicting-los-angeles-traffic-with-graph-neural-networks-52652bc643b1

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




## **Objective**
The goal is to reduce the number of points of interest (POIs) in the dataset while preserving key relationships and influential locations. This will facilitate better interpretability and computational efficiency while allowing a refined model to capture spatiotemporal dynamics more effectively.

## **Approach**
1. **Train an Initial ST-GAT Model (Single GAT Layer Over Aggregated Features)**  
   - Apply a **single GAT layer** to the aggregated node features over all time steps.
   - This ensures that the attention values encode **both spatial and temporal significance**.
   - The model learns to weigh important POIs based on long-term influence rather than short-term fluctuations.

2. **Extract Attention Scores to Rank POIs**  
   - Use the trained model’s attention values to compute a measure of **significance** for each POI.
   - Aggregate attention weights across multiple time steps to get a robust ranking.
   - Normalize attention values if necessary to prevent bias from outlier POIs.

3. **Select the Top-Ranked POIs**  
   - Define a POI’s overall significance as the average attention weight it receives across all other POIs. Only incoming attention is considered by taking a mean over the columns of the attention matrix
   - Define a selection criterion based on the computed significance (e.g., top 20% of POIs).
   - This ensures that only the most **influential** POIs are retained for further analysis.

4. **Train a Refined ST-GAT Model (GAT Over Each Time Step)**  
   - Use the reduced set of POIs to train a new model where **GAT is applied at each time step**.
   - This allows for a more precise temporal understanding of influential POIs.

5. **Analyze Relationships & Predictions**  
   - Investigate whether selected POIs correlate with geographical features.
   - Check for anomalies and emerging patterns in the refined model’s predictions.
   - Compare results against the full dataset model to validate improvements.

## **Advantages of This Approach**
- **Captures Long-Term Spatiotemporal Significance**: [Temporal Attention] The first GAT model ensures attention values reflect both spatial and temporal importance before filtering POIs.
- **Computational Efficiency**: Reducing the number of POIs lowers the complexity of the final model, leading to faster training and inference.
- **Improved Interpretability**: Selecting POIs based on learned attention weights provides a principled way to focus on significant locations.

## **Considerations & Potential Adjustments**
- **Attention Score Normalization**: Normalize or aggregate scores across multiple runs to ensure robustness.
- **Dynamic POI Selection**: Some POIs might be important only at specific times; an alternative approach could involve dynamic selection per time interval.
- **Reintroducing Removed POIs**: After training the refined model, check whether any removed POIs contribute to error reduction.

This methodology ensures an optimal balance between computational efficiency and model effectiveness while allowing for meaningful analysis of POI relationships.

