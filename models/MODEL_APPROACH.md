# POI Selection Strategy for The Spatio-Temporal Graph Attention Network

## **Objective**
The goal is to reduce the number of points of interest (POIs) in the dataset while preserving key relationships and influential locations. This will facilitate better interpretability and computational efficiency while allowing a refined model to capture spatiotemporal dynamics more effectively.

## **Approach**
1. **Train the ST-GAT Model**  
   - Apply a **single GAT layer** and pass every time step through that layer.
   - This ensures that the attention values encode **spatial significance** across the entire training data time period.
   - The model learns to weigh important POIs based on long-term influence.
   - Perform grid-search with expanding window cross validation approach (time-series friendly) to select proper hyperparameters
   - Implement the cosine annealing (no restarts) learning rate scheduler to decay the learning rate as the model approaches convergence.
   - Freeze the GAT layer weights in attempt to independantly fine-tune the 2 LSTM layers and final linear layer.

2. **Extract Attention Scores to Rank POIs**  
   - Use the trained model’s GAT attention values to compute a measure of **significance** for each POI.
   - Since the GAT attention computation procedure applied the softmax transform, the column sum of the attention matrix serves as a measure of POI importance
   - Define a POI’s overall significance as the average attention weight it receives across all other POIs.
   - Define a selection criterion based on the computed significance (e.g., top 20% of POIs).
   - This ensures that only the most **influential** POIs are retained for  analysis procedures.

5. **Analyze Relationships & Predictions**  
   - Investigate whether selected POIs correlate with geographical features and events occurring during the training data time period.
   - Check for anomalies and emerging patterns in the trained model’s predictions.

This methodology ensures an optimal balance between computational efficiency and model effectiveness and interpretability while allowing for meaningful analysis of POI relationships.

