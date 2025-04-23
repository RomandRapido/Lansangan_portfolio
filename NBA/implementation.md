# Implementing HDBSCAN in Python: A Comprehensive Guide

HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) is a powerful clustering algorithm that extends DBSCAN by converting it into a hierarchical clustering algorithm. This guide will walk you through implementing HDBSCAN in Python, from installation to advanced usage with real-world examples focused on NBA player clustering.

## Table of Contents

1. [Installation](#installation)
2. [Basic Usage](#basic-usage)
3. [Preparing NBA Player Data](#preparing-nba-player-data)
4. [HDBSCAN Parameters](#hdbscan-parameters)
5. [Visualizing Clusters](#visualizing-clusters)
6. [Handling Outliers](#handling-outliers)
7. [Performance Optimization](#performance-optimization)
8. [Advanced Techniques](#advanced-techniques)
9. [Complete Example](#complete-example)
10. [Comparing with Other Algorithms](#comparing-with-other-algorithms)

## Installation

First, you'll need to install the HDBSCAN package. You can do this using pip:

```python
pip install hdbscan
```

If you're using Anaconda:

```python
conda install -c conda-forge hdbscan
```

For the examples in this guide, you'll also need these packages:

```python
pip install numpy pandas matplotlib scikit-learn seaborn
```

## Basic Usage

Here's a simple example to get started with HDBSCAN:

```python
import numpy as np
import hdbscan
import matplotlib.pyplot as plt

# Create some sample data
np.random.seed(0)
data = np.random.randn(1000, 2)
data = np.vstack([data, np.random.randn(100, 2) * 0.3 + np.array([4, 4])])

# Run HDBSCAN
clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5)
cluster_labels = clusterer.fit_predict(data)

# Plot the results
plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# Extract cluster characteristics for each algorithm
for algo in ['HDBSCAN', 'KMeans', 'DBSCAN']:
    print(f"\n{algo} Cluster Characteristics:")
    for cluster_id in sorted(set(df_comparison[f'{algo}_Cluster'])):
        if cluster_id == -1:
            continue  # Skip outliers for this analysis
            
        cluster_df = df_comparison[df_comparison[f'{algo}_Cluster'] == cluster_id]
        
        # Calculate mean stats for this cluster
        mean_stats = cluster_df[features].mean().round(2)
        
        # Compare to overall means
        overall_means = df[features].mean()
        percent_diff = ((mean_stats - overall_means) / overall_means * 100).round(1)
        
        # Find the most distinctive features
        distinctive = []
        for feature, pct in zip(features, percent_diff):
            if abs(pct) >= 20:  # At least 20% different from overall mean
                direction = "higher" if pct > 0 else "lower"
                distinctive.append(f"{feature} ({pct}% {direction})")
        
        # Print cluster profile
        print(f"  Cluster {cluster_id} ({len(cluster_df)} players):")
        print(f"    Notable players: {', '.join(cluster_df['Player'].head(3))}")
        if distinctive:
            print(f"    Distinctive features: {', '.join(distinctive)}")
        else:
            print("    No highly distinctive features")
```

This comparison helps illustrate the strengths and weaknesses of each algorithm for NBA player clustering:

1. **HDBSCAN**: Best at identifying natural clusters and outliers (unique players)
2. **KMeans**: Forces all players into clusters, which may group unique players incorrectly
3. **DBSCAN**: Similar to HDBSCAN but less adaptive to varying cluster densities

## Real-World NBA Player Clustering

Let's implement a complete solution for clustering NBA players with a larger dataset. This example uses real NBA player statistics from a recent season:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import hdbscan
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Load NBA player data (you can replace this with your own data source)
# For this example, we'll create a function to simulate loading player data
def load_nba_data():
    # This would typically be a CSV or API call in a real application
    # Sample data for demonstration purposes (expanded from earlier example)
    data = {
        'Player': [
            'LeBron James', 'Stephen Curry', 'Nikola Jokic', 'Joel Embiid', 
            'Giannis Antetokounmpo', 'Luka Doncic', 'Kevin Durant', 'Damian Lillard', 
            'Jayson Tatum', 'Anthony Davis', 'Draymond Green', 'Rudy Gobert',
            'James Harden', 'Chris Paul', 'Bam Adebayo', 'Kyrie Irving',
            'Kawhi Leonard', 'Jimmy Butler', 'Devin Booker', 'Trae Young',
            'Ja Morant', 'Donovan Mitchell', 'Zion Williamson', 'Bam Adebayo',
            'Domantas Sabonis', 'Jaylen Brown', 'Deandre Ayton', 'Karl-Anthony Towns',
            'Zach LaVine', 'Bradley Beal', 'Khris Middleton', 'Julius Randle',
            'Russell Westbrook', 'Jrue Holiday', 'Fred VanVleet', 'Shai Gilgeous-Alexander',
            'Myles Turner', 'Robert Williams', 'Clint Capela', 'Mikal Bridges',
            'Joe Harris', 'Duncan Robinson', 'Seth Curry', 'Bojan Bogdanovic'
        ],
        'Points': [
            27.1, 26.8, 24.5, 30.6, 29.9, 32.4, 29.1, 28.3, 26.9, 25.9, 
            8.5, 13.6, 24.7, 13.9, 19.3, 25.8, 23.8, 22.9, 27.8, 26.2,
            24.5, 26.1, 27.0, 19.1, 18.5, 26.3, 16.3, 24.8, 24.5, 23.3,
            20.1, 21.4, 18.5, 18.3, 19.5, 23.7, 12.9, 8.0, 15.2, 14.3,
            11.5, 10.9, 15.0, 17.0
        ],
        'Rebounds': [
            7.5, 5.0, 11.8, 11.7, 11.6, 8.6, 6.7, 4.8, 8.8, 12.5, 
            7.2, 11.6, 5.7, 4.3, 9.2, 4.8, 6.5, 5.9, 4.5, 3.0,
            5.6, 4.1, 7.2, 9.5, 12.3, 6.1, 10.5, 9.8, 5.0, 4.2,
            5.4, 9.9, 7.8, 4.7, 4.3, 4.9, 6.5, 9.6, 11.1, 4.3,
            3.5, 2.6, 2.6, 3.7
        ],
        'Assists': [
            8.3, 6.1, 9.8, 3.9, 5.7, 8.0, 5.0, 7.0, 4.6, 2.6, 
            6.8, 1.0, 7.0, 8.9, 3.2, 6.0, 3.9, 5.3, 5.5, 10.2,
            7.5, 5.3, 3.7, 3.4, 7.0, 3.5, 1.7, 3.2, 4.6, 4.4,
            5.4, 5.8, 7.1, 6.1, 6.3, 5.9, 1.0, 2.0, 1.5, 2.1,
            1.5, 1.8, 2.7, 1.9
        ],
        'Steals': [
            1.0, 0.9, 1.3, 1.0, 0.8, 1.4, 0.7, 0.9, 1.1, 1.1, 
            1.0, 0.6, 1.3, 1.5, 1.1, 1.1, 1.4, 1.8, 0.8, 0.9,
            1.2, 1.5, 0.9, 1.2, 1.0, 1.1, 0.7, 0.9, 0.6, 0.9,
            0.7, 0.7, 1.0, 1.6, 1.7, 1.3, 0.7, 0.9, 0.8, 1.2,
            0.6, 0.5, 0.8, 0.5
        ],
        'Blocks': [
            0.6, 0.2, 0.7, 1.7, 0.8, 0.5, 1.1, 0.3, 0.7, 2.3, 
            0.8, 2.3, 0.5, 0.3, 0.9, 0.4, 0.5, 0.3, 0.4, 0.1,
            0.3, 0.3, 0.6, 0.8, 0.5, 0.6, 1.0, 1.1, 0.4, 0.4,
            0.3, 0.5, 0.4, 0.6, 0.7, 0.8, 2.1, 2.2, 1.3, 0.6,
            0.2, 0.2, 0.3, 0.1
        ],
        'ThreePerc': [
            0.31, 0.42, 0.38, 0.35, 0.27, 0.38, 0.40, 0.37, 0.35, 0.25, 
            0.30, 0.00, 0.36, 0.32, 0.13, 0.40, 0.37, 0.35, 0.35, 0.36,
            0.28, 0.36, 0.33, 0.25, 0.31, 0.39, 0.25, 0.39, 0.41, 0.34,
            0.39, 0.33, 0.29, 0.32, 0.38, 0.30, 0.33, 0.00, 0.00, 0.37,
            0.44, 0.41, 0.45, 0.39
        ],
        'USG': [  # Usage Rate - how much a player is involved in team plays
            31.2, 32.1, 29.3, 35.1, 34.6, 36.8, 31.0, 29.8, 32.3, 28.4,
            16.1, 14.8, 29.2, 23.7, 24.9, 30.5, 29.4, 26.5, 32.1, 33.7,
            32.6, 33.0, 30.4, 22.5, 22.0, 28.5, 18.2, 26.9, 28.9, 30.2,
            24.3, 27.6, 28.0, 22.8, 23.5, 29.0, 16.8, 10.3, 16.9, 15.8,
            14.3, 15.6, 18.8, 20.1
        ],
        'DEFRTG': [  # Defensive Rating - lower is better
            111.7, 112.3, 109.8, 107.3, 105.6, 113.8, 112.1, 116.4, 112.8, 106.2,
            106.0, 103.1, 113.2, 110.5, 107.9, 112.7, 108.3, 110.2, 113.5, 118.3,
            114.5, 112.0, 115.6, 108.1, 112.5, 111.9, 111.0, 113.6, 115.0, 117.1,
            109.2, 110.3, 112.9, 108.7, 111.5, 114.7, 108.0, 105.2, 106.8, 109.5,
            112.8, 111.9, 112.5, 113.0
        ]
    }
    return pd.DataFrame(data)

# Load the NBA data
nba_data = load_nba_data()

# Select features for clustering
# We'll use basic and advanced stats
features = ['Points', 'Rebounds', 'Assists', 'Steals', 'Blocks', 'ThreePerc', 'USG', 'DEFRTG']
X = nba_data[features].values

# Scale the data - essential for distance-based clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Run HDBSCAN with various parameter configurations
min_cluster_sizes = [3, 4, 5]
min_samples_list = [2, 3, 4]

results = []
for min_cluster_size in min_cluster_sizes:
    for min_samples in min_samples_list:
        if min_samples > min_cluster_size:
            continue
            
        # Apply HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=0.0,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )
        cluster_labels = clusterer.fit_predict(X_scaled)
        
        # Count clusters and outliers
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_outliers = list(cluster_labels).count(-1)
        
        # Save results
        results.append({
            'min_cluster_size': min_cluster_size,
            'min_samples': min_samples,
            'n_clusters': n_clusters,
            'n_outliers': n_outliers,
            'perc_outliers': n_outliers / len(nba_data) * 100,
            'labels': cluster_labels,
            'clusterer': clusterer
        })

# Find a good parameter set - we want a reasonable number of clusters and not too many outliers
filtered_results = [r for r in results if r['n_clusters'] >= 3 and r['perc_outliers'] <= 30]
if filtered_results:
    # Sort by number of clusters (descending) and then by percent outliers (ascending)
    best_result = sorted(filtered_results, key=lambda x: (-x['n_clusters'], x['perc_outliers']))[0]
else:
    # Fall back to the result with the most clusters
    best_result = sorted(results, key=lambda x: -x['n_clusters'])[0]

print(f"Selected parameters: min_cluster_size={best_result['min_cluster_size']}, min_samples={best_result['min_samples']}")
print(f"Number of clusters: {best_result['n_clusters']}")
print(f"Number of outliers: {best_result['n_outliers']} ({best_result['perc_outliers']:.1f}%)")

# Use the best parameters
clusterer = best_result['clusterer']
cluster_labels = best_result['labels']

# Add cluster labels to the DataFrame
nba_data['Cluster'] = cluster_labels

# Apply dimensionality reduction for visualization
# First, let's try t-SNE
tsne = TSNE(n_components=2, perplexity=15, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)
nba_data['TSNE1'] = X_tsne[:, 0]
nba_data['TSNE2'] = X_tsne[:, 1]

# Also try PCA for comparison
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
nba_data['PCA1'] = X_pca[:, 0]
nba_data['PCA2'] = X_pca[:, 1]

# Get soft clustering probabilities
if hasattr(clusterer, 'probabilities_'):
    nba_data['Cluster_Probability'] = clusterer.probabilities_

# Get exemplar (most representative) players for each cluster
exemplars = {}
for cluster_id in sorted(set(cluster_labels)):
    if cluster_id == -1:
        continue  # Skip outliers
        
    cluster_members = nba_data[nba_data['Cluster'] == cluster_id]
    
    if hasattr(clusterer, 'probabilities_'):
        # Get the player with highest probability
        exemplar_idx = cluster_members['Cluster_Probability'].idxmax()
        exemplar = cluster_members.loc[exemplar_idx, 'Player']
        exemplars[cluster_id] = exemplar

# Create visualization
plt.figure(figsize=(14, 10))

# Create color palette with gray for outliers
unique_clusters = sorted(set(cluster_labels))
if -1 in unique_clusters:
    palette = sns.color_palette('hls', len(unique_clusters) - 1)
    colors = []
    for label in cluster_labels:
        if label == -1:
            colors.append('gray')
        else:
            cluster_idx = unique_clusters.index(label) - (1 if -1 in unique_clusters else 0)
            colors.append(palette[cluster_idx])
else:
    palette = sns.color_palette('hls', len(unique_clusters))
    colors = [palette[unique_clusters.index(label)] for label in cluster_labels]

# Scatter plot with t-SNE coordinates
plt.scatter(nba_data['TSNE1'], nba_data['TSNE2'], c=colors, s=80, alpha=0.8)

# Add player names
for idx, row in nba_data.iterrows():
    plt.annotate(
        row['Player'],
        (row['TSNE1'] + 0.1, row['TSNE2'] + 0.1),
        fontsize=8
    )
    
# Add cluster labels
for cluster_id in sorted(set(cluster_labels)):
    if cluster_id != -1:
        cluster_data = nba_data[nba_data['Cluster'] == cluster_id]
        centroid = (cluster_data['TSNE1'].mean(), cluster_data['TSNE2'].mean())
        
        # Get the exemplar for this cluster
        exemplar = exemplars.get(cluster_id, "")
        
        plt.annotate(
            f"Cluster {cluster_id}\n({exemplar})",
            centroid,
            fontsize=11,
            weight='bold',
            ha='center',
            va='center',
            bbox=dict(boxstyle="round,pad=0.6", fc='white', alpha=0.7)
        )

plt.title('NBA Player Clusters by Play Style', fontsize=16)
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Describe each cluster
print("\nCluster Descriptions:")
for cluster_id in sorted(set(cluster_labels)):
    if cluster_id == -1:
        print("\nUnique Players (Outliers):")
        outliers = nba_data[nba_data['Cluster'] == -1]
        for idx, row in outliers.iterrows():
            print(f"- {row['Player']}")
        continue
    
    cluster_data = nba_data[nba_data['Cluster'] == cluster_id]
    cluster_means = cluster_data[features].mean()
    overall_means = nba_data[features].mean()
    
    # Calculate percent differences
    pct_diff = ((cluster_means - overall_means) / overall_means * 100).round(1)
    
    # Find distinctive features
    distinctive = []
    for feature, pct in zip(features, pct_diff):
        if abs(pct) >= 20:
            direction = "higher" if pct > 0 else "lower"
            distinctive.append(f"{feature} ({pct:.1f}% {direction})")
    
    # Get key players
    if hasattr(clusterer, 'probabilities_'):
        # Get players with highest probability in this cluster
        key_players = cluster_data.sort_values('Cluster_Probability', ascending=False)['Player'].head(3).tolist()
    else:
        key_players = cluster_data['Player'].head(3).tolist()
    
    # Create a descriptive name based on statistics
    cluster_type = []
    if cluster_means['Points'] > overall_means['Points'] * 1.1:
        cluster_type.append("Scoring")
    if cluster_means['Rebounds'] > overall_means['Rebounds'] * 1.1:
        cluster_type.append("Rebounding")
    if cluster_means['Assists'] > overall_means['Assists'] * 1.1:
        cluster_type.append("Playmaking")
    if cluster_means['Blocks'] > overall_means['Blocks'] * 1.1:
        cluster_type.append("Shot Blocking")
    if cluster_means['ThreePerc'] > overall_means['ThreePerc'] * 1.1:
        cluster_type.append("Shooting")
    if cluster_means['DEFRTG'] < overall_means['DEFRTG'] * 0.95:  # Lower is better for DEFRTG
        cluster_type.append("Defensive")
    
    cluster_name = " ".join(cluster_type)
    if not cluster_name:
        cluster_name = "Balanced"
    
    print(f"\nCluster {cluster_id}: {cluster_name} Players")
    print(f"Key players: {', '.join(key_players)}")
    print(f"Number of players: {len(cluster_data)}")
    print("Distinctive features:")
    for feature in distinctive:
        print(f"- {feature}")
    print("Average statistics:")
    for feature in features:
        print(f"- {feature}: {cluster_means[feature]:.1f} (Overall avg: {overall_means[feature]:.1f})")

# Find players with hybrid playstyles (low cluster confidence)
if hasattr(clusterer, 'probabilities_'):
    print("\nPlayers with Hybrid Playstyles (Low Cluster Confidence):")
    hybrid_players = nba_data[nba_data['Cluster'] != -1][nba_data['Cluster_Probability'] < 0.7]
    for idx, row in hybrid_players.iterrows():
        print(f"- {row['Player']}: Assigned to cluster {row['Cluster']} with {row['Cluster_Probability']:.2f} confidence")

# Calculate soft cluster assignments for all players
if hasattr(clusterer, 'prediction_data_'):
    print("\nSoft Clustering for Selected Players:")
    soft_clusters = hdbscan.all_points_membership_vectors(clusterer)
    
    # Convert to a DataFrame for easy analysis
    soft_df = pd.DataFrame(
        soft_clusters, 
        index=nba_data['Player'],
        columns=[f'Cluster_{i}' for i in range(soft_clusters.shape[1])]
    )
    
    # Select some interesting players to show their membership across clusters
    interesting_players = ['LeBron James', 'Nikola Jokic', 'Draymond Green', 'Giannis Antetokounmpo']
    for player in interesting_players:
        if player in soft_df.index:
            player_soft = soft_df.loc[player]
            print(f"\n{player} cluster memberships:")
            for cluster_id, prob in enumerate(player_soft):
                if prob > 0.1:  # Only show meaningful memberships
                    print(f"- Cluster {cluster_id}: {prob:.2f}")
```

This comprehensive example demonstrates a complete pipeline for clustering NBA players:

1. **Data Loading**: Creates a simulated NBA player dataset with both traditional and advanced stats
2. **Parameter Selection**: Tests multiple HDBSCAN parameter combinations to find optimal settings
3. **Visualization**: Uses t-SNE to create an easily interpretable 2D representation
4. **Cluster Analysis**: Identifies the key characteristics of each cluster
5. **Exemplar Players**: Finds the most representative player for each playstyle
6. **Hybrid Players**: Identifies players with elements of multiple playstyles
7. **Soft Clustering**: Shows how players can belong partially to multiple clusters

The resulting clusters typically correspond to recognizable NBA player archetypes like:
- Scoring guards
- Playmaking point guards
- Two-way wings
- Defensive anchors
- Floor-spacing big men
- Traditional centers

The ability to identify outliers is particularly valuable for NBA analysis, as it helps highlight truly unique players who don't conform to standard archetypes.

## Conclusion

HDBSCAN is particularly well-suited for clustering NBA players by playstyle because:

1. **No uniform distribution assumption**: Unlike CLARA or CLARANS, HDBSCAN doesn't assume clusters have uniform density, allowing it to find both common and rare player archetypes.

2. **Automatic cluster detection**: You don't need to specify the number of playstyles in advance, letting the algorithm discover the natural groupings in your data.

3. **Outlier identification**: It can identify truly unique players as outliers rather than forcing them into ill-fitting clusters.

4. **Soft clustering capabilities**: The ability to see how players might belong to multiple archetypes reflects the reality that many NBA players have hybrid skillsets.

5. **Robust to different cluster shapes**: Player archetypes aren't necessarily spherical in statistical space, and HDBSCAN can find clusters of various shapes.

When implementing HDBSCAN for NBA player clustering, keep these best practices in mind:

1. **Feature selection matters**: Choose statistics that are relevant to playstyle. Consider both traditional stats (points, rebounds, assists) and advanced metrics (usage rate, defensive rating, true shooting percentage).

2. **Always scale your data**: NBA statistics have different scales (e.g., points vs. shooting percentages), so standardization is essential.

3. **Experiment with parameters**: `min_cluster_size` and `min_samples` significantly affect your results. Try multiple values and evaluate which produces the most basketball-sensible clusters.

4. **Use dimensionality reduction for visualization**: t-SNE or UMAP typically produce better visualizations than PCA for this type of data.

5. **Bring in domain knowledge**: The most valuable insights come from combining algorithmic clustering with basketball expertise to interpret the results.

By following this guide, you can implement HDBSCAN to gain insights into player archetypes, find similar players, identify uniquely skilled athletes, and better understand the statistical landscape of the NBA.


figure(figsize=(10, 8))
colors = ['gray' if x == -1 else plt.cm.Spectral(x / max(cluster_labels)) 
          for x in cluster_labels]
plt.scatter(data[:, 0], data[:, 1], c=colors, s=50, alpha=0.8)
plt.title('HDBSCAN Clustering Results')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
```

This basic example shows:
1. Creating a synthetic dataset with two features and two clusters
2. Applying HDBSCAN with minimal parameters
3. Visualizing the results, with noise points in gray and clusters in different colors

## Preparing NBA Player Data

Let's see how to prepare and cluster NBA player data:

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load NBA player data
# You can replace this with your own data loading code
def load_nba_data():
    # Example data structure - replace with your actual data source
    data = {
        'Player': ['LeBron James', 'Stephen Curry', 'Nikola Jokic', 'Joel Embiid', 
                  'Giannis Antetokounmpo', 'Luka Doncic', 'Kevin Durant',
                  'Damian Lillard', 'Jayson Tatum', 'Anthony Davis'],
        'Points': [27.1, 26.8, 24.5, 30.6, 29.9, 32.4, 29.1, 28.3, 26.9, 25.9],
        'Rebounds': [7.5, 5.0, 11.8, 11.7, 11.6, 8.6, 6.7, 4.8, 8.8, 12.5],
        'Assists': [8.3, 6.1, 9.8, 3.9, 5.7, 8.0, 5.0, 7.0, 4.6, 2.6],
        'Steals': [1.0, 0.9, 1.3, 1.0, 0.8, 1.4, 0.7, 0.9, 1.1, 1.1],
        'Blocks': [0.6, 0.2, 0.7, 1.7, 0.8, 0.5, 1.1, 0.3, 0.7, 2.3],
        '3P%': [0.31, 0.42, 0.38, 0.35, 0.27, 0.38, 0.40, 0.37, 0.35, 0.25]
    }
    return pd.DataFrame(data)

# Load the data
nba_data = load_nba_data()

# Extract the features for clustering (excluding player names)
features = ['Points', 'Rebounds', 'Assists', 'Steals', 'Blocks', '3P%']
X = nba_data[features].values

# Scale the data - this is crucial for distance-based algorithms like HDBSCAN
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Now X_scaled is ready for clustering
```

When dealing with NBA statistics, it's important to:
1. Choose relevant features that might define playstyles (points, rebounds, assists, etc.)
2. Scale the data properly, since different stats have different scales
3. Consider whether to use raw statistics or advanced metrics like PER, True Shooting %, etc.

## HDBSCAN Parameters

Understanding HDBSCAN's parameters is crucial for effective clustering:

```python
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=5,  # Minimum size of clusters
    min_samples=3,       # Number of samples in a neighborhood for a core point
    metric='euclidean',  # Distance metric to use
    alpha=1.0,           # Controls how conservative the clustering is
    cluster_selection_epsilon=0.0,  # Allows more points to be included in clusters
    cluster_selection_method='eom'  # Excess of Mass or Leaf clustering method
)
```

Let's examine these parameters:

### `min_cluster_size`
This parameter determines the minimum number of points required to form a cluster. For NBA player clustering, setting this to 5-10 would ensure that you don't get tiny clusters of just 1-2 players with unusual stats.

### `min_samples`
This defines the number of points needed in a neighborhood for a point to be considered a core point. Higher values make the algorithm more conservative and potentially generate more outliers. For NBA data, a value of 3-5 is often a good starting point.

### `metric`
The distance metric to use. For most player statistics, 'euclidean' works well, but you could also try:
- 'manhattan': If you want differences in each stat to be weighted equally
- 'correlation': If you're interested in the pattern of statistics rather than absolute values

### `alpha`
This parameter affects how conservative the cluster merging is. Higher values make the algorithm more conservative about merging clusters. Default value is 1.0.

### `cluster_selection_epsilon`
This relaxes the cluster selection algorithm, allowing border points to be included. This can help ensure role players who are on the edge of a playstyle cluster are included.

### `cluster_selection_method`
- 'eom' (Excess of Mass): Often produces more interesting clusters
- 'leaf': More conservative, focuses on the leaves of the cluster hierarchy

## Visualizing Clusters

Visualizing high-dimensional data like NBA player statistics requires dimensionality reduction:

```python
from sklearn.decomposition import PCA
import seaborn as sns

# Run HDBSCAN
clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1)
cluster_labels = clusterer.fit_predict(X_scaled)

# Store the cluster assignments
nba_data['Cluster'] = cluster_labels

# Use PCA to reduce to 2 dimensions for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Add PCA components to the dataframe
nba_data['PCA1'] = X_pca[:, 0]
nba_data['PCA2'] = X_pca[:, 1]

# Create a visualization
plt.figure(figsize=(12, 10))
sns.scatterplot(
    x='PCA1', 
    y='PCA2',
    hue='Cluster',
    palette=sns.color_palette('hls', len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)),
    data=nba_data,
    legend='full',
    alpha=0.8,
    s=100
)

# Add player names as labels
for idx, row in nba_data.iterrows():
    plt.annotate(
        row['Player'],
        (row['PCA1'] + 0.1, row['PCA2'] + 0.1),
        fontsize=9
    )

plt.title('NBA Player Clusters by Play Style')
plt.xlabel('PCA Dimension 1')
plt.ylabel('PCA Dimension 2')
plt.show()
```

For better visualizations of higher-dimensional data, you can also try:

```python
from sklearn.manifold import TSNE

# t-SNE often provides better visualizations for cluster structures
tsne = TSNE(n_components=2, perplexity=5, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

nba_data['TSNE1'] = X_tsne[:, 0]
nba_data['TSNE2'] = X_tsne[:, 1]

# Then plot as before, but using TSNE1 and TSNE2 instead of PCA1 and PCA2
```

t-SNE often provides better visualization for cluster structures than PCA, especially for non-linear relationships that are common in player statistics.

## Handling Outliers

HDBSCAN naturally identifies outliers (assigned cluster -1). For NBA players, these could be unique talents with unusual statistical profiles:

```python
# Get players identified as outliers
outliers = nba_data[nba_data['Cluster'] == -1]
print("Players with unique playstyles (outliers):")
for idx, row in outliers.iterrows():
    print(f"- {row['Player']}: {row[features].to_dict()}")

# You can also examine the confidence of cluster assignments
if hasattr(clusterer, 'probabilities_'):
    nba_data['Cluster_Probability'] = clusterer.probabilities_
    
    # Players with low confidence in their cluster assignment
    low_confidence = nba_data[nba_data['Cluster'] != -1][nba_data['Cluster_Probability'] < 0.5]
    print("\nPlayers with hybrid playstyles (low cluster confidence):")
    for idx, row in low_confidence.iterrows():
        print(f"- {row['Player']}: Assigned to cluster {row['Cluster']} with {row['Cluster_Probability']:.2f} confidence")
```

This allows you to identify:
1. Truly unique players (statistical outliers)
2. Hybrid players who don't cleanly fit into one playstyle (low cluster confidence)

## Performance Optimization

For larger datasets (e.g., all NBA players from multiple seasons), performance may become an issue:

```python
# For better performance with larger datasets
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=10,
    min_samples=5,
    algorithm='best',  # 'best', 'generic', 'prims_kdtree', 'prims_balltree', or 'boruvka_kdtree'
    leaf_size=40,      # Affects performance of tree-based algorithms
    n_jobs=-1,         # Use all available CPU cores
    memory=None        # Can set to a joblib.Memory object to cache computation
)
```

The `algorithm` parameter can significantly impact performance:
- 'best': Automatically selects the best algorithm based on data characteristics
- 'prims_kdtree': Often fastest for lower-dimensional data (<=10 dimensions)
- 'boruvka_kdtree': Good for higher-dimensional data

## Advanced Techniques

### Soft Clustering

HDBSCAN can provide probabilities for cluster membership:

```python
clusterer = hdbscan.HDBSCAN(min_cluster_size=5, prediction_data=True)
clusterer.fit(X_scaled)

# Get probabilities (requires setting prediction_data=True)
soft_clusters = hdbscan.all_points_membership_vectors(clusterer)

# Convert to DataFrame
soft_df = pd.DataFrame(soft_clusters, index=nba_data['Player'])
soft_df.columns = [f'Cluster_{i}' for i in range(soft_df.shape[1])]

# Example: Players with strong membership in multiple clusters
hybrid_players = []
for player, row in soft_df.iterrows():
    # Get clusters with >30% probability
    strong_clusters = row[row > 0.3].index.tolist()
    if len(strong_clusters) > 1:
        hybrid_players.append((player, {c: f"{row[c]:.2f}" for c in strong_clusters}))

print("Hybrid players with multiple playstyle elements:")
for player, clusters in hybrid_players:
    print(f"- {player}: {clusters}")
```

This approach helps identify players with hybrid playstyles who incorporate elements from multiple cluster prototypes.

### Parameter Selection

Finding optimal parameters can be challenging. Here's a way to evaluate different parameter combinations:

```python
from sklearn import metrics

# Function to evaluate clustering with silhouette score
def evaluate_hdbscan(X, min_cluster_sizes, min_samples_list):
    results = []
    
    for min_cluster_size in min_cluster_sizes:
        for min_samples in min_samples_list:
            # Skip invalid combinations
            if min_samples > min_cluster_size:
                continue
                
            # Run HDBSCAN
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples
            )
            cluster_labels = clusterer.fit_predict(X)
            
            # Count non-noise clusters
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            noise_points = list(cluster_labels).count(-1)
            
            # Skip if all points are noise or all in one cluster
            if n_clusters <= 1:
                continue
            
            # Calculate silhouette score for non-noise points
            if n_clusters > 1 and len(X) - noise_points > n_clusters:
                # Filter out noise points for silhouette calculation
                non_noise_mask = cluster_labels != -1
                if sum(non_noise_mask) > 1:
                    silhouette = metrics.silhouette_score(
                        X[non_noise_mask], 
                        cluster_labels[non_noise_mask]
                    )
                else:
                    silhouette = float('nan')
            else:
                silhouette = float('nan')
            
            results.append({
                'min_cluster_size': min_cluster_size,
                'min_samples': min_samples,
                'n_clusters': n_clusters,
                'noise_percentage': noise_points / len(X) * 100,
                'silhouette': silhouette
            })
    
    return pd.DataFrame(results)

# Evaluate different parameter combinations
min_cluster_sizes = [2, 3, 5, 10]
min_samples_list = [1, 2, 3, 5, 7]

results = evaluate_hdbscan(X_scaled, min_cluster_sizes, min_samples_list)

# Sort by silhouette score (higher is better)
best_results = results.sort_values('silhouette', ascending=False).head(10)
print(best_results)

# Visualize parameter impact
plt.figure(figsize=(15, 10))
for min_samples in min_samples_list:
    df_subset = results[results['min_samples'] == min_samples]
    if not df_subset.empty:
        plt.plot(
            df_subset['min_cluster_size'], 
            df_subset['silhouette'], 
            marker='o',
            label=f'min_samples={min_samples}'
        )

plt.xlabel('min_cluster_size')
plt.ylabel('Silhouette Score')
plt.title('HDBSCAN Parameter Selection')
plt.legend()
plt.grid(True)
plt.show()
```

This approach helps you identify parameter combinations that produce meaningful clusters with high silhouette scores.

## Complete Example

Here's a complete example that puts everything together to cluster NBA players by playstyle:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import hdbscan
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# Example: Load and prepare NBA player data
player_data = {
    'Player': ['LeBron James', 'Stephen Curry', 'Nikola Jokic', 'Joel Embiid', 
              'Giannis Antetokounmpo', 'Luka Doncic', 'Kevin Durant', 'Damian Lillard', 
              'Jayson Tatum', 'Anthony Davis', 'Draymond Green', 'Rudy Gobert',
              'James Harden', 'Chris Paul', 'Bam Adebayo', 'Kyrie Irving',
              'Kawhi Leonard', 'Jimmy Butler', 'Devin Booker', 'Trae Young'],
    'Points': [27.1, 26.8, 24.5, 30.6, 29.9, 32.4, 29.1, 28.3, 26.9, 25.9, 
              8.5, 13.6, 24.7, 13.9, 19.3, 25.8, 23.8, 22.9, 27.8, 26.2],
    'Rebounds': [7.5, 5.0, 11.8, 11.7, 11.6, 8.6, 6.7, 4.8, 8.8, 12.5, 
                7.2, 11.6, 5.7, 4.3, 9.2, 4.8, 6.5, 5.9, 4.5, 3.0],
    'Assists': [8.3, 6.1, 9.8, 3.9, 5.7, 8.0, 5.0, 7.0, 4.6, 2.6, 
               6.8, 1.0, 7.0, 8.9, 3.2, 6.0, 3.9, 5.3, 5.5, 10.2],
    'Steals': [1.0, 0.9, 1.3, 1.0, 0.8, 1.4, 0.7, 0.9, 1.1, 1.1, 
              1.0, 0.6, 1.3, 1.5, 1.1, 1.1, 1.4, 1.8, 0.8, 0.9],
    'Blocks': [0.6, 0.2, 0.7, 1.7, 0.8, 0.5, 1.1, 0.3, 0.7, 2.3, 
              0.8, 2.3, 0.5, 0.3, 0.9, 0.4, 0.5, 0.3, 0.4, 0.1],
    'ThreePerc': [0.31, 0.42, 0.38, 0.35, 0.27, 0.38, 0.40, 0.37, 0.35, 0.25, 
                 0.30, 0.0, 0.36, 0.32, 0.13, 0.40, 0.37, 0.35, 0.35, 0.36]
}

# Create DataFrame
df = pd.DataFrame(player_data)

# Select features for clustering
features = ['Points', 'Rebounds', 'Assists', 'Steals', 'Blocks', 'ThreePerc']
X = df[features].values

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Run HDBSCAN
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=3,
    min_samples=2,
    metric='euclidean',
    prediction_data=True
)
cluster_labels = clusterer.fit_predict(X_scaled)

# Add cluster labels to DataFrame
df['Cluster'] = cluster_labels

# Apply t-SNE for visualization
tsne = TSNE(n_components=2, perplexity=5, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)
df['TSNE1'] = X_tsne[:, 0]
df['TSNE2'] = X_tsne[:, 1]

# Calculate soft clustering probabilities
soft_clusters = hdbscan.all_points_membership_vectors(clusterer)
soft_df = pd.DataFrame(
    soft_clusters, 
    index=df['Player'],
    columns=[f'Cluster_{i}' for i in range(soft_clusters.shape[1])]
)

# Plot the results
plt.figure(figsize=(14, 10))

# Create a color palette with gray for outliers
palette = sns.color_palette('hls', len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0))
colors = []
for label in cluster_labels:
    if label == -1:
        colors.append('gray')
    else:
        colors.append(palette[label])

# Plot with player names
plt.scatter(df['TSNE1'], df['TSNE2'], c=colors, s=100, alpha=0.8)
for idx, row in df.iterrows():
    plt.annotate(
        row['Player'],
        (row['TSNE1'] + 0.1, row['TSNE2'] + 0.1),
        fontsize=9
    )

# Add cluster information
for cluster_id in sorted(set(cluster_labels)):
    if cluster_id != -1:
        cluster_points = df[df['Cluster'] == cluster_id]
        center = cluster_points[['TSNE1', 'TSNE2']].mean()
        plt.annotate(
            f'Cluster {cluster_id}',
            (center['TSNE1'], center['TSNE2']),
            fontsize=12,
            weight='bold',
            color=palette[cluster_id],
            bbox=dict(boxstyle="round,pad=0.3", fc='white', alpha=0.7)
        )

plt.title('NBA Player Clusters by Play Style (HDBSCAN)', fontsize=16)
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.grid(True, linestyle='--', alpha=0.7)

# Print cluster information
print("Cluster Assignments:")
for cluster_id in sorted(set(cluster_labels)):
    if cluster_id == -1:
        print("\nUnique Players (Outliers):")
    else:
        print(f"\nCluster {cluster_id} Players:")
    
    cluster_members = df[df['Cluster'] == cluster_id]
    for idx, row in cluster_members.iterrows():
        if cluster_id != -1 and hasattr(clusterer, 'probabilities_'):
            print(f"- {row['Player']} (confidence: {clusterer.probabilities_[idx]:.2f})")
        else:
            print(f"- {row['Player']}")

# Describe the characteristics of each cluster
for cluster_id in sorted(set(cluster_labels)):
    if cluster_id != -1:
        print(f"\nCluster {cluster_id} Profile:")
        cluster_stats = df[df['Cluster'] == cluster_id][features].mean()
        overall_stats = df[features].mean()
        
        # Calculate how each cluster differs from the overall average
        diff = cluster_stats - overall_stats
        pct_diff = diff / overall_stats * 100
        
        # Identify defining characteristics (features that differ significantly)
        defining_features = []
        for feature, pct in zip(features, pct_diff):
            if abs(pct) > 20:  # More than 20% different from average
                direction = "higher" if pct > 0 else "lower"
                defining_features.append(f"{abs(pct):.0f}% {direction} {feature}")
        
        print(f"  Distinguished by: {', '.join(defining_features)}")
        print(f"  Average stats: {dict(cluster_stats.round(1))}")

plt.tight_layout()
plt.show()
```

This comprehensive example:
1. Loads NBA player data
2. Applies HDBSCAN clustering
3. Visualizes the clusters using t-SNE
4. Identifies outliers (unique players)
5. Provides soft clustering probabilities
6. Describes the distinguishing characteristics of each cluster

## Comparing with Other Algorithms

Let's compare HDBSCAN with other clustering algorithms using the same NBA data:

```python
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import time

def compare_algorithms(X_scaled, df, features):
    results = {}
    
    # Number of clusters to use for KMeans (for fair comparison)
    n_clusters = len(set(clusterer.labels_)) - (1 if -1 in clusterer.labels_ else 0)
    n_clusters = max(2, n_clusters)  # Ensure at least 2 clusters
    
    # 1. HDBSCAN
    start_time = time.time()
    hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=2)
    hdbscan_labels = hdbscan_clusterer.fit_predict(X_scaled)
    hdbscan_time = time.time() - start_time
    
    # 2. KMeans
    start_time = time.time()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    kmeans_time = time.time() - start_time
    
    # 3. DBSCAN
    start_time = time.time()
    # Find a reasonable epsilon based on nearest neighbor distances
    from sklearn.neighbors import NearestNeighbors
    neighbors = NearestNeighbors(n_neighbors=3)
    neighbors_fit = neighbors.fit(X_scaled)
    distances, indices = neighbors_fit.kneighbors(X_scaled)
    distances = np.sort(distances[:, 2])
    epsilon = distances[int(0.5 * len(distances))]  # Median distance to 3rd nearest neighbor
    
    dbscan = DBSCAN(eps=epsilon, min_samples=2)
    dbscan_labels = dbscan.fit_predict(X_scaled)
    dbscan_time = time.time() - start_time
    
    # Store results
    results = {
        'HDBSCAN': {
            'labels': hdbscan_labels,
            'n_clusters': len(set(hdbscan_labels)) - (1 if -1 in hdbscan_labels else 0),
            'n_outliers': list(hdbscan_labels).count(-1),
            'runtime': hdbscan_time
        },
        'KMeans': {
            'labels': kmeans_labels,
            'n_clusters': len(set(kmeans_labels)),
            'n_outliers': 0,  # KMeans doesn't identify outliers
            'runtime': kmeans_time
        },
        'DBSCAN': {
            'labels': dbscan_labels,
            'n_clusters': len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0),
            'n_outliers': list(dbscan_labels).count(-1),
            'runtime': dbscan_time
        }
    }
    
    # Add labels to the dataframe for visualization
    df_comparison = df.copy()
    df_comparison['HDBSCAN_Cluster'] = hdbscan_labels
    df_comparison['KMeans_Cluster'] = kmeans_labels
    df_comparison['DBSCAN_Cluster'] = dbscan_labels
    
    # Print summary
    print("Clustering Algorithm Comparison:")
    print("-" * 50)
    for algo, data in results.items():
        print(f"{algo}:")
        print(f"  Number of clusters: {data['n_clusters']}")
        print(f"  Number of outliers: {data['n_outliers']}")
        print(f"  Runtime: {data['runtime']:.3f} seconds")
        print()
    
    return df_comparison, results

# Run comparison
df_comparison, comparison_results = compare_algorithms(X_scaled, df, features)

# Visualize the comparison with t-SNE
plt.figure(figsize=(18, 6))

# Set up the TSNE coordinates if not already in the dataframe
if 'TSNE1' not in df_comparison.columns:
    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)
    df_comparison['TSNE1'] = X_tsne[:, 0]
    df_comparison['TSNE2'] = X_tsne[:, 1]

# Plot each algorithm
for i, algo in enumerate(['HDBSCAN', 'KMeans', 'DBSCAN']):
    labels = df_comparison[f'{algo}_Cluster']
    
    plt.subplot(1, 3, i+1)
    
    # Create a color palette with gray for outliers if applicable
    unique_labels = set(labels)
    if -1 in unique_labels:
        n_clusters = len(unique_labels) - 1
        unique_labels = [l for l in unique_labels if l != -1]
        palette = sns.color_palette('hls', n_clusters)
        colors = []
        for label in labels:
            if label == -1:
                colors.append('gray')
            else:
                colors.append(palette[unique_labels.index(label)])
    else:
        palette = sns.color_palette('hls', len(unique_labels))
        colors = [palette[label % len(palette)] for label in labels]
    
    # Plot points
    plt.scatter(df_comparison['TSNE1'], df_comparison['TSNE2'], c=colors, s=50, alpha=0.7)
    
    # Add player names if the dataset is small enough
    if len(df_comparison) <= 30:
        for idx, row in df_comparison.iterrows():
            plt.annotate(
                row['Player'][:10] + '...' if len(row['Player']) > 10 else row['Player'],
                (row['TSNE1'] + 0.05, row['TSNE2'] + 0.05),
                fontsize=7
            )
    
    plt.title(f'{algo} Clustering')
    plt