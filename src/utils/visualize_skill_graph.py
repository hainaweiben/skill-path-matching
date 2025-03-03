"""
Skill Graph Visualization Script
"""
import networkx as nx
import matplotlib.pyplot as plt
import json
import os
import numpy as np

# Load skill graph data
with open('/home/u2021201733/test/skill_path_matching/data/processed/skill_graph.json', 'r') as f:
    data = json.load(f)

# Create graph
G = nx.Graph()

# Add nodes and edges
for node in data['nodes']:
    G.add_node(node['id'], name=node['name'])
for edge in data['edges']:
    G.add_edge(edge['source'], edge['target'], weight=edge['weight'], type=edge['type'])

# Calculate node degree centrality
degree_centrality = nx.degree_centrality(G)

# Create output directory
output_dir = '/home/u2021201733/test/skill_path_matching/data/visualizations'
os.makedirs(output_dir, exist_ok=True)

# ================================
# 1. Main Skill Relationship Graph
# ================================
plt.figure(figsize=(20, 20), dpi=300)
plt.title('Skill Relationship Graph (Node size and color represent degree centrality)', fontsize=24, pad=20)

# Use Kamada-Kawai layout for better node distribution
pos = nx.kamada_kawai_layout(G)

# Draw edges with adjustable width based on weight
edge_weights = [G[u][v]['weight'] / 2 for u, v in G.edges()]
nx.draw_networkx_edges(
    G, pos,
    alpha=0.3,
    edge_color='gray',
    width=edge_weights
)

# Draw nodes with degree centrality-based size and color
node_sizes = [degree_centrality[node] * 5000 for node in G.nodes()]
nodes = nx.draw_networkx_nodes(
    G, pos,
    node_size=node_sizes,
    node_color=[degree_centrality[node] for node in G.nodes()],
    cmap=plt.cm.viridis,
    alpha=0.8,
    edgecolors='black',  # Add black borders for better visibility
    linewidths=1
)

# Add labels for top 10 nodes by degree centrality
top_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
labels = {node: G.nodes[node]['name'] for node, _ in top_nodes}
nx.draw_networkx_labels(G, pos, labels=labels, font_size=14, font_weight='bold')

plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'skill_graph.png'), dpi=300, bbox_inches='tight')
plt.close()

# ================================
# 2. Community Structure Visualization
# ================================
print("Detecting community structure...")
communities = nx.community.greedy_modularity_communities(G)
print(f"Found {len(communities)} communities")

# Assign a color to each community
community_colors = {}
for i, community in enumerate(communities):
    for node in community:
        community_colors[node] = i

plt.figure(figsize=(20, 20), dpi=300)
plt.title('Skill Community Structure (Different colors represent different communities)', fontsize=24, pad=20)

# Use Kamada-Kawai layout for consistency
pos = nx.kamada_kawai_layout(G)

# Draw edges with reduced opacity
edge_weights = [G[u][v]['weight'] / 2 for u, v in G.edges()]
nx.draw_networkx_edges(
    G, pos,
    alpha=0.2,
    edge_color='gray',
    width=edge_weights
)

# Draw nodes with community colors and degree centrality-based sizes
node_sizes = [degree_centrality[node] * 5000 for node in G.nodes()]
nodes = nx.draw_networkx_nodes(
    G, pos,
    node_size=node_sizes,
    node_color=[community_colors[node] for node in G.nodes()],
    cmap=plt.cm.tab20,
    alpha=0.8,
    edgecolors='black',  # Add black borders for better visibility
    linewidths=1
)

# Add labels for top 3 nodes in each community
community_top_nodes = {}
for i, community in enumerate(communities):
    community_nodes = list(community)
    sorted_nodes = sorted([(node, degree_centrality[node]) for node in community_nodes], key=lambda x: x[1], reverse=True)
    top_n = min(3, len(sorted_nodes))
    for j in range(top_n):
        community_top_nodes[sorted_nodes[j][0]] = G.nodes[sorted_nodes[j][0]]['name']

# Adjust label positions to avoid overlap
labels = {node: G.nodes[node]['name'] for node in community_top_nodes}
for node, (x, y) in pos.items():
    if node in labels:
        plt.text(x, y + 0.05, labels[node], fontsize=12, ha='center', va='bottom', weight='bold', color='black')

plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'skill_communities.png'), dpi=300, bbox_inches='tight')
plt.close()

# ================================
# 3. Skill Relationship Heatmap
# ================================
print("Creating skill relationship heatmap...")
top_skills = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:20]
top_skill_ids = [node for node, _ in top_skills]
top_skill_names = [G.nodes[node]['name'] for node in top_skill_ids]

# Create adjacency matrix
adjacency_matrix = np.zeros((len(top_skill_ids), len(top_skill_ids)))
for i, skill1 in enumerate(top_skill_ids):
    for j, skill2 in enumerate(top_skill_ids):
        if G.has_edge(skill1, skill2):
            adjacency_matrix[i, j] = G[skill1][skill2]['weight']

# Draw heatmap
plt.figure(figsize=(15, 12), dpi=300)
plt.imshow(adjacency_matrix, cmap='YlOrRd', interpolation='nearest')
plt.colorbar(label='Relationship Strength', fraction=0.046, pad=0.04)

# Add axis labels
plt.xticks(range(len(top_skill_names)), top_skill_names, rotation=90, fontsize=10)
plt.yticks(range(len(top_skill_names)), top_skill_names, fontsize=10)

plt.title('Relationship Strength Between Top 20 Skills', fontsize=20, pad=20)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'skill_relationship_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close()

# ================================
# 4. Community Analysis Report
# ================================
print("\nCommunity analysis:")
for i, community in enumerate(communities):
    print(f"\nCommunity {i+1}:")
    skill_names = [G.nodes[node]['name'] for node in community]
    for skill in skill_names:
        print(f"  - {skill}")

# Save community analysis results
with open(os.path.join(output_dir, 'community_analysis.txt'), 'w', encoding='utf-8') as f:
    f.write("Skill Community Analysis\n")
    f.write("=" * 50 + "\n\n")
    
    for i, community in enumerate(communities):
        f.write(f"Community {i+1}:\n")
        skills_with_centrality = [(G.nodes[node]['name'], degree_centrality[node]) for node in community]
        skills_with_centrality.sort(key=lambda x: x[1], reverse=True)
        
        for skill, centrality in skills_with_centrality:
            f.write(f"  - {skill} (centrality: {centrality:.4f})\n")
        f.write("\n")

print(f"\nVisualization results saved to {output_dir} directory")