from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score

iris = fetch_ucirepo(id=53)

X = iris.data.features
y = iris.data.targets

agg = AgglomerativeClustering(n_clusters=3)
y_agg = agg.fit_predict(X)


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

y_labels = y_agg 
unique_labels, counts = np.unique(y_labels, return_counts=True)
colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))

plt.figure(figsize=(8, 6))
for label, count, color in zip(unique_labels, counts, colors):
    idx = y_labels == label
    plt.scatter(X_pca[idx, 0], X_pca[idx, 1],
                color=color,
                label=f'Cluster {label} ({count} pts)',
                edgecolor='k',
                alpha=0.7,
                s=50)

plt.title('Clusters utilizando AgglomerativeClustering = 3')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend(title='Clusters', loc='best')
plt.grid(True)
plt.tight_layout()
plt.savefig("./graficos/agrupamento_hierarquico")

sil_score = silhouette_score(X, y_agg)
ari = adjusted_rand_score(y.values.ravel(), y_agg)

print(f"Silhouette Score: {sil_score:.3f}")
print(f"Adjusted Rand Index (ARI): {ari:.3f}")