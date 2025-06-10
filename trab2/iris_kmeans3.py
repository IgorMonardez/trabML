from ucimlrepo import fetch_ucirepo
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score

#carregar dataset iris
iris = fetch_ucirepo(id=53)

X = iris.data.features
y = iris.data.targets

for i in range(2,4):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init='auto')
    y_kmeans = kmeans.fit_predict(X) 

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    y_labels = y_kmeans  
    unique_labels, counts = np.unique(y_labels, return_counts=True)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
    plt.figure(figsize=(8, 6))
    for label, count, color in zip(unique_labels, counts, colors):
        idx = y_labels == label
        plt.scatter(X_pca[idx, 0], X_pca[idx, 1],
                    color=color,
                    label=f'Cluster {label} ({count} pts)',
                    edgecolor='k',
                    alpha=1,
                    s=50)

    plt.title(f'Clusterização utilizando KMeans - K = {i}')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.legend(title='Clusters', loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"./graficos/kmeans{i}")

    sil_score = silhouette_score(X, y_kmeans)
    ari = adjusted_rand_score(y.values.ravel(), y_kmeans)

    print(f"Silhouette Score: {sil_score:.3f}")
    print(f"Adjusted Rand Index (ARI): {ari:.3f}")

    