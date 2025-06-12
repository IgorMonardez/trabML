from ucimlrepo import fetch_ucirepo
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#carregar dataset iris
iris = fetch_ucirepo(id=53)

X = iris.data.features
X = X[['petal length', 'petal width']]
y = iris.data.targets


wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Método do Cotovelo')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inércia (WCSS)')
plt.xticks(range(1, 11))
plt.grid(True)
plt.tight_layout()
plt.savefig("./graficos/kmeans_cotovelo_2_features")