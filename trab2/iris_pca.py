from ucimlrepo import fetch_ucirepo
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#carregar dataset iris
iris = fetch_ucirepo(id=53)

X = iris.data.features
y = iris.data.targets

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c='gray', edgecolor='k', alpha=0.7)
plt.title('PCA sem rótulos - Projeção dos dados em 2D')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.grid(True)
plt.tight_layout()
plt.savefig("./graficos/PCA_sem_rotulos")
