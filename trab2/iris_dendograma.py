from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo


iris = fetch_ucirepo(id=53)

X = iris.data.features
y = iris.data.targets

linked = linkage(X, method='ward')

plt.figure(figsize=(10, 6))
dendrogram(linked, truncate_mode='lastp', p=30, leaf_rotation=90., leaf_font_size=10.)
plt.title('Dendrograma - AgglomerativeClustering')
plt.xlabel('Amostras ou subgrupos')
plt.ylabel('Distância (Ward)')
plt.grid(True)
plt.tight_layout()
plt.savefig("./graficos/Dendograma")