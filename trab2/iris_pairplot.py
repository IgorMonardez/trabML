from ucimlrepo import fetch_ucirepo
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#carregar dataset iris
iris = fetch_ucirepo(id=53)

X = iris.data.features
y = iris.data.targets

sns.pairplot(pd.DataFrame(X, columns=iris.feature_names), diag_kind='kde')
plt.suptitle('Pairplot sem rótulos - Análise Visual de Grupos Naturais', y=1.02)
plt.savefig("./graficos/Pairplot_sem_rotulos")
