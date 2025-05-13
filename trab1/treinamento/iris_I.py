from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.tree import export_text
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo

# fetch dataset
iris = fetch_ucirepo(id=53)

# data (as pandas dataframes)
X = iris.data.features
y = iris.data.targets

# Via treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.33,
    random_state=42
)

# Inicializar a árvore de decisão
clf = DecisionTreeClassifier(
    criterion='gini',  # ou 'entropy' para ganho de informação
    max_depth=3,        # profundidade máxima
    min_samples_leaf=1,
    random_state=42
)

# Treinar o modelo
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Acurácia:", metrics.accuracy_score(y_test, y_pred))
print("\nMatriz de Confusão:\n", metrics.confusion_matrix(y_test, y_pred))
print("\nRelatório de Classificação:\n", metrics.classification_report(y_test, y_pred))

# Texto ASCII
tree_rules = export_text(clf, feature_names=iris.feature_names)
print(tree_rules)

# Gráfico usando matplotlib
plt.figure(figsize=(12,8))
plot_tree(clf,
          feature_names=iris.feature_names,
          class_names=iris.target_names,
          filled=True)
plt.show()