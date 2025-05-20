from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn import metrics
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo

# -------------------------------
# 1. Carregamento do dataset Iris
# -------------------------------
iris = fetch_ucirepo(id=53)

X = iris.data.features
y = iris.data.targets

# -------------------------------
# 2. Divisão entre treino e teste
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.33,
    random_state=42,
    stratify=y  # Garante que a proporção de classes seja mantida
)

# -------------------------------
# 3. Treinamento do modelo
# -------------------------------
clf = DecisionTreeClassifier(
    criterion='gini',
    max_depth=3,
    min_samples_leaf=1,
    random_state=42
)

clf.fit(X_train, y_train)

# -------------------------------
# 4. Avaliação do modelo
# -------------------------------
y_pred = clf.predict(X_test)

print("===== Avaliação no Conjunto de Teste =====")
print("Acurácia:", metrics.accuracy_score(y_test, y_pred))
print("\nMatriz de Confusão:\n", metrics.confusion_matrix(y_test, y_pred))
print("\nRelatório de Classificação:\n", metrics.classification_report(y_test, y_pred))

# -------------------------------
# 5. Visualização da Árvore
# -------------------------------
# Texto com as regras da árvore
print("===== Regras da Árvore =====")
tree_rules = export_text(clf, feature_names=iris.feature_names)
print(tree_rules)

# Gráfico da árvore de decisão
plt.figure(figsize=(12, 8))
plot_tree(
    clf,
    feature_names=X.columns,
    class_names=clf.classes_,
    filled=True
)

plt.show()
plt.savefig("grafico.png")
