from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Carregar o dataset Iris
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
colunas = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
df = pd.read_csv(url, header=None, names=colunas)

X = df.drop(columns=['class'])
y = df['class']

# -----------------------
# 1. Validação cruzada
# -----------------------
clf_cv = DecisionTreeClassifier(
    criterion='gini',
    max_depth=3,
    min_samples_leaf=1,
    random_state=42
)

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

scores = cross_validate(clf_cv, X, y, cv=cv, scoring=scoring)

print("===== Validação Cruzada =====")
for metric in scoring:
    mean_score = np.mean(scores[f'test_{metric}'])
    std_score = np.std(scores[f'test_{metric}'])
    print(f"{metric}: {mean_score:.3f} (±{std_score:.3f})")
print()

# -----------------------
# 2. Treinamento final e avaliação com teste
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y  # garante a proporção das classes
)

clf = DecisionTreeClassifier(
    criterion='gini',
    max_depth=3,
    min_samples_leaf=1,
    random_state=42
)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("===== Avaliação no Conjunto de Teste =====")
print("Acurácia:", metrics.accuracy_score(y_test, y_pred))
print("\nMatriz de Confusão:\n", metrics.confusion_matrix(y_test, y_pred))
print("\nRelatório de Classificação:\n", metrics.classification_report(y_test, y_pred))

# -----------------------
# 3. Visualização da Árvore
# -----------------------
# Texto ASCII da árvore
print("===== Regras da Árvore =====")
tree_rules = export_text(clf, feature_names=X.columns.tolist())
print(tree_rules)

# Gráfico da árvore
plt.figure(figsize=(12, 8))
plot_tree(clf,
          feature_names=X.columns.tolist(),
          class_names=clf.classes_,
          filled=True)


plt.show()
plt.savefig("grafico2.png")
