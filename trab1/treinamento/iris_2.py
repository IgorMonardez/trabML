from sklearn import datasets
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.tree import export_text
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
colunas = [
    'sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'
]

df = pd.read_csv(url, header=None, names=colunas)

X = df.drop(columns=['class'])
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)

# Inicializar a árvore de decisão
clf = DecisionTreeClassifier(
    criterion='gini',  # ou 'entropy' para ganho de informação
    max_depth=3,        # profundidade máxima
    min_samples_leaf=1,
    random_state=42
)

# Definir a validação cruzada (10 folds)
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)  # Mantém a proporção das classes
scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
scores = cross_validate(
    estimator=clf,
    X=X,
    y=y,
    cv=cv,
    scoring=scoring
)

# Exibir resultados
for metric in scoring:
    mean_score = np.mean(scores[f'test_{metric}'])
    std_score = np.std(scores[f'test_{metric}'])
    print(f"{metric}: {mean_score:.3f} (±{std_score:.3f})")

y_pred = clf.predict(X_test)

print("Acurácia:", metrics.accuracy_score(y_test, y_pred))
print("\nMatriz de Confusão:\n", metrics.confusion_matrix(y_test, y_pred))
print("\nRelatório de Classificação:\n", metrics.classification_report(y_test, y_pred))

# Texto ASCII
tree_rules = export_text(clf, feature_names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
print(tree_rules)

# Gráfico usando matplotlib
plt.figure(figsize=(12,8))
plot_tree(clf,
          feature_names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
          class_names=['class'],
          filled=True)
plt.show()