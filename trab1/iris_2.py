from sklearn import datasets
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.tree import export_text
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
import numpy as np

# fetch dataset
iris = fetch_ucirepo(id=53)

# data (as pandas dataframes)
X = iris.data.features
y = iris.data.targets


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
    cv=cv,          # 10 folds
    scoring=scoring
)

# Exibir resultados
for metric in scoring:
    mean_score = np.mean(scores[f'test_{metric}'])
    std_score = np.std(scores[f'test_{metric}'])
    print(f"{metric}: {mean_score:.3f} (±{std_score:.3f})")