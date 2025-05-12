from sklearn import datasets
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
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


# Inicializar a árvore de decisão
clf = DecisionTreeClassifier(
    criterion='gini',  # ou 'entropy' para ganho de informação
    max_depth=2,        # profundidade máxima
    min_samples_leaf=1,
    random_state=42
)

# Definir a validação cruzada (10 folds)
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)  # Mantém a proporção das classes
scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
scores = cross_val_score(
    estimator=clf,
    X=X,
    y=y,
    cv=cv,          # 10 folds
    scoring='accuracy'
)

print("Acurácia:", scores.mean().round(3))
print("Desvio padrão:", scores.std().round(3))
print("\nAcurácia por fold:")
print(scores.round(3))

