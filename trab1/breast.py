from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from ucimlrepo import fetch_ucirepo
import numpy as np

# fetch dataset
breast_cancer = fetch_ucirepo(id=17)

# data (as pandas dataframes)
X = breast_cancer.data.features
y = breast_cancer.data.targets

param_grid = {
    'max_depth': [2, 3, 4, 5],
    'min_samples_split': [5, 10, 15],
    'min_samples_leaf': [1, 5, 10, 15, 20],
}

# Inicializar a árvore de decisão
clf = DecisionTreeClassifier(random_state=42)




# Definir a validação cruzada (10 folds)
grid_search = GridSearchCV(
    estimator= clf,
    param_grid=param_grid,
    cv=30,
    scoring='accuracy'
)

grid_search.fit(X, y)

# Resultados
print("Melhores hiperparâmetros: ", grid_search.best_params_)
print("Melhor acurácia: ", grid_search.best_score_)

scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

clf = DecisionTreeClassifier(
    max_depth=5,
    min_samples_leaf=1,
    min_samples_split=5,
    random_state=42
)


cv = StratifiedKFold(n_splits=30, shuffle=True, random_state=42)  # Mantém a proporção das classes

# Executar cross-validation
results = cross_validate(clf, X, y, cv=cv, scoring=scoring)

# Exibir resultados
for metric in scoring:
    mean_score = np.mean(results[f'test_{metric}'])
    std_score = np.std(results[f'test_{metric}'])
    print(f"{metric}: {mean_score:.3f} (±{std_score:.3f})")

# for metric in scoring:
#     mean_score = np.mean(scores[f'test_{metric}'])
#     std_score = np.std(scores[f'test_{metric}'])
#     print(f"{metric}: {mean_score:.3f} (±{std_score:.3f})")
