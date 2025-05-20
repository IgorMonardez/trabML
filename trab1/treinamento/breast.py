from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
colunas = [
    'id', 'diagnosis',
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean',
    'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
    'smoothness_se', 'compactness_se', 'concavity_se', 'concave_points_se',
    'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
    'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst',
    'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

df = pd.read_csv(url, header=None, names=colunas)
df.replace('?', np.nan, inplace=True)
df_limpo = df.dropna(axis=0)

X = df_limpo.drop(columns=['id', 'diagnosis'])
y = df_limpo['diagnosis']

param_grid = {
    'max_depth': [2, 3, 4, 5],
    'min_samples_split': [5, 10, 15],
    'min_samples_leaf': [1, 5, 10, 15, 20],
}

clf = DecisionTreeClassifier(random_state=42)

grid_search = GridSearchCV(
    estimator= clf,
    param_grid=param_grid,
    cv=30,
    scoring='accuracy'
)

grid_search.fit(X, y)

print("Melhores hiperparâmetros: ", grid_search.best_params_)
print("Melhor acurácia: ", grid_search.best_score_)

scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

clf = DecisionTreeClassifier(
    max_depth=5,
    min_samples_leaf=1,
    min_samples_split=5,
    random_state=42
)


cv = StratifiedKFold(n_splits=30, shuffle=True, random_state=42)

results = cross_validate(clf, X, y, cv=cv, scoring=scoring)

for metric in scoring:
    mean_score = np.mean(results[f'test_{metric}'])
    std_score = np.std(results[f'test_{metric}'])
    print(f"{metric}: {mean_score:.3f} (±{std_score:.3f})")