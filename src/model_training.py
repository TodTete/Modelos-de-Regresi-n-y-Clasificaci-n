from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pickle

def train_regression_model(X_train, y_train):
    """Entrenamiento de modelos de regresión"""
    # Modelo baseline
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    # Modelo avanzado con optimización
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20]
    }
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    
    # Guardar mejor modelo
    with open('../models/regression_model.pkl', 'wb') as f:
        pickle.dump(grid_search.best_estimator_, f)
    
    return lr, grid_search.best_estimator_

def train_classification_model(X_train, y_train):
    """Entrenamiento de modelos de clasificación"""
    # Modelo baseline
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)
    
    # Modelo avanzado con optimización
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None],
        'class_weight': ['balanced', {0:1, 1:2}]
    }
    rf_clf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf_clf, param_grid, cv=5, scoring='roc_auc')
    grid_search.fit(X_train, y_train)
    
    # Guardar mejor modelo
    with open('../models/classification_model.pkl', 'wb') as f:
        pickle.dump(grid_search.best_estimator_, f)
    
    return logreg, grid_search.best_estimator_