from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_regression(model, X_test, y_test, model_name):
    """Evaluación de modelos de regresión"""
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    # Gráfico de residuos
    plt.figure(figsize=(10,6))
    sns.residplot(x=preds, y=y_test, lowess=True, line_kws={'color': 'red'})
    plt.title(f'Gráfico de Residuos - {model_name}')
    plt.savefig(f'../reports/residuals_{model_name}.png')
    
    return {'Model': model_name, 'MSE': mse, 'R2': r2}

def evaluate_classification(model, X_test, y_test, model_name):
    """Evaluación de modelos de clasificación"""
    preds = model.predict(X_test)
    pred_probs = model.predict_proba(X_test)[:,1]
    
    # Métricas
    accuracy = accuracy_score(y_test, preds)
    roc_auc = roc_auc_score(y_test, pred_probs)
    report = classification_report(y_test, preds, output_dict=True)
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matriz de Confusión - {model_name}')
    plt.savefig(f'../reports/confusion_matrix_{model_name}.png')
    
    return {
        'Model': model_name,
        'Accuracy': accuracy,
        'ROC_AUC': roc_auc,
        'Precision_0': report['0']['precision'],
        'Recall_0': report['0']['recall'],
        'Precision_1': report['1']['precision'],
        'Recall_1': report['1']['recall']
    }