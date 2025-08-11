import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

def preprocess_regression_data(filepath):
    """Preprocesamiento para datos de regresión"""
    df = pd.read_csv(filepath)
    # Limpieza de outliers
    Q1 = df['target'].quantile(0.25)
    Q3 = df['target'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df['target'] < (Q1 - 1.5*IQR)) | (df['target'] > (Q3 + 1.5*IQR)))]
    
    # Escalado
    scaler = StandardScaler()
    X = df.drop('target', axis=1)
    y = df['target']
    X_scaled = scaler.fit_transform(X)
    
    return train_test_split(X_scaled, y, test_size=0.3, random_state=42)

def preprocess_classification_data(filepath):
    """Preprocesamiento para datos de clasificación"""
    df = pd.read_csv(filepath)
    # Manejo de valores faltantes
    df.fillna(df.median(), inplace=True)
    
    # Balanceo de clases
    from imblearn.over_sampling import SMOTE
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    smote = SMOTE()
    X_res, y_res = smote.fit_resample(X, y)
    
    # Escalado
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_res)
    
    return train_test_split(X_scaled, y_res, test_size=0.3, random_state=42, stratify=y_res)