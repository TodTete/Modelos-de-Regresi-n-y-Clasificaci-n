# **Proyecto de Modelos Predictivos: Regresión y Clasificación**  
**Autor:** Ricardo Vallejo Sánchez  

## **📌 Descripción del Proyecto**  
Este repositorio implementa un **flujo completo de ciencia de datos** para resolver dos problemas fundamentales en *machine learning*:  
1. **🔢 Regresión**: Predicción de precios de viviendas utilizando el *Boston Housing Dataset*.  
2. **🏷️ Clasificación**: Diagnóstico de diabetes (pacientes diabéticos vs no diabéticos) con el *Pima Indians Diabetes Dataset*.  

El proyecto abarca desde el **análisis exploratorio (EDA)** hasta el **despliegue de modelos** en una API REST, siguiendo buenas prácticas de *MLOps*.  

---

## **📂 Estructura del Repositorio**  
```bash
proyecto_ml/
│
├── data/               # Datos crudos y procesados
├── notebooks/          # Jupyter Notebooks para EDA y modelado
├── src/                # Scripts Python (preprocesamiento, entrenamiento)
├── models/             # Modelos serializados (.pkl)
├── reports/            # Resultados y visualizaciones
├── app/                # API Flask para despliegue
├── tests/              # Pruebas unitarias
├── .github/            # Configuración CI/CD (opcional)
├── requirements.txt    # Dependencias
└── README.md           # Documentación principal
```

---

## **⚙️ Instalación y Configuración**  

### **Requisitos Previos**  
- Python 3.8+  
- Git  

### **Pasos para Ejecutar el Proyecto**  
1. **Clonar el repositorio**:  
   ```bash
   git clone https://github.com/tu-usuario/proyecto-ml.git
   cd proyecto-ml
   ```

2. **Crear y activar entorno virtual** (recomendado):  
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   .venv\Scripts\activate     # Windows
   ```

3. **Instalar dependencias**:  
   ```bash
   pip install -r requirements.txt
   ```

4. **Ejecutar el flujo de trabajo**:  
   - **Análisis exploratorio**: Abrir los notebooks en `notebooks/` en orden numérico.  
   - **Entrenamiento de modelos**: Ejecutar los scripts en `src/`.  
   - **Desplegar la API**:  
     ```bash
     cd app
     python app.py
     ```
     La API estará disponible en `http://localhost:5000`.

---

## **📊 Datos Utilizados**  

### **1. Boston Housing Dataset (Regresión)**  
- **Objetivo**: Predecir el valor mediano de viviendas en Boston (1978).  
- **Características**: 13 variables (CRIM: tasa de criminalidad, RM: habitaciones por vivienda, etc.).  
- **Fuente**: [Scikit-learn datasets](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html) *(Nota: En versiones recientes, reemplazado por `fetch_california_housing` debido a problemas éticos)*.  

### **2. Pima Indians Diabetes Dataset (Clasificación)**  
- **Objetivo**: Clasificar pacientes en diabéticos (1) o no diabéticos (0).  
- **Características**: 8 variables médicas (glucosa, IMC, edad, etc.).  
- **Fuente**: [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).  

---

## **🚀 Uso Avanzado**  

### **Cómo Contribuir**  
1. **Reportar issues**: Errores o mejoras.  
2. **Enviar *pull requests***:  
   ```bash
   git checkout -b feature/nueva-funcionalidad
   git add .
   git commit -m "Descripción de los cambios"
   git push origin feature/nueva-funcionalidad
   ```

### **Pruebas Unitarias**  
Ejecutar tests desde la raíz:  
```bash
python -m pytest tests/
```

### **Despliegue en Producción**  
- **Opción 1**: Usar Flask con Gunicorn:  
  ```bash
  gunicorn --bind 0.0.0.0:5000 app:app
  ```
- **Opción 2**: Contenerizar con Docker:  
  ```dockerfile
  FROM python:3.8-slim
  WORKDIR /app
  COPY . .
  RUN pip install -r requirements.txt
  CMD ["python", "app/app.py"]
  ```

---

## **📌 Autores y Contacto**  
**👨💻 Ricardo Vallejo Sánchez**  
- 🌟 **¡Contribuciones son bienvenidas!**  

---

## **📜 Licencia**  
Este proyecto está bajo la licencia **MIT**. Ver [LICENSE](LICENSE) para más detalles.  

---

### **🔗 Enlaces Útiles**  
- [Documentación de Scikit-learn](https://scikit-learn.org/stable/documentation.html)  
- [Flask API Tutorial](https://flask.palletsprojects.com/en/2.0.x/tutorial/)  

--- 

**✨ ¡Gracias por tu interés en este proyecto!**
