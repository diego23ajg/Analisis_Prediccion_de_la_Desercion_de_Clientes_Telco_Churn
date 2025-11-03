# Predicción de deserción de clientes (Churn) en telecomunicaciones

**Autor:** Diego Julca  
**Fecha:** Julio 2025  

## Introducción

En la industria de telecomunicaciones, retener a los clientes existentes suele ser más económico que adquirir nuevos. Entender por qué los usuarios abandonan (churn) ayuda a las empresas a actuar a tiempo y mejorar la satisfacción del cliente.

Este proyecto utiliza técnicas de aprendizaje automático para predecir qué clientes tienen mayor probabilidad de abandonar el servicio, basado en datos reales de clientes que incluyen información de uso del servicio, facturación e información demográfica.

## Objetivo

El objetivo principal es entrenar un modelo que pueda predecir el churn de los clientes utilizando características como el tipo de contrato, uso de internet, cargos mensuales y tiempo de permanencia con la empresa. Esto permitirá a la empresa tomar medidas preventivas y mejorar la retención de clientes.

## Herramientas y Tecnologías

- **Python** (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, XGBoost, SHAP)
- **Jupyter Notebook**
- **Dataset:** [Telco Customer Churn - Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Modelos de Clasificación:** Regresión Logística, Random Forest, XGBoost

## Pasos del Proyecto

### 1. Carga de Datos y Exploración Inicial

Se cargaron los datos y se exploró la estructura del dataset. Esto incluyó inspeccionar las primeras filas, verificar valores nulos, y analizar los tipos de datos.

### 2. Limpieza de Datos

Se verificaron los valores nulos, duplicados, y se aseguraron que los tipos de columnas fueran apropiados. También se eliminaron columnas irrelevantes y se trató cualquier anomalía en los datos.

### 3. Separación de Características

Se separaron las características en numéricas y categóricas para aplicar transformaciones específicas a cada tipo de datos. 

### 4. Codificación de Características y Escalado

Las características categóricas fueron codificadas mediante **One-Hot Encoding**, y las características numéricas fueron estandarizadas utilizando **StandardScaler**.

### 5. Entrenamiento de Modelos

Se entrenaron tres modelos para predecir el churn de los clientes:

- **Regresión Logística**: Un modelo simple y interpretable.
- **Random Forest**: Un modelo de ensamble que maneja bien las relaciones no lineales.
- **XGBoost**: Un modelo de boosting conocido por su alto rendimiento en tareas de clasificación.

### 6. Evaluación del Rendimiento de los Modelos

Se evaluaron los modelos utilizando métricas de clasificación como precisión, recall y F1-score. Se analizó el rendimiento en términos de cómo cada modelo maneja los verdaderos positivos, falsos positivos, y la tasa de falsos negativos.

### 7. Visualización del Rendimiento de los Modelos

Se visualizó la matriz de confusión y las curvas ROC de cada modelo para evaluar su rendimiento y balance en la clasificación de churners y no-churners.

### 8. Importancia de las Características

Se analizó la importancia de las características utilizando el modelo Random Forest, y se visualizó cuál de las características (como cargo mensual y tiempo de permanencia) tiene más influencia en la predicción de churn.

También se utilizaron valores **SHAP** para interpretar el impacto de cada característica en las predicciones del modelo **XGBoost**.

### 9. Conclusiones

- Todos los modelos mostraron un buen rendimiento, con **XGBoost** ligeramente superando a los demás.
- Las características más importantes para predecir el churn fueron el cargo mensual, la velocidad de conexión y el tiempo de vida del cliente.
- El análisis puede ayudar a la empresa a tomar medidas preventivas y mejorar la retención de clientes, especialmente enfocándose en clientes con un alto cargo mensual y baja permanencia.

### 10. Recomendaciones

- Enfocar los esfuerzos en los clientes con altos cargos mensuales y poco tiempo de servicio, ya que tienen más probabilidades de abandonar.
- Ofrecer programas de fidelización o descuentos para mejorar la retención.
- Actualizar el modelo continuamente con nuevos datos para mantener la precisión de las predicciones a lo largo del tiempo.

## Código

El código se encuentra en el archivo **`notebook.ipynb`**. El flujo de trabajo sigue estos pasos:

1. **Carga y limpieza de datos**.
2. **Transformación de características** (escala y codificación).
3. **Entrenamiento de modelos** (Regresión Logística, Random Forest, XGBoost).
4. **Evaluación de rendimiento** (matriz de confusión, métricas de clasificación).
5. **Visualización de la importancia de las características**.
6. **Interpretación de predicciones** mediante valores SHAP.

## Requisitos

Para ejecutar este proyecto, asegúrate de tener instaladas las siguientes bibliotecas:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost shap
