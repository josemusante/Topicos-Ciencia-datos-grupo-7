import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
import warnings
import math


# Ignorar advertencias
warnings.filterwarnings("ignore")

# 1. Cargar los datos
file_path = 'E:/Topicos en Ciencia de datos/topicosdatos-main/test.csv'
df = pd.read_csv(file_path)

# 2. Convertir la columna de fecha a formato datetime
df['date'] = pd.to_datetime(df['date'])

# Inicializar lista para guardar métricas
metrics_list = []

# 3. Agrupar por 'substation' y modelar ARIMA
substations = df['substation'].unique()

for substation in substations:
    # Filtrar los datos por substation
    data = df[df['substation'] == substation].copy()
    data.set_index('date', inplace=True)
    data = data['consumption'].asfreq('H')  # Frecuencia horaria

    # Manejar valores faltantes
    data = data.fillna(method='ffill')

    # Dividir los datos
    train = data[:-24]  # Entrenamiento: todas menos las últimas 24 horas
    test = data[-24:]   # Prueba: últimas 24 horas

    # 4. Crear y ajustar el modelo ARIMA
    model = ARIMA(train, order=(2, 1, 2))
    fitted_model = model.fit()

    # 5. Predicción
    forecast = fitted_model.forecast(steps=24)

    # 6. Calcular métricas
    mae = mean_absolute_error(test, forecast)
    mse = mean_squared_error(test, forecast)
    rmse = math.sqrt(mse)
    # Clasificación binaria: consumo alto o bajo
    threshold = train.mean()  # Umbral = media del consumo de entrenamiento
    y_true = (test > threshold).astype(int)
    y_pred = (forecast > threshold).astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    

    # Guardar métricas
    metrics_list.append({
        'Substation': substation,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'Accuracy': accuracy,
        'Precision': precision,
    })

# 7. Guardar resultados en un archivo CSV
metrics_df = pd.DataFrame(metrics_list)
output_path = 'E:/Topicos en Ciencia de datos/topicosdatos-main/valores_3.csv'
metrics_df.to_csv(output_path, index=False)

print(f"Las métricas se han guardado en: {output_path}")

# 8. Visualizar resultados
print(metrics_df)