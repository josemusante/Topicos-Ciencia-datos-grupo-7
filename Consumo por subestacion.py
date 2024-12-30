import pandas as pd

# Cargar el archivo proporcionado por el usuario
file_path = 'E:/Topicos en Ciencia de datos/topicosdatos-main/test.csv'
data = pd.read_csv(file_path)

# Mostrar las primeras filas para analizar la estructura
data.head(), data.info()


# Convertir la columna 'date' a formato datetime
data['date'] = pd.to_datetime(data['date'])

# Crear un consumo diario sumando por día y subestación
data['day'] = data['date'].dt.date  # Extraer solo la fecha sin la hora
daily_data = data.groupby(['substation', 'day'])['consumption'].sum().reset_index()

# Comprobar la estructura del nuevo conjunto de datos
daily_data.head()

from statsmodels.tsa.arima.model import ARIMA
import warnings

# Ignorar advertencias generadas por el modelo
warnings.filterwarnings("ignore")

# Seleccionar datos de una subestación (ejemplo: 'AJAHUEL')
substation_data = daily_data[daily_data['substation'] == 'LOSALME']
substation_data.set_index('day', inplace=True)

# Entrenar un modelo ARIMA
model = ARIMA(substation_data['consumption'], order=(1, 1, 1))
fitted_model = model.fit()

# Proyectar los próximos 7 días
forecast = fitted_model.forecast(steps=7)
forecast

import matplotlib.pyplot as plt

# Preparar el gráfico
plt.figure(figsize=(12, 6))
plt.plot(substation_data.index, substation_data['consumption'], label='Consumo Histórico', color='blue')
plt.plot(forecast.index, forecast, label='Proyección (7 días)', color='orange', linestyle='--')
plt.title('Consumo Diario y Proyección - Subestación LOSALME', fontsize=16)
plt.xlabel('Fecha', fontsize=12)
plt.ylabel('Consumo', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()