import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import warnings

# Ignorar advertencias
warnings.filterwarnings("ignore")

# 1. Cargar los datos
file_path = 'C:/Users/Jose Musante/Downloads/topicosdatos-main/train.csv'  # Asegúrate de que la ruta es correcta
df = pd.read_csv(file_path)

# 2. Convertir la columna de fecha a datetime
df['date'] = pd.to_datetime(df['date'])

# 3. Agrupar por 'substation' y modelar ARIMA
substations = df['substation'].unique()

# Crear una figura para visualizar los pronósticos
plt.figure(figsize=(12, 8))

# Número de pasos para 12 meses en datos horarios (12 meses = 8640 horas)
forecast_steps = 12 * 30 * 24

for substation in substations:
    # Filtrar los datos por substation
    data = df[df['substation'] == substation].copy()
    data.set_index('date', inplace=True)  # Establecer la fecha como índice
    data = data['consumption'].asfreq('H')  # Asegurarse de que los datos sean horarios
    
    # Manejar valores faltantes (si existen)
    data = data.fillna(method='ffill')
    
    # 4. Ajustar el modelo ARIMA
    model = ARIMA(data, order=(2, 1, 2))  # Puedes ajustar el orden ARIMA(p, d, q)
    fitted_model = model.fit()
    
    # 5. Realizar el forecast para 9 meses (8640 horas)
    forecast = fitted_model.forecast(steps=forecast_steps)
    
    # Crear un índice de fechas para el forecast
    forecast_index = pd.date_range(start=data.index[-1], periods=forecast_steps + 1, freq='H')[1:]
    
    # 6. Visualizar resultados
    plt.plot(data.index, data, label=f'{substation} - Real')
    plt.plot(forecast_index, forecast, '--', label=f'{substation} - Forecast 12 meses')

# Personalización del gráfico
plt.title('Forecast de Consumo por Substation para 12 meses usando ARIMA')
plt.xlabel('Fecha')
plt.ylabel('Consumo')
plt.legend()
plt.grid(True)

# Mostrar gráfico
plt.show()