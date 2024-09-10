import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from keras._tf_keras.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler


# Generar datos de ejemplo (seno con ruido)
def generate_data(timesteps):
    np.random.seed(0)
    t = np.linspace(0, 100, timesteps)
    data = np.sin(t) + 0.1 * np.random.randn(timesteps)
    return data

def create_dataset(data, look_back=1):
    X, y = [], []
    for i in range(len(data) - look_back - 1):
        a = data[i:(i + look_back)]
        X.append(a)
        y.append(data[i + look_back])
    return np.array(X), np.array(y)

# Parámetros
timesteps = 1000
look_back = 10

# Generar y preparar los datos
data = generate_data(timesteps)
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data.reshape(-1, 1)).flatten()
X, y = create_dataset(data, look_back)

# Redimensionar los datos para la LSTM
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Construir el modelo LSTM
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Entrenar el modelo
model.fit(X, y, epochs=100, batch_size=32, verbose=2)

# Predecir
train_predict = model.predict(X)

# Invertir la normalización para ver los resultados en la escala original
train_predict = scaler.inverse_transform(train_predict)
original_data = scaler.inverse_transform(data.reshape(-1, 1))

# Visualizar los resultados
plt.figure(figsize=(12, 6))
plt.plot(original_data, label='Datos originales')
plt.plot(np.arange(look_back, look_back + len(train_predict)), train_predict, label='Predicción', color='red')
plt.xlabel('Timestep')
plt.ylabel('Valor')
plt.legend()
plt.show()


# Calcular el MSE
mse = mean_squared_error(original_data[look_back:look_back + len(train_predict)], train_predict)
print(f"Error Cuadrático Medio (MSE): {mse}")


print(len(train_predict))
print(len(original_data))


