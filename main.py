import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Carregar e pré-processar o conjunto de dados MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalizar entre 0 e 1
y_train, y_test = to_categorical(y_train), to_categorical(y_test)  # One-hot encoding das classes

# Criar o modelo
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Achatar a entrada 2D (imagens) para 1D
    Dense(128, activation='relu'),  # Camada oculta com 128 neurônios e ativação ReLU
    Dense(64, activation='relu'),   # Outra camada oculta
    Dense(10, activation='softmax') # Camada de saída com 10 neurônios (10 classes)
])

# Compilar o modelo
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Treinar o modelo
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Avaliar o modelo
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Acurácia no teste: {test_acc}")
