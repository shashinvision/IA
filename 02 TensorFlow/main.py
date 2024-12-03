import tensorflow as tf
import os

# Ruta del conjunto de datos
dataset_path = './IMGs/'
print("******************************************")

print(f" Carpeta actual: {os.getcwd()}")
print("******************************************")

# Cargar los datos con división en entrenamiento y validación
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    labels='inferred',
    label_mode='int',
    color_mode='rgb',
    image_size=(224, 224),
    batch_size=32,
    shuffle=True,
    seed=1234,
    validation_split=0.2,  # 80% entrenamiento, 20% validación
    subset='training'
)

val_data = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    labels='inferred',
    label_mode='int',
    color_mode='rgb',
    image_size=(224, 224),
    batch_size=32,
    shuffle=True,
    seed=1234,
    validation_split=0.2,
    subset='validation'
)

# Aumento de datos para entrenamiento
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.1)
])

# Aplicar aumento de datos solo al conjunto de entrenamiento
train_data = train_data.map(lambda x, y: (data_augmentation(x, training=True), y))

# Cargar modelo preentrenado MobileNetV2
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,  # Excluir las capas densas superiores
    weights='imagenet'  # Usar pesos preentrenados en ImageNet
)

base_model.trainable = False  # Congelar las capas del modelo base

# Crear el modelo
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')  # Ajustar al número de clases
])

# Compilar el modelo
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Mostrar el resumen del modelo
model.summary()

# Entrenar el modelo
model.fit(train_data, validation_data=val_data, epochs=10)

# Evaluar el modelo en el conjunto de validación
loss, accuracy = model.evaluate(val_data)
print("******************************************")
print(f'Validation Accuracy: {accuracy:.2f}, Loss: {loss:.2f}')
print("******************************************")

# Descongelar capas del modelo base para ajuste fino
base_model.trainable = True

# Recompilar el modelo con una tasa de aprendizaje baja para evitar dañar los pesos preentrenados
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Entrenar nuevamente el modelo con ajuste fino
model.fit(train_data, validation_data=val_data, epochs=5)

# Evaluar el modelo después del ajuste fino
loss, accuracy = model.evaluate(val_data)
print("******************************************")
print(f'Fine-Tuned Validation Accuracy: {accuracy:.2f}, Loss: {loss:.2f}')
print("******************************************")
