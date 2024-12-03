import tensorflow as tf

# Cargar los datos de entrenamiento y prueba
# Se asume que el conjunto de datos esta almacenado en el disco

train_data = tf.preprocessing.image_dataset_from_directory(
    'data/train',
    labels='inferred',
    label_mode='int',
    color_mode='rgb',
    image_size=(224, 224),
    batch_size=32,
    shuffle=True,
    seed=1234
)

test_data = tf.preprocessing.image_dataset_from_directory(
    'data/test',
    labels='inferred',
    label_mode='int',
    color_mode='rgb',
    image_size=(224, 224),
    batch_size=32,
    shuffle=True,
    seed=1234
)

# Crear el modelo de redes neuronales
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compilar el modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])