import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

def build_model():
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Freeze MOST layers (very important)
    for layer in base_model.layers[:-10]:
        layer.trainable = False

    # Fine-tune last 10 layers
    for layer in base_model.layers[-10:]:
        layer.trainable = True

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),

        Dense(256, activation="relu"),
        Dropout(0.5),

        Dense(64, activation="relu"),
        Dropout(0.3),

        Dense(6, activation="softmax")
    ])

    model.compile(
        optimizer=Adam(learning_rate=5e-6),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top2_acc")
        ]
    )

    return model