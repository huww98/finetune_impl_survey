import numpy as np

import tensorflow as tf
from tensorflow.keras import layers

def build_base_model():
    base_model = tf.keras.Sequential([
        layers.Dense(16, activation='relu', input_shape=(2,)),
        layers.Dense(16, activation='relu'),
        layers.BatchNormalization(name='bn'),
        layers.Dense(16),
    ])
    return base_model

def build_model():
    base_model = build_base_model()
    model = tf.keras.Sequential([
        base_model,
        layers.Dense(1),
    ])
    return base_model, model

def test(train_mode_base_model: bool, all_params_to_optimizer: bool, freeze_features: bool):
    base_model, model = build_model()
    initial_base_model = build_base_model()
    initial_base_model.set_weights(base_model.get_weights())

    if freeze_features:
        for l in base_model.layers:
            l.trainable = False

    base_model.get_layer('bn').trainable = train_mode_base_model

    base_model.compile()
    model.compile()
    # base_model.summary()
    # model.summary()

    test_input_1 = np.random.normal(size=(128, 2))
    test_input_2 = np.random.normal(size=(128, 2))
    initial_feature_1 = base_model(test_input_1, training=False)
    assert(tf.reduce_all(initial_feature_1 == initial_base_model(test_input_1, training=False)))

    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)

    with tf.GradientTape() as tape:
        y = model(test_input_2, training=True)
        loss = tf.reduce_sum(y)

    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    feature_1 = base_model(test_input_1, training=False)

    bn_changed = bool(tf.reduce_any(initial_base_model.get_layer('bn').moving_mean != base_model.get_layer('bn').moving_mean) or
        tf.reduce_any(initial_base_model.get_layer('bn').moving_variance != base_model.get_layer('bn').moving_variance))
    feature_changed = bool(tf.reduce_any(initial_feature_1 != feature_1))

    print(bn_changed, feature_changed)


if __name__ == "__main__":
    test(True, True, True)
    test(False, True, True)
    test(True, True, False)
    test(False, True, False)
