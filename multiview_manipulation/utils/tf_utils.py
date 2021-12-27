import tensorflow as tf


def tf_setup(device=0, use_gpu=True):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.experimental.set_visible_devices(gpus[device], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[device], True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)
