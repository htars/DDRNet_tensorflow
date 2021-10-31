import numpy as np
import tensorflow as tf


def h5_evaluate():
    model = tf.keras.models.load_model("model.h5")

    x = np.random.randn(1024*1024)
    x = x.reshape(1, 1024, 1024, 1)
    y = model.predict(x)
    print(y.shape)

    
def saved_model_evaluate():
    model = tf.keras.models.load_model("saved_model")
    
    x = np.random.randn(1024*1024)
    x = x.reshape(1, 1024, 1024, 1)
    y = model.predict(x)
    print(y.shape)

    
if __name__ == "__main__":
    saved_model_evaluate()
