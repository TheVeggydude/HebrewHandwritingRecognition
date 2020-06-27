import keras
import numpy as np
from tensorflow.keras.models import load_model

# It can be used to reconstruct the model identically.
reconstructed_model = load_model("char_model_loss")

# Number of images to predict
n_samples = 10

# Let's check:
x_new = np.random.rand(n_samples, 64, 64, 3)
print(f"Shape XNEW: {x_new.shape}")

# Make model predictions and save them
y_new = np.argmax(reconstructed_model.predict(x_new), axis = -1)

# Print predictions
print(f"y_new: {y_new}")