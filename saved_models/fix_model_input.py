# fix_model_input.py
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input
import os

# ⚠️ Set these according to your original model
TIMESTEPS = 100   # Number of timesteps in your sequence
FEATURES = 17     # Number of features per timestep (must match original model)

# Paths
original_model_path = "base_cnn_lstm.h5"
fixed_model_path = "base_cnn_lstm_fixed.h5"

# Load the original model without compiling
model = load_model(original_model_path, compile=False)
print("Original model loaded.")

# Create a new input layer with correct shape
new_input = Input(shape=(TIMESTEPS, FEATURES), name="fixed_input")

# Connect new input to the original model's layers
x = new_input
for layer in model.layers[1:]:  # skip original input layer
    x = layer(x)

# Create a new fixed model
fixed_model = Model(inputs=new_input, outputs=x)
print("Fixed model created with corrected input shape.")

# Save fixed model
fixed_model.save(fixed_model_path)
print(f"Fixed model saved to: {fixed_model_path}")
