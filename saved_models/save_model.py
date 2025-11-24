from tensorflow.keras.models import load_model

# Load your fixed model
model = load_model('ml/saved_models/base_cnn_lstm_fixed.h5')

# Save full model (architecture + weights)
model.save('ml/saved_models/base_cnn_lstm_full.h5')
print("Full model saved at ml/saved_models/base_cnn_lstm_full.h5")
