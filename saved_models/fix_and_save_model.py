// server.js
import express from 'express';
import * as tf from '@tensorflow/tfjs-node';
import dotenv from 'dotenv';
import bodyParser from 'body-parser';
import { MongoClient } from 'mongodb';

// Load environment variables from .env
dotenv.config();

// Create Express app
const app = express();
const PORT = process.env.PORT || 4000;

// Middleware
app.use(bodyParser.json());

// MongoDB connection
const mongoUrl = process.env.MONGO_URI || 'mongodb://localhost:27017';
const dbName = process.env.DB_NAME || 'nwarehouse';
let db;

MongoClient.connect(mongoUrl)
  .then((client) => {
    db = client.db(dbName);
    console.log('✅ MongoDB connected');
  })
  .catch((err) => {
    console.error('❌ MongoDB connection error:', err);
  });

// Load TensorFlow.js model
const MODEL_PATH = 'file://ml/saved_models/model_tfjs/model.json';
let model;

async function loadModel() {
  try {
    model = await tf.loadLayersModel(MODEL_PATH);
    console.log('✅ TF.js model loaded successfully');
  } catch (err) {
    console.error('❌ Model Load Error:', err);
  }
}

loadModel();

// Prediction endpoint
app.post('/predict', async (req, res) => {
  if (!model) {
    return res.status(500).json({ error: 'Model not loaded yet' });
  }

  try {
    const inputData = req.body.data; // Expecting 2D array: [batch, features] or 3D if time series
    const inputTensor = tf.tensor(inputData);

    const prediction = model.predict(inputTensor);
    const predictionData = await prediction.array();

    res.json({ prediction: predictionData });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Prediction failed', details: err.message });
  }
});

// Test endpoint
app.get('/', (req, res) => {
  res.send('🟢 Server is running!');
});

// Start server
app.listen(PORT, () => {
  console.log(`🚀 Server running at http://127.0.0.1:${PORT}`);
});
