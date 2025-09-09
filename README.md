# Car Price Prediction App
A Dash web application for predicting prices using a trained Machine Learning model.  
The app provides an interactive interface to input features and receive predictions.

---
## 🚀 How to Run (Run with Docker Compose)
```
step 1: cd app 
step 2: docker compose up --build
step 3: Open the browser at: 👉 http://localhost:8050
```
---
## 📂 App Structure 
```
├─app/code
  ├─ app.py # Main entry point for the Dash app
  ├─ pages/
  │  ├─ home.py # Home page: description of features & model
  │  └─ prediction_model.py # Input form & price prediction
  └─ model/
  │   ├─ model.pkl # Trained ML model
  │   └─ scaler.pkl # Scaler for numeric features
  └─ .Dockerfile
  └─ .dockercompose.yml
```

### Aphisit Jaemyaem st126130
