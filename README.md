# ML Price Prediction App

A Dash web application for predicting prices using a trained Machine Learning model.  
The app provides an interactive interface to input features and receive predictions.

---

## 📂 App Structure 
```
├─app/code
│ ├─ app.py # Main entry point for the Dash app
│ ├─ pages/
│ │  ├─ home.py # Home page: description of features & model
│ │  └─ prediction_model.py # Input form & price prediction
│ └─ model/
│    ├─ model.pkl # Trained ML model
│    └─ scaler.pkl # Scaler for numeric features
│ └─ .Dockerfile
  └─ .dockercompose.yml
├─README.md
```

