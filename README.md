# Food Delivery Churn Prediction

## Overview

An **end-to-end Machine Learning project** that predicts whether a customer will continue using a food delivery service or churn. The model is trained using structured customer, behavioral, and service-related data and is served using a **Flask web application and REST API**.

This project focuses on **clean architecture, modular ML pipelines, and real-world deployment practices**.

---

## Tech Stack

* Python
* Flask
* Scikit-learn
* Pandas, NumPy

---

## Project Structure

```
FOOD_DELIVERY_PREDICTION/
├── app.py                  # Flask app (UI + API)
├── templates/              # HTML pages
├── src/
│   ├── components/         # Ingestion, transformation, training
│   ├── pipeline/           # Training & prediction pipelines
│   ├── logger.py
│   ├── exception.py
│   └── utils.py
├── artifacts/              # Saved models & transformers
└── requirements.txt
```

---

## Key Features

* End-to-end ML pipeline (training + inference)
* Flask-based web interface for predictions
* REST API for programmatic access
* Modular, scalable project design
* Custom logging and exception handling

---

## Flask Endpoints

### Web UI

```
GET  /
GET  /predict_datapoint
POST /predict_datapoint
```

### API

```
POST /api/predict
```

**Sample Request**

```json
{
  "age": 25,
  "family_size": 3,
  "gender": "Male",
  "marital_status": "Single",
  "occupation": "Student"
}
```

### Health Check

```
GET /health
```

---

## How to Run

```bash
git clone https://github.com/Shivamsp13/FOOD_DELIVERY_PREDICTION.git
cd FOOD_DELIVERY_PREDICTION
pip install -r requirements.txt
python app.py
```

App runs at: `http://localhost:5000`

---

## Author

**Shivam Pandey**

---

## Notes

This project demonstrates **industry-style ML engineering and deployment**, making it suitable for internships and entry-level roles.
