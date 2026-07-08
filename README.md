Real-Time Financial Fraud Detection and Telemetry Platform
==========================================================

A lightweight, full-stack machine learning architecture designed to evaluate financial transactions in real time (under 5 milliseconds) while providing mathematical proof and explainability for every prediction.

Overview
--------

Modern payment gateways require sub-second transaction evaluation. Traditional rule-based engines generate high false-positive rates, while complex deep learning pipelines introduce unacceptable latency and act as regulatory black boxes.

This project solves the latency versus accuracy dilemma by uniting a deterministic fintech rules engine, stateful in-memory behavioral profiling, gradient-boosted decision trees, and game-theory explainability into a deployable microservice architecture.

Key Features
------------

-   **Stateful Behavioral Profiling (Redis):** Resolves stateless API limitations by querying an in-memory Redis cache to fetch 30-day rolling customer spending averages in under 1 millisecond.

-   **Hybrid Defense Architecture:** Places a sub-millisecond deterministic rules engine ahead of the machine learning model to immediately block extreme velocity spikes, impossible physical travel distances, and hard monetary limits without wasting CPU compute.

-   **Low-Latency ML Inference:** Utilizes an optimized XGBoost ensemble trained with Stratified K-Fold cross-validation to handle heavy class imbalance and execute classification in milliseconds.

-   **Explainable AI (SHAP):** Integrates SHAP (SHapley Additive exPlanations) directly into the backend inference pipeline to generate dynamic JSON arrays detailing the exact mathematical reasoning behind every fraud score.

-   **Live Telemetry Dashboard:** Asynchronously logs all transaction evaluations to a PostgreSQL database, powering an interactive Streamlit frontend for risk analysts to monitor fraud rates and prevented losses in real time.

## Fraud-Detection Lifecycle:
![Fraud-Detection Lifecycle](Fraud%20Detection%20Lifecycle.png)


System Architecture
-------------------

1.  **Ingestion:** A client application submits a live transaction JSON payload to the FastAPI `/predict` endpoint.

2.  **State Lookup:** The backend queries Redis for the user's historical spending baseline using an Exponential Moving Average (EMA).

3.  **Rule Evaluation:** The transaction passes through deterministic safety guardrails (e.g., hard limits, physical travel velocity).

4.  **ML Inference:** If rules are passed, XGBoost calculates the fraud probability while TreeSHAP extracts top feature risk drivers.

5.  **Telemetry Storage:** The API returns the verdict to the client and asynchronously logs the execution audit trail into PostgreSQL.

6.  **Live Visualization:** Risk analysts monitor system-wide metrics and inspect flagged transaction SHAP explanations via Streamlit.

Tech Stack
----------

-   **Backend API:** FastAPI, Uvicorn, Pydantic (Schema Validation)

-   **Machine Learning:** XGBoost, Scikit-Learn, SHAP, Pandas, NumPy

-   **Caching & Storage:** Redis (In-Memory State Store), PostgreSQL (Audit Database)

-   **Frontend Dashboard:** Streamlit

-   **DevOps & Containerization:** Docker, Docker Compose

Getting Started
---------------

### Prerequisites

-   Python 3.9 or higher

-   Docker and Docker Compose

-   Git

### Installation & Setup

1.  **Clone the Repository**

Bash

```
git clone https://github.com/23f3004075/Real-time-Fraud-Detection-System.git
cd Fraud_Detection_System

```

1.  **Start Database and Cache Services**

    Start the PostgreSQL database and Redis container in the background:

Bash

```
docker compose up db redis -d

```

1.  **Set Up Python Environment**

    Create a virtual environment and install dependencies:

Bash

```
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt

```

1.  **Start the FastAPI Backend Server**

    In your first terminal window, launch the API:

Bash

```
uvicorn app.main:app --reload --port 8000

```

1.  **Start the Streamlit Telemetry Dashboard**

    Open a second terminal window and launch the frontend:

Bash

```
streamlit run dashboard/app.py --server.port 8501

```

Testing the API
---------------

Once the backend is running, open your browser and navigate to the interactive Swagger UI:

`http://localhost:8000/docs`

Click on the `POST /predict` endpoint, click **Try it out**, and submit one of the test payloads below.

### Example 1: Normal Approved Transaction

JSON

```
{
  "user_id": 1042,
  "amount": 24.50,
  "merchant": "Starbucks",
  "category": "food_dining",
  "gender": "F",
  "city": "Seattle",
  "state": "WA",
  "lat": 47.6062,
  "long": -122.3321,
  "city_pop": 737015,
  "job": "Software Engineer",
  "dob": "1988-04-12",
  "merch_lat": 47.6080,
  "merch_long": -122.3350,
  "device": "iOS"
}

```

**Expected Response:** `is_fraud: 0`, low probability score, and standard profile explanation.

### Example 2: High-Risk Flagged Transaction (Guardrail / ML Block)

JSON

```
{
  "user_id": 9999,
  "amount": 15000.00,
  "merchant": "Unknown_Overseas_Electronics",
  "category": "shopping_net",
  "gender": "F",
  "city": "Seattle",
  "state": "WA",
  "lat": 47.6062,
  "long": -122.3321,
  "city_pop": 737015,
  "job": "Software Engineer",
  "dob": "1988-04-12",
  "merch_lat": 51.5074,
  "merch_long": -0.1278,
  "device": "Android"
}

```

**Expected Response:** `is_fraud: 1`, high probability score (`0.99`), and explicit SHAP/Rule block explanation strings.

Project Structure
-----------------

Plaintext

```
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI application and route definitions
│   ├── ml.py            # XGBoost inference, Redis EMA lookup, and SHAP logic
│   └── schemas.py       # Pydantic validation models
├── dashboard/
│   └── app.py           # Streamlit dashbord
├── model/
│   ├── best_model.pkl   
│   ├── scaler.pkl       
│   └── encoder.pkl      
├── docker-compose.yml   
├── requirements.txt     
├── .gitignore           
└── README.md            

```

License
-------

This project is open-source and available under the MIT License.
