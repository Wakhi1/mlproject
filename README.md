
# ERS Compliance Risk Prediction API

A FastAPI-based machine learning service that predicts taxpayer compliance risk. Includes user authentication (JWT), single prediction, bulk prediction via file upload, and model versioning with retraining capabilities.

## Features

- 🔐 **JWT Authentication** – Register and login to obtain a token.
- 📈 **Single Prediction** – Submit one taxpayer record and get risk level.
- 📊 **Bulk Prediction** – Upload CSV/Excel files for batch predictions (download results).
- 🤖 **Model Versioning** – Track trained models with performance metrics (AUC, accuracy, etc.).
- 🔄 **Retraining** – Upload labeled data to retrain the model and create a new version.

## Tech Stack

- **FastAPI** – Web framework.
- **SQLAlchemy** + **MySQL** (XAMPP) – User and model version storage.
- **Scikit-learn** – RandomForest classifier.
- **Pandas** – Data processing.
- **JWT** – Authentication.
- **bcrypt** – Password hashing.



## Setup Instructions

### 1. Prerequisites

- Python 3.9+
- XAMPP (MySQL) running locally, MariaDB when containerized
- Git and GitHub for CI/CD(github actions)

### 2. Clone Repository

```bash
git clone https://github.com/Wakhi1/mlproject.git
cd mlproject
```

### 3. Create Virtual Environment

```bash
python -m venv mlops-env
# Windows
mlops-env\Scripts\activate
# Linux/Mac
source mlops-env/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure Database (XAMPP)

- Start MySQL from XAMPP Control Panel.
- Create database `compliance_db` (via phpMyAdmin or SQL command).
- The tables (`users`, `model_versions`) will be created automatically on first run (see `app/main.py`).

### 6. Environment Variables

Create a `.env` file in the project root:

```env
# Database
DB_USER=user
DB_PASSWORD=password
DB_HOST=localhost
DB_PORT=3306
DB_NAME=compliance_db

# JWT
JWT_SECRET=your-very-secret-key-change-this
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

*sample database
```
--
-- Database: `compliance_db`
--

-- --------------------------------------------------------

--
-- Table structure for table `model_versions`
--

CREATE TABLE `model_versions` (
  `id` int(11) NOT NULL,
  `version` int(11) NOT NULL,
  `created_at` datetime DEFAULT NULL,
  `metrics` text DEFAULT NULL,
  `notes` varchar(255) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `users`
--

CREATE TABLE `users` (
  `id` int(11) NOT NULL,
  `username` varchar(50) NOT NULL,
  `email` varchar(100) NOT NULL,
  `full_name` varchar(100) DEFAULT NULL,
  `hashed_password` varchar(255) NOT NULL,
  `disabled` tinyint(1) DEFAULT 0,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `users`
--

INSERT INTO `users` (`id`, `username`, `email`, `full_name`, `hashed_password`, `disabled`, `created_at`) VALUES
(1, 'admin', 'admin@system.com', 'System Administrator', '$2a$12$Qj4Et29Lrf0jzQaZ.DfgT.XCHHOJM/qwVp2cB3M7n22nEkYkCYt/i', 0, '2026-03-01 21:50:31'),
(2, 'inspector', 'inspector@system.com', 'System Inspector', '$2a$12$Qj4Et29Lrf0jzQaZ.DfgT.XCHHOJM/qwVp2cB3M7n22nEkYkCYt/i', 0, '2026-03-01 21:50:31'),
(3, 'custom_officer', 'custom.officer@system.com', 'Custom Officer', '$2a$12$Qj4Et29Lrf0jzQaZ.DfgT.XCHHOJM/qwVp2cB3M7n22nEkYkCYt/i', 0, '2026-03-01 21:50:31'),
(4, 'tester', 'tester@system.com', 'System Tester', '$2a$12$Qj4Et29Lrf0jzQaZ.DfgT.XCHHOJM/qwVp2cB3M7n22nEkYkCYt/i', 0, '2026-03-01 21:50:31');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `model_versions`
--
ALTER TABLE `model_versions`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `version` (`version`),
  ADD KEY `ix_model_versions_id` (`id`);

--
-- Indexes for table `users`
--
ALTER TABLE `users`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `username` (`username`),
  ADD UNIQUE KEY `email` (`email`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `model_versions`
--
ALTER TABLE `model_versions`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `users`
--
ALTER TABLE `users`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=5;
COMMIT;
```

*Note: If MySQL has a password, set `DB_PASSWORD` accordingly.*

### 7. Train Initial Model

Place the sample dataset `ers_sample_dataset.csv` in the project root, then run:

```bash
cd training
python train.py
```

This creates the model artifacts in `ml_artifacts/`.

### 8. Run the API Server

```bash
uvicorn app.main:app --reload
```

Server will be available at `http://127.0.0.1:8000`.

### 9. Import Postman Collection

Import the provided `Compliance_Risk_API.postman_collection.json` into Postman to test all endpoints.

---

## API Endpoints

All protected endpoints require a Bearer token obtained from `/auth/login`.

### Health Check

- `GET /health` – No authentication. Returns API status.

### Authentication

- `POST /auth/register` – Register new user.
  ```json
  {
    "username": "testuser",
    "email": "test@example.com",
    "full_name": "Test User",
    "password": "secret"
  }
  ```

- `POST /auth/login` – Login, returns JWT token.
  - Body (form-urlencoded): `username` and `password`.

### Prediction

- `POST /predict` – Single prediction.
  - Headers: `Authorization: Bearer <token>`
  - Body: JSON matching the training features (see example in collection).

- `POST /predict/bulk` – Batch prediction.
  - Headers: `Authorization: Bearer <token>`
  - Body: `form-data` with key `file` (CSV or Excel file).
  - Returns a CSV file with predictions appended.

### Model Management

- `GET /model/versions` – List all trained model versions with metrics.
- `POST /model/retrain` – Retrain model with new labeled data.
  - Headers: `Authorization: Bearer <token>`
  - Body: `form-data`
    - `file`: CSV file containing all features plus `is_non_compliant`.
    - `notes` (optional): description of the retraining.
  - Returns new version number and evaluation metrics.

---

## Example Usage

### 1. Register a user

```bash
curl -X POST "http://127.0.0.1:8000/auth/register" \
  -H "Content-Type: application/json" \
  -d '{"username":"user1","email":"user1@example.com","full_name":"User One","password":"pass123"}'
```

### 2. Login

```bash
curl -X POST "http://127.0.0.1:8000/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=user1&password=pass123"
```

Save the returned `access_token`.

### 3. Single prediction

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "taxpayer_type": "Company",
    "region": "Manzini",
    "industry_sector": "Retail",
    "years_registered": 5,
    "annual_turnover_szl": 1200000,
    "vat_registered": true,
    "paye_registered": true,
    "num_employees_declared": 30,
    "filings_due_last_12m": 12,
    "filings_submitted_last_12m": 10,
    "late_filings_count": 3,
    "amended_returns_count": 2,
    "outstanding_tax_szl": 78000,
    "penalty_count": 2,
    "prior_audit_flag": true,
    "prior_audit_finding": true,
    "days_since_last_payment": 95,
    "payment_plan_active": false,
    "cross_border_transactions": false
  }'
```

### 4. Bulk prediction

```bash
curl -X POST "http://127.0.0.1:8000/predict/bulk" \
  -H "Authorization: Bearer <token>" \
  -F "file=@/path/to/your/data.csv" \
  --output predictions.csv
```

### 5. Retrain model

```bash
curl -X POST "http://127.0.0.1:8000/model/retrain" \
  -H "Authorization: Bearer <token>" \
  -F "file=@/path/to/labeled_data.csv" \
  -F "notes=Retrained with March 2026 data"
```

---

## File Structure

```
compliance-risk-model/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app with routers
│   ├── database.py           # DB connection
│   ├── models.py             # SQLAlchemy models + Pydantic schemas
│   ├── auth.py               # JWT & bcrypt functions
│   ├── ml_utils.py           # Shared ML functions
│   └── routes/
│       ├── auth.py           # /auth endpoints
│       ├── predict.py        # single prediction
│       ├── bulk.py           # bulk prediction
│       └── model.py          # versioning & retraining
├── ml_artifacts/             # trained model files
├── training/
│   └── train.py              # initial model training
├── .env
├── requirements.txt
└── postman_collection.json
```

---

## Model Versioning

Each retraining creates a new entry in the `model_versions` table with:

- `version`: auto-incremented integer.
- `created_at`: timestamp.
- `metrics`: JSON with `accuracy`, `precision`, `recall`, `f1_score`, `auc`, `test_size`.
- `notes`: user‑provided description.

The model artifacts (`model.pkl`, `encoders.pkl`, `columns.pkl`) are overwritten. For production, consider archiving previous versions separately.

---

## Troubleshooting

- **Database connection error**: Verify MySQL is running and credentials in `.env` are correct.
- **ModuleNotFoundError**: Ensure all packages installed (`pip install -r requirements.txt`).
- **JWT authentication fails**: Check that `JWT_SECRET` is set and the token is passed correctly.
- **Bulk upload fails**: File must contain all required feature columns (same as training). For retraining, must include `is_non_compliant`.

---


## Running the Full Stack
Create the .env file with your secrets.

Ensure ports are free (3307, 5000, 8000, 9090, 3000). If conflicts, adjust docker-compose.yml.

## Start all services:
```bash
docker-compose up -d
```

## Check logs:
```bash
docker-compose logs -f app
```

## Access services:

### API docs: http://localhost:8000/docs

### MLflow UI: http://localhost:5000

### Prometheus: http://localhost:9090

### Grafana: http://localhost:3000 (admin/admin)


## Train the initial model (inside the app container or locally with MLflow running):
```bash
docker exec -it compliance_api python training/train.py
```


This logs the model to MLflow and registers it.

Register a user via the API, login, and test predictions.

## CI/CD (GitHub Actions)
The .github/workflows/ci-cd.yml file provided earlier will run tests, build the Docker image, and deploy on push to main. Make sure to set repository secrets: DOCKER_USERNAME, DOCKER_PASSWORD, SERVER_HOST, SERVER_USER, SSH_PRIVATE_KEY.

## Final Notes
Drift detection: Implement a periodic task that computes drift (e.g., PSI) between training and recent predictions, and updates the drift_score gauge.

Model version metrics: After each retraining, update the model_auc, model_accuracy, etc., gauges using the metrics from MLflow.

Authentication metrics: Increment login_attempts and registrations counters in your auth routes.

Your MLOps pipeline is now complete: containerized, with monitoring, experiment tracking, and CI/CD. Let me know if you need help with any specific implementation detail!

## License

[MIT](LICENSE)
