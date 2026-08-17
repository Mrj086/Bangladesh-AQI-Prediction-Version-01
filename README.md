# 🌏 Bangladesh Air Quality Analytics

An industry-style **data analytics project for Bangladesh air quality** using historical AQI and pollutant data.

This project transforms air-quality data into analytical insights, business-oriented findings, and an interactive Streamlit dashboard for portfolio and GitHub presentation.

---

## 📌 Project Overview

The project analyzes historical air-quality observations to understand:

- 🌫️ AQI patterns and trends
- 🏙️ Differences between cities
- 📅 Daily, monthly, and seasonal behavior
- 🧪 Relationships between AQI and major pollutants
- 🚨 High-pollution periods
- 📊 Business and operational insights

The current version focuses on **data analytics and decision support**.

---

## 🎯 Objectives

1. Clean and prepare air-quality data.
2. Perform exploratory and statistical analysis.
3. Identify temporal and geographic AQI patterns.
4. Analyze relationships between pollutants and AQI.
5. Compare air quality across locations.
6. Translate findings into practical business decisions.
7. Present results through an interactive executive-style dashboard.

---

## 🗂️ Project Structure

```text
aqi-bangladesh-analytics/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── aqi_data_analysis.ipynb
│
├── src/
│   └── analysis/
│
├── reports/
│   ├── figures/
│   └── metrics/
│
├── dashboard/
│   └── app.py
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

> Large raw datasets and generated artifacts can be excluded from the public repository using `.gitignore`.

---

## 📊 Dataset

The analysis uses historical Bangladesh air-quality observations containing measurements such as:

- AQI
- PM2.5
- PM10
- Carbon monoxide (CO)
- Carbon dioxide (CO₂)
- Nitrogen dioxide (NO₂)
- Sulphur dioxide (SO₂)
- Ozone (O₃)
- Geographic coordinates
- City/location
- Timestamp information

The data is processed into analysis-ready tables for statistical and visual analysis.

---

## 🔍 Analytics Performed

### 🧹 Data Preparation

- Missing-value analysis
- Data-type validation
- Duplicate detection
- Timestamp processing
- Feature preparation
- Data-quality checks

### 📈 Exploratory Data Analysis

The project examines:

- AQI distribution
- AQI trends over time
- Monthly and seasonal behavior
- Hourly patterns
- City-level differences
- Pollutant distributions
- Pollutant-to-AQI relationships
- High-AQI periods

### 🏙️ Geographic Analysis

City-level analysis identifies locations with:

- Higher average AQI
- Greater pollution variability
- More frequent high-pollution observations
- Different pollutant profiles

### 🧪 Pollutant Analysis

Major pollutants are analyzed to understand their relationship with AQI and identify the strongest associations with air-quality changes.

---

## 💼 Business & Decision Insights

The analysis translates environmental data into practical decision-support areas:

| Area | Example Decision |
|---|---|
| 🏭 Operations | Adjust outdoor operations during high-pollution periods |
| 👷 Workforce | Plan outdoor work and protective measures |
| 🚚 Logistics | Consider pollution conditions when scheduling outdoor activity |
| 🏙️ City Management | Prioritize locations with persistent poor AQI |
| 📢 Communication | Increase air-quality alerts during high-risk periods |
| 📊 Monitoring | Track recurring pollution patterns and pollutant trends |

The goal is to move from **raw data → analysis → actionable insight**.

---

## 📊 Streamlit Executive Dashboard

The project includes an interactive Streamlit dashboard with an executive analytics style.

Dashboard capabilities include:

- 📌 KPI cards
- 📈 AQI trend visualization
- 🏙️ City comparison
- 🧪 Pollutant analysis
- 📅 Time-based filtering
- 🚨 High-AQI analysis
- 📊 Interactive charts
- 🔎 Data exploration

### ▶️ Run the Dashboard

From the project root:

```bash
streamlit run dashboard/app.py
```

Then open the local Streamlit URL shown in the terminal, normally:

```text
http://localhost:8501
```

---

## 💻 Tech Stack

- 🐍 Python
- 🐼 Pandas
- 🔢 NumPy
- 📊 Matplotlib
- 📈 Plotly
- 🎨 Streamlit
- 📓 Jupyter Notebook
- 🗃️ Parquet / CSV
- 🔧 Git & GitHub

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd aqi-bangladesh-analytics
```

### 2. Create an environment

Using Conda:

```bash
conda create -n aqi-analytics python=3.11
conda activate aqi-analytics
```

Or using Python:

```bash
python -m venv .venv
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the notebook

```bash
jupyter notebook
```

Open:

```text
notebooks/aqi_data_analysis.ipynb
```

### 5. Run the dashboard

```bash
streamlit run dashboard/app.py
```

---

## 📌 Key Deliverables

- 📓 Reproducible analytics notebook
- 📊 Exploratory data analysis
- 📈 Statistical visualizations
- 🏙️ City-level analysis
- 🧪 Pollutant analysis
- 💼 Business-oriented insights
- 🎨 Interactive Streamlit dashboard
- 🗂️ Organized analytical artifacts
- 📄 GitHub-ready documentation

---

## 👤 Author

**MD. MIRAJ-UL-ISLAM**

🔗 LinkedIn: **www.linkedin.com/in/md-miraj-ul-islam-77b30b26a**

---

## 📄 License

This project is released under the **MIT License**.

See [`LICENSE`](LICENSE) for details.
