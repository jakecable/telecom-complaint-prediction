# End-to-End Customer Complaint Analysis & Prediction API

This project is a full-cycle data science project designed to analyze the drivers of customer complaints and build predictive models. The process involved a multi-stage pipeline: data engineering, interactive visualization, statistical analysis, advanced modeling, and deployment as a REST API.

## Tech Stack
*   **Data Engineering:** Python, Pandas
*   **Data Visualization:** Microsoft Power BI
*   **Statistical Analysis:** R, ggplot2
*   **Machine Learning:** Scikit-learn, XGBoost
*   **API Deployment:** Flask

---

## Project Structure
```
customer_churn/
│
├── images/
│   ├── power_bi_dashboard.png
│   ├── histogram_complaint_volume.png
│   └── scatter_population_vs_complaints.png
│
├── src/
│   ├── data_processing_pipeline.py   # ETL script to unify all data
│   ├── analysis.R                      # R script for statistical analysis
│   ├── model_training.py             # Python script to train all 4 ML models
│   ├── app.py                        # Flask API to serve the models
│   ├── test_api.py                   # Script to test the live API
│   ├── *.csv                         # Raw and processed data files
│   └── *.joblib                      # Saved, trained model files
│
├── README.md                         # This file
└── requirements.txt                  # Python dependencies
```

---

## Phase 1: Data Engineering & ETL

### The Challenge
The project began with four separate and messy datasets: a log of thousands of individual customer complaints, a wide census file, broadband quality data aggregated by "Census Place", and a geographic crosswalk file to fix a geographic mismatch between the datasets.

### My Solution
I built an ETL (Extract, Transform, Load) pipeline using Python and Pandas to merge these sources into a single, clean dataset.

1.  **Extract & Clean:** I read in the raw CSVs, handling non-standard file encodings, skipping metadata rows, and standardizing data types to preserve data integrity by keeping leading zeros on ZIP codes.
2.  **Transform & Aggregate:** I aggregated the thousands of complaint tickets into a single metric: `complaint_volume` per ZIP code. I also parsed the census file to extract key demographic features like `total_population` and `median_age`.
3.  **Solve the Geographic Mismatch:** To solve the core challenge of joining ZIP code data with "Census Place" data, I implemented a weighted-average allocation. Using a geographic crosswalk file, I accurately distributed broadband statistics from a single city across its multiple constituent ZIP codes, ensuring a high-fidelity final dataset.

**The Outcome:** A single, unified CSV file (`unified_dataset.csv`), ready for analysis, where each row represented a ZIP code with its associated complaint volume, demographics, and broadband quality metrics.

---

## Phase 2: Visualization with Power BI

The next step was to visualize the data. I used Microsoft Power BI to create a one-page dashboard designed to answer key business questions at a glance. This dashboard successfully translated the raw numbers into an interactive, visual story, making the insights accessible.
Key Visuals:
1.  **Geographic Map ("Zip Code Hotspots")** A map of Texas with ZIP codes sized by complaint volume, to draw attention to the biggest problem areas.
2.  **KPI Cards** Cards displaying "Total Complaints", "Average Complaints per ZIP", and the Top 5 Complaining ZIP codes.
3.  **Scratter Plot ("Average Age per Complaint")** A plot of median_age vs complaint_volume, which visualized the relationship between demographics and service issues.
4.  **Slicer ("State Abbreviations")** A slicer was made to filter the data between the selected state abbreviation. For now there is only data for Texas, but if more data is added in the future then more states can be selected.
![Power BI Dashboard Screenshot](images/power_bi_dashboard.png)

---

## Phase 3: Statistical Analysis in R

I used R to create a Linear Regression Model and a scatter plot showing the connection between total population and total complaints.

### Key Findings from a Linear Regression Model:
*   **Population is the Strongest Predictor:** The model confirmed with statistical significance (`p-value < 0.001`) that `total_population` is the primary driver of complaint volume.
*   **A Surprising Insight:** After accounting for population, broadband availability was *not* a statistically significant predictor. This finding suggests that simply improving speeds might not be the most effective way to reduce complaints.
*   **Model Performance:** The linear model achieved an R-squared of **0.45**, indicating it could explain about 45% of the variance in complaints—a strong baseline but with clear room for improvement.

![Population vs Complaints Scatter Plot](images/scatter_population_vs_complaints.png)

---

## Phase 4: Advanced Modeling & API Deployment

Next I wanted to create a predictive tool that was accessible for users to test.

### 1. Building More Models
Recognizing the limitations of a linear model, I trained and compared four different machine learning models in Python using Scikit-learn and XGBoost:
*   Linear Regression
*   Support Vector Regressor (SVR)
*   Random Forest
*   XGBoost (Gradient Boosting)

The non-linear ensemble models performed significantly better, with **XGBoost achieving an R-squared of over 0.65**, proving its ability to capture the  non-linear patterns in the data.

### 2. Deploying the Models as a REST API
A model is only useful if it can be used. I used **Flask** to wrap all four trained models into a single, flexible REST API. The API exposes a `/predict` endpoint that accepts a JSON payload with a ZIP code's features and returns a real-time prediction of the expected complaint volume.

---

## How to Run This Project

### Prerequisites
*   Python 3.8+
*   R and RStudio
*   Access to a Power BI Desktop client

### 1. Setup
First, clone the repository and create a Python virtual environment.
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
Install the required Python packages from the `requirements.txt` file.
```bash
pip install -r requirements.txt
```
*(Note: You will need to create a `requirements.txt` file with the project dependencies: `pandas`, `scikit-learn`, `xgboost`, `flask`, `requests`)*

### 2. Run the Data Pipeline
Execute the ETL script to generate the unified dataset.
```bash
python src/data_processing_pipeline.py
```

### 3. Train the ML Models
Run the training script to train all four models and save them as `.joblib` files.
```bash
python src/model_training.py
```

### 4. Launch the API
Start the Flask server. The API will be available at `http://127.0.0.1:5000`.
```bash
python src/app.py
```

### 5. Test the API
In a new terminal, run the test script to send a sample request to the live API.
```bash
python src/test_api.py
```
You should see the JSON responses from all four models printed to the console.

---

## Conclusion: Creating Business Value

This project successfully delivered an end-to-end data science solution. It provides not just a historical analysis but a forward-looking predictive service. A business can now use the API to:

*   **Proactively Allocate Resources:** Identify future complaint hotspots and pre-emptively increase customer support staffing in those areas.
*   **Target Network Investments:** Pinpoint areas with disproportionately high complaints relative to their size, signaling underlying infrastructure issues.
*   **Launch Data-Driven Retention Campaigns:** Engage with customers in high-risk areas *before* they complain, reducing churn and improving customer satisfaction.
