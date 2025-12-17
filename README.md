# AI Business Strategy Simulator

### *Predict, Analyze, and Optimize Business Decisions using Machine Learning*

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange)

## Overview
The **Business Strategy Simulator** is an interactive Decision Support System (DSS) designed to help managers and strategists visualize the impact of their decisions before spending a single dollar.

Unlike static spreadsheets, this application uses a **Machine Learning model (XGBoost)** trained on industry-specific synthetic data to predict **Demand**, **Revenue**, and **Profit**. It accounts for complex non-linear relationships like *Price Elasticity of Demand*, *Diminishing Returns on Marketing*, and *Competitive Pressure*.

The system culminates in an **"AI Consultant"**, an optimization engine that runs grid-search simulations to mathematically determine the perfect pricing and marketing strategy for maximum profit.

## Key Features

### 1. Multi-Industry Physics Engine
The simulator doesn't just fit data; it simulates the economic "physics" of 5 distinct sectors:
* **Tech (Gadgets):** High price sensitivity, moderate marketing impact.
* **Luxury Fashion:** Inelastic demand (price drops don't help), high brand reliance.
* **FMCG (Food):** Razor-thin margins (15%), extreme price sensitivity.
* **FMCG (Cosmetics):** High margins (80%), driven by brand image.
* **FMCG (Personal Care):** Balanced volume and loyalty.

### 2. Scenario Comparison
* **Current vs. Future:** Input your current market conditions and compare them side-by-side with a simulated strategy.
* **Real-time Deltas:** Instantly see the financial impact (e.g., *Revenue +$12,000*, *Profit -$400*) with color-coded indicators.

### 3. Glass-Box AI (Explainability)
* **SHAP Integration:** Uses SHAP (SHapley Additive exPlanations) to demystify the "Black Box."
* **Visual Insights:** A bar chart visualizes exactly which factors (e.g., "High Competitor Price" or "Low Marketing Spend") are driving demand up or down.

### 4. The AI Consultant (Optimization)
* **Automated Strategy:** A built-in optimizer runs **1,000+ simulations** in milliseconds via Grid Search.
* **Recommendation Engine:** It identifies the exact "Sweet Spot" for Price and Marketing to mathematically maximize profit, effectively "solving" the game for the user.

## 🛠️ Tech Stack
* **Frontend:** [Streamlit](https://streamlit.io/) (Interactive Web UI)
* **Machine Learning:** [XGBoost](https://xgboost.readthedocs.io/) (Gradient Boosting Regressor)
* **Explainability:** [SHAP](https://shap.readthedocs.io/) (Model Interpretability)
* **Data Processing:** Pandas, NumPy
* **Visualization:** Matplotlib

## How It Works (The Logic)
The app generates synthetic training data on the fly based on economic principles:

| Industry | Elasticity | Marketing Impact | COGS (Margin) | Behavior |
| :--- | :--- | :--- | :--- | :--- |
| **FMCG (Food)** | **High (-6.5)** | Low | High Cost (15% Margin) | Volume Game |
| **Tech** | Medium (-2.0) | Medium | Medium Cost (60% Margin) | Balanced |
| **Luxury** | **Inelastic (-0.6)** | **Very High** | Low Cost (85% Margin) | Brand Game |

* **Demand Curve:** Modeled using a Power Law ($P^E$) to simulate elasticity.
* **Marketing Saturation:** Modeled using a Logarithmic Function ($\ln(x)$) to simulate diminishing returns.

## Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/business-decision-simulator.git](https://github.com/yourusername/business-decision-simulator.git)
    cd business-decision-simulator
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the app:**
    ```bash
    streamlit run app.py
    ```

## Project Structure
```text
├── app.py               # Main application logic (UI + ML + Physics)
├── requirements.txt     # Python dependencies
└── README.md            # Documentation
