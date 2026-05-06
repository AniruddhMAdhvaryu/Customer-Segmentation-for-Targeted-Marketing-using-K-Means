# Customer Segmentation for Targeted Marketing using K Means

## Overview
This project analyzes mall customer data to identify distinct customer segments based on their annual income and spending behavior.
Using **K-Means clustering**, the application groups customers into meaningful clusters that help businesses make data-driven marketing decisions.
An **interactive Streamlit dashboard** is built to visualize these segments dynamically and provide actionable insights.


## Features
* Interactive customer segmentation using K-Means
* Dynamic selection of number of clusters (K)
* Elbow Method visualization to determine optimal K
* Interactive and colorful Plotly visualizations
* Real-time clustering updates
* Customer input simulation (predict cluster for new customer)
* Download clustered dataset
* Clean and modern UI with responsive design

---

## Tech Stack

* **Python**
* **Pandas** – Data preprocessing
* **Scikit-learn** – K-Means clustering
* **Plotly** – Interactive visualizations
* **Streamlit** – Web application framework


## Dataset
The dataset contains mall customer information with features such as:
* Customer ID
* Gender
* Age
* Annual Income (k$)
* Spending Score (1–100)


## How It Works
1. Load and preprocess the dataset
2. Apply K-Means clustering algorithm
3. Determine optimal clusters using the Elbow Method
4. Visualize clusters using interactive scatter plots
5. Provide insights for each customer segment


## Key Insights
* Identifies **high-income low-spending customers** (potential targets)
* Detects **low-income high-spending customers**
* Segments customers for **personalized marketing strategies**


##  Use Cases
* Customer segmentation for retail businesses
* Targeted marketing campaigns
* Business intelligence dashboards
* Data science portfolio project


##  Future Improvements
* Add more clustering algorithms (DBSCAN, Hierarchical)
* Include more features (age-based segmentation)
* Improve UI with animations and themes
* Deploy with authentication for business use


