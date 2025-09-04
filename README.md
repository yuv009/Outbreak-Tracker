Disease Outbreak Prediction & Geospatial Analysis 🩺
====================================================

This project offers a comprehensive, data-driven tool for understanding and predicting disease patterns in India. It moves beyond simple prediction by integrating real-world, location-based data and providing powerful geospatial visualizations to uncover hidden trends.

The core mission is to create an accessible web application that serves two primary functions: first, to provide an intelligent, context-aware disease prediction based on symptoms, and second, to identify and visualize endemic hotspots of various diseases across the country.

\## 🎯 Key Objectives
---------------------

*   **Intelligent Prediction**: To develop a robust machine learning model that accurately predicts potential diseases from a set of user-provided symptoms.
    
*   **Contextual Accuracy**: To significantly enhance the prediction model's relevance by factoring in real-world disease prevalence data from the user's specific state.
    
*   **Geospatial Insight**: To apply unsupervised clustering techniques to a large-scale dataset to automatically identify and map geographical disease hotspots.
    
*   **User-Friendly Interface**: To wrap these complex analytical tools in a simple, interactive, and easy-to-use Streamlit web application.
    

\## ⚙️ How It Works
-------------------

The application's functionality is split into two sophisticated components: a **Predictive Model** and a **Geospatial Cluster Analysis** map.

### \### 1. The Predictive Model: Beyond Basic Symptoms

The prediction feature is more than just a simple symptom checker. It employs a two-stage process to deliver a nuanced and realistic assessment.

*   **Stage 1: Machine Learning Core**A **Random Forest Classifier** serves as the predictive engine. This powerful ensemble model, composed of hundreds of individual decision trees, is trained on a vast dataset linking symptoms to diagnoses. It analyzes the user's input symptoms to calculate a baseline probability for a wide range of diseases.
    
*   **Stage 2: Location-Aware Scoring**This is the project's key innovation. The model understands that a disease's likelihood is not just about symptoms—it's also about location. The baseline probabilities from the Random Forest are intelligently adjusted using **real-world active case data**. The final score for each predicted disease is a weighted combination of the model's symptom-based confidence and the actual prevalence of that disease in the user's selected state. This makes the predictions far more practical and grounded in reality.
    

### \### 2. Geospatial Cluster Analysis: Finding the Hotspots

The "Interactive Map" tab reveals macro-level insights about disease distribution across India.

*   **Unsupervised Clustering**: The system uses the **K-Means clustering algorithm** on a dataset containing geo-coordinates (latitude/longitude), disease diagnoses, and environmental factors. K-Means is an unsupervised method, meaning it finds natural patterns and groups in the data without being told what to look for.
    
*   **Visualizing the Results**: The algorithm groups the data into 10 distinct clusters. Each cluster represents a geographical area with a unique disease profile. These clusters are then plotted on an interactive **Folium map**, with each cluster assigned a different color. The map features a detailed legend that automatically analyzes and describes each cluster's characteristics, such as "Plains Dengue & Malaria Zone (Monsoon)"—transforming raw data into actionable geographical intelligence.
