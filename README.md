# COMP3610-Project
# Predicting Flight Delays Using Machine Learning

**Problem Statement:**

Flight delays are a major challenge in the aviation industry, leading to passenger inconvenience, financial losses, and operational inefficiencies for airlines and airports. Our project aims to develop a machine learning model to predict flight delays by leveraging historical flight records, aircraft specifications, weather conditions, carrier performance, route details, and airport congestion metrics.

**Dataset:**

https://developer.ibm.com/exchanges/data/all/airline/

**Expected Outcomes:**

- A machine learning model capable of accurately predicting flight delays based on flight parameters.
- A web application for delay predictions.
- A comprehensive report detailing the methodology, data processing steps, model performance, and evaluation metrics.
- Recommendations for optimizing airline scheduling and resource management based on predictive insights.

**Repository Notebook Structure**

```
COMP3610-Project/
├── eda.ipynb
├── cleaning.ipynb
├── delay_feature_analysis.ipynb
├── modeling.ipynb
```

1) eda.ipynb
Purpose: Initial exploration of the dataset to understand the features, distributions, missing values, and identify correlations.

2) cleaning.ipynb
Purpose: Cleaning the data by handling missing values, fixing inconsistencies, and preparing the dataset for modeling.

3) delay_feature_analysis.ipynb
Purpose: Focus on analyzing and refining the features specifically related to flight delays — identifying which factors most significantly impact delays.

4) modeling.ipynb
Purpose: Implementing the predictive models, training, and evaluating performance (accuracy, F1, etc.) based on the selected features.

**How to Run the Project**

To run the project and predict flight delays using the web application:

1) Install Dependencies
If you haven't already, make sure to install the required dependencies listed in requirements.txt:

```bash
pip install -r requirements.txt
```

2) Run the Flask Application
In your terminal, navigate to the project directory and run the following command:

```bash
flask run
```

This will start the Flask development server.

3) Access the Web Application
Once the Flask server is running, open your web browser and go to:

```bash
http://127.0.0.1:5000/
```
Here, you'll be able to interact with the web application, input flight details, and see the prediction for flight delays.
