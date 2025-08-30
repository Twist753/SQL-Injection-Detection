# SQL Injection Detection using Machine Learning

This project demonstrates the development of a machine learning model to detect SQL injection (SQLi) attacks. Using a dataset of over 30,000 SQL queries, this model is trained to classify a given query as either **malicious (1)** or **safe (0)** with high accuracy.

The entire implementation, from data cleaning to model evaluation, is detailed in the Jupyter Notebook: `SQL_injection_detection.ipynb`.


## Project Results

The final model, a **Logistic Regression** classifier, achieved excellent performance on the unseen test data:

* **Accuracy**: **97.86%**
* **Precision (Malicious)**: **1.00** (When it predicts an attack, it's always correct)
* **Recall (Malicious)**: **0.94** (It successfully identifies 94% of all actual attacks)

<img width="813" height="869" alt="image" src="https://github.com/user-attachments/assets/47d5e80b-3758-45b9-be17-bbc1c52c8d0e" />

These results indicate a highly effective and reliable model for identifying SQLi threats, with a very low rate of false alarms.

## Project Structure

```
.
├── dataset/
│   └── SQLiV3.csv          # The dataset file
├── SQL_injection_detection.ipynb  # Main notebook with all the code
└── README.md               # You are here
```

##  Methodology

The project follows a standard machine learning workflow:

1.  **Data Loading and Initial Analysis**: The dataset (`SQLiV3.csv`) is loaded into a pandas DataFrame. The initial inspection revealed messy labels and extra columns.

2.  **Data Cleaning and Preprocessing**: This was a critical step. The `Label` column contained noise, including non-numeric text and `NaN` values. The data was cleaned by:
    * Removing extraneous columns.
    * Coercing the `Label` column to a numeric type, which turned all text-based noise into `NaN`.
    * Dropping all rows with `NaN` labels to ensure a clean dataset for training.
    * The SQL query sentences were standardized by converting them to lowercase.

3.  **Exploratory Data Analysis (EDA)**: A count plot was generated to visualize the distribution of safe vs. malicious queries, confirming a reasonably balanced dataset after cleaning.
    
4.  **Feature Engineering (Vectorization)**: To make the text data understandable for a machine learning model, the SQL queries were converted into numerical vectors using the **TF-IDF (Term Frequency-Inverse Document Frequency)** technique.
    * `TfidfVectorizer` was used with a `char_wb` analyzer and an `ngram_range` of (3, 5). This approach is highly effective for SQLi detection as it captures character-level patterns (like `' or '`, `1=1`, `--`) that are indicative of an attack.

5.  **Model Training**:
    * The dataset was split into training (80%) and testing (20%) sets.
    * A **Logistic Regression** model was chosen as a strong and efficient baseline for this text classification task. The model was trained on the training data.

6.  **Model Evaluation**: The model's performance was evaluated on the unseen test set using key metrics:
    * **Accuracy Score**: To measure overall correctness.
    * **Classification Report**: To analyze precision, recall, and F1-score for each class.
    * **Confusion Matrix**: To visualize the model's predictions and identify where it made errors (e.g., False Positives vs. False Negatives).

## How to Use

1.  **Clone the repository**:
    ```bash
    git clone [https://github.com/](https://github.com/)<your-username>/<your-repo-name>.git
    cd <your-repo-name>
    ```

2.  **Install dependencies**:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```

3.  **Run the Jupyter Notebook**:
    Open and run the `SQL_injection_detection.ipynb` notebook in a Jupyter environment to see the full process and reproduce the results.

## Future Improvements

* **Try Different Models**: Experiment with more complex models like Random Forest, Gradient Boosting (XGBoost, LightGBM), or even deep learning models (e.g., LSTMs) to potentially improve the recall score.
* **Hyperparameter Tuning**: Use techniques like `GridSearchCV` or `RandomizedSearchCV` to find the optimal parameters for the model and vectorizer.
* **Deploy the Model**: Wrap the trained model in a simple API using a framework like Flask or FastAPI to allow for real-time SQL query checks.
