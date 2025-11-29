#Spam SMS Detection
This repository, spam-sms-detection, contains a machine learning project to detect spam SMS messages using text classification techniques.[1]

Project Overview
Spam messages are unsolicited SMS texts that can be annoying or malicious, making automatic detection important for user safety and experience.[1] This project builds and evaluates a classifier that distinguishes between legitimate (ham) and spam SMS messages based on their content.[1]

Dataset
Source: SMS spam dataset (commonly known as the “spam.csv” dataset) loaded from a CSV file.[1]
Initial columns: v1 (label: ham/spam), v2 (raw message text) along with some unused metadata columns.[1]
Processed columns:
Label – mapped to 0 for ham and 1 for spam.[1]
Message – SMS text content.[1]
Note: The notebook currently reads the file from a local path; update this path or place spam.csv in the project directory when running the code.[1]

Approach
Data loading and selection
Load the CSV into a pandas DataFrame and keep only the relevant columns v1 and v2.[1]
Rename columns to Label and Message for clarity.[1]
Preprocessing
Convert text labels to numerical form by mapping ham → 0 and spam → 1.[1]
(Optional) Additional cleaning such as lowercasing, removing special characters, or stopwords can be added if required.[1]
Feature extraction
Use TfidfVectorizer to convert SMS text into numerical feature vectors, with English stop words removed.[1]
Define x as the TF-IDF features and y as the numerical labels.[1]
Train–test split
Split the dataset into training and test sets using train_test_split with a test_size of 0.2 and a fixed random_state for reproducibility.[1]
Modeling
Train a Multinomial Naive Bayes classifier, which is well-suited for text classification problems.[1]
Evaluation
Evaluate model performance on the test set using:
Accuracy score.[1]
Confusion matrix.[1]
Classification report (precision, recall, f1-score) for both spam and ham classes.[1]
Repository Structure
Naive-Byes.ipynb – Main Jupyter Notebook containing data preprocessing, feature extraction, model training, and evaluation.[1]
data/spam.csv – SMS spam dataset file (recommended relative path; add this folder and file when publishing).[1]
Requirements
Key Python libraries used:

pandas.[1]
scikit-learn (model_selection, feature_extraction.text, naive_bayes, metrics).[1]
Install them with:

pip install -r requirements.txt
How to Run
Clone the repository:
git clone https://github.com//spam-sms-detection.git
cd spam-sms-detection
Set up environment and install dependencies:
python -m venv venv
source venv/bin/activate (Linux/Mac)
venv\Scripts\activate (Windows)
pip install -r requirements.txt
Place the dataset:
Create a data/ folder in the project root and add spam.csv.[1]
Update the notebook path to: df = pd.read_csv("data/spam.csv", encoding="ISO-8859-1").[1]
Run the notebook:
jupyter notebook
Open Naive-Byes.ipynb and run all cells in order.
Future Improvements
Add more preprocessing steps such as stemming/lemmatization and advanced text cleaning.[1]
Experiment with alternative models (e.g., Linear SVM, Logistic Regression) and compare performance.[1]
Perform hyperparameter tuning and add evaluation metrics such as ROC-AUC and precision–recall curves.[1]
License
This project respects all relevant intellectual property and copyright for datasets and libraries used.
Specify your chosen open-source license here (e.g., MIT License) depending on your needs.
