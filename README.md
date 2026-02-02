# Heart Disease Prediction using Machine Learning

## Building a Life-Saving Predictor: Machine Learning Meets Healthcare

A comprehensive machine learning classification project that predicts the severity of heart disease in patients based on medical attributes. This end-to-end workflow demonstrates the power of AI in healthcare, moving from raw patient data to an accurate and interpretable predictive model.

## 🎯 Project Objective

To build a machine learning model that can accurately predict whether a patient has heart disease based on a set of medical attributes. This project serves as a comprehensive introduction to classification, one of the most common types of machine learning problems, with real-world healthcare applications.

## 🔍 Key Findings

1. **Model Performance**
- **Logistic Regression** achieved the highest accuracy at **59.24%**, proving that simpler models can outperform complex ones
- All three models (Logistic Regression, Random Forest, SVM) performed similarly (~57-59%), suggesting the feature set captures most available signal
- Models excel at identifying healthy patients (class 0) with **87-89% recall**

 2. **The Challenge of Multiclass Classification**
- This is a **5-class problem** (severity levels 0-4), making it inherently more difficult than binary classification
- Class imbalance heavily impacts performance - most patients fall into class 0 (no disease)
- Distinguishing between severity levels 1, 2, 3, and 4 proved challenging for all models

 3. **Feature Importance**
- The Random Forest model identified the **most predictive medical factors** for heart disease
- Top features align with medical knowledge, validating the model's learning process
- Key indicators include: number of major vessels (ca), maximum heart rate (thalach), thalassemia type (thal), and chest pain type (cp)

 4. **Medical Insights**
- Models are picking up on **clinically relevant patterns**, not just statistical noise
- High performance on class 0 makes the model useful for screening purposes
- The confusion between severity levels suggests these categories may have overlapping clinical presentations

# 📊 Analysis Components

# Core Concepts Covered:

1. **Classification Fundamentals**
   - Understanding the goal of predicting a discrete category
   - Binary vs. multiclass classification

2. **Exploratory Data Analysis (EDA) for Classification**
   - Analyzing features to find patterns that distinguish between classes
   - Visualizing relationships between medical indicators and disease severity

3. **Data Preprocessing**
   - Preparing data for classification using encoding and feature scaling
   - Handling missing values through imputation
   - Creating robust preprocessing pipelines

4. **Model Building**
   - Training and comparing three algorithms: Logistic Regression, Random Forest, and SVM
   - Understanding different approaches to classification problems

5. **Model Evaluation**
   - Mastering key classification metrics: Accuracy, Precision, Recall, F1-Score
   - Interpreting the Confusion Matrix to understand model behavior
   - Recognizing the importance of multiple metrics in medical contexts

6. **Feature Importance**
   - Identifying the most influential medical factors for predicting heart disease
   - Validating model insights against medical knowledge

# 📁 Dataset

**Source:** UCI Heart Disease Dataset (via Kaggle)

## Dataset Statistics:
- **920 patient records** with comprehensive medical information
- **13 clinical features** covering demographics, symptoms, and test results
- **Target variable:** Heart disease severity (0 = no disease, 1-4 = increasing severity)

## Features Breakdown:

**Numerical Features:**
- `age` - Patient age in years
- `trestbps` - Resting blood pressure (mm Hg)
- `chol` - Serum cholesterol (mg/dl)
- `thalach` - Maximum heart rate achieved
- `oldpeak` - ST depression induced by exercise
- `ca` - Number of major vessels colored by fluoroscopy

**Categorical Features:**
- `sex` - Patient gender
- `cp` - Chest pain type (4 categories)
- `fbs` - Fasting blood sugar > 120 mg/dl
- `restecg` - Resting electrocardiographic results
- `exang` - Exercise induced angina
- `slope` - Slope of peak exercise ST segment
- `thal` - Thalassemia (blood disorder)

## 🛠️ Tools & Libraries Used

- **Python 3.12** (Google Colab environment)
- **Pandas & NumPy** - Data manipulation and numerical operations
- **Matplotlib & Seaborn** - Statistical data visualization
- **Scikit-learn** - Machine learning algorithms and evaluation
- **KaggleHub** - Seamless dataset integration

## 🚀 Project Workflow

### Step 1: Exploratory Data Analysis (EDA) 🔎
Understanding the data before building models:
- **Visualized the distribution** of heart disease severity across the dataset
- **Plotted relationships** between key features (age, max heart rate, cholesterol, chest pain) and disease severity
- **Generated a correlation heatmap** to identify relationships between numerical features
- **Identified data quality issues** and missing values

## Step 2: Data Preprocessing 🧹
Preparing clean, model-ready data:
- **Handled missing values** through systematic imputation strategies
- **Split the dataset** into training (80%) and testing (20%) sets
- **Applied StandardScaler** to normalize numerical features for optimal model performance
- **Preserved all 13 features** to capture maximum predictive signal

## Step 3: Model Training & Comparison ⚙️
Trained and evaluated three classification algorithms:

**🎯 Logistic Regression**
- **Accuracy:** 59.24%
- **Precision:** 0.55 (weighted avg)
- **Recall:** 0.59 (weighted avg)
- **F1-Score:** 0.56 (weighted avg)
- *Best overall performer despite being the simplest model*

**🌲 Random Forest Classifier**
- **Accuracy:** 57%
- **Precision:** 0.52 (weighted avg)
- **Recall:** 0.57 (weighted avg)
- *Excellent at identifying class 0 (88% recall), but struggled with minority classes*

**🔷 Support Vector Machine (SVM)**
- **Accuracy:** 58.15%
- **Precision:** 0.52 (weighted avg)
- **Recall:** 0.58 (weighted avg)
- **F1-Score:** 0.55 (weighted avg)
- *Balanced performance across metrics*

## Step 4: Model Evaluation 📈
Applied comprehensive evaluation metrics:
- **Confusion Matrix:** Analyzed where models make correct/incorrect predictions across all 5 classes
- **Classification Report:** Examined precision, recall, and F1-scores for each severity level
- **Accuracy Score:** Measured overall prediction correctness
- **Insight:** Models consistently excelled at identifying healthy patients (class 0) but found it challenging to distinguish between the four disease severity levels

## Step 5: Feature Importance Analysis 🔬
Extracted insights from the Random Forest model:
- **Identified top 10 most influential features** for heart disease prediction
- **Visualized feature importances** to understand which medical measurements drive predictions
- **Validated findings** against medical knowledge to ensure clinical relevance
- **Key predictors:** ca (major vessels), thalach (max heart rate), thal (thalassemia), cp (chest pain type)


## 🎯 Skills Demonstrated

- ✅ **Exploratory Data Analysis (EDA)** - Understanding data before modeling
- ✅ **Data Preprocessing** - Cleaning, scaling, and preparing data
- ✅ **Classification Algorithms** - Logistic Regression, Random Forest, SVM
- ✅ **Model Evaluation** - Confusion matrices, classification reports, multiple metrics
- ✅ **Feature Importance Analysis** - Interpreting model decisions
- ✅ **Data Visualization** - Communicating insights through plots
- ✅ **Medical Domain Knowledge** - Understanding clinical context
- ✅ **Critical Thinking** - Recognizing model limitations and proposing improvements
- 

## 👤 Author
Vaishnavi Dua
