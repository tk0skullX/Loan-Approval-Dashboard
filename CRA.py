#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


loan_df = pd.read_csv('loan.csv')

# Display the first few rows of the dataset to verify
loan_df.head()


# In[3]:


# Check for missing values
loan_df.isnull().sum()


# In[10]:


# Check the data types of the columns
loan_df.info()


# In[4]:


# Summary statistics
loan_df.describe()


# In[5]:


# Calculate the threshold (10% of total entries)
threshold = len(loan_df) * 0.10

# Drop columns where the number of null values exceeds the threshold
loan_df_cleaned = loan_df.dropna(thresh=threshold, axis=1)

# Display the shape of the cleaned dataset to verify
loan_df_cleaned.shape


# In[6]:


# Get a list of numeric columns
numeric_cols = loan_df_cleaned.select_dtypes(include=['float64', 'int64']).columns

# Display the numeric columns
numeric_cols


# In[7]:


# Replace missing values in numeric columns with the mean (you can change to median if needed)
for col in numeric_cols:
    if loan_df_cleaned[col].isnull().sum() > 0:  # Only process columns with missing values
        # Check skewness of the column to decide between mean and median
        if loan_df_cleaned[col].skew() > 1:  # Highly skewed data, use median
            loan_df_cleaned[col].fillna(loan_df_cleaned[col].median(), inplace=True)
        else:  # Not highly skewed, use mean
            loan_df_cleaned[col].fillna(loan_df_cleaned[col].mean(), inplace=True)

# Verify there are no missing values in the numeric columns
loan_df_cleaned[numeric_cols].isnull().sum()


# In[8]:


# Check if there are any missing values left in the dataset
loan_df_cleaned.isnull().sum()


# In[9]:


# List of unnecessary columns to drop (excluding the ones you want to retain)
columns_to_drop = ['emp_title', 'url', 'desc', 'title', 'policy_code', 'member_id', 'id']

# Drop the unnecessary columns while keeping next_pymnt_d, mths_since_last_delinq, and mths_since_last_record
loan_df_cleaned = loan_df_cleaned.drop(columns=columns_to_drop, axis=1)

# Verify the result
loan_df_cleaned.info()


# In[10]:


# Identify object columns
object_columns = loan_df_cleaned.select_dtypes(include=['object']).columns

# Get the unique values for each object column
for col in object_columns:
    print(f"Unique values in '{col}':\n{loan_df_cleaned[col].unique()}\n")


# In[11]:


from sklearn.preprocessing import LabelEncoder

# Label Encoding for columns with few unique values
label_cols = ['term', 'grade', 'sub_grade', 'emp_length', 'home_ownership', 'verification_status',
              'loan_status', 'pymnt_plan', 'application_type', 'initial_list_status']

label_encoder = LabelEncoder()

# Apply label encoding to these columns
for col in label_cols:
    loan_df_cleaned[col] = label_encoder.fit_transform(loan_df_cleaned[col])


# In[12]:


# One-Hot Encoding for 'purpose', 'addr_state', 'zip_code'
loan_df_encoded = pd.get_dummies(loan_df_cleaned, columns=['purpose', 'addr_state', 'zip_code'], drop_first=True)


# In[13]:


# Convert date columns to datetime format
loan_df_cleaned['earliest_cr_line'] = pd.to_datetime(loan_df_cleaned['earliest_cr_line'], format='%b-%Y', errors='coerce')
loan_df_cleaned['next_pymnt_d'] = pd.to_datetime(loan_df_cleaned['next_pymnt_d'], format='%b-%Y', errors='coerce')
loan_df_cleaned['last_credit_pull_d'] = pd.to_datetime(loan_df_cleaned['last_credit_pull_d'], format='%b-%Y', errors='coerce')
loan_df_cleaned['issue_d'] = pd.to_datetime(loan_df_cleaned['issue_d'], format='%b-%Y', errors='coerce')

# Create new features from the date columns
loan_df_cleaned['years_since_earliest_cr_line'] = pd.to_datetime('today').year - loan_df_cleaned['earliest_cr_line'].dt.year
loan_df_cleaned['days_until_next_pymnt'] = (loan_df_cleaned['next_pymnt_d'] - pd.to_datetime('today')).dt.days
loan_df_cleaned['days_since_last_credit_pull'] = (pd.to_datetime('today') - loan_df_cleaned['last_credit_pull_d']).dt.days
loan_df_cleaned['years_since_issue'] = pd.to_datetime('today').year - loan_df_cleaned['issue_d'].dt.year

# Drop original date columns as we've extracted useful information
loan_df_cleaned = loan_df_cleaned.drop(columns=['earliest_cr_line', 'next_pymnt_d', 'last_credit_pull_d', 'issue_d'])


# In[14]:


# Check the transformed DataFrame
loan_df_encoded.info()

# Preview the first few rows
loan_df_encoded.head()


# In[15]:


# Drop the encoded zip_code columns
zip_code_columns = [col for col in loan_df_encoded.columns if 'zip_code' in col]

# Drop these columns from the dataset
loan_df_encoded = loan_df_encoded.drop(columns=zip_code_columns)

# Check the shape of the data after dropping the zip_code columns
loan_df_encoded.shape


# In[16]:


get_ipython().system('pip install xgboost')


# In[22]:


# Convert the date columns to datetime format
loan_df_encoded['earliest_cr_line'] = pd.to_datetime(loan_df_encoded['earliest_cr_line'], format='%b-%Y', errors='coerce')
loan_df_encoded['next_pymnt_d'] = pd.to_datetime(loan_df_encoded['next_pymnt_d'], format='%b-%Y', errors='coerce')
loan_df_encoded['last_credit_pull_d'] = pd.to_datetime(loan_df_encoded['last_credit_pull_d'], format='%b-%Y', errors='coerce')
loan_df_encoded['last_pymnt_d'] = pd.to_datetime(loan_df_encoded['last_pymnt_d'], format='%b-%Y', errors='coerce')
loan_df_encoded['issue_d'] = pd.to_datetime(loan_df_encoded['issue_d'], format='%b-%Y', errors='coerce')

# Create new features from the date columns (e.g., number of years since earliest credit line, etc.)
loan_df_encoded['years_since_earliest_cr_line'] = pd.to_datetime('today').year - loan_df_encoded['earliest_cr_line'].dt.year
loan_df_encoded['days_until_next_pymnt'] = (loan_df_encoded['next_pymnt_d'] - pd.to_datetime('today')).dt.days
loan_df_encoded['days_since_last_credit_pull'] = (pd.to_datetime('today') - loan_df_encoded['last_credit_pull_d']).dt.days
loan_df_encoded['years_since_issue'] = pd.to_datetime('today').year - loan_df_encoded['issue_d'].dt.year

# Drop the original date columns
loan_df_encoded = loan_df_encoded.drop(columns=['earliest_cr_line', 'next_pymnt_d', 'last_credit_pull_d', 'last_pymnt_d', 'issue_d'])


# In[23]:


from sklearn.model_selection import train_test_split

# Define the target variable (loan_status) and features (everything except loan_status)
X = loan_df_encoded.drop(columns=['loan_status'])  # Replace 'loan_status' with your actual target column
y = loan_df_encoded['loan_status']

# Split the data into training (80%) and testing sets (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the shape of the training and testing sets
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[25]:


print(loan_df_encoded.columns)


# In[26]:


from sklearn.model_selection import train_test_split

# Assume 'loan_status' is the target column and the rest are features
# Define X (features) and y (target)
X = loan_df_encoded.drop(columns=['loan_status'])  # drop target column from features
y = loan_df_encoded['loan_status']  # target column

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[27]:


from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# Initialize XGBoost
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

# Train the model
xgb_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = xgb_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))


# In[48]:


get_ipython().system('pip install imbalanced-learn')
get_ipython().system('pip install --upgrade scikit-learn')
get_ipython().system('pip install --upgrade imbalanced-learn')


# In[33]:


import pandas as pd
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Define X and y
X = loan_df_encoded.drop(columns=['loan_status'])
y = loan_df_encoded['loan_status']

# Check for missing values
missing_cols = X.columns[X.isnull().any()]
print(f"Columns with missing values: {missing_cols}")

# Handle missing values
# Impute missing values for numeric columns
numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
numeric_imputer = SimpleImputer(strategy='mean')
X[numeric_cols] = numeric_imputer.fit_transform(X[numeric_cols])

# Impute missing values for categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns

# Remove columns if they contain only NaN
X = X.dropna(axis=1, how='all')

# Ensure all categorical columns are strings
X[categorical_cols] = X[categorical_cols].astype(str)

# Impute categorical columns with the most frequent values
if len(categorical_cols) > 0:
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    X[categorical_cols] = pd.DataFrame(categorical_imputer.fit_transform(X[categorical_cols]), columns=categorical_cols)

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to balance the training set
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Initialize XGBoost with slightly tuned parameters for accuracy
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', 
                          learning_rate=0.1, max_depth=6, n_estimators=200)

# Train the model
xgb_model.fit(X_train_smote, y_train_smote)

# Make predictions on the test set
y_pred = xgb_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))


# In[34]:


import joblib

# Save the model
joblib.dump(xgb_model, 'xgb_model.pkl')

# To load the model later
xgb_model = joblib.load('xgb_model.pkl')


# In[36]:


pip install voila


# In[48]:


import joblib

# Assuming `xgb_model` is your trained XGBoost model
joblib.dump(xgb_model, 'xgb_model.pkl')


# In[49]:


import os
print(os.getcwd())  # This will print the current working directory path


# In[52]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, HTML
import joblib
from sklearn.preprocessing import LabelEncoder

# Add custom CSS styling for the layout
custom_style = """
<style>
    .container {
        width: 90% !important;
        margin: auto;
    }
    .section-title {
        color: #444;
        font-family: Arial, sans-serif;
        font-size: 24px;
        margin-top: 30px;
        margin-bottom: 15px;
        text-align: center;
    }
    .card {
        background-color: #f9f9f9;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 25px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }
    .button-style {
        background-color: #4CAF50 !important;
        color: white !important;
    }
</style>
"""
display(HTML(custom_style))

# Section 1: Static Visualizations in "Card" Style
def display_static_visuals():
    output_static1_2 = widgets.Output()
    output_static3_4 = widgets.Output()
    output_static5 = widgets.Output()
    output_static6 = widgets.Output()

    with output_static1_2:
        # Row 1: Side-by-side visualizations 1 and 2
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Feature 1: Loan Approval Rate
        approval_rate = loan_df['loan_status'].value_counts(normalize=True) * 100
        approval_rate.plot(kind='bar', color=['green', 'red'], ax=ax1)
        ax1.set_title('Loan Approval vs Denial Rate')
        ax1.set_ylabel('Percentage')

        # Feature 2: Distribution of Loan Amounts
        sns.histplot(loan_df['loan_amnt'], bins=30, kde=False, color='blue', ax=ax2)
        ax2.set_title('Distribution of Loan Amounts')
        ax2.set_xlabel('Loan Amount')
        ax2.set_ylabel('Frequency')

        plt.tight_layout()
        plt.show()

    with output_static3_4:
        # Row 2: Side-by-side visualizations 3 and 4
        fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 6))

        # Feature 3: Interest Rate Distribution
        sns.histplot(loan_df['int_rate'], bins=30, kde=True, color='purple', ax=ax3)
        ax3.set_title('Interest Rate Distribution')
        ax3.set_xlabel('Interest Rate (%)')
        ax3.set_ylabel('Frequency')

        # Feature 4: Home Ownership Analysis
        loan_df['home_ownership'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['#ff9999', '#66b3ff', '#99ff99'], ax=ax4)
        ax4.set_title('Home Ownership Distribution')

        plt.tight_layout()
        plt.show()

    with output_static5:
        # Row 3: Full-width visualization 5
        plt.figure(figsize=(10, 6))
        sns.barplot(x='loan_status', y='annual_inc', data=loan_df, errorbar=None)
        plt.title('Average Annual Income by Loan Status')
        plt.xlabel('Loan Status')
        plt.ylabel('Average Annual Income')
        plt.tight_layout()
        plt.show()

    with output_static6:
        # Row 4: Full-width visualization 6
        plt.figure(figsize=(10, 6))
        sns.histplot(data=loan_df, x='loan_amnt', hue='loan_status', multiple='stack', kde=True)
        plt.title('Loan Amount Distribution by Loan Status')
        plt.xlabel('Loan Amount')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()

    return widgets.VBox([output_static1_2, output_static3_4, output_static5, output_static6])

# Section 2: Interactive Widgets for Loan Prediction in "Card" Style
def display_interactive_widgets():
    # Load the saved model
    model = joblib.load('xgb_model.pkl')

    # Get the expected feature names from the model
    expected_features = model.feature_names_in_

    # Create input widgets
    loan_amnt = widgets.IntSlider(min=1000, max=40000, step=1000, description="Loan Amount:", style={'description_width': '120px'})
    int_rate = widgets.FloatSlider(min=5.0, max=30.0, step=0.1, description="Interest Rate:", style={'description_width': '120px'})
    emp_length = widgets.IntSlider(min=0, max=40, step=1, description="Employment Length:", style={'description_width': '120px'})
    annual_income = widgets.IntSlider(min=10000, max=500000, step=10000, description="Annual Income:", style={'description_width': '120px'})
    home_ownership = widgets.Dropdown(options=['RENT', 'OWN', 'MORTGAGE'], description="Home Ownership:", style={'description_width': '120px'})
    days_until_next_pymnt = widgets.IntSlider(min=0, max=365, step=1, description="Days Until Next Payment:", style={'description_width': '120px'})
    days_since_last_credit_pull = widgets.IntSlider(min=0, max=365, step=1, description="Days Since Last Credit Pull:", style={'description_width': '120px'})

    # Create a button to trigger the prediction
    button = widgets.Button(description="Predict Loan Approval", button_style='success', layout=widgets.Layout(width='250px'))
    
    # Define a function to run when the button is clicked
    def on_button_click(b):
        # Input data should have the same features as the model was trained on
        input_data = pd.DataFrame({
            'loan_amnt': [loan_amnt.value],
            'int_rate': [int_rate.value],
            'emp_length': [emp_length.value],
            'annual_inc': [annual_income.value],
            'home_ownership': [home_ownership.value],
            'days_until_next_pymnt': [days_until_next_pymnt.value],
            'days_since_last_credit_pull': [days_since_last_credit_pull.value]
        })

        # Encode the categorical column 'home_ownership'
        le = LabelEncoder()
        input_data['home_ownership'] = le.fit_transform(input_data['home_ownership'])

        # Add missing columns
        missing_cols = set(expected_features) - set(input_data.columns)
        for col in missing_cols:
            input_data[col] = 0  # Adjust default values based on your dataset

        # Reorder columns to match the order expected by the model
        input_data = input_data[expected_features]

        # Predict using the model
        prediction = model.predict(input_data)

        if prediction[0] == 1:
            output_prediction.clear_output()
            with output_prediction:
                display(HTML("<h4 style='color:green;'>Loan Approved ✅</h4>"))
        else:
            output_prediction.clear_output()
            with output_prediction:
                display(HTML("<h4 style='color:red;'>Loan Denied ❌</h4>"))

    # Link the button to the prediction function
    button.on_click(on_button_click)

    # Display the widgets and the button in a grid layout
    form_items = [
        widgets.VBox([loan_amnt, int_rate, emp_length]),
        widgets.VBox([annual_income, home_ownership, days_until_next_pymnt, days_since_last_credit_pull])
    ]
    form_box = widgets.HBox(form_items)

    # Output section for prediction result
    global output_prediction
    output_prediction = widgets.Output()

    return widgets.VBox([form_box, button, output_prediction], layout=widgets.Layout(align_items='center'))

# Section 3: Putting Everything Together in a Beautiful Layout
header = widgets.HTML(value="<h2 style='text-align:center;'>Loan Prediction Dashboard</h2>")

# Split the dashboard into two sections: static and interactive
static_visuals = display_static_visuals()
interactive_widgets = display_interactive_widgets()

# Arrange them in a vertical layout
dashboard_layout = widgets.VBox([
    header,
    widgets.HTML("<div class='section-title'>Static Visualizations</div>"),
    static_visuals,  # Correctly showing the static visualizations without HTML wrapping
    widgets.HTML("<div class='section-title'>Interactive Loan Prediction</div>"),
    interactive_widgets
])

# Display the full dashboard layout
dashboard_layout


# In[54]:


pip install streamlit


# In[55]:


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the saved model
model = joblib.load('xgb_model.pkl')

# Title of the App
st.title('Loan Approval Prediction Dashboard')

# Sidebar with Input Fields
st.sidebar.header("Loan Details")
loan_amnt = st.sidebar.slider('Loan Amount', min_value=1000, max_value=40000, step=1000, value=10000)
int_rate = st.sidebar.slider('Interest Rate (%)', min_value=5.0, max_value=30.0, step=0.1, value=10.5)
emp_length = st.sidebar.slider('Employment Length (Years)', min_value=0, max_value=40, step=1, value=5)
annual_income = st.sidebar.slider('Annual Income ($)', min_value=10000, max_value=500000, step=10000, value=50000)
home_ownership = st.sidebar.selectbox('Home Ownership', ['RENT', 'OWN', 'MORTGAGE'])
days_until_next_pymnt = st.sidebar.slider('Days Until Next Payment', min_value=0, max_value=365, step=1, value=30)
days_since_last_credit_pull = st.sidebar.slider('Days Since Last Credit Pull', min_value=0, max_value=365, step=1, value=90)

# Organize data for prediction
input_data = pd.DataFrame({
    'loan_amnt': [loan_amnt],
    'int_rate': [int_rate],
    'emp_length': [emp_length],
    'annual_inc': [annual_income],
    'home_ownership': [home_ownership],
    'days_until_next_pymnt': [days_until_next_pymnt],
    'days_since_last_credit_pull': [days_since_last_credit_pull]
})

# Encode the categorical column 'home_ownership'
le = LabelEncoder()
input_data['home_ownership'] = le.fit_transform(input_data['home_ownership'])

# Add missing columns (ensure that all features are present)
expected_features = model.feature_names_in_
missing_cols = set(expected_features) - set(input_data.columns)
for col in missing_cols:
    input_data[col] = 0

# Reorder columns to match the order expected by the model
input_data = input_data[expected_features]

# Predict using the model
prediction = model.predict(input_data)

# Display the Prediction
if prediction[0] == 1:
    st.success("Loan Approved")
else:
    st.error("Loan Denied")

# Static visualizations
st.header("Static Visualizations")

# Feature 1: Loan Approval Rate
st.subheader("Loan Approval vs Denial Rate")
approval_rate = loan_df['loan_status'].value_counts(normalize=True) * 100
fig1, ax1 = plt.subplots()
approval_rate.plot(kind='bar', ax=ax1, color=['green', 'red'])
st.pyplot(fig1)

# Feature 2: Distribution of Loan Amounts
st.subheader("Distribution of Loan Amounts")
fig2, ax2 = plt.subplots()
sns.histplot(loan_df['loan_amnt'], bins=30, kde=False, color='blue', ax=ax2)
st.pyplot(fig2)

# Feature 3: Interest Rate Distribution
st.subheader("Interest Rate Distribution")
fig3, ax3 = plt.subplots()
sns.histplot(loan_df['int_rate'], bins=30, kde=True, color='purple', ax=ax3)
st.pyplot(fig3)

# Add more visualizations as needed...


# In[ ]:




