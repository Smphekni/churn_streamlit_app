# Standalone Code: churnnew_dash.py (Integrated Solution with Heatmap and Security Placeholders)

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import shap
import joblib

# Security Placeholder: Encryption, Access Control, and MFA policies would be enforced at deployment level (e.g., GitHub private repos, TLS/SSL in Streamlit Cloud).

# Load or simulate dataset
@st.cache_data
def load_data():
    num_samples = 1000
    customer_ids = np.arange(1, num_samples + 1)
    monthly_charges = np.round(np.random.uniform(20, 120, num_samples), 2)
    tenure = np.random.randint(1, 72, num_samples)
    total_charges = np.round(monthly_charges * tenure + np.random.uniform(0, 50, num_samples), 2)
    contract_types = np.random.choice(['Month-to-Month', 'One Year', 'Two Year'], num_samples)
    payment_methods = np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], num_samples)
    paperless_billing = np.random.choice([True, False], num_samples).astype(int)
    has_dependents = np.random.choice([True, False], num_samples).astype(int)
    churn = np.random.choice([0, 1], num_samples, p=[0.8, 0.2])

    data = pd.DataFrame({
        'customer_id': customer_ids,
        'monthly_charges': monthly_charges,
        'tenure': tenure,
        'total_charges': total_charges,
        'contract_type': contract_types,
        'payment_method': payment_methods,
        'paperless_billing': paperless_billing,
        'has_dependents': has_dependents,
        'churn': churn
    })

    # --- Additional Data Cleaning Steps ---
    # Drop duplicates
    data.drop_duplicates(inplace=True)

    # IQR Outlier Removal for 'monthly_charges'
    Q1 = data['monthly_charges'].quantile(0.25)
    Q3 = data['monthly_charges'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data = data[(data['monthly_charges'] >= lower_bound) & (data['monthly_charges'] <= upper_bound)]

    # Logical Consistency Check: Adjust total_charges if misaligned
    expected_total = data['monthly_charges'] * data['tenure']
    data['total_charges'] = np.where(
        abs(data['total_charges'] - expected_total) > 50,
        expected_total,
        data['total_charges']
    )

    return data

data = load_data()

# Preprocess Data
def preprocess_data(data):
    numerical_features = ['monthly_charges', 'tenure', 'total_charges']
    categorical_features = ['contract_type', 'payment_method']

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, numerical_features),
        ('cat', cat_pipeline, categorical_features)
    ], remainder='passthrough')

    X = data.drop(['customer_id', 'churn'], axis=1)
    y = data['churn']

    X_processed = preprocessor.fit_transform(X)
    return X_processed, np.array(y), preprocessor

X, y, preprocessor = preprocess_data(data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train deep learning model
def build_model(input_shape):
    model = Sequential()
    model.add(Input(shape=(input_shape,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = build_model(X_train.shape[1])
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

try:
    model.fit(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[early_stop], verbose=0)
except Exception as e:
    st.error(f"Model training failed: {e}")

joblib.dump(preprocessor, 'preprocessor.pkl')
model.save('churn_model.h5')

st.title("Customer Churn Prediction Dashboard")

# Authentication Placeholder
# In production, enable MFA on GitHub and Streamlit Cloud and use Role-Based Access Control (RBAC).

tabs = st.tabs(["Predict Churn", "Data Analysis", "Reports"])

with tabs[0]:
    st.header("Predict Customer Churn")

    tenure = st.number_input("Tenure (months)", min_value=1, max_value=72, value=24)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=20.0, max_value=120.0, value=50.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=1200.0)
    contract_type = st.selectbox("Contract Type", ['Month-to-Month', 'One Year', 'Two Year'])
    payment_method = st.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'])
    paperless_billing = st.checkbox("Paperless Billing")
    has_dependents = st.checkbox("Has Dependents")

    if st.button("Predict Churn Probability"):
        input_df = pd.DataFrame([{
            'monthly_charges': monthly_charges,
            'tenure': tenure,
            'total_charges': total_charges,
            'contract_type': contract_type,
            'payment_method': payment_method,
            'paperless_billing': int(paperless_billing),
            'has_dependents': int(has_dependents)
        }])

        preprocessor = joblib.load('preprocessor.pkl')
        from tensorflow.keras.models import load_model
        model = load_model('churn_model.h5')
        model_input = preprocessor.transform(input_df)
        prediction = model.predict(model_input)

        st.success(f"Predicted Churn Probability: {prediction[0][0]:.2%}")

with tabs[1]:
    st.header("Interactive Data Analysis")
    st.subheader("Dataset Overview")
    st.dataframe(data.head())

    st.subheader("Churn Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x=data['churn'], ax=ax)
    st.pyplot(fig)

    column_to_plot = st.selectbox("Select column for histogram", data.columns)
    fig, ax = plt.subplots()
    sns.histplot(data[column_to_plot], kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    numerical_data = data[['monthly_charges', 'tenure', 'total_charges']]
    correlation = numerical_data.corr()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

with tabs[2]:
    st.header("Web-Based Report Generation")
    if st.button("Generate Report"):
        st.subheader("Model Performance Metrics")
        y_pred = (model.predict(X_test) > 0.5).astype(int)
        st.write("Accuracy:", accuracy_score(y_test, y_pred))
        st.write("Precision:", precision_score(y_test, y_pred))
        st.write("Recall:", recall_score(y_test, y_pred))
        st.write("F1 Score:", f1_score(y_test, y_pred))
        st.write("ROC-AUC:", roc_auc_score(y_test, y_pred))

        st.subheader("Feature Importance using SHAP")
        try:
            background = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]
            explainer = shap.DeepExplainer(model, background)
            shap_values = explainer.shap_values(X_test)
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values[0], X_test, show=False)
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"SHAP failed to generate: {e}")

        st.subheader("Dataset Statistics")
        st.dataframe(data.describe())

# --- Security Summary ---
# This application assumes the use of GitHub private repositories, SSL-enabled Streamlit Cloud deployment,
# Role-Based Access Control (RBAC), Multi-Factor Authentication (MFA), and AES-256 encryption for production environments.
# Ensure compliance with GDPR and CCPA for future use with real-world datasets.


