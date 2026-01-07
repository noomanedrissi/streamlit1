# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 20:53:48 2026

@author: noomane.drissi
"""

# my_app.py
import streamlit as st
import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="ML Suite 2026", layout="wide")
st.title("🤖 Data Suite: Profile, Clean & Predict")

uploaded_file = st.file_uploader("Upload CSV (Max 1GB)", type=["csv"])

if uploaded_file:
    # 1. Read subset
    df = pd.read_csv(uploaded_file, nrows=1000)
    st.success(f"Loaded {len(df):,} rows.")

    # 2. Cleaning & Encoding Pipeline
    if st.button("🚀 Step 1: Clean & Encode Data"):
        df.drop_duplicates(inplace=True)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in df.columns:
            if col in numeric_cols:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")
        
        # Remove outliers
        df = df[(np.abs(stats.zscore(df[numeric_cols])) < 3).all(axis=1)]
        
        # Encode (Saving for session state to use in ML)
        df_encoded = pd.get_dummies(df, drop_first=True)
        st.session_state['data'] = df_encoded
        st.success("Data Cleaned and Encoded!")
        st.dataframe(df_encoded.head())

    # 3. Profiling
    if st.checkbox("Generate Deep Insights"):
#        profile=ProfileReport(df,title='Pandas Profiling Report',explorative=True)#,dataset={'description':'this data contains Expresso Churn Prediction Challenge',
#                                                                                             'creator':'N.DRISSI'})#,samples=None)#sample none for safe report to not show data

        st.components.v1.html(profile.to_html(), height=600, scrolling=True)

    # 4. Machine Learning Section
    if 'data' in st.session_state:
        st.divider()
        st.header("🧠 Step 2: Train Classifier")
        
        data = st.session_state['data']
        target_col = st.selectbox("Select Target Column (Y)", data.columns)
        
        if st.button("🎯 Train Model"):
            X = data.drop(columns=[target_col])
            y = data[target_col]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            
            # Metrics
            acc = accuracy_score(y_test, model.predict(X_test))
            st.metric("Model Accuracy", f"{acc:.2%}")
            
            # Store model and features for prediction
            st.session_state['model'] = model
            st.session_state['features'] = X.columns.tolist()

    # 5. Prediction Form
    if 'model' in st.session_state:
        st.divider()
        st.header("🔮 Step 3: Make Predictions")
        
        with st.form("prediction_form"):
            st.write("Enter feature values:")
            user_inputs = {}
            # Dynamically create input fields based on features
            cols = st.columns(3)
            for i, feat in enumerate(st.session_state['features']):
                with cols[i % 3]:
                    user_inputs[feat] = st.number_input(f"{feat}", value=0.0)
            
            submit = st.form_submit_button("Validate & Predict")
            
            if submit:
                input_df = pd.DataFrame([user_inputs])
                prediction = st.session_state['model'].predict(input_df)
                st.success(f"**Prediction Result:** {prediction[0]}")

