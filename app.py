import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Naive Bayes Classifier - Credit Dataset")
st.write("Train and evaluate a Naive Bayes classifier on Credit.csv")

# Load dataset directly
try:
    df = pd.read_csv("Credit.csv")
    st.write("### Dataset Preview")
    st.write(df.head())
    st.write(f"Dataset shape: {df.shape}")
    
    # Select target column
    target_column = st.selectbox("Select target column", df.columns)
    
    # Select feature columns
    feature_columns = st.multiselect(
        "Select feature columns", 
        [col for col in df.columns if col != target_column],
        default=[col for col in df.columns if col != target_column]
    )
    
    if feature_columns and target_column:
        # Prepare data
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        
        # Handle non-numeric data
        le_dict = {}
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                le_dict[col] = le
        
        # Encode target if categorical
        if y.dtype == 'object':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y.astype(str))
        
        # Train-test split
        test_size = st.slider("Test size (%)", 10, 50, 20) / 100
        random_state = st.number_input("Random state", value=42, step=1)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=int(random_state)
        )
        
        st.write(f"### Data Split")
        st.write(f"Training samples: {len(X_train)}")
        st.write(f"Testing samples: {len(X_test)}")
        
        # Train model
        if st.button("Train Naive Bayes Classifier"):
            # Create and train model
            model = GaussianNB()
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            
            st.write("### Results")
            st.metric("Accuracy", f"{accuracy:.4f} ({accuracy*100:.2f}%)")
            
            # Confusion Matrix
            st.write("### Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix')
            st.pyplot(fig)
            
            # Display confusion matrix as table
            st.write("#### Confusion Matrix (Table)")
            cm_df = pd.DataFrame(cm, 
                                columns=[f"Predicted {i}" for i in range(cm.shape[1])],
                                index=[f"Actual {i}" for i in range(cm.shape[0])])
            st.dataframe(cm_df)
            
            # Additional metrics
            st.write("### Additional Information")
            st.write(f"True Positives: {np.diag(cm).sum()}")
            st.write(f"Total Predictions: {cm.sum()}")
            st.write(f"Misclassifications: {cm.sum() - np.diag(cm).sum()}")

except FileNotFoundError:
    st.error("Credit.csv file not found. Please ensure the file is in the same directory as this script.")
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
