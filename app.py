import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------- Page Config --------------------
st.set_page_config(page_title="Naive Bayes Classifier", layout="wide")

st.title("🧠 Naive Bayes Classifier Dashboard")
st.write("Upload a dataset and train a Gaussian Naive Bayes model interactively.")

# -------------------- File Upload --------------------
uploaded_file = st.file_uploader("📂 Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        st.subheader("📊 Dataset Preview")
        st.dataframe(df.head())
        st.write(f"Dataset Shape: {df.shape}")

        # -------------------- Column Selection --------------------
        target_column = st.selectbox("🎯 Select Target Column", df.columns)

        feature_columns = st.multiselect(
            "📝 Select Feature Columns",
            [col for col in df.columns if col != target_column],
            default=[col for col in df.columns if col != target_column]
        )

        if feature_columns and target_column:

            X = df[feature_columns].copy()
            y = df[target_column].copy()

            # -------------------- Encoding --------------------
            st.subheader("🔄 Encoding Categorical Variables")

            le_dict = {}
            for col in X.columns:
                if X[col].dtype == 'object':
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    le_dict[col] = le

            if y.dtype == 'object':
                le_target = LabelEncoder()
                y = le_target.fit_transform(y.astype(str))

            st.success("Encoding completed.")

            # -------------------- Split Controls --------------------
            col1, col2 = st.columns(2)

            with col1:
                test_size = st.slider("Test Size (%)", 10, 50, 20) / 100

            with col2:
                random_state = st.number_input("Random State", value=42, step=1)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=int(random_state)
            )

            st.write("### 📦 Data Split Summary")
            st.write(f"Training Samples: {len(X_train)}")
            st.write(f"Testing Samples: {len(X_test)}")

            # -------------------- Train Button --------------------
            if st.button("🚀 Train Naive Bayes Model"):

                model = GaussianNB()
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred)

                st.subheader("📈 Model Performance")
                st.metric("Accuracy", f"{accuracy:.4f} ({accuracy*100:.2f}%)")

                # -------------------- Confusion Matrix --------------------
                st.subheader("📊 Confusion Matrix")

                cm = confusion_matrix(y_test, y_pred)

                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                ax.set_title("Confusion Matrix")
                st.pyplot(fig)

                # Table version
                cm_df = pd.DataFrame(
                    cm,
                    columns=[f"Predicted {i}" for i in range(cm.shape[1])],
                    index=[f"Actual {i}" for i in range(cm.shape[0])]
                )

                st.dataframe(cm_df)

                # -------------------- Classification Report --------------------
                st.subheader("📑 Classification Report")
                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)

                # -------------------- Extra Stats --------------------
                st.subheader("📌 Additional Metrics")
                st.write(f"Correct Predictions: {np.diag(cm).sum()}")
                st.write(f"Total Predictions: {cm.sum()}")
                st.write(f"Misclassifications: {cm.sum() - np.diag(cm).sum()}")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

else:
    st.info("Please upload a CSV file to begin.")
