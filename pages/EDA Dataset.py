import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def app():
    st.set_page_config(page_title="EDA - Heart Failure", layout="wide")
    st.title("📊 Exploratory Data Analysis - Heart Failure Dataset")

    # --- Load dataset safely ---
    file_path = os.path.join(os.path.dirname(__file__), "heart.csv")
    if not os.path.exists(file_path):
        st.error("❌ Dataset file not found. Please make sure 'heart.csv' is in the same folder as this file.")
        st.stop()

    data = pd.read_csv(file_path)

    # --- Target column fix ---
    if 'Heart Disease' in data.columns:
        data.rename(columns={'Heart Disease': 'HeartFailure'}, inplace=True)

    # If still named HeartDisease, convert it to HeartFailure for UI consistency
    if 'HeartDisease' in data.columns:
        data.rename(columns={'HeartDisease': 'HeartFailure'}, inplace=True)

    # --- Dataset Overview ---
    st.header("🧾 Dataset Overview")
    st.write("**Shape of dataset:**", data.shape)
    st.dataframe(data.head())

    # --- Summary Statistics ---
    st.subheader("📈 Summary Statistics")
    st.write(data.describe(include='all'))

    # --- Missing Values ---
    st.subheader("🚫 Missing Values")
    st.write(data.isnull().sum())

    # --- Feature Separation ---
    categorical_features = []
    numerical_features = []

    for col in data.columns:
        if data[col].dtype == 'object' or data[col].nunique() <= 6:
            categorical_features.append(col)
        else:
            numerical_features.append(col)

    st.write("**Categorical Features:**", categorical_features)
    st.write("**Numerical Features:**", numerical_features)

    # --- Target Distribution ---
    if "HeartFailure" in data.columns:
        st.header("🎯 Target Variable: Heart Failure Distribution")
        col1, col2 = st.columns([1, 1])

        # --- Pie Chart ---
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            data['HeartFailure'].value_counts().plot.pie(
                autopct='%1.1f%%', 
                colors=['#FDD20E', '#F93822'],
                ax=ax, 
                startangle=90, 
                explode=(0.1, 0), 
                textprops={'fontsize': 12}
            )
            ax.set_ylabel('')
            ax.set_title("Heart Failure Ratio", fontsize=16, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

        # --- Count Plot ---
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.countplot(
                x='HeartFailure',
                data=data,
                palette=['#FDD20E', '#F93822'],
                edgecolor='black',
                ax=ax
            )
            ax.set_xticklabels(['No Heart Failure', 'Heart Failure'], fontsize=12)
            ax.set_title("Heart Failure Count", fontsize=16, fontweight='bold')
            ax.set_ylabel("Count")
            ax.set_xlabel("")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

    # --- Numerical Feature Distributions ---
    st.header("📊 Numerical Feature Distributions")
    for i in range(0, len(numerical_features), 2):
        cols = st.columns([1, 1])
        for j in range(2):
            if i + j < len(numerical_features):
                col = numerical_features[i + j]
                with cols[j]:
                    fig, ax = plt.subplots(figsize=(12, 5))
                    sns.histplot(data[col], kde=True, color='#F93822', ax=ax)
                    ax.set_title(f'{col} Distribution', fontsize=14, fontweight='bold')
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)

    # --- Categorical Features vs Target ---
    st.header("📉 Categorical Features vs Heart Failure")
    for i in range(0, len(categorical_features), 2):
        cols = st.columns([1, 1])
        for j in range(2):
            if i + j < len(categorical_features):
                col = categorical_features[i + j]
                if col != "HeartFailure" and "HeartFailure" in data.columns:
                    with cols[j]:
                        fig, ax = plt.subplots(figsize=(12, 5))
                        sns.countplot(
                            x=col, 
                            hue='HeartFailure', 
                            data=data,
                            palette=['#FDD20E', '#F93822'], 
                            edgecolor='black', 
                            ax=ax
                        )
                        plt.xticks(rotation=30)
                        ax.set_title(f"{col} vs HeartFailure", fontsize=14, fontweight='bold')
                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=True)

    # --- Correlation Heatmap ---
    st.header("🔗 Correlation Heatmap (Numerical Features Only)")
    numeric_df = data.select_dtypes(include=['int64', 'float64'])
    if not numeric_df.empty:
        fig, ax = plt.subplots(figsize=(14, 4))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        ax.set_title("Feature Correlation Matrix", fontsize=16, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
    else:
        st.warning("No numeric columns available for correlation heatmap.")

    st.markdown("<br>", unsafe_allow_html=True)


# Run directly for testing
if __name__ == "__main__":
    app()