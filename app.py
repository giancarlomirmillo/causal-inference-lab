#!/usr/bin/env python3
"""
Streamlit-based web application for interactive data analysis.

Users can upload CSV or Excel datasets, define a research objective, and perform a range
of analyses including exploratory data analysis, descriptive statistics, correlations,
regression, classification, and a simple path analysis. The interface is designed to be
colorful, modern, and professional, with navigation controls and a final report
summarising insights and actionable suggestions.

To run the app locally, install the required libraries (pandas, numpy, seaborn, matplotlib,
plotly, scikit-learn, networkx) and execute `streamlit run app.py`.
"""

import os
import io
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
)

import streamlit as st


def load_dataset(file) -> pd.DataFrame:
    """Load a dataset from an uploaded CSV or Excel file.

    Args:
        file: A file-like object representing the uploaded dataset.

    Returns:
        A pandas DataFrame containing the data.
    """
    name = file.name.lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(file)
        elif name.endswith(('.xls', '.xlsx')):
            return pd.read_excel(file)
        else:
            raise ValueError("Unsupported file format. Please upload CSV or Excel files.")
    except Exception as exc:
        st.error(f"Error loading file: {exc}")
        raise


def exploratory_data_analysis(df: pd.DataFrame) -> None:
    """Display exploratory data analysis on the provided DataFrame via Streamlit.

    Shows shape, data types, missing values, descriptive stats, and interactive plots.
    """
    st.write("**Shape of dataset:**", df.shape)
    st.write("**Column types:**")
    st.dataframe(df.dtypes.rename('dtype'))
    st.write("**Missing values per column:**")
    st.dataframe(df.isnull().sum().rename('missing_values'))
    st.write("**Descriptive statistics:**")
    st.dataframe(df.describe(include='all').T)

    # Numeric variable distribution
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        col_to_plot = st.selectbox(
            "Select a numeric column to explore its distribution:",
            options=numeric_cols,
        )
        if col_to_plot:
            fig, ax = plt.subplots()
            sns.histplot(df[col_to_plot].dropna(), kde=True, ax=ax)
            ax.set_title(f"Distribution of {col_to_plot}")
            st.pyplot(fig)

        # Pairplot selection
        selected_cols = st.multiselect(
            "Select numeric columns for a pairplot (up to 5):",
            numeric_cols,
            default=numeric_cols[: min(3, len(numeric_cols))],
        )
        if selected_cols and st.button("Show Pairplot"):
            # pairplot is heavy; use a sample of rows if dataset is large
            subset = df[selected_cols].dropna().copy()
            if len(subset) > 1000:
                subset = subset.sample(1000, random_state=42)
            pair_fig = sns.pairplot(subset)
            st.pyplot(pair_fig)


def descriptive_analysis(df: pd.DataFrame) -> None:
    """Display descriptive statistics for the DataFrame via Streamlit."""
    st.write("### Descriptive Statistics")
    st.dataframe(df.describe(include='all').T)
    st.write(
        "The above table shows measures of central tendency (mean, median), dispersion (std),"
        " and counts for each column where applicable."
    )


def correlation_analysis(df: pd.DataFrame) -> None:
    """Display correlation matrix and scatter plot selection via Streamlit."""
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols or len(numeric_cols) < 2:
        st.warning("Need at least two numeric columns for correlation analysis.")
        return

    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Matrix")
    st.pyplot(fig)
    st.write("Heatmap showing pairwise correlations among numeric variables.")

    x_var = st.selectbox("Select X variable for scatter plot:", numeric_cols)
    # Provide a default index not equal to x_var for y variable
    y_options = [col for col in numeric_cols if col != x_var]
    y_var = st.selectbox("Select Y variable for scatter plot:", y_options)
    if x_var and y_var:
        fig2, ax2 = plt.subplots()
        sns.scatterplot(data=df, x=x_var, y=y_var, ax=ax2)
        ax2.set_title(f"Scatter Plot: {y_var} vs {x_var}")
        st.pyplot(fig2)


def run_regression(df: pd.DataFrame) -> None:
    """Perform a simple linear regression using selected features and target."""
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) < 2:
        st.warning("Not enough numeric columns for regression analysis.")
        return

    target = st.selectbox("Select target (dependent) variable:", numeric_cols)
    feature_candidates = [col for col in numeric_cols if col != target]
    features = st.multiselect(
        "Select features (independent variables):",
        options=feature_candidates,
    )
    if features and st.button("Run Regression"):
        X = df[features]
        y = df[target]
        # Drop rows with any missing values in selected columns
        combined = pd.concat([X, y], axis=1).dropna()
        X = combined[features]
        y = combined[target]
        if X.empty:
            st.error("No data available after removing missing values. Unable to run regression.")
            return
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.write(f"**Mean Squared Error (MSE):** {mse:.4f}")
        st.write(f"**R² Score:** {r2:.4f}")
        coeff_df = pd.DataFrame({
            "Feature": features,
            "Coefficient": model.coef_,
        })
        st.write("### Coefficients")
        st.dataframe(coeff_df)
        # Plot actual vs predicted
        fig = px.scatter(
            x=y_test,
            y=y_pred,
            labels={"x": "Actual", "y": "Predicted"},
            title="Actual vs Predicted Values",
        )
        st.plotly_chart(fig, use_container_width=True)


def run_classification(df: pd.DataFrame) -> None:
    """Perform a basic classification using a RandomForestClassifier."""
    # Identify categorical targets
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    if not cat_cols:
        st.warning("No categorical target columns found for classification.")
        return

    target = st.selectbox("Select target (categorical) variable:", cat_cols)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    features = st.multiselect(
        "Select numeric features for classification:", numeric_cols
    )
    if features and st.button("Run Classification"):
        X = df[features]
        y = df[target]
        combined = pd.concat([X, y], axis=1).dropna()
        X = combined[features]
        y = combined[target]
        if X.empty:
            st.error("No data available after removing missing values. Unable to run classification.")
            return
        # Encode categorical labels
        y_encoded, uniques = pd.factorize(y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.write(f"**Accuracy:** {acc:.4f}")
        cm = confusion_matrix(y_test, y_pred)
        cm_fig = px.imshow(
            cm,
            text_auto=True,
            labels={"x": "Predicted", "y": "Actual"},
            title="Confusion Matrix",
            color_continuous_scale="Blues",
        )
        st.plotly_chart(cm_fig, use_container_width=True)
        report_dict = classification_report(
            y_test, y_pred, target_names=[str(u) for u in uniques], output_dict=True
        )
        st.write("### Classification Report")
        st.dataframe(pd.DataFrame(report_dict).T)


def path_analysis(df: pd.DataFrame) -> None:
    """Perform a simple correlation-based path analysis visualization.

    This function constructs a network graph where nodes represent numeric variables and
    edges represent correlations above a user-defined threshold. It's not a true
    structural equation model but provides a visual cue of variable relationships.
    """
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) < 2:
        st.warning("Not enough numeric columns to perform path analysis.")
        return

    corr = df[numeric_cols].corr()
    threshold = st.slider(
        "Correlation threshold to display edges:",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
    )
    G = nx.Graph()
    for col in numeric_cols:
        G.add_node(col)
    # Add edges for correlations exceeding threshold
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            weight = corr.iloc[i, j]
            if abs(weight) >= threshold:
                G.add_edge(numeric_cols[i], numeric_cols[j], weight=weight)
    # Draw the graph
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    fig, ax = plt.subplots(figsize=(8, 6))
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    # Normalize colors based on weights
    cmap = plt.cm.coolwarm
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color='skyblue',
        edge_color=weights,
        edge_cmap=cmap,
        ax=ax,
    )
    # Edge labels
    labels = nx.get_edge_attributes(G, 'weight')
    labels_formatted = {k: f"{v:.2f}" for k, v in labels.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels_formatted, ax=ax)
    ax.set_title("Correlation-based Path Analysis")
    st.pyplot(fig)
    st.write(
        "Nodes represent numeric variables. Edges connect variables with correlations above the threshold;"
        " edge labels show the correlation coefficients."
    )


def generate_final_report(df: pd.DataFrame, research_obj: str) -> None:
    """Generate a final report summarising the objective, dataset, and suggestions."""
    st.subheader("Final Report")
    st.write("### Research Objective")
    st.write(research_obj if research_obj else "No research objective defined.")
    st.write("### Dataset Overview")
    st.write(f"Shape: {df.shape}")
    st.dataframe(df.dtypes.rename('dtype'))
    st.write("### Key Insights & Suggestions")
    if research_obj:
        st.markdown(
            f"Based on your objective, **{research_obj}**, focus on variables that show strong relationships with your target."
            " Use the results from the analyses (correlations, regression coefficients, classification feature importances)"
            " to guide decision-making and further investigate unexpected patterns. Consider data preprocessing (e.g.,"
            " scaling or encoding) and model tuning for improved performance."
        )
    else:
        st.write("Define a research objective to receive tailored suggestions.")


def main() -> None:
    st.set_page_config(
        page_title="Interactive Data Analysis", page_icon="📊", layout="wide"
    )
    st.title("📊 Interactive Data Analysis Web App")
    st.markdown(
        "Upload your dataset, define your research objective, and choose from various analyses to gain insights."
    )

    research_obj = st.text_input("Define your research objective:")
    uploaded_file = st.file_uploader(
        "Upload your dataset (CSV or Excel):", type=["csv", "xls", "xlsx"]
    )

    if uploaded_file:
        df = load_dataset(uploaded_file)
        st.success(
            f"Loaded {uploaded_file.name} successfully. Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns."
        )
        if st.checkbox("Preview dataset"):
            st.dataframe(df.head())

        analysis_option = st.radio(
            "Select analysis type:",
            options=[
                "Exploratory Data Analysis (EDA)",
                "Descriptive Analysis",
                "Correlation Analysis",
                "Regression",
                "Classification",
                "Path Analysis",
            ],
            horizontal=True,
        )

        # Display selected analysis
        if analysis_option == "Exploratory Data Analysis (EDA)":
            st.header("Exploratory Data Analysis")
            exploratory_data_analysis(df)
        elif analysis_option == "Descriptive Analysis":
            st.header("Descriptive Analysis")
            descriptive_analysis(df)
        elif analysis_option == "Correlation Analysis":
            st.header("Correlation Analysis")
            correlation_analysis(df)
        elif analysis_option == "Regression":
            st.header("Regression Analysis")
            run_regression(df)
        elif analysis_option == "Classification":
            st.header("Classification Analysis")
            run_classification(df)
        elif analysis_option == "Path Analysis":
            st.header("Path Analysis")
            path_analysis(df)

        # Final report button
        st.divider()
        if st.button("Generate Final Report"):
            generate_final_report(df, research_obj)
    else:
        st.info(
            "Awaiting a dataset upload. Please upload a CSV or Excel file to begin analysis."
        )


if __name__ == "__main__":
    main()