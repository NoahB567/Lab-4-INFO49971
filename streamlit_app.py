import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import streamlit as st

# Streamlit File Upload
file_path = st.file_uploader("Upload CSV", type=["csv"])

if file_path is not None:
    # Load the dataset
    data = pd.read_csv(file_path)

    # Display basic information about the dataset
    st.text(str(data.info()))

    # Preview the first few rows
    st.write(data.head())

    # List of cost-related columns
    cost_columns = ['On Demand', 'Linux Reserved cost', 'Linux Spot Minimum cost', 'Windows On Demand cost', 'Windows Reserved cost']

    # Remove '$' and 'hourly' and convert columns to numeric
    for column in cost_columns:
        data[column] = pd.to_numeric(data[column].str.replace('[$, hourly]', '', regex=True), errors='coerce')

    # Check for missing values after conversion
    st.write(data[cost_columns].isnull().sum())

    # Generate summary statistics for cost-related columns
    cost_summary = data[cost_columns].describe()
    st.write(cost_summary)

    # Set up the plotting style
    sns.set(style="whitegrid")

    # Create a boxplot to visualize the distribution of costs
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=data[cost_columns], palette="Set2")

    # Set the plot labels and title
    plt.title('Cost Comparison of Amazon EC2 Instances (Hourly)', fontsize=16)
    plt.ylabel('Cost (USD)', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=12)

    # Display the plot
    plt.tight_layout()
    st.pyplot(fig)

    # Function to identify outliers using the IQR method
    def detect_outliers(column):
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return data[(data[column] < lower_bound) | (data[column] > upper_bound)]

    # Find outliers in the On-Demand cost column
    outliers_on_demand = detect_outliers('On Demand')
    st.write(outliers_on_demand)

    # Compare Reserved and On-Demand costs
    cost_comparison = data[['Name', 'On Demand', 'Linux Reserved cost']].dropna().sort_values('On Demand')
    st.write(cost_comparison.head(10))  # Display the top 10 cost-effective instances

    # Filter for specific instance families
    def filter_instance_family(family):
        return data[data['Name'].str.startswith(family)]

    # Example: Filter for T2 and T3 instance families
    t2_instances = filter_instance_family('T2')
    t3_instances = filter_instance_family('T3')

    # Compare summary statistics for T2 and T3 instances
    t2_summary = t2_instances[cost_columns].describe()
    t3_summary = t3_instances[cost_columns].describe()

    st.write("T2 Instance Costs Summary:\n", t2_summary)
    st.write("\nT3 Instance Costs Summary:\n", t3_summary)

    # Visualize cost comparison for T2 and T3 instance families
    fig, ax = plt.subplots(figsize=(12, 6))

    # Add T2 instances
    sns.boxplot(data=t2_instances[cost_columns], palette="Blues", showmeans=True)
    plt.title('Cost Distribution for T2 Instances', fontsize=16)
    plt.ylabel('Cost (USD)', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    st.pyplot(fig)

    # Add T3 instances
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=t3_instances[cost_columns], palette="Greens", showmeans=True)
    plt.title('Cost Distribution for T3 Instances', fontsize=16)
    plt.ylabel('Cost (USD)', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    st.pyplot(fig)

    # Compare On-Demand and Reserved costs for T2 and T3 families
    comparison = pd.concat([t2_instances[['Name', 'On Demand', 'Linux Reserved cost']],t3_instances[['Name', 'On Demand', 'Linux Reserved cost']]])

    # Sort by On-Demand costs
    comparison_sorted = comparison.dropna().sort_values('On Demand')
    st.write(comparison_sorted.head(10))  # Display the 10 lowest-cost instances

    # Convert memory from string (e.g., '0.5 GiB') to numeric by stripping ' GiB'
    data['Instance Memory'] = pd.to_numeric(data['Instance Memory'].str.replace(' GiB', ''))

    # Extract the number of vCPUs from the 'vCPUs' column (e.g., '2 vCPUs')
    data['vCPUs'] = pd.to_numeric(data['vCPUs'].str.extract('(\d+)', expand=False))

    # Check the data after conversion
    st.write(data[['Instance Memory', 'vCPUs']].head())

    # Drop rows with missing values in the relevant columns (for simplicity)
    data_cleaned = data.dropna(subset=['On Demand', 'Instance Memory', 'vCPUs'])

    # Verify that the cleaned data has no missing values
    st.write(data_cleaned.isnull().sum())

    # Define features and target
    X = data_cleaned[['Instance Memory', 'vCPUs']]
    y = data_cleaned['On Demand']

    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.write(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Print the coefficients of the model
    st.write(f"Intercept: {model.intercept_}")
    st.write(f"Coefficients: {model.coef_}")

    # Predict the costs for the test data
    y_pred = model.predict(X_test)

    # Calculate performance metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5

    st.write(f"Mean Absolute Error (MAE): {mae}")
    st.write(f"Mean Squared Error (MSE): {mse}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse}")

    # Plot actual vs predicted values
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, color='b')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.title('Actual vs Predicted On-Demand Costs')
    plt.xlabel('Actual On-Demand Cost')
    plt.ylabel('Predicted On-Demand Cost')
    st.pyplot(fig)

    # Example: Predict cost for a new instance with 4 GiB memory and 2 vCPUs
    new_instance = [[4, 2]]
    predicted_cost = model.predict(new_instance)
    final_cost = max(0.0, predicted_cost[0]) # Predicted cost won't go below 0.
    st.write(f"Predicted On-Demand Cost for 4 GiB, 2 vCPUs: ${final_cost:.4f}")