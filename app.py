import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

# Generate dummy energy demand data for demonstration
def load_energy_demand_data():
    # Replace with your actual dataset path
    try:
        energy_demand_df = pd.read_csv("dataset/spg.csv")
        st.success("Energy demand data loaded successfully.")
    except Exception as e:
        st.error(f"Failed to load energy demand data. Error: {e}")
        return None

    # Check that 'generated_power_kw' exists
    if 'generated_power_kw' not in energy_demand_df.columns:
        st.error("Missing 'generated_power_kw' column in the dataset.")
        return None

    # Preprocessing (if necessary)
    # Ensure 'Hour' column exists and is properly formatted
    if 'Hour' not in energy_demand_df.columns:
        #st.warning("No 'Hour' column found. Creating a default Hour column based on row index.")
        energy_demand_df['Hour'] = np.arange(len(energy_demand_df))

    energy_demand_df['generated_power_kw'] = energy_demand_df['generated_power_kw'].fillna(energy_demand_df['generated_power_kw'].mean())  # Handle missing values
    return energy_demand_df

# Function to train and save model
def train_and_save_model():
    st.warning("Training a new model. This might take a moment...")

    # Example: Create synthetic data for training
    X, y = make_regression(n_samples=100, n_features=20, noise=0.1, random_state=42)

    # Train the model
    model = RandomForestRegressor()
    model.fit(X, y)

    # Save the model to a file
    model_path = os.path.join("models", "modelRF.pkl")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

    st.success("Model trained and saved successfully!")
    return model

# Function to load pre-trained model or train a new one
def load_model():
    model_path = os.path.join("models", "modelRF.pkl")

    if not os.path.exists(model_path):
        st.warning(f"The model file '{model_path}' does not exist.")
        return train_and_save_model()

    try:
        model = joblib.load(model_path)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.warning("Re-training the model...")
        return train_and_save_model()

# Function to predict using the trained model
def predict_output(model, input_data):
    prediction = model.predict([input_data])
    return prediction

# Function to get user input for all features
def get_user_input():
    input_values = []
    features = [
        'temperature_2_m_above_gnd', 'relative_humidity_2_m_above_gnd',
        'mean_sea_level_pressure_MSL', 'total_precipitation_sfc',
        'snowfall_amount_sfc', 'total_cloud_cover_sfc',
        'high_cloud_cover_high_cld_lay', 'medium_cloud_cover_mid_cld_lay',
        'low_cloud_cover_low_cld_lay', 'shortwave_radiation_backwards_sfc',
        'wind_speed_10_m_above_gnd', 'wind_direction_10_m_above_gnd',
        'wind_speed_80_m_above_gnd', 'wind_direction_80_m_above_gnd',
        'wind_speed_900_mb', 'wind_direction_900_mb',
        'wind_gust_10_m_above_gnd', 'angle_of_incidence', 'zenith', 'azimuth'
    ]
    for feature in features:
        input_value = st.text_input(f"Enter value for {feature}:", value="0")
        try:
            input_values.append(float(input_value))
        except ValueError:
            st.error(f"Invalid input for {feature}. Please enter a numeric value.")
            return None
    return input_values

# Function to detect power theft
def detect_power_theft(actual_demand, predicted_demand):
    # Calculate difference between actual and predicted demand
    difference = np.abs(actual_demand - predicted_demand)
    theft_threshold = 200  # Threshold difference indicating possible theft
    if difference > theft_threshold:
        return True, difference
    return False, difference

# Function for load balancing (simple approach)
# Function for load balancing (simple approach)
def load_balancing(energy_demand_df, demand_adjustment):
    # Adjust energy demand based on the user input adjustment percentage
    adjusted_demand = energy_demand_df['generated_power_kw'] * (1 + demand_adjustment / 100)
    energy_demand_df['Adjusted Demand (MW)'] = adjusted_demand
    return energy_demand_df


# Automated Load Balancing and Recommendations
# Automated Load Balancing and Recommendations
def automated_load_balancing(predicted_demand):
    """Automated load balancing based on predicted demand."""
    num_zones = 100

    # Simulate different demand distribution for each zone (randomly for demonstration)
    # You can replace this with more complex logic or based on user input
    demand_per_zone = np.random.rand(num_zones)  # Random values to simulate fluctuation
    demand_per_zone = demand_per_zone / demand_per_zone.sum()  # Normalize to make sure they sum to 1
    load_distribution = {f"Zone {i+1}": predicted_demand * demand_per_zone[i] for i in range(num_zones)}

    # Generating recommendations based on load balancing
    recommendations = []
    for zone, demand in load_distribution.items():
        if demand > 750:  # Assume 750 MW as a threshold for high demand
            recommendations.append(f"{zone} is overloaded! Consider optimizing or reducing demand.")
        else:
            recommendations.append(f"{zone} is within optimal demand range.")

    return load_distribution, recommendations


def main():
    st.title("Smart Grid Management using AIML")
    st.markdown("Distribution of the energy based on the prediction and power theft detection.")

    # Generate and display energy demand data
    energy_demand_df = load_energy_demand_data()
    if energy_demand_df is None:
        return  # Exit if data loading failed

    st.subheader("Real-Time Energy Distribution")
    st.line_chart(energy_demand_df.set_index("Hour")['generated_power_kw'])
    st.subheader("Hourly Demand DataFrame")
    st.write(energy_demand_df)

    # Adjust Demand slider
    user_input_slider = st.slider("Adjust Demand (%)", min_value=-100, max_value=100, value=0)

    # Apply the demand adjustment and update the chart and DataFrame
    adjusted_energy_demand_df = load_balancing(energy_demand_df.copy(), user_input_slider)

    # Display the adjusted energy demand
    st.subheader("Adjusted Energy Demand Distribution")
    st.line_chart(adjusted_energy_demand_df.set_index("Hour")['Adjusted Demand (MW)'])
    
    st.subheader("Adjusted Hourly Demand DataFrame")
    st.write(adjusted_energy_demand_df)

    # Load or train the model
    model = load_model()
    if not model:
        return  # Exit if model loading or training failed

    # Displaying grid management section
    st.subheader("Power Generation Prediction")
    user_input = get_user_input()
    if user_input is not None:
        if st.button("Predict"):
            try:
                predicted_output = predict_output(model, user_input)
                st.success(f"Predicted Output: {predicted_output[0]:.2f}")

                # Simulating actual demand for theft detection
                actual_demand = np.random.randint(100, 1000)  # Example actual demand
                theft_detected, theft_difference = detect_power_theft(actual_demand, predicted_output[0])
                if theft_detected:
                    st.error(f"Power theft detected! Difference: {theft_difference:.2f} MW")
                else:
                    st.info("No power theft detected.")

                # Load balancing section
                st.subheader("Load Balancing")
                #demand_adjustment = st.slider("Adjust the load balancing (%)", min_value=-20, max_value=20, value=0)
                
                # Apply load balancing to the already adjusted energy demand
                balanced_df = adjusted_energy_demand_df.copy()
                
                
    
    
    

                st.write("Balanced Energy Demand DataFrame:")
                st.write(balanced_df)

                # Automated Load Balancing and Recommendations
                st.subheader("Automated Load Balancing")
                load_distribution, recommendations = automated_load_balancing(predicted_output[0])
                
                st.write("Automated Load Distribution:")
                st.write(load_distribution)
                
                st.write("Recommendations for Optimizing Load Distribution:")
                for recommendation in recommendations:
                    st.write(f"- {recommendation}")
                    
            except Exception as e:
                st.error(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()
