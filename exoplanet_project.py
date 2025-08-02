import streamlit as st
import pandas as pd
import numpy as np

# --- Data Loading with Caching ---
@st.cache_data
def load_data(filepath):
    """
    Loads the exoplanet data from a CSV file.
    Using @st.cache_data ensures the data is loaded only once.
    """
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        st.error(f"Error: The file '{filepath}' was not found. Please make sure it's in the same directory as the app.py file.")
        return None

# --- Main Application Logic ---
def main():
    """
    The main function that sets up and runs the Streamlit application.
    """
    st.set_page_config(layout="wide", page_title="Exoplanet Analysis Tool", page_icon="🪐")

    st.title("Exoplanet Analysis Tool 🪐")

    # Load the dataset from the CSV file
    df = load_data("planetary_system.csv")

    if df is None:
        st.warning("Dataset could not be loaded. The application cannot proceed.")
        return # Stop the app if the data isn't available

    # --- Sidebar for User Input ---
    st.sidebar.header("🎯 Target Selection")

    # Mode selection for the app's main functionality
    app_mode = st.sidebar.radio(
        "Choose your analysis mode:",
        ("Select Planet from Archive", "Analyze a Hypothetical Planet")
    )

    # --- Main Panel Display ---
    if app_mode == "Select Planet from Archive":
        st.header("🔭 Analysis from Exoplanet Archive")
        planet_name_list = sorted(df['pl_name'].unique())
        selected_planet = st.selectbox(
            "Select a planet to analyze:",
            options=planet_name_list
        )

        st.markdown("---")
        st.subheader(f"Planetary Profile for: **{selected_planet}**")
        planet_data = df[df['pl_name'] == selected_planet].iloc[0]

        # --- Objective 1 & 6: Predicted Classification & Density ---
        
        # TODO: Replace this with your actual classification model prediction
        # Example: predicted_classification = your_model.predict(planet_features)
        predicted_classification = "Super-Earth" # Placeholder value
        st.write(f"**Predicted Classification:** `{predicted_classification}`")

        # Button to show classification probabilities
        if st.button("View Classification Probabilities"):
            # TODO: Replace this with your model's actual probabilities
            # Example: probs = your_model.predict_proba(planet_features)
            st.info("This is placeholder data. Your model's output should replace this chart.")
            probabilities = pd.DataFrame({
                "Planet Type": ["Super-Earth", "Neptune-like", "Terrestrial", "Gas Giant"],
                "Probability": [0.78, 0.15, 0.07, 0.0]
            })
            st.bar_chart(probabilities.set_index("Planet Type"))

        # TODO: Replace this with your actual regression model prediction
        # Example: predicted_density_val = your_density_model.predict(planet_features)
        predicted_density = 6.2 # Placeholder value
        st.write(f"**Predicted Density:** `{predicted_density} g/cm³`")


    elif app_mode == "Analyze a Hypothetical Planet":
        st.header("🔬 Analyze a Hypothetical Planet")
        st.write("Manually enter parameters to perform a 'what-if' scenario.")

        with st.form(key="hypothetical_planet_form"):
            col1, col2 = st.columns(2)
            with col1:
                orbital_period = st.number_input("Orbital Period (days)", min_value=0.1, value=10.0, step=1.0, format="%.2f")
                stellar_temperature = st.number_input("Stellar Temperature (K)", min_value=2000, value=5778, step=100)
            with col2:
                planet_radius = st.number_input("Planet Radius (Earth Radii)", min_value=0.1, value=1.0, step=0.1, format="%.2f")
                planet_mass = st.number_input("Planet Mass (Earth Masses)", min_value=0.1, value=1.0, step=0.1, format="%.2f")

            analyze_button = st.form_submit_button(label="Analyze This Planet")

            if analyze_button:
                st.markdown("---")
                st.subheader("Hypothetical Planetary Profile")
                st.info("The following are placeholder predictions based on your input. Integrate your models for real analysis.")

                # --- Objective 1 & 6: Predicted Classification & Density ---
                
                # TODO: Feed inputs into your classification model
                # Example: predicted_class = your_model.predict([[orbital_period, stellar_temp, ...]])
                hypothetical_class = "Terrestrial"
                st.write(f"**Predicted Classification:** `{hypothetical_class}`")
                
                # TODO: Feed inputs into your regression model
                # For this example, we just calculate density from mass and radius.
                earth_avg_density = 5.51 # in g/cm³
                # Density = mass / volume; Volume is proportional to radius^3
                hypothetical_density = (planet_mass / (planet_radius**3)) * earth_avg_density if planet_radius > 0 else 0
                st.write(f"**Predicted Density:** `{hypothetical_density:.2f} g/cm³`")


if __name__ == "__main__":
    main()
