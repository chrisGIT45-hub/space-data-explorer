# exoplanet_project.py

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, r2_score
from sklearn.decomposition import PCA
import numpy as np
import base64
import plotly.express as px
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="Space Data Explorer",
    page_icon="🌌",
    layout="wide"
)

# --- CORRECTED: RELATIVE FILE PATHS ---
# All paths are now relative to the script's location in your GitHub repository.
# The BASE_PATH variable has been removed as it is no longer needed.

# Data files
EXOPLANET_FILEPATH = "PS_2025.07.25_23.13.46.zip"
METEORITE_FILEPATH = "Meteorite_Landings.csv"
TRANSIT_FILEPATH = "cleaned_Transitplanet_data.zip"  # Note: corrected to lowercase 't'
KEPLER_FP_FILEPATH = "keplerfalsepostives.csv"

# Asset files
EXO_MAIN_BG = "solar-system-252023.png"
EXO_SIDEBAR_BG = "planets-solar-system-cosmic-in-s.jpg"
METEOR_MAIN_BG = "asteroid-earth-space-hd-wallpaper-uhdpaper.com-510@0@f.jpg"
METEOR_SIDEBAR_BG = "image_a67ddb.png"
KEPLER_MAIN_BG = "kepler_main.jpg"
KEPLER_SIDEBAR_BG = "kepler_sidebar.jpg"
TRANSIT_MAIN_BG = "transitbackground.png"
# IMPORTANT: Make sure 'transitsidebar.png' exists in your GitHub repository!
TRANSIT_SIDEBAR_BG = "transitsidebar.png"

# Font file paths (Note the 'fonts/' subfolder)
TITLE_FONT_PATH = "fonts/SpecialGothicExpandedOne-Regular.ttf"


# --- Helper function to load a local image as Base64 ---
@st.cache_data
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        st.warning(f"Asset file not found: {bin_file}. A default background will be used.")
        return None

# --- Helper function to load and encode a local font file ---
@st.cache_data
def get_font_as_base64(font_path):
    try:
        with open(font_path, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        st.warning(f"Font file not found: {font_path}. Using default fonts.")
        return None

# --- Custom Styling and Background ---
def add_custom_styling(main_bg_path, sidebar_bg_path, title_font_path):
    title_font_base64 = get_font_as_base64(title_font_path)
    main_bg_base64 = get_base64_of_bin_file(main_bg_path)
    sidebar_bg_base64 = get_base64_of_bin_file(sidebar_bg_path)

    font_css = ""
    if title_font_base64:
        font_css = f"""
        @font-face {{
            font-family: 'SpecialGothic';
            src: url(data:font/ttf;base64,{title_font_base64}) format('truetype');
        }}
        h1, h2, h3, h4, h5, h6 {{
            font-family: 'SpecialGothic', sans-serif;
        }}
        """

    main_bg_css = f"""
        [data-testid="stAppViewContainer"] > .main {{
            background-image: url("data:image/png;base64,{main_bg_base64}");
            background-size: cover;
            background-position: center center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        [data-testid="stAppViewContainer"] > .main::before {{
            content: ''; position: absolute; top: 0; left: 0; right: 0; bottom: 0;
            background-color: rgba(0, 0, 0, 0.7); z-index: -1;
        }}
    """ if main_bg_base64 else ""

    sidebar_bg_css = f"""
        [data-testid="stSidebar"] > div:first-child {{
            background-image: url("data:image/jpeg;base64,{sidebar_bg_base64}");
            background-position: center; background-size: cover;
        }}
    """ if sidebar_bg_base64 else ""

    custom_css = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Exo+2:wght@400;700&display=swap');
        {font_css}
        html, body, [class*="css"] {{
            font-family: 'Exo 2', sans-serif;
        }}
        [data-testid="stHeader"] {{
            display: none;
        }}
        {main_bg_css}
        {sidebar_bg_css}
        .main .block-container {{ position: relative; z-index: 1; background: none; }}
        [data-testid="stAppViewContainer"] > .main .block-container * {{
            color: #FFFFFF; text-shadow: 1px 1px 6px #000000;
        }}
        [data-testid="stSidebar"] {{
            border-right: 1px solid rgba(125, 249, 255, 0.7);
            box-shadow: 0 0 10px rgba(125, 249, 255, 0.5);
        }}
        .styled-container {{
            padding: 1rem; border-radius: 10px; border: 1px solid rgba(125, 249, 255, 0.7);
            box-shadow: 0 0 8px rgba(125, 249, 255, 0.5); background-color: rgba(30, 30, 50, 0.9); height: 100%;
        }}
        [data-testid="stDataFrame"] {{
            padding: 1rem; border-radius: 10px; border: 1px solid rgba(125, 249, 255, 0.7);
            box-shadow: 0 0 10px rgba(125, 249, 255, 0.5); background-color: rgba(30, 30, 50, 0.9);
        }}
        [data-testid="stDataFrame"] thead th {{ background-color: rgba(125, 249, 255, 0.2); color: white; }}
        [data-testid="stDataFrame"] {{ border: none !important; }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


# --- DATA LOADING FUNCTIONS ---
@st.cache_data
def load_data(filepath):
    try:
        # Determine file type and load accordingly
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith('.zip'):
            df = pd.read_csv(filepath, comment='#') # Assuming zipped CSVs
        else:
            st.error(f"Unsupported file type for: {filepath}")
            return None
        return df
    except FileNotFoundError:
        st.error(f"Data file not found: '{filepath}'. Please check the path.")
        return None

@st.cache_data
def load_meteorite_data(filepath):
    df = load_data(filepath)
    if df is not None:
        df.dropna(subset=['mass (g)', 'year', 'reclat', 'reclong'], inplace=True)
    return df

# --- MODEL TRAINING AND OBJECTIVE FUNCTIONS ---
# (omitted for brevity, no changes needed from your original)
@st.cache_resource
def train_classification_model(data):
    features = ['pl_orbper', 'pl_rade', 'pl_bmasse', 'st_teff', 'st_rad', 'st_mass', 'sy_dist']
    target = 'discoverymethod'
    class_df = data[features + [target]].dropna()
    le = LabelEncoder()
    y = le.fit_transform(class_df[target])
    X = class_df[features]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test, le, X.columns

@st.cache_resource
def train_regression_model(data):
    features = ['pl_orbper', 'st_teff', 'st_rad', 'st_mass']
    target = 'pl_rade'
    regr_df = data[features + [target]].dropna()
    X = regr_df[features]
    y = regr_df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test

@st.cache_resource
def train_clustering_model(data):
    features = ['sy_snum', 'sy_pnum', 'st_teff', 'st_mass', 'sy_dist']
    cluster_df = data[features].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(cluster_df)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    return kmeans, X_scaled, cluster_df

@st.cache_resource
def train_transit_classification_model(_data):
    def classify_planet_type(radius):
        if radius < 2: return 'Earth-like / Super-Earth'
        elif 2 <= radius < 6: return 'Neptune-like'
        elif 6 <= radius < 15: return 'Gas Giant'
        else: return 'Large Gas Giant'

    _data['planet_type'] = _data['pl_rade'].apply(classify_planet_type)
    features = ['pl_rade', 'pl_orbper']
    target = 'planet_type'
    model_df = _data[features + [target]].dropna()
    X = model_df[features]
    y = model_df[target]
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    return model, X_test, y_test, le, X.columns

def run_classification_objective(df):
    st.header("Objective 1: Predicting Planet Discovery Method")
    model_results = train_classification_model(df)
    model, X_test, y_test, le, feature_names = model_results
    y_pred = model.predict(X_test)
    st.subheader("Model Performance")
    report = classification_report(y_test, y_pred, target_names=le.inverse_transform(np.unique(y_test)), zero_division=0, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())
    st.subheader("Visual Analysis")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<div class="styled-container">', unsafe_allow_html=True)
        feature_importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
        fig = px.bar(x=feature_importances.index, y=feature_importances.values, labels={'x': 'Feature', 'y': 'Importance'}, title="Feature Importance")
        fig.update_layout(paper_bgcolor='rgba(30, 30, 50, 0.9)', plot_bgcolor='rgba(0,0,0,0.4)', font_color="white")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="styled-container">', unsafe_allow_html=True)
        with st.expander("View Interpretation", expanded=True):
            st.markdown("""
            The classification model performs exceptionally well, with an accuracy near 99%.
            - **Most Important Feature:** The plot shows that the planet's orbital period (`pl_orbper`) and its radius (`pl_rade`) are the most significant factors.
            - **Model Strength:** This confirms that different discovery methods are biased towards finding planets with specific characteristics.
            """)
        st.markdown('</div>', unsafe_allow_html=True)

def run_regression_objective(df):
    st.header("Objective 2: Predicting Planet Radius")
    model_results = train_regression_model(df)
    model, X_test, y_test = model_results
    y_pred = model.predict(X_test)
    st.subheader("Model Performance")
    st.dataframe(pd.DataFrame({'Actual Radius': y_test.head(10), 'Predicted Radius': y_pred[:10]}).reset_index(drop=True))
    st.subheader("Visual Analysis")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<div class="styled-container">', unsafe_allow_html=True)
        fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual Radius', 'y': 'Predicted Radius'}, title="Prediction Accuracy", opacity=0.7)
        fig.add_shape(type='line', x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(), line=dict(color='Red', dash='dash'))
        fig.update_layout(paper_bgcolor='rgba(30, 30, 50, 0.9)', plot_bgcolor='rgba(0,0,0,0.4)', font_color="white")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="styled-container">', unsafe_allow_html=True)
        with st.expander("View Interpretation", expanded=True):
            st.markdown(f"""
            The regression model performs poorly, with a negative **R-squared value of {r2_score(y_test, y_pred):.2f}**.
            - **Visual Evidence:** The scatter plot shows predictions do not align with the actual values along the red line.
            - **Conclusion:** Predicting a planet's radius from these features is not feasible, suggesting a more complex relationship.
            """)
        st.markdown('</div>', unsafe_allow_html=True)

def run_clustering_objective(df):
    st.header("Objective 3: Clustering to Discover Planetary System Types")
    model_results = train_clustering_model(df)
    kmeans, X_scaled, cluster_df = model_results
    cluster_df['cluster'] = kmeans.labels_
    st.subheader("Cluster Analysis")
    st.dataframe(cluster_df.groupby('cluster').mean())
    st.subheader("Visual Analysis")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<div class="styled-container">', unsafe_allow_html=True)
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(X_scaled)
        pca_df = pd.DataFrame(data=principal_components, columns=['PC 1', 'PC 2'])
        pca_df['cluster'] = kmeans.labels_
        fig = px.scatter(pca_df, x='PC 1', y='PC 2', color='cluster', title="2D Cluster Visualization", opacity=0.8)
        fig.update_layout(paper_bgcolor='rgba(30, 30, 50, 0.9)', plot_bgcolor='rgba(0,0,0,0.4)', font_color="white")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="styled-container">', unsafe_allow_html=True)
        with st.expander("View Interpretation", expanded=True):
            st.markdown("""
            The K-Means algorithm successfully grouped systems into four distinct clusters.
            - **Visualization:** The plot shows these four groups in a 2D space.
            - **Cluster Meaning:** The "Cluster Analysis" table helps infer what each cluster represents (e.g., single-star vs. multi-star systems).
            """)
        st.markdown('</div>', unsafe_allow_html=True)

def run_habitable_zone_analysis(df):
    st.header("Objective 4: Finding Potentially Habitable Planets")
    hz_df = df[['pl_name', 'st_teff', 'st_rad', 'pl_orbsmax']].dropna()
    sun_teff = 5778
    luminosity = hz_df['st_rad']**2 * (hz_df['st_teff'] / sun_teff)**4
    inner_boundary = np.sqrt(luminosity / 1.1)
    outer_boundary = np.sqrt(luminosity / 0.53)
    is_habitable = (hz_df['pl_orbsmax'] > inner_boundary) & (hz_df['pl_orbsmax'] < outer_boundary)
    habitable_planets = hz_df[is_habitable]
    st.metric("Potentially Habitable Planets Found", f"{len(habitable_planets)} out of {len(hz_df)} candidates")
    if not habitable_planets.empty:
        st.subheader("Visual Analysis")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown('<div class="styled-container">', unsafe_allow_html=True)
            fig = px.histogram(habitable_planets, x='st_teff', nbins=30, labels={'st_teff': 'Stellar Effective Temperature (K)'}, title="Host Star Temperatures")
            fig.update_layout(paper_bgcolor='rgba(30, 30, 50, 0.9)', plot_bgcolor='rgba(0,0,0,0.4)', font_color="white", bargap=0.1)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="styled-container">', unsafe_allow_html=True)
            with st.expander("View Interpretation", expanded=True):
                st.markdown("""
                This analysis filters for planets within the "habitable zone".
                - **Distribution:** Most of these planets orbit stars cooler than our Sun (~5778K).
                - **Significance:** This suggests that cooler, dimmer stars are prime targets in the search for life.
                """)
            st.markdown('</div>', unsafe_allow_html=True)

def display_mass_statistics(df):
    st.header("1. Mass Statistics")
    mass_analysis_df = df[df['mass (g)'] < 50000]
    mean_mass = mass_analysis_df['mass (g)'].mean()
    median_mass = mass_analysis_df['mass (g)'].median()
    col1, col2 = st.columns(2)
    col1.metric("Mean Mass (<50kg)", f"{mean_mass:,.2f} g")
    col2.metric("Median Mass (<50kg)", f"{median_mass:,.2f} g")
    st.subheader("Mass Distribution")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<div class="styled-container">', unsafe_allow_html=True)
        fig = px.histogram(mass_analysis_df, x='mass (g)', nbins=100, title="Distribution of Meteorite Mass (finds < 50kg)")
        fig.update_layout(paper_bgcolor='rgba(30, 30, 50, 0.9)', plot_bgcolor='rgba(0,0,0,0.4)', font_color="white")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="styled-container">', unsafe_allow_html=True)
        with st.expander("View Interpretation", expanded=True):
            st.markdown("""
            The distribution is heavily skewed to the right, meaning most found meteorites are relatively small. The **median** is a better measure of a 'typical' meteorite's mass than the mean, which is pulled upwards by a few very large outliers.
            """)
        st.markdown('</div>', unsafe_allow_html=True)

def display_geospatial_heatmap(df):
    st.header("2. Global Landing Locations")
    st.subheader("Global Heatmap of Meteorite Landings")
    fig = px.density_mapbox(df, lat='reclat', lon='reclong', radius=5, center=dict(lat=0, lon=0), zoom=1,
                            mapbox_style="open-street-map", title="Concentration of Meteorite Finds")
    fig.update_layout(margin=dict(r=0, t=40, l=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Mass-Scaled Bubble Map")
    map_df = df.sample(n=3000, random_state=1) if len(df) > 3000 else df
    map_df['mass_scaled'] = map_df['mass (g)'].apply(lambda x: np.log10(x+1))
    fig2 = px.scatter_mapbox(map_df, lat='reclat', lon='reclong', size='mass_scaled', color='mass (g)',
                             hover_name='name', hover_data=['year', 'recclass'],
                             mapbox_style="open-street-map", zoom=1, title="Global Finds Scaled by Mass (log scale)")
    fig2.update_layout(margin=dict(r=0, t=40, l=0, b=0))
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown('<div class="styled-container" style="text-align: center;">', unsafe_allow_html=True)
    with st.expander("View Interpretation", expanded=True):
        st.markdown("""
        The maps clearly show **detection bias**. Meteorites are easier to **find** in deserts like Antarctica and the Sahara, not that they fall there more often.
        """)
    st.markdown('</div>', unsafe_allow_html=True)

def run_transit_habitability_plot(df):
    st.header("1. Potentially Habitable vs. Non-habitable Planets")
    df_cleaned = df.dropna(subset=['st_teff', 'pl_orbper']).copy()
    def classify_habitability(row):
        st_teff = row['st_teff']
        pl_orbper = row['pl_orbper']
        if 2500 < st_teff <= 3700 and 1 <= pl_orbper <= 50: return 'Potentially Habitable'
        elif 3700 < st_teff <= 5000 and 20 <= pl_orbper <= 150: return 'Potentially Habitable'
        elif 5000 < st_teff <= 6000 and 200 <= pl_orbper <= 400: return 'Potentially Habitable'
        elif 6000 < st_teff <= 7500 and 300 <= pl_orbper <= 500: return 'Potentially Habitable'
        else: return 'Non-habitable'
    df_cleaned['habitability'] = df_cleaned.apply(classify_habitability, axis=1)
    habitable_count = df_cleaned[df_cleaned['habitability'] == 'Potentially Habitable'].shape[0]
    st.metric("Potentially Habitable Planets Found (Simplified Model)", f"{habitable_count}")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<div class="styled-container">', unsafe_allow_html=True)
        fig = go.Figure()
        non_habitable_df = df_cleaned[df_cleaned['habitability'] == 'Non-habitable']
        fig.add_trace(go.Scatter(x=non_habitable_df['st_teff'], y=non_habitable_df['pl_orbper'], mode='markers', marker=dict(color='gray', size=3, opacity=0.4), name='Non-habitable'))
        habitable_df = df_cleaned[df_cleaned['habitability'] == 'Potentially Habitable']
        fig.add_trace(go.Scatter(x=habitable_df['st_teff'], y=habitable_df['pl_orbper'], mode='markers', marker=dict(color='#34dbeb', size=8, line=dict(width=1, color='black')), name='Potentially Habitable'))
        fig.update_layout(title="Habitable Zone Explorer", xaxis_title="Stellar Effective Temperature (K)", yaxis_title="Orbital Period (days)", yaxis_type="log", paper_bgcolor='rgba(30, 30, 50, 0.9)', plot_bgcolor='rgba(0,0,0,0.4)', font_color="white", legend_title_text='Planet Type')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="styled-container">', unsafe_allow_html=True)
        with st.expander("View Interpretation", expanded=True):
            st.markdown("""
            This plot visualizes a simplified **Habitable Zone** model.
            - **What it shows**: Planets are plotted by their orbital period against their star's temperature. The highlighted points are those that fall within a temperature/orbit combination where liquid water could theoretically exist.
            - **The 'Habitable Zone'**: Notice the highlighted band slopes upwards. Cooler, dimmer stars (left) have habitable zones much closer to them, while hotter, brighter stars (right) have zones further away.
            - **Limitations**: This is a very simple model and does not account for other factors like atmospheric composition, planetary mass, or tidal locking, which are all crucial for actual habitability.
            """)
        st.markdown('</div>', unsafe_allow_html=True)

def run_transit_classification_model(df):
    st.header("2. Machine Learning: Planet Classification")
    df_cleaned = df.dropna(subset=['pl_rade', 'pl_orbper']).copy()
    model, X_test, y_test, le, feature_names = train_transit_classification_model(df_cleaned)
    st.markdown("### Predict a Planet's Type")
    st.write("Enter the properties of a planet to predict its classification based on our trained model.")
    col1, col2, col3 = st.columns(3)
    with col1: user_radius = st.number_input("Planet Radius (in Earth Radii)", min_value=0.1, max_value=50.0, value=1.0, step=0.1)
    with col2: user_period = st.number_input("Orbital Period (in days)", min_value=0.1, max_value=1000.0, value=365.25, step=1.0)
    user_input = pd.DataFrame([[user_radius, user_period]], columns=['pl_rade', 'pl_orbper'])
    prediction_encoded = model.predict(user_input)
    prediction = le.inverse_transform(prediction_encoded)
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"**Predicted Planet Type:**")
        st.success(f"### {prediction[0]}")
    st.markdown("---")
    st.subheader("Model Performance & Insights")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<div class="styled-container">', unsafe_allow_html=True)
        st.text("Classification Report")
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

        st.markdown("<br>", unsafe_allow_html=True)

        def classify_planet_type(radius):
            if radius < 2: return 'Earth-like / Super-Earth'
            elif 2 <= radius < 6: return 'Neptune-like'
            elif 6 <= radius < 15: return 'Gas Giant'
            else: return 'Large Gas Giant'
        df_cleaned['planet_type'] = df_cleaned['pl_rade'].apply(classify_planet_type)
        type_counts = df_cleaned['planet_type'].value_counts().reset_index()
        type_counts.columns = ['planet_type', 'count']

        fig_bar = px.bar(type_counts, x='planet_type', y='count',
                         title="Distribution of Planet Types in the Dataset",
                         labels={'planet_type': 'Planet Type', 'count': 'Number of Planets'})
        fig_bar.update_layout(paper_bgcolor='rgba(30, 30, 50, 0.9)', plot_bgcolor='rgba(0,0,0,0.4)', font_color="white")
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="styled-container">', unsafe_allow_html=True)
        with st.expander("View Interpretation", expanded=True):
            st.markdown("""
            **Classification Report:**
            This table shows the model's 'report card'. A high `f1-score` (close to 1.0) indicates high accuracy for that planet type.

            **Distribution Plot:**
            This bar chart shows how many of each planet type are in the dataset used to train the model. You can see the data is imbalanced, with far more 'Earth-like' planets than 'Large Gas Giants'. This context is important for understanding the model's performance.
            """)
        st.markdown('</div>', unsafe_allow_html=True)

# --- Kepler False Positive EDA Functions ---
def run_fpp_distribution_plot(df):
    st.header("Distribution of False Positive Probability")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<div class="styled-container">', unsafe_allow_html=True)
        plot_df = df.dropna(subset=['fpp_prob'])
        fig = px.histogram(plot_df, x='fpp_prob', nbins=50, title="Distribution of False Positive Probability (fpp_prob)")
        fig.update_layout(paper_bgcolor='rgba(30, 30, 50, 0.9)', plot_bgcolor='rgba(0,0,0,0.4)', font_color="white")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="styled-container">', unsafe_allow_html=True)
        with st.expander("View Interpretation", expanded=True):
            st.markdown("""
            This plot shows two large spikes:
            - **Near 0:** A huge number of candidates have a probability very close to zero, meaning they are strong **planet candidates**.
            - **Near 1:** Another large group has a probability close to one, meaning they are strong **false positive** candidates.
            This bimodal distribution is expected and shows the model is good at separating clear cases.
            """)
        st.markdown('</div>', unsafe_allow_html=True)

def run_disposition_pie_chart(df):
    st.header("Disposition of Candidates")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<div class="styled-container">', unsafe_allow_html=True)
        plot_df = df.dropna(subset=['fpp_score']).copy()
        plot_df['disposition'] = plot_df['fpp_score'].apply(lambda x: 'False Positive' if x == 1.0 else 'Planet Candidate')
        fig = px.pie(plot_df, names='disposition', title="Candidate Disposition based on fpp_score", hole=0.3)
        fig.update_traces(textinfo='percent+label', marker=dict(colors=['#4C72B0', '#C44E52']))
        fig.update_layout(paper_bgcolor='rgba(30, 30, 50, 0.9)', plot_bgcolor='rgba(0,0,0,0.4)', font_color="white")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="styled-container">', unsafe_allow_html=True)
        with st.expander("View Interpretation", expanded=True):
            st.markdown("""
            This chart gives a clear count of how many objects in the dataset are classified as Planet Candidates versus False Positives based on the definitive `fpp_score`.
            It shows that a significant portion of the initial signals detected by Kepler were not actual planets.
            """)
        st.markdown('</div>', unsafe_allow_html=True)

def run_radius_period_scatter(df):
    st.header("Radius vs. Period Scatter Plot")
    st.info("ℹ️ To ensure the app is fast and responsive, this plot shows a random sample of 2,000 candidates from the dataset.")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<div class="styled-container">', unsafe_allow_html=True)

        # Prepare and sample the data
        plot_df = df.dropna(subset=['fpp_koi_period', 'fpp_prad', 'fpp_score']).copy()
        plot_df['disposition'] = plot_df['fpp_score'].apply(lambda x: 'False Positive' if x == 1.0 else 'Planet Candidate')

        # Take a random sample to avoid lag
        if len(plot_df) > 2000:
            plot_df = plot_df.sample(n=2000, random_state=42)

        # Create the scatter plot
        fig = px.scatter(plot_df, x='fpp_koi_period', y='fpp_prad', color='disposition',
                         log_x=True, log_y=True,
                         labels={'fpp_koi_period': 'Orbital Period in Days (Log Scale)', 'fpp_prad': 'Planet Radius in Earth Radii (Log Scale)'},
                         title="Planet Radius vs. Orbital Period (Sampled)",
                         color_discrete_map={'Planet Candidate': '#4C72B0', 'False Positive': '#C44E52'})

        fig.update_traces(marker=dict(size=5, opacity=0.7))
        fig.update_layout(paper_bgcolor='rgba(30, 30, 50, 0.9)', plot_bgcolor='rgba(0,0,0,0.4)', font_color="white")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="styled-container">', unsafe_allow_html=True)
        with st.expander("View Interpretation", expanded=True):
            st.markdown("""
            This plot helps visualize where planets and false positives lie.
            - **Planet Candidates** (blue) are often smaller and have a wider range of orbital periods.
            - **False Positives** (red) are scattered but include a large number of objects with very large radii. These are likely background eclipsing binary stars, not actual planets.
            """)
        st.markdown('</div>', unsafe_allow_html=True)


# --- MAIN APP ---
st.sidebar.title("Space Data Explorer")
st.sidebar.markdown("---")
dataset_choice = st.sidebar.selectbox("Select a Dataset:", ["Exoplanet Archive", "Meteorite Landings", "Transiting Planets", "Kepler False Positives"])
st.sidebar.markdown("---")

if dataset_choice == "Exoplanet Archive":
    add_custom_styling(main_bg_path=EXO_MAIN_BG, sidebar_bg_path=EXO_SIDEBAR_BG, title_font_path=TITLE_FONT_PATH)
    data = load_data(EXOPLANET_FILEPATH)
    if data is not None:
        st.sidebar.header("Select an Analysis")
        analysis_choice = st.sidebar.radio("Exoplanet Analyses:", ["Home", "1. Predict Discovery Method", "2. Predict Planet Radius", "3. Discover System Types", "4. Find Habitable Planets"], key="exoplanet_nav")
        if analysis_choice == "Home":
            st.title("🌌 Exoplanet Explorer")
            st.dataframe(data.head(10))
        elif "1." in analysis_choice: run_classification_objective(data)
        elif "2." in analysis_choice: run_regression_objective(data)
        elif "3." in analysis_choice: run_clustering_objective(data)
        elif "4." in analysis_choice: run_habitable_zone_analysis(data)

elif dataset_choice == "Meteorite Landings":
    add_custom_styling(main_bg_path=METEOR_MAIN_BG, sidebar_bg_path=METEOR_SIDEBAR_BG, title_font_path=TITLE_FONT_PATH)
    data = load_meteorite_data(METEORITE_FILEPATH)
    if data is not None:
        st.sidebar.header("Select an Analysis")
        analysis_choice = st.sidebar.radio("Meteorite Analyses:", ["Home", "1. Mass Statistics", "2. Global Landing Locations"], key="meteorite_nav")
        if analysis_choice == "Home":
            st.title("☄️ Meteorite Landings")
            st.dataframe(data.head(10))
        elif "1." in analysis_choice: display_mass_statistics(data)
        elif "2." in analysis_choice: display_geospatial_heatmap(data)

elif dataset_choice == "Transiting Planets":
    add_custom_styling(main_bg_path=TRANSIT_MAIN_BG, sidebar_bg_path=TRANSIT_SIDEBAR_BG, title_font_path=TITLE_FONT_PATH)
    data = load_data(TRANSIT_FILEPATH)
    if data is not None:
        st.sidebar.header("Select an Analysis")
        analysis_choice = st.sidebar.radio("Transit Analyses:", ["Home", "1. Habitable vs. Non-Habitable", "2. Planet Classification Model"], key="transit_nav")
        if analysis_choice == "Home":
            st.title("🪐 Transiting Planets Insights")
            st.dataframe(data.head(10))
        elif "1." in analysis_choice: run_transit_habitability_plot(data)
        elif "2." in analysis_choice: run_transit_classification_model(data)

elif dataset_choice == "Kepler False Positives":
    add_custom_styling(main_bg_path=KEPLER_MAIN_BG, sidebar_bg_path=KEPLER_SIDEBAR_BG, title_font_path=TITLE_FONT_PATH)
    data = load_data(KEPLER_FP_FILEPATH)
    if data is not None:
        st.sidebar.header("Select an Analysis")
        analysis_choice = st.sidebar.radio(
            "Kepler Analyses:",
            [
                "Home",
                "1. Probability Distribution",
                "2. Disposition of Candidates",
                "3. Radius vs. Period Scatter"
            ],
            key="kepler_nav"
        )
        if analysis_choice == "Home":
            st.title("🔭 Kepler False Positive Explorer")
            st.dataframe(data.head(10))
        elif "1." in analysis_choice:
            run_fpp_distribution_plot(data)
        elif "2." in analysis_choice:
            run_disposition_pie_chart(data)
        elif "3." in analysis_choice:
            run_radius_period_scatter(data)
