import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
import time

# ----------------------------------------------------------------------------
# 1. Configuration & Custom CSS
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Segmentation App",
    page_icon="bar-chart",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_css():
    """Loads custom CSS for styling."""
    try:
        with open('style.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        pass # Silently pass if CSS is not found

load_css()

# ----------------------------------------------------------------------------
# 2. Data Processing Functions
# ----------------------------------------------------------------------------
@st.cache_data
def load_data():
    """Loads and preprocesses the dataset."""
    try:
        df = pd.read_csv("mall-customers-data.csv")
    except FileNotFoundError:
        st.error("Dataset not found. Please ensure 'mall-customers-data.csv' is in the directory.")
        st.stop()
        
    # Standardize column names
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    if 'annual_income_(k$)' in df.columns:
        df.rename(columns={'annual_income_(k$)': 'annual_income'}, inplace=True)
    if 'spending_score_(1-100)' in df.columns:
        df.rename(columns={'spending_score_(1-100)': 'spending_score'}, inplace=True)
        
    # Handle missing values by dropping them
    df.dropna(inplace=True)
    return df

@st.cache_data
def preprocess_data(df, features):
    """Scales features using StandardScaler."""
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[features])
    return scaled_features, scaler

# ----------------------------------------------------------------------------
# 3. Model & Analysis Functions
# ----------------------------------------------------------------------------
def calculate_wcss(scaled_data, max_k=10):
    """Calculates Within-Cluster Sum of Squares for the Elbow Method."""
    wcss = []
    for i in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(scaled_data)
        wcss.append(kmeans.inertia_)
    return wcss

def perform_clustering(scaled_data, k):
    """Performs K-Means clustering and returns the model and labels."""
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_data)
    return kmeans, clusters

def generate_insights(df, features, cluster_col='cluster'):
    """Generates automated insights comparing cluster means to overall means."""
    insights = []
    cluster_means = df.groupby(cluster_col)[features].mean()
    overall_means = df[features].mean()
    
    for cluster in cluster_means.index:
        desc = f"**Cluster {cluster}** Profile:\n\n"
        for feature in features:
            val = cluster_means.loc[cluster, feature]
            avg = overall_means[feature]
            feature_name = feature.replace('_', ' ').title()
            if val > avg * 1.15:
                desc += f"**High** {feature_name} (Avg: {val:.1f})\n"
            elif val < avg * 0.85:
                desc += f"**Low** {feature_name} (Avg: {val:.1f})\n"
            else:
                desc += f"**Average** {feature_name} (Avg: {val:.1f})\n"
        insights.append(desc)
    return insights

# ----------------------------------------------------------------------------
# 4. Main UI Application
# ----------------------------------------------------------------------------
def main():
    # --- Sidebar ---
    with st.sidebar:
        st.markdown("<h2 style='text-align: center;'> Settings & Filters</h2>", unsafe_allow_html=True)
        st.markdown("---")
        
        # Feature Selection
        st.subheader("1. Select Features")
        available_features = ['age', 'annual_income', 'spending_score']
        selected_features = st.multiselect(
            "Choose dimensions for clustering:",
            available_features,
            default=['annual_income', 'spending_score']
        )
        
        if len(selected_features) < 2:
            st.warning("⚠️ Please select at least 2 features.")
            st.stop()
            
        # K selection
        st.subheader("2. Number of Clusters (k)")
        k = st.slider("Select K", min_value=2, max_value=10, value=5, step=1)
        
        st.markdown("---")
        
        # User Prediction Form
        st.subheader("Predict Customer Segment")
        st.write("Enter custom values to see which cluster they belong to:")
        with st.form("predict_form"):
            input_data = {}
            for feature in selected_features:
                if feature == 'age':
                    input_data[feature] = st.number_input("Age", min_value=18, max_value=100, value=30)
                elif feature == 'annual_income':
                    input_data[feature] = st.number_input("Annual Income (k$)", min_value=0, max_value=200, value=50)
                elif feature == 'spending_score':
                    input_data[feature] = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50)
                    
            predict_submitted = st.form_submit_button("Predict Cluster")

    # --- Main Content Area ---
    st.markdown("<h1 class='main-header'>Mall Customer Segmentation</h1>", unsafe_allow_html=True)
    st.markdown("Discover actionable insights from customer behavior using machine learning and K-Means clustering. "
                "Adjust settings in the sidebar to see real-time updates.")
    
    # Load and process data
    with st.spinner("Loading and processing data..."):
        time.sleep(0.4) # Simulate loading for visual polish
        df = load_data()
        scaled_data, scaler = preprocess_data(df, selected_features)
        kmeans, clusters = perform_clustering(scaled_data, k)
        
        df_clustered = df.copy()
        df_clustered['cluster'] = clusters
        # Convert cluster to string for categorical coloring in Plotly
        df_clustered['cluster_name'] = 'Cluster ' + df_clustered['cluster'].astype(str)

    # --- KPIs Dashboard ---
    st.markdown("### Key Performance Indicators")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric(label="Total Customers", value=len(df))
    with kpi2:
        st.metric(label="Avg Annual Income", value=f"${df['annual_income'].mean():.1f}k")
    with kpi3:
        st.metric(label="Avg Spending Score", value=f"{df['spending_score'].mean():.1f}")
    with kpi4:
        st.metric(label="Selected Clusters (k)", value=k)
        
    st.markdown("<br>", unsafe_allow_html=True)

    # --- Visualizations Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs(["Cluster Visualization", "Elbow Method", "Data & Export", "Cluster Profiling"])
    
    with tab1:
        st.markdown("### Interactive Cluster Plot")
        if len(selected_features) == 2:
            fig = px.scatter(
                df_clustered, 
                x=selected_features[0], 
                y=selected_features[1], 
                color='cluster_name',
                hover_data=['customer_id', 'gender', 'age'],
                title=f"2D Customer Segments (k={k})",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                template="plotly_white"
            )
            fig.update_traces(marker=dict(size=12, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
            fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), hovermode="closest")
            st.plotly_chart(fig, width="stretch")
            
        elif len(selected_features) == 3:
            fig = px.scatter_3d(
                df_clustered, 
                x=selected_features[0], 
                y=selected_features[1], 
                z=selected_features[2],
                color='cluster_name',
                hover_data=['customer_id', 'gender'],
                title=f"3D Customer Segments (k={k})",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                template="plotly_white"
            )
            fig.update_traces(marker=dict(size=6, opacity=0.8, line=dict(width=0.5, color='DarkSlateGrey')))
            fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, width="stretch")
            
        # Insights Panel
        st.markdown("### Automated Insights & Business Recommendations")
        insights = generate_insights(df_clustered, selected_features, 'cluster')
        
        # Display insights in a dynamic grid
        cols = st.columns(min(3, k))
        for i, insight in enumerate(insights):
            col_idx = i % min(3, k)
            with cols[col_idx]:
                st.info(insight)
                
    with tab2:
        st.markdown("### Optimal k Selection: The Elbow Method")
        st.markdown("The **Elbow Method** calculates the Within-Cluster Sum of Squares (WCSS). "
                    "The optimal number of clusters is usually at the 'elbow' point of the graph, "
                    "where adding another cluster doesn't significantly improve the variance explained.")
        
        with st.spinner("Calculating WCSS..."):
            wcss = calculate_wcss(scaled_data, max_k=10)
            
        fig_elbow = px.line(
            x=range(1, 11), 
            y=wcss, 
            markers=True, 
            labels={'x': 'Number of Clusters (k)', 'y': 'WCSS'},
            title="Elbow Method Curve",
            template="plotly_white"
        )
        fig_elbow.update_traces(line=dict(color="#FF6B6B", width=3), marker=dict(size=8, color="#4ECDC4"))
        fig_elbow.add_vline(x=k, line_dash="dash", line_color="#2c3e50", annotation_text=f"Selected k={k}")
        st.plotly_chart(fig_elbow, width="stretch")
        
    with tab3:
        st.markdown("### Clustered Dataset")
        st.dataframe(df_clustered.style.background_gradient(cmap='viridis', subset=['annual_income', 'spending_score']), 
                     width="stretch")
        
        # Download button
        csv = df_clustered.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Dataset with Clusters (CSV)",
            data=csv,
            file_name=f'customer_segments_k{k}.csv',
            mime='text/csv',
        )

    with tab4:
        st.markdown("### Cluster Distributions")
        st.write("Analyze how features vary across different clusters.")
        for feature in selected_features:
            fig_box = px.box(
                df_clustered, 
                x='cluster_name', 
                y=feature, 
                color='cluster_name',
                title=f"Distribution of {feature.replace('_', ' ').title()} by Cluster",
                template="plotly_white"
            )
            st.plotly_chart(fig_box, width="stretch")
            
        st.markdown("### Cluster Sizes")
        cluster_counts = df_clustered['cluster_name'].value_counts().reset_index()
        cluster_counts.columns = ['cluster_name', 'count']
        fig_bar = px.bar(
            cluster_counts, 
            x='cluster_name', 
            y='count', 
            color='cluster_name',
            title="Number of Customers per Cluster",
            template="plotly_white"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # --- Handle Prediction Submission ---
    if predict_submitted:
        # Format input data
        input_df = pd.DataFrame([input_data])
        scaled_input = scaler.transform(input_df)
        predicted_cluster = kmeans.predict(scaled_input)[0]
        distances = kmeans.transform(scaled_input)[0]
        
        st.balloons()
        st.markdown("---")
        st.success(f"### Prediction Result: **Cluster {predicted_cluster}**")
        st.write(f"Based on the input profile, the new customer aligns with the behaviors of **Cluster {predicted_cluster}**.")
        st.write("Review the Insights Panel in the 'Cluster Visualization' tab to understand this cluster's characteristics and suggested marketing approaches.")
        
        st.markdown("#### Prediction Confidence (Distances to Cluster Centers)")
        st.write("Lower distance indicates higher similarity to the cluster profile.")
        dist_df = pd.DataFrame({'Cluster': [f'Cluster {i}' for i in range(k)], 'Distance': distances})
        dist_df = dist_df.sort_values('Distance')
        
        fig_dist = px.bar(
            dist_df,
            x='Cluster',
            y='Distance',
            title='Distance to Each Cluster Center (Lower is Closer)',
            template='plotly_white',
            color='Distance',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_dist, use_container_width=True)

if __name__ == "__main__":
    main()
