import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="🔧 Pelatihan Model", layout="wide")
st.title("🔧 Pelatihan Model Regresi dan Clustering Penjualan Supermarket")

# 1. Load data
df = pd.read_csv("dataset/Supermarket_Sales.csv")

# 2. Aggregate
grouped = df.groupby(['City', 'Product line']).agg({
    'Quantity': 'sum',
    'Total': 'sum',
    'gross income': 'sum'
}).reset_index()

# 3. Normalisasi minmax 
scaler = MinMaxScaler()
grouped['Quantity_norm'] = scaler.fit_transform(grouped[['Quantity']])

# 4. Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
grouped['Cluster'] = kmeans.fit_predict(grouped[['Quantity_norm']])

# Label Clusters berdasarkan rata-rata Quantity
cluster_map = grouped.groupby('Cluster')['Quantity'].mean().sort_values(ascending=False)
cluster_labels = {
    cluster: label for cluster, label in zip(cluster_map.index, ['0', '1', '2'])  # 0 = Sering Diminati
}
grouped['Popularity'] = grouped['Cluster'].map(cluster_labels)

# 4. Tampilkan data hasil clustering
st.subheader("📊 Hasil Agregasi & Clustering")
st.dataframe(grouped[['City', 'Product line', 'Quantity', 'gross income', 'Popularity']])

# Visualisasi Clustering Lengkap
st.subheader("📦 Visualisasi Clustering Berdasarkan Quantity dan Product Line")

fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(
    data=grouped,
    x='Cluster',
    y='Quantity',
    hue='Product line',   # warna berdasarkan jenis produk
    style='City',         # bentuk marker berdasarkan kota
    palette='tab10',
    s=100,
    ax=ax
)
ax.set_title("📦 Visualisasi Clustering Berdasarkan Quantity dan Product Line")
ax.set_xlabel("Cluster")
ax.set_ylabel("Quantity")
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # legend di luar plot
st.pyplot(fig)
plt.close(fig)

# 5. One-Hot Encoding
X = pd.get_dummies(grouped[['City', 'Product line', 'Quantity']], drop_first=True)
y = grouped['gross income']

# 6. Sidebar untuk memilih proporsi data training
st.sidebar.title("⚙️ Pengaturan Model")
test_size = st.sidebar.slider("Proporsi Data Test (%)", min_value=10, max_value=50, value=20, step=5) / 100

# 7. Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

st.sidebar.markdown(f"📚 Data Latih: {len(X_train)} baris")
st.sidebar.markdown(f"🧪 Data Uji: {len(X_test)} baris")

# 8. Train Model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# 9. Evaluasi Model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("📈 Evaluasi Model Regresi")
col1, col2 = st.columns(2)
col1.metric("📉 MAE (Mean Absolute Error)", f"{mae:.2f}")
col2.metric("📈 R² Score", f"{r2:.2f}")

# 10. Simpan model dan hasil clustering
if st.button("💾 Simpan Model & Data"):
    joblib.dump(model, "regresi_model.pkl")
    cluster_output = grouped[['City', 'Product line', 'Quantity', 'Popularity', 'gross income']]
    cluster_output.to_csv("cluster_label.csv", index=False)
    st.success("Model dan hasil clustering berhasil disimpan!")


# # 10. Visualisasi Feature Importance
# st.subheader("🔍 Pentingnya Fitur dalam Model")
# importances = model.feature_importances_
# feature_names = X.columns

# feat_imp_df = pd.DataFrame({
#     'Fitur': feature_names,
#     'Importance': importances
# }).sort_values(by='Importance', ascending=False)

# fig, ax = plt.subplots(figsize=(10, 5))
# sns.barplot(data=feat_imp_df, x='Importance', y='Fitur', palette='viridis', ax=ax)
# ax.set_title("🔍 Feature Importance - Random Forest")
# ax.set_xlabel("Tingkat Kepentingan")
# ax.set_ylabel("Fitur")
# st.pyplot(fig)

st.subheader("🏙️ Feature Importance per Kota")

cities = grouped['City'].unique()
for city in cities:
    st.markdown(f"### 📍 Kota: {city}")
    
    # Filter data per kota
    city_data = grouped[grouped['City'] == city]
    
    # Preprocessing
    X_city = pd.get_dummies(city_data[['Product line', 'Quantity']], drop_first=True)
    y_city = city_data['gross income']
    
    if len(X_city) < 2:  # tidak cukup data
        st.warning(f"Tidak cukup data untuk kota {city}, lewati...")
        continue

    # Train model
    model_city = RandomForestRegressor(random_state=42)
    model_city.fit(X_city, y_city)
    
    # Feature importance
    importances = model_city.feature_importances_
    feature_names = X_city.columns
    importance_df = pd.DataFrame({
        'Fitur': feature_names,
        'Kepentingan': importances
    }).sort_values(by='Kepentingan', ascending=False)

    # Visualisasi
    fig, ax = plt.subplots()
    sns.barplot(x='Kepentingan', y='Fitur', data=importance_df, ax=ax, palette="crest")
    ax.set_title(f"🔍 Feature Importance - Kota {city}")
    st.pyplot(fig)

