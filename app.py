import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score, silhouette_score
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
import joblib
from io import BytesIO
import numpy as np

# Fungsi hitung WCSS untuk Elbow
def get_elbow_wcss(data, max_k=10):
    wcss = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    return wcss

# Fungsi train KMeans dengan jumlah cluster tertentu
def train_kmeans(data, n_clusters):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(data)
    data = data.copy()
    data['Cluster'] = labels
    return model, data

# Konfigurasi halaman
st.set_page_config(page_title="📊 Analisis Supermarket", layout="wide")
st.title("📊 Analisis Penjualan Supermarket Berdasarkan Produk dan Kota")

# Sidebar Navigation
st.sidebar.title("Navigasi")
page = st.sidebar.selectbox("Pilih Halaman", [
    "🗂️ Data Awal",
    "📊 Hasil Clustering & Prediksi",
    "🔬 Normalisasi",
    "📈Silhouette",
])

# Load Data
df_raw = pd.read_csv("dataset/supermarket_Sales.csv")

if page == "🗂️ Data Awal":
    st.title("🗂️ Tabel Data Awal")
    st.write("Berikut adalah data asli sebelum dilakukan preprocessing:")
    st.dataframe(df_raw)

elif page == "📊 Hasil Clustering & Prediksi":
    st.subheader("📊 Hasil Clustering dan Prediksi Regresi")

    # Load data hasil clustering dan model regresi
    df = pd.read_csv("cluster_label.csv")
    model_regresi = joblib.load("regresi_model.pkl")

    # Tambahkan deskripsi Popularity
    df['Popularity_Desc'] = df['Popularity'].map({
        0: "Sering Diminati",
        1: "Diminati",
        2: "Kurang Diminati"
    })

    # Buat fitur regresi kembali (harus sama dengan saat training)
    df_model = pd.get_dummies(df[['City', 'Product line', 'Quantity']], drop_first=True)
    df['Predicted Income'] = model_regresi.predict(df_model)

    # Evaluasi Regresi
    target = df['gross income']
    mae = mean_absolute_error(target, df['Predicted Income'])
    r2 = r2_score(target, df['Predicted Income'])

    # Sidebar filter kota
    city = st.sidebar.selectbox("Pilih Kota", df['City'].unique())
    filtered = df[df['City'] == city]

    # Tabel hasil prediksi dan cluster
    st.subheader(f"📍 Data Penjualan di Kota: {city}")
    st.write("Keterangan Popularity: 0 = Sering Diminati, 1 = Diminati, 2 = Kurang Diminati")
    st.dataframe(filtered[['Product line', 'Quantity', 'Popularity', 'Popularity_Desc', 'gross income', 'Predicted Income']])

    # Visualisasi Scatter Plot Prediksi vs Aktualisasi Gross Income
    st.subheader("🔹 Visualisasi Prediksi vs Aktualisasi Gross Income")
    fig_pred_act, ax_pred_act = plt.subplots(figsize=(10, 5))
    sns.scatterplot(x=target, y=df['Predicted Income'], ax=ax_pred_act)
    ax_pred_act.plot([target.min(), target.max()], [target.min(), target.max()], 'r--')  # garis ideal
    ax_pred_act.set_xlabel("Gross Income Aktual")
    ax_pred_act.set_ylabel("Gross Income Prediksi")
    ax_pred_act.set_title("Prediksi vs Aktual Gross Income")
    st.pyplot(fig_pred_act)

    # Visualisasi Popularitas
    st.subheader("🔹 Visualisasi Popularitas Produk")
    fig, ax = plt.subplots(figsize=(10, 3))
    sns.barplot(data=filtered, x='Product line', y='Quantity', hue='Popularity', ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Visualisasi Prediksi
    st.subheader("🔹 Prediksi Total Keuntungan Produk")
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    sns.barplot(data=filtered, x='Product line', y='Predicted Income', ax=ax2)
    plt.xticks(rotation=43)
    st.pyplot(fig2)

    # Export ke Excel
    st.subheader("🔹 Export Data Cluster ke Excel")

    def to_excel(df):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Cluster_Data')
        return output.getvalue()

    excel_data = to_excel(df)
    st.download_button(
        label="📥 Download Data sebagai Excel",
        data=excel_data,
        file_name='cluster_data.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

elif page == "🔬 Normalisasi":
    st.subheader("🔍 Hasil Normalisasi Jumlah Penjualan Produk per Kota")

    grouped = df_raw.groupby(['City', 'Product line']).agg({
        'Quantity': 'sum'
    }).reset_index()

    scaler = MinMaxScaler()
    grouped['Quantity_norm'] = scaler.fit_transform(grouped[['Quantity']])

    normalisasi_df = grouped[['Product line', 'City', 'Quantity', 'Quantity_norm']]

    styled_df = normalisasi_df.style\
        .format({"Quantity": "{:.0f}", "Quantity_norm": "{:.4f}"})\
        .set_properties(**{'text-align': 'center'})\
        .set_table_styles([
            {'selector': 'th', 'props': [('text-align', 'center')]},
            {'selector': 'td', 'props': [('text-align', 'center')]},
        ])

    st.dataframe(styled_df, use_container_width=True)

elif page == "📈Silhouette":
    st.subheader("📈Evaluasi Elbow dan Silhouette Coefficient")

    # Siapkan data grouped dan normalisasi
    grouped = df_raw.groupby(['City', 'Product line']).agg({'Quantity': 'sum'}).reset_index()
    scaler = MinMaxScaler()
    quantity_scaled = scaler.fit_transform(grouped[['Quantity']])
    df_scaled = pd.DataFrame(quantity_scaled, columns=['Quantity_norm'])

    # Slider untuk memilih jumlah cluster harus dipanggil terlebih dahulu
    n_clusters = st.slider("Pilih jumlah cluster", 2, 10, 3)

    # Plot Elbow Method sesuai dengan nilai slider
    wcss = get_elbow_wcss(df_scaled.values, max_k=n_clusters)
    fig_elbow, ax_elbow = plt.subplots(figsize=(10, 4))
    ax_elbow.plot(range(1, n_clusters + 1), wcss, marker='o')
    ax_elbow.set_title('Metode Elbow')
    ax_elbow.set_xlabel('Jumlah Cluster')
    ax_elbow.set_ylabel('WCSS (Within-Cluster Sum of Squares)')
    ax_elbow.grid(True)
    st.pyplot(fig_elbow)

    # Jalankan clustering dengan jumlah cluster yang dipilih
    kmeans_model, df_clustered = train_kmeans(df_scaled.copy(), n_clusters)

    # Tambahkan label cluster ke data grouped asli
    grouped['Cluster'] = df_clustered['Cluster']

    # Hitung silhouette score (hanya jika cluster > 1)
    try:
        silhouette = silhouette_score(df_scaled, grouped['Cluster'])
    except:
        silhouette = None

    st.write(f"Jumlah cluster yang dipilih: {n_clusters}")
    if silhouette is not None:
        st.write(f"Silhouette Coefficient: {silhouette:.3f}")
    else:
        st.warning("Silhouette Coefficient tidak bisa dihitung (cluster kurang dari 2)")

# 12. Feature Importance Per Kota
st.subheader("🏙️ Feature Importance per Kota")

try:
    df_clustered = pd.read_csv("cluster_label.csv")
    cities = df_clustered['City'].unique()

    for city in cities:
        st.markdown(f"### 📍 Kota: {city}")

        # Filter data untuk kota tertentu
        df_city = df_clustered[df_clustered['City'] == city]

        # Buat fitur dan target
        X_city = pd.get_dummies(df_city[['Product line', 'Quantity']], drop_first=True)
        y_city = df_city['gross income']

        # Skip jika data terlalu sedikit
        if len(X_city) < 2:
            st.warning(f"❌ Data terlalu sedikit di kota {city}, dilewati...")
            continue

        # Train model Random Forest
        model_city = RandomForestRegressor(random_state=42)
        model_city.fit(X_city, y_city)

        # Ambil feature importance
        importances = model_city.feature_importances_
        feature_names = X_city.columns

        # Buat DataFrame visualisasi
        feat_imp_df = pd.DataFrame({
            'Fitur': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        # Visualisasi
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=feat_imp_df, x='Importance', y='Fitur', palette='crest', ax=ax)
        ax.set_title(f"🔍 Feature Importance - Kota {city}")
        ax.set_xlabel("Tingkat Kepentingan")
        ax.set_ylabel("Fitur")
        st.pyplot(fig)

except Exception as e:
    st.error(f"❌ Gagal menghitung feature importance per kota: {e}")
