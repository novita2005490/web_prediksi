import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from io import BytesIO

# Fungsi untuk memuat model SARIMA terbaik dari file
def load_model(filename):
    with open(filename, 'rb') as pkl_file:
        model = pickle.load(pkl_file)
    return model

# Fungsi untuk memprediksi 120 bulan ke depan
def predict_future(model, start_date, steps):
    predictions = model.get_forecast(steps=steps).predicted_mean
    future_dates = pd.date_range(start=start_date, periods=steps, freq='M')
    return predictions, future_dates

# Fungsi untuk memuat dan mengubah data dari file Excel
def load_and_resample_data(filename):
    dataset = pd.read_excel(filename, index_col=0, parse_dates=True)
    monthly_data = dataset.resample('M').mean()
    return monthly_data

# Fungsi untuk mengunduh data sebagai CSV
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# Setup Streamlit
st.title('📈 Prediksi Volume Produksi Perikanan Tangkap Laut')
st.markdown('## PPP Tegalsari Kota Tegal')
st.markdown('### Prediksi Volume Produksi untuk 10 Tahun ke Depan')

# Memuat model SARIMA terbaik
best_model_fit = load_model('project/best_sarima_model.pkl')

# Tanggal awal untuk prediksi
start_date = pd.Timestamp('2023-06-30')

# Prediksi 120 bulan ke depan
predictions, future_dates = predict_future(best_model_fit, start_date, 120)

# Memuat data asli dari file Excel dan meresample ke bulanan
monthly_data = load_and_resample_data('project/dataset.xlsx')

# Membuat DataFrame untuk tabel di Streamlit
pred_df = pd.DataFrame({
    'Tanggal': future_dates,
    'Volume Produksi (ton)': predictions
})

# Membulatkan angka pada DataFrame menjadi dua angka di belakang koma dan menghilangkan ".00"
pred_df['Volume Produksi (ton)'] = pred_df['Volume Produksi (ton)'].apply(lambda x: f"{x:.2f}".rstrip('0').rstrip('.'))

# Komponen Streamlit
year_to_predict = st.slider('Tentukan Bulan:', min_value=1, max_value=120, value=24)
if st.button('Predict'):
    # Visualisasi hasil prediksi
    fig, ax = plt.subplots(figsize=(12, 6))

    # Menentukan rentang data untuk visualisasi, mulai dari 2023-01-31
    visualization_start_date = pd.Timestamp('2023-01-31')
    filtered_monthly_data = monthly_data[monthly_data.index >= visualization_start_date]

    sns.lineplot(data=filtered_monthly_data, x=filtered_monthly_data.index, y='volume_produksi', ax=ax, label='Data Asli', color='royalblue')
    sns.lineplot(x=future_dates[:year_to_predict], y=predictions[:year_to_predict], ax=ax, label='Prediksi', linestyle='--', color='orange')
    
    ax.set_xlabel('Bulan', fontsize=12)
    ax.set_ylabel('Volume Produksi (ton)', fontsize=12)
    ax.set_title(f"Prediksi Volume Produksi Perikanan Tangkap {year_to_predict} Bulan ke Depan", fontsize=14)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

    # Menampilkan plot di Streamlit
    st.pyplot(fig)

    # Menampilkan tabel prediksi di Streamlit
    st.markdown('### Tabel Hasil Prediksi' )
    st.table(pred_df.head(year_to_predict).reset_index(drop=True))

    # Menambahkan tombol untuk mengunduh tabel prediksi
    csv = convert_df_to_csv(pred_df.head(year_to_predict).reset_index(drop=True))
    st.download_button(
        label="Download tabel prediksi sebagai CSV",
        data=csv,
        file_name='prediksi_volume_produksi.csv',
        mime='text/csv',
    )

# Menambahkan gaya pada aplikasi
st.markdown(
    """
    <style>
    .stButton button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
    }
    .stSlider .st-cg {
        color: #FF4B4B;
    }
    .stMarkdown h2 {
        color: #FF4B4B;
    }
    </style>
    """,
    unsafe_allow_html=True
)
