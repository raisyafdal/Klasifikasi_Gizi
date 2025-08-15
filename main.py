import streamlit as st
import pandas as pd
from datetime import datetime, date
import numpy as np

# Konfigurasi halaman
st.set_page_config(
    page_title="Klasifikasi Status Gizi",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS untuk styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .classification-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        border: none;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .classification-card:hover {
        transform: translateY(-2px);
    }
    
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem;
        border-radius: 20px;
        margin: 2rem 0;
        border: 3px solid #6366f1;
        text-align: center;
        box-shadow: 0 15px 35px rgba(99, 102, 241, 0.2);
    }
    
    .normal-status {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        border-color: #10b981;
        color: #065f46;
    }
    
    .underweight-status {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        border-color: #f59e0b;
        color: #92400e;
    }
    
    .overweight-status {
        background: linear-gradient(135deg, #fad0c4 0%, #ffd1ff 100%);
        border-color: #f97316;
        color: #9a3412;
    }
    
    .obesity-status {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        border-color: #ef4444;
        color: #991b1b;
    }
    
    .info-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-top: 6.25rem;
        border: none;
        box-shadow: 0 5px 15px rgba(168, 237, 234, 0.3);
    }
    
    .metric-card {
        background-color: #f8fafc;
        border: 2px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
            
    @media (max-width: 480px), (max-width: 640px) {
    .info-section {
        margin-top: 1.5rem;
    }
    }

</style>
""", unsafe_allow_html=True)

def calculate_age_months(birth_date, measure_date):
    """Menghitung umur dalam bulan"""
    age_years = measure_date.year - birth_date.year
    age_months = measure_date.month - birth_date.month
    
    if measure_date.day < birth_date.day:
        age_months -= 1
    
    total_months = age_years * 12 + age_months
    return total_months

def calculate_bmi(weight, height):
    """Menghitung BMI/IMT"""
    height_m = height / 100  # convert cm to m
    bmi = weight / (height_m ** 2)
    return round(bmi, 1)

def classify_nutrition_status_zscore(age_months):
    """Klasifikasi berdasarkan Z-Score untuk usia 5-14 tahun"""
    if 60 <= age_months <= 168:  # 5-14 tahun
        return "z_score"
    else:
        return None

def classify_nutrition_status_bmi(age_months):
    """Klasifikasi berdasarkan IMT untuk usia 15 tahun ke atas"""
    if age_months >= 180:  # 15 tahun ke atas
        return "bmi"
    else:
        return None

def get_bmi_classification(bmi):
    """Klasifikasi berdasarkan IMT"""
    if bmi < 18.5:
        return "Kurus", "underweight-status"
    elif 18.5 <= bmi < 24.9:
        return "Normal", "normal-status"
    elif 25.0 <= bmi < 27.0:
        return "Berat Badan Lebih", "overweight-status"
    else:
        return "Obesitas", "obesity-status"

# Header utama
st.markdown("""
<div class="main-header">
    <h1>Aplikasi Klasifikasi Status Gizi</h1>
    <p>Sistem Penilaian Status Gizi Berdasarkan Z-Score dan IMT</p>
</div>
""", unsafe_allow_html=True)

# Sidebar untuk input
st.sidebar.header("Input Data Pengukuran")

with st.sidebar:
    # Input data
    jk = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
    bb = st.number_input("Berat Badan (kg)", min_value=1.0, max_value=200.0, value=50.0, step=0.1)
    tb = st.number_input("Tinggi Badan (cm)", min_value=50.0, max_value=250.0, value=160.0, step=0.1)
    tgl_lahir = st.date_input("Tanggal Lahir", value=date(2010, 1, 1))
    tgl_ukur = st.date_input("Tanggal Ukur", value=date.today())
    
    submit_button = st.button("Analisis Status Gizi", type="primary")

# Konten utama
col1, col2 = st.columns([2, 1])

with col1:
    # Informasi Klasifikasi
    st.header("Informasi Klasifikasi Status Gizi")
    
    # Klasifikasi Z-Score
    st.markdown("""
    <div class="classification-card">
        <h3>Klasifikasi Z-Score (Usia 5-14 Tahun)</h3>
        <table style="width:100%; border-collapse: collapse;">
            <tr style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
                <th style="border: 1px solid #ddd; padding: 12px; text-align: left;">Klasifikasi</th>
                <th style="border: 1px solid #ddd; padding: 12px; text-align: left;">Z-Score</th>
            </tr>
            <tr>
                <td style="border: 1px solid #ddd; padding: 10px;">Kurus</td>
                <td style="border: 1px solid #ddd; padding: 10px;">-3 SD sd < -2 SD</td>
            </tr>
            <tr>
                <td style="border: 1px solid #ddd; padding: 10px;">Normal</td>
                <td style="border: 1px solid #ddd; padding: 10px;">-2 SD sd +1 SD</td>
            </tr>
            <tr>
                <td style="border: 1px solid #ddd; padding: 10px;">Berat Badan Lebih</td>
                <td style="border: 1px solid #ddd; padding: 10px;">+1 SD sd +2 SD</td>
            </tr>
            <tr>
                <td style="border: 1px solid #ddd; padding: 10px;">Obesitas</td>
                <td style="border: 1px solid #ddd; padding: 10px;">> +2 SD</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)
    
    # Klasifikasi IMT
    st.markdown("""
    <div class="classification-card">
        <h3>Klasifikasi IMT (Usia 15 Tahun ke Atas)</h3>
        <table style="width:100%; border-collapse: collapse;">
            <tr style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
                <th style="border: 1px solid #ddd; padding: 12px; text-align: left;">Klasifikasi</th>
                <th style="border: 1px solid #ddd; padding: 12px; text-align: left;">IMT</th>
            </tr>
            <tr>
                <td style="border: 1px solid #ddd; padding: 10px;">Kurus</td>
                <td style="border: 1px solid #ddd; padding: 10px;">< 18,5</td>
            </tr>
            <tr>
                <td style="border: 1px solid #ddd; padding: 10px;">Normal</td>
                <td style="border: 1px solid #ddd; padding: 10px;">≥ 18,5 - < 24,9</td>
            </tr>
            <tr>
                <td style="border: 1px solid #ddd; padding: 10px;">Berat Badan Lebih</td>
                <td style="border: 1px solid #ddd; padding: 10px;">≥ 25,0 - < 27,0</td>
            </tr>
            <tr>
                <td style="border: 1px solid #ddd; padding: 10px;">Obesitas</td>
                <td style="border: 1px solid #ddd; padding: 10px;">≥ 27,0</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

with col2:
    # Info tambahan
    st.markdown("""
    <div class="info-section">
        <h4>Informasi Penting</h4>
        <ul>
            <li><strong>Z-Score:</strong> Digunakan untuk anak usia 5-14 tahun</li>
            <li><strong>IMT:</strong> Digunakan untuk remaja dan dewasa usia 15+ tahun</li>
            <li><strong>SD:</strong> Standard Deviasi</li>
            <li><strong>IMT:</strong> Indeks Massa Tubuh (BMI)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Hasil analisis
if submit_button:
    # Validasi tanggal
    if tgl_lahir >= tgl_ukur:
        st.error("Tanggal lahir tidak boleh sama dengan atau setelah tanggal ukur!")
    else:
        # Hitung umur dalam bulan
        age_months = calculate_age_months(tgl_lahir, tgl_ukur)
        age_years = age_months / 12
        
        # Hitung IMT
        bmi = calculate_bmi(bb, tb)
        
        # Tentukan metode klasifikasi
        classification_method = None
        if 60 <= age_months <= 168:  # 5-14 tahun
            classification_method = "z_score"
        elif age_months >= 180:  # 15 tahun ke atas
            classification_method = "bmi"
        
        # Tampilkan hasil
        st.header("Hasil Analisis")
        
        # Data dasar
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Jenis Kelamin", jk)
        with col2:
            st.metric("Umur", f"{age_years:.1f} tahun")
        with col3:
            st.metric("Berat Badan", f"{bb} kg")
        with col4:
            st.metric("Tinggi Badan", f"{tb} cm")
        
        st.metric("IMT/BMI", f"{bmi}")
        
        # Hasil klasifikasi
        if classification_method == "bmi":
            status, status_class = get_bmi_classification(bmi)
            method_name = "IMT (Indeks Massa Tubuh)"
            
            st.markdown(f"""
            <div class="result-card {status_class}">
                <h3>Hasil Klasifikasi</h3>
                <h2>{status}</h2>
                <p><strong>Metode:</strong> {method_name}</p>
                <p><strong>IMT:</strong> {bmi}</p>
                <p><strong>Kategori Usia:</strong> 15 tahun ke atas</p>
            </div>
            """, unsafe_allow_html=True)
            
        elif classification_method == "z_score":
            st.markdown(f"""
            <div class="result-card">
                <h3>Hasil Klasifikasi</h3>
                <h2>Memerlukan Data Referensi Z-Score</h2>
                <p><strong>Metode:</strong> Z-Score</p>
                <p><strong>IMT:</strong> {bmi}</p>
                <p><strong>Kategori Usia:</strong> 5-14 tahun</p>
                <p><em>Untuk klasifikasi akurat pada usia ini, diperlukan data referensi Z-Score WHO/nasional</em></p>
            </div>
            """, unsafe_allow_html=True)
            
        else:
            st.markdown(f"""
            <div class="result-card overweight-status">
                <h3>Peringatan</h3>
                <h2>Usia di Luar Rentang Klasifikasi</h2>
                <p><strong>Umur saat ini:</strong> {age_years:.1f} tahun</p>
                <p><strong>IMT:</strong> {bmi}</p>
                <p><em>Sistem ini hanya dapat mengklasifikasi untuk usia 5 tahun ke atas</em></p>
            </div>
            """, unsafe_allow_html=True)
        
        # Rekomendasi
        st.markdown("""
        <div class="info-section">
            <h4>Rekomendasi</h4>
            <p>Konsultasikan hasil ini dengan tenaga kesehatan profesional untuk mendapatkan saran yang tepat mengenai status gizi dan kesehatan.</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>Aplikasi Klasifikasi Status Gizi | Dikembangkan untuk Analisis Kesehatan</p>
</div>
""", unsafe_allow_html=True)