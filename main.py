import streamlit as st
import pandas as pd
from datetime import datetime, date
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Klasifikasi Status Gizi",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
        color: white;
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
        color: white;
    }
    
    .metric-card {
        background-color: #f8fafc;
        border: 2px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    
    .prediction-confidence {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        color: white;
        text-align: center;
    }
            
    @media (max-width: 480px), (max-width: 640px) {
    .info-section {
        margin-top: 1.5rem;
    }
    }

</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_scaler():
    """Load model KNN dan scaler yang sudah dilatih"""
    try:
        with open('model_child.h5', 'rb') as file:
            model_child = joblib.load(file)
       
        with open('model_teen.h5', 'rb') as file:
            model_teen = joblib.load(file)
        
        try:
            with open('scaler_child.pkl', 'rb') as file:
                scaler_child = joblib.load(file)
           
            with open('scaler_teen.pkl', 'rb') as file:
                scaler_teen = joblib.load(file)
        except FileNotFoundError:
            st.warning("Scaler tidak ditemukan. Menggunakan scaler default.")
            scaler = StandardScaler()
        
        return model_child, model_teen, scaler_child, scaler_teen
    except FileNotFoundError:
        st.error("Model knn_model.pkl tidak ditemukan. Pastikan file model ada di direktori yang sama.")
        return None, None

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

lms_lk = pd.read_excel("laki_laki.xlsx")
lms_pr = pd.read_excel("perempuan.xlsx")

lms_lk.rename(columns={'Month': 'Umur_Bulan'}, inplace=True)
lms_pr.rename(columns={'Month': 'Umur_Bulan'}, inplace=True)

def calculate_baz(bmi, age_months, jk):
    """
    Menghitung BAZ (BMI-for-Age Z-score)
    """

    jk = 1 if jk == "Laki-laki" else 0

    if jk == 1:
        lms_row = lms_lk[lms_lk['Umur_Bulan'] == age_months]
    else:
        lms_row = lms_pr[lms_pr['Umur_Bulan'] == age_months]

    if lms_row.empty:
        print(f"Data LMS untuk umur ini tidak ditemukan untuk kategori {jk}.")
        return 0
    else:
        L = lms_row['L'].values[0]
        M = lms_row['M'].values[0]
        S = lms_row['S'].values[0]

        if L != 0:
            baz = ((bmi / M) ** L - 1) / (L * S)
        else:
            baz = np.log(bmi / M) / S
        
    return round(baz, 2)

def prepare_features(jk, tb, bb, age_years, age_months, bmi, baz):
    """Menyiapkan fitur untuk prediksi model"""
    gender_encoded = 1 if jk == "Laki-laki" else 0
    features = np.array([[gender_encoded, tb, bb, age_years, age_months, bmi, baz]])
    
    return features
def get_status_class(prediction):
    """Mendapatkan class CSS berdasarkan prediksi"""
    status_mapping = {
        "Normal": "normal-status",
        "Kurus": "underweight-status", 
        "Berat Badan Lebih": "overweight-status",
        "Obesitas": "obesity-status",
    }
    return status_mapping.get(prediction, "result-card")

st.markdown("""
<div class="main-header">
    <h1>Aplikasi Klasifikasi Status Gizi dengan AI</h1>
    <p>Sistem Prediksi Status Gizi Menggunakan Machine Learning (KNN)</p>
</div>
""", unsafe_allow_html=True)

model_child, model_teen, scaler_child, scaler_teen = load_model_and_scaler()

st.sidebar.header("Input Data Pengukuran")

with st.sidebar:
    jk = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
    bb = st.number_input("Berat Badan (kg)", min_value=1.0, max_value=200.0, value=50.0, step=1.0)
    tb = st.number_input("Tinggi Badan (cm)", min_value=50.0, max_value=250.0, value=160.0, step=1.0)
    tgl_lahir = st.date_input(
        "Tanggal Lahir",
        value=date(2010, 1, 1),
        min_value=date(1900, 1, 1), 
        max_value=date.today()
    )
    tgl_ukur = st.date_input("Tanggal Ukur", value=date.today())
    
    st.markdown("---")
    
    use_ai_prediction = st.checkbox("Gunakan Prediksi AI (Model KNN)", value=True)
    
    submit_button = st.button("Analisis Status Gizi", type="primary")

col1, col2 = st.columns([2, 1])

with col1:
    st.header("Informasi Klasifikasi Status Gizi")
    
    st.markdown("""
    <div class="classification-card">
        <h3>Klasifikasi Z-Score 5 Tahun ( 60 - 68 Bulan )</h3>
        <table style="width:100%; border-collapse: collapse;">
            <tr style="background: rgba(255,255,255,0.2); color: white;">
                <th style="border: 1px solid rgba(255,255,255,0.3); padding: 12px; text-align: left;">Klasifikasi</th>
                <th style="border: 1px solid rgba(255,255,255,0.3); padding: 12px; text-align: left;">Z-Score</th>
            </tr>
            <tr style="color: white;">
                <td style="border: 1px solid rgba(255,255,255,0.3); padding: 10px;">Kurus</td>
                <td style="border: 1px solid rgba(255,255,255,0.3); padding: 10px;">-3 SD sd < -2 SD</td>
            </tr>
            <tr style="color: white;">
                <td style="border: 1px solid rgba(255,255,255,0.3); padding: 10px;">Normal</td>
                <td style="border: 1px solid rgba(255,255,255,0.3); padding: 10px;">-2 SD sd +1 SD</td>
            </tr>
            <tr style="color: white;">
                <td style="border: 1px solid rgba(255,255,255,0.3); padding: 10px;">Berat Badan Lebih</td>
                <td style="border: 1px solid rgba(255,255,255,0.3); padding: 10px;">+1 SD sd +2 SD</td>
            </tr>
            <tr style="color: white;">
                <td style="border: 1px solid rgba(255,255,255,0.3); padding: 10px;">Obesitas</td>
                <td style="border: 1px solid rgba(255,255,255,0.3); padding: 10px;">> +2 SD</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="classification-card">
        <h3>Klasifikasi IMT 17 - 20 Tahun</h3>
        <table style="width:100%; border-collapse: collapse;">
            <tr style="background: rgba(255,255,255,0.2); color: white;">
                <th style="border: 1px solid rgba(255,255,255,0.3); padding: 12px; text-align: left;">Klasifikasi</th>
                <th style="border: 1px solid rgba(255,255,255,0.3); padding: 12px; text-align: left;">IMT</th>
            </tr>
            <tr style="color: white;">
                <td style="border: 1px solid rgba(255,255,255,0.3); padding: 10px;">Kurus</td>
                <td style="border: 1px solid rgba(255,255,255,0.3); padding: 10px;">17 - < 18,5</td>
            </tr>
            <tr style="color: white;">
                <td style="border: 1px solid rgba(255,255,255,0.3); padding: 10px;">Normal</td>
                <td style="border: 1px solid rgba(255,255,255,0.3); padding: 10px;">≥ 18,5 - < 25</td>
            </tr>
            <tr style="color: white;">
                <td style="border: 1px solid rgba(255,255,255,0.3); padding: 10px;">Berat Badan Lebih</td>
                <td style="border: 1px solid rgba(255,255,255,0.3); padding: 10px;">≥ 25,0 - < 27</td>
            </tr>
            <tr style="color: white;">
                <td style="border: 1px solid rgba(255,255,255,0.3); padding: 10px;">Obesitas</td>
                <td style="border: 1px solid rgba(255,255,255,0.3); padding: 10px;">≥ 27</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-section">
        <h4>Model AI Information</h4>
        <ul>
            <li><strong>Model:</strong> K-Nearest Neighbors (KNN)</li>
            <li><strong>Fitur:</strong> 7 parameter antropometri</li>
            <li><strong>Preprocessing:</strong> StandardScaler normalization</li>
            <li><strong>Target:</strong> 4 kategori status gizi</li>
        </ul>
        <br>
        <h4>Informasi Klasifikasi</h4>
        <ul>
            <li><strong>Z-Score:</strong> Usia 5 Tahun ( 60-68 bulan )</li>
            <li><strong>IMT:</strong> Usia 17 - 20 tahun</li>
            <li><strong>BAZ:</strong> BMI-for-Age Z-score</li>
            <li><strong>SD:</strong> Standard Deviasi</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

if submit_button:
    if tgl_lahir >= tgl_ukur:
        st.error("Tanggal lahir tidak boleh sama dengan atau setelah tanggal ukur!")
    else:
        # Hitung parameter dasar
        age_months = calculate_age_months(tgl_lahir, tgl_ukur)
        age_years = round(age_months / 12, 1)
        bmi = calculate_bmi(bb, tb)
        baz = calculate_baz(bmi, age_months, jk)
        
        st.header("Hasil Analisis")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Jenis Kelamin", jk)
        with col2:
            st.metric("Umur", f"{age_years} tahun")
        with col3:
            st.metric("Berat Badan", f"{bb} kg")
        with col4:
            st.metric("Tinggi Badan", f"{tb} cm")

        if 204 <= age_months <= 251:
            with col5:
                st.metric("IMT", f"{bmi}")
        elif 60 <= age_months <= 68:
            st.metric("BAZ (BMI-for-Age Z-score)", f"{baz}")

        is_predict = False

        if (60 <= age_months <= 68) or (204 <= age_months <= 251):
            is_predict = True

        if is_predict and use_ai_prediction and (model_child or model_teen) is not None and (scaler_child or scaler_teen) is not None:
            try:

                if 60 <= age_months <= 68:
                    features = prepare_features(jk, tb, bb, age_years, age_months, bmi, baz)
                    features_scaled = scaler_child.transform(features)
                    prediction = model_child.predict(features_scaled)[0]
                elif 204 <= age_months <= 251:
                    features = prepare_features(jk, tb, bb, age_years, age_months, bmi, baz)
                    features_scaled = scaler_teen.transform(features)
                    prediction = model_teen.predict(features_scaled)[0]
                
                try:
                    if 60 <= age_months <= 68:
                        prediction_proba = model_child.predict_proba(features_scaled)[0]
                    elif 204 <= age_months <= 251:
                        prediction_proba = model_teen.predict_proba(features_scaled)[0]

                    confidence = max(prediction_proba) * 100
                except:
                    confidence = None
                
                label = ['Berat Badan Lebih', 'Kurus', 'Normal', 'Obesitas']

                status_class = get_status_class(label[prediction])

                if age_years > 15:
                    result_card = f"IMT:</strong> {bmi}"
                else:
                    result_card = f"<strong>BAZ:</strong> {baz}"
                
                st.markdown(f"""
                <div class="result-card {status_class}">
                    <h3>Hasil Prediksi AI</h3>
                    <h2>{label[prediction]}</h2>
                    <p><strong>Metode:</strong> Machine Learning (KNN)</p>
                    <p><strong>{result_card}</p>
                    <p><strong>Umur:</strong> {age_years} tahun ({age_months} bulan)</p>
                </div>
                """, unsafe_allow_html=True)
                
                if confidence is not None:
                    st.markdown(f"""
                    <div class="prediction-confidence">
                        <h4>Confidence Score</h4>
                        <h3>{confidence:.1f}%</h3>
                        <p>Tingkat kepercayaan model terhadap prediksi ini</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with st.expander("🔍 Detail Fitur yang Digunakan Model"):
                    gender_display = "1 (Laki-laki)" if jk == "Laki-laki" else "0 (Perempuan)"

                    if 204 <= age_months <= 251:
                    
                        feature_df = pd.DataFrame({
                            'Fitur': ['Jenis Kelamin', 'IMT'],
                            'Nilai Asli': [
                                gender_display,
                                f"{bmi}",
                            ],
                            'Nilai Normalized': [
                                f"{features_scaled[0][0]:.3f}",
                                f"{features_scaled[0][1]:.3f}",
                            ]
                        })
                        st.dataframe(feature_df, use_container_width=True)
                    elif 60 <= age_months <= 68:
                        feature_df = pd.DataFrame({
                            'Fitur': ['Jenis Kelamin', 'Tinggi Badan', 'Berat Badan', 'Umur Bulan', 'BAZ'],
                            'Nilai Asli': [
                                gender_display,
                                f"{tb} cm",
                                f"{bb} kg", 
                                f"{age_months} bulan",
                                f"{baz}"
                            ],
                            'Nilai Normalized': [
                                f"{features_scaled[0][0]:.3f}",
                                f"{features_scaled[0][1]:.3f}",
                                f"{features_scaled[0][2]:.3f}",
                                f"{features_scaled[0][3]:.3f}",
                                f"{features_scaled[0][4]:.3f}",
                            ]
                        })

                        st.dataframe(feature_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error dalam prediksi AI: {str(e)}")
                st.info("Menggunakan klasifikasi manual sebagai fallback.")
                use_ai_prediction = False
        else:
            if age_years > 15:
                result_card = f"BMI:</strong> {bmi}"
            else:
                result_card = f"<strong>BAZ:</strong> {baz}"

            st.markdown(f"""
                <div class="result-card">
                    <h3>Mohon maaf prediksi gagal di lakukan</h3>
                    <h2>Data yang anda masukan di luar kategori <br> yang di tetapkan</h2>
                    <p><strong>Metode:</strong> Machine Learning (KNN)</p>
                    <p><strong>{result_card}</p>
                    <p><strong>Umur:</strong> {age_years} tahun ({age_months} bulan)</p>
                </div>
            """, unsafe_allow_html=True)

        if 204 <= age_months <= 251:
            st.markdown("### Interpretasi IMT")

            if bmi < 18.5:
                bmi_interpretation = "Kurus ( 17 - < 18,5 )"
                bmi_class = "underweight-status"
            elif 18.5 <= bmi < 25:
                bmi_interpretation = "Normal ( ≥ 18,5 - < 25 )"
                bmi_class = "normal-status"
            elif 25 <= bmi < 27:
                bmi_interpretation = "Berat Badan Lebih ( ≥ 25,0 - < 27 )"
                bmi_class = "overweight-status"
            else:
                bmi_interpretation = "Obesitas ( ≥ 27 )"
                bmi_class = "obesity-status"

            st.markdown(f"""
            <div style="padding: 1rem; border-radius: 10px; margin: 1rem 0;" class="{bmi_class}">
                <strong>Interpretasi IMT/BMI:</strong> {bmi_interpretation}
            </div>
            """, unsafe_allow_html=True)

        elif 60 <= age_months <= 68:
            st.markdown("### Interpretasi BAZ (BMI-for-Age Z-score)")
            if baz < -2:
                baz_interpretation = "Kurus (BAZ < -2 SD)"
                baz_class = "underweight-status"
            elif -2 <= baz <= 1:
                baz_interpretation = "Normal (-2 SD ≤ BAZ ≤ +1 SD)"
                baz_class = "normal-status"
            elif 1 < baz <= 2:
                baz_interpretation = "Berat Badan Lebih (+1 SD < BAZ ≤ +2 SD)"
                baz_class = "overweight-status"
            else:
                baz_interpretation = "Obesitas (BAZ > +2 SD)"
                baz_class = "obesity-status"
        
            st.markdown(f"""
            <div style="padding: 1rem; border-radius: 10px; margin: 1rem 0;" class="{baz_class}">
                <strong>Interpretasi BAZ:</strong> {baz_interpretation}
            </div>
            """, unsafe_allow_html=True)

if model_teen is not None and model_child is not None:
    with st.expander("Informasi Detail Model"):
        st.write("**Model Remaja berhasil dimuat:**")
        st.write(f"- Tipe Model: {type(model_teen).__name__}")

        if hasattr(model_teen, 'n_neighbors'):
            st.write(f"- Jumlah Neighbors (K): {model_teen.n_neighbors}")
        if hasattr(model_teen, 'metric'):
            st.write(f"- Metrik Jarak: {model_teen.metric}")
        
        st.write("**Fitur Input Model:**")
        st.write("1. Jenis Kelamin (0: Perempuan, 1: Laki-laki)")
        st.write("2. IMT")
        
        st.write("")
        st.write("="*30)
        st.write("")

        st.write("**Model Anak berhasil dimuat:**")
        st.write(f"- Tipe Model: {type(model_child).__name__}")

        if hasattr(model_child, 'n_neighbors'):
            st.write(f"- Jumlah Neighbors (K): {model_child.n_neighbors}")
        if hasattr(model_child, 'metric'):
            st.write(f"- Metrik Jarak: {model_child.metric}")
        
        st.write("**Fitur Input Model:**")
        st.write("1. Jenis Kelamin (0: Perempuan, 1: Laki-laki)")
        st.write("2. Tinggi Badan (cm)")
        st.write("3. Berat Badan (kg)")
        st.write("4. Umur Bulan")
        st.write("5. Umur Tahun")
        st.write("6. BAZ (BMI-for-Age Z-score)")
else:
    st.warning("Model KNN tidak tersedia. Pastikan file 'knn_model.pkl' ada di direktori yang sama dengan aplikasi.")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>Aplikasi Klasifikasi Status Gizi dengan AI | Powered by Machine Learning (KNN)</p>
    <p><em>Hasil prediksi AI harus dikonfirmasi dengan tenaga kesehatan profesional</em></p>
</div>
""", unsafe_allow_html=True)

