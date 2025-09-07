import streamlit as st
import pandas as pd
from datetime import datetime, date
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import warnings
from io import BytesIO

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
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        color: white;
        text-align: center;
    }
    
    .batch-upload {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        border: none;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
            
    @media (max-width: 480px), (max-width: 640px) {
    .info-section {
        margin-top: 1.5rem;
    }
    }

</style>
""", unsafe_allow_html=True)

PATH_MODEL_ANAK = "model_child.h5"
PATH_MODEL_REMAJA = "model_teen.h5"
PATH_LMS_LK = "laki_laki.xlsx"
PATH_LMS_PR = "perempuan.xlsx"

label_map = {0: "Kurus", 1: "Normal", 2: "BB Lebih", 3: "Obesitas"}

@st.cache_resource
def load_models_and_lms():
    """Load model dan data LMS yang sudah dilatih"""
    try:
        model_child = joblib.load(PATH_MODEL_ANAK)
        model_teen = joblib.load(PATH_MODEL_REMAJA)
        
        lms_lk = pd.read_excel(PATH_LMS_LK).copy()
        lms_pr = pd.read_excel(PATH_LMS_PR).copy()
        
        lms_lk.rename(columns={"Month": "Umur_Bulan"}, inplace=True)
        lms_pr.rename(columns={"Month": "Umur_Bulan"}, inplace=True)
        lms_lk["Umur_Bulan"] = lms_lk["Umur_Bulan"].astype(int)
        lms_pr["Umur_Bulan"] = lms_pr["Umur_Bulan"].astype(int)
        
        for df_ in (lms_lk, lms_pr):
            for c in ["L","M","S"]:
                df_[c] = df_[c].astype(float)
        
        lms_lk["Jenis_Kelamin"] = 1
        lms_pr["Jenis_Kelamin"] = 0
        
        lms_all = pd.concat([lms_lk, lms_pr], ignore_index=True)
        umur_min = int(lms_all["Umur_Bulan"].min())
        umur_max = int(lms_all["Umur_Bulan"].max())
        
        return model_child, model_teen, lms_all, umur_min, umur_max
    except FileNotFoundError as e:
        st.error(f"File tidak ditemukan: {str(e)}")
        return None, None, None, None, None

def _norm_sex(x):
    """Map Jenis_Kelamin → 1 untuk laki-laki, 0 untuk perempuan"""
    if pd.isna(x): return np.nan
    if isinstance(x, (int, np.integer)): return 1 if x == 1 else 0
    s = str(x).strip().upper()
    if s in ["M","MALE","L","LAKI-LAKI","LK"]: return 1
    if s in ["F","FEMALE","P","PEREMPUAN","PR"]: return 0
    try:
        return 1 if int(s) == 1 else 0
    except:
        return 1 if s.startswith('1') else 0

def tambah_imt_baz(df, lms_all, umur_min, umur_max,
                   col_jk="Jenis_Kelamin",
                   col_tb="Tinggi_Badan", 
                   col_bb="Berat_Badan",
                   col_umur="Umur_Bulan"):
    """Hitung IMT dan BAZ untuk dataset"""
    out = df.copy()
    out["IMT"] = out[col_bb] / ((out[col_tb] / 100.0) ** 2)
    
    mask_anak = out[col_umur].between(61, 68)
    
    if mask_anak.any():
        out_anak = out.loc[mask_anak].copy()
        out_anak["_SexMF"] = out_anak[col_jk].apply(_norm_sex)
        out_anak["_UmurClip"] = out_anak[col_umur].astype(int).clip(umur_min, umur_max)
        
        merged = out_anak.merge(
            lms_all[["Umur_Bulan","Jenis_Kelamin","L","M","S"]],
            how="left",
            left_on=["_UmurClip","_SexMF"],
            right_on=["Umur_Bulan","Jenis_Kelamin"]
        )
        
        L = merged["L"].astype(float)
        M = merged["M"].astype(float) 
        S = merged["S"].astype(float)
        BMI = merged["IMT"].astype(float)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            num = np.power(BMI / M, L) - 1
            baz = np.where(L != 0, num / (L * S), np.log(BMI / M) / S)
        
        out.loc[mask_anak, "BAZ"] = baz
    
    out.loc[~mask_anak, "BAZ"] = np.nan
    return out.drop(columns=["_SexMF","_UmurClip"], errors="ignore")

def _expected_features(est):
    """Ambil daftar fitur yang diharapkan estimator"""
    if hasattr(est, "feature_names_in_"):
        return list(est.feature_names_in_)
    if hasattr(est, "named_steps"):
        for step in est.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return None

def build_X_ikut_model(df, est, feats_manual=None):
    """Pilih kolom df agar sama persis seperti saat training"""
    feats = _expected_features(est)
    if feats is None:
        if feats_manual is None:
            raise ValueError("Model tidak menyimpan daftar fitur. Isi feats_manual sesuai training.")
        feats = feats_manual
    
    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom hilang untuk prediksi: {missing}\nModel mengharapkan fitur: {feats}")
    return df[feats]

def prediksi_batch(data_baru, model_child, model_teen, lms_all, umur_min, umur_max,
                   feats_anak_manual=None, feats_remaja_manual=None):
    """Prediksi batch untuk anak dan remaja"""
    df = data_baru.copy()
    df = tambah_imt_baz(df, lms_all, umur_min, umur_max)
    
    mask_anak = df["Umur_Bulan"].between(61, 68)
    mask_remaja = df["Umur_Bulan"].between(204, 251)
    
    out_parts = []
    
    if mask_anak.any():
        anak = df.loc[mask_anak].copy()
        X_anak = build_X_ikut_model(anak, model_child, feats_manual=feats_anak_manual)
        anak["Prediksi_Status"] = model_child.predict(X_anak)
        anak["Keterangan"] = anak["Prediksi_Status"].map(label_map)
        out_parts.append(anak)
    
    if mask_remaja.any():
        rem = df.loc[mask_remaja].copy()
        X_rem = build_X_ikut_model(rem, model_teen, feats_manual=feats_remaja_manual)
        rem["Prediksi_Status"] = model_teen.predict(X_rem)
        rem["Keterangan"] = rem["Prediksi_Status"].map(label_map)
        out_parts.append(rem)
    
    if not out_parts:
        raise ValueError("Tidak ada baris yang masuk rentang anak (61-68 bln) atau remaja (204-240 bln).")
    
    return pd.concat(out_parts).sort_index()

def calculate_age_months(birth_date, measure_date):
    """Menghitung umur dalam bulan"""
    age_years = measure_date.year - birth_date.year
    age_months = measure_date.month - birth_date.month
    
    if measure_date.day < birth_date.day:
        age_months -= 1
    
    total_months = age_years * 12 + age_months
    return total_months

def calculate_bmi(weight: float, height: float):
    """Hitung BMI/IMT dari berat (kg) dan tinggi (cm)"""
    if weight is None or height is None or height == 0:
        return np.nan
    height_m = height / 100
    bmi = weight / (height_m ** 2)
    return bmi

def get_lms_single(age_months: int, jk: int, lms_all):
    """Ambil nilai L, M, S dari tabel LMS untuk single prediction"""
    lms_row = lms_all[(lms_all['Umur_Bulan'] == age_months) & (lms_all['Jenis_Kelamin'] == jk)]
    
    if lms_row.empty:
        return np.nan, np.nan, np.nan
    
    return (
        lms_row['L'].values[0],
        lms_row['M'].values[0], 
        lms_row['S'].values[0],
    )

def calculate_baz_single(bmi: float, age_months: int, jk: str | int, lms_all):
    """Hitung BAZ untuk single sample"""
    if isinstance(jk, str):
        jk = 1 if jk.lower() == "laki-laki" else 0
    
    L, M, S = get_lms_single(age_months, jk, lms_all)
    
    if np.isnan(L) or np.isnan(M) or np.isnan(S):
        return np.nan
    
    if L != 0:
        baz = ((bmi / M) ** L - 1) / (L * S)
    else:
        baz = np.log(bmi / M) / S
    
    return baz

def get_status_class(prediction):
    """Mendapatkan class CSS berdasarkan prediksi"""
    status_mapping = {
        "Normal": "normal-status",
        "Kurus": "underweight-status",
        "BB Lebih": "overweight-status", 
        "Obesitas": "obesity-status",
    }
    return status_mapping.get(prediction, "result-card")

model_child, model_teen, lms_all, UMUR_MIN, UMUR_MAX = load_models_and_lms()

st.markdown("""
<div class="main-header">
    <h1>Aplikasi Klasifikasi Status Gizi dengan AI</h1>
    <p>Sistem Prediksi Status Gizi Menggunakan Machine Learning (KNN)</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Prediksi Tunggal", "Prediksi Batch"])

with tab1:
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
                <li><strong>Fitur:</strong> Parameter antropometri</li>
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
            age_months = calculate_age_months(tgl_lahir, tgl_ukur)
            age_years = round(age_months / 12, 1)
            bmi = calculate_bmi(bb, tb)
            
            if lms_all is not None:
                baz = calculate_baz_single(bmi, age_months, jk, lms_all)
            else:
                baz = np.nan
            
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
                    st.metric("IMT", f"{bmi:.2f}")
            elif 60 <= age_months <= 68:
                with col5:
                    st.metric("BAZ", f"{baz:.2f}" if not np.isnan(baz) else "N/A")
            
            if ((60 <= age_months <= 68) or (204 <= age_months <= 251)) and use_ai_prediction and model_child and model_teen:
                try:
                    gender_encoded = 1 if jk == "Laki-laki" else 0
                    single_data = pd.DataFrame([{
                        "Jenis_Kelamin": gender_encoded,
                        "Tinggi_Badan": tb,
                        "Berat_Badan": bb,
                        "Umur_Bulan": age_months
                    }])
                    
                    hasil = prediksi_batch(single_data, model_child, model_teen, lms_all, UMUR_MIN, UMUR_MAX)
                    
                    prediction = hasil["Keterangan"].iloc[0]
                    status_class = get_status_class(prediction)
                    
                    if age_months >= 204:
                        result_detail = f"<strong>IMT:</strong> {bmi:.2f}"
                    else:
                        result_detail = f"<strong>BAZ:</strong> {baz:.2f}"
                    
                    st.markdown(f"""
                    <div class="result-card {status_class}">
                        <h3>Hasil Prediksi AI</h3>
                        <h2>{prediction}</h2>
                        <p><strong>Metode:</strong> Machine Learning (KNN)</p>
                        <p>{result_detail}</p>
                        <p><strong>Umur:</strong> {age_years} tahun ({age_months} bulan)</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error dalam prediksi AI: {str(e)}")
            else:
                if not ((60 <= age_months <= 68) or (204 <= age_months <= 251)):
                    st.markdown(f"""
                        <div class="result-card">
                            <h3>Mohon maaf prediksi gagal dilakukan</h3>
                            <h2>Data yang anda masukkan di luar kategori<br>yang ditetapkan</h2>
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

with tab2:
    st.header("Prediksi Batch Status Gizi")
    
    st.markdown("""
    <div class="batch-upload">
        <h3>Upload File CSV/Excel untuk Prediksi Batch</h3>
        <p>File harus mengandung kolom: <strong>Jenis_Kelamin, Tinggi_Badan, Berat_Badan, Umur_Bulan</strong></p>
        <p>Format Jenis_Kelamin: 1 = Laki-laki, 0 = Perempuan (atau "Laki-laki"/"Perempuan")</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload file yang berisi data untuk prediksi batch"
    )
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"File berhasil diupload! {len(df)} baris data ditemukan.")
            
            st.subheader("Preview Data:")
            st.dataframe(df.head())
            
            required_cols = ["Jenis_Kelamin", "Tinggi_Badan", "Berat_Badan", "Umur_Bulan"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"Kolom yang hilang: {missing_cols}")
                st.info("Pastikan file memiliki kolom: Jenis_Kelamin, Tinggi_Badan, Berat_Badan, Umur_Bulan")
            else:
                if st.button("Mulai Prediksi Batch", type="primary"):
                    try:
                        with st.spinner("Sedang memproses prediksi batch..."):
                            hasil_batch = prediksi_batch(df, model_child, model_teen, lms_all, UMUR_MIN, UMUR_MAX)
                        
                        st.success("Prediksi batch selesai!")
                        
                        st.subheader("Hasil Prediksi Batch:")
                        
                        display_cols = ["Jenis_Kelamin", "Tinggi_Badan", "Berat_Badan", "Umur_Bulan", 
                                      "IMT", "BAZ", "Prediksi_Status", "Keterangan"]
                        hasil_display = hasil_batch[[col for col in display_cols if col in hasil_batch.columns]]
                        
                        hasil_display = hasil_display.copy()
                        if "IMT" in hasil_display.columns:
                            hasil_display["IMT"] = hasil_display["IMT"].round(2)
                        if "BAZ" in hasil_display.columns:
                            hasil_display["BAZ"] = hasil_display["BAZ"].round(2)
                        
                        column_mapping = {
                            "Jenis_Kelamin": "Gender",
                            "Tinggi_Badan": "Tinggi (cm)",
                            "Berat_Badan": "Berat (kg)",
                            "Umur_Bulan": "Umur (bulan)",
                            "Prediksi_Status": "Status Code",
                            "Keterangan": "Status Gizi"
                        }
                        hasil_display.rename(columns=column_mapping, inplace=True)
                        
                        if "Gender" in hasil_display.columns:
                            hasil_display["Gender"] = hasil_display["Gender"].map({1: "Laki-laki", 0: "Perempuan"})
                        
                        st.dataframe(hasil_display, use_container_width=True)
                        
                        st.subheader("📊 Ringkasan Hasil:")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        status_counts = hasil_batch["Keterangan"].value_counts()
                        total_data = len(hasil_batch)
                        
                        with col1:
                            kurus_count = status_counts.get("Kurus", 0)
                            st.metric("Kurus", f"{kurus_count}", f"{kurus_count/total_data*100:.1f}%")
                        
                        with col2:
                            normal_count = status_counts.get("Normal", 0)
                            st.metric("Normal", f"{normal_count}", f"{normal_count/total_data*100:.1f}%")
                        
                        with col3:
                            bb_lebih_count = status_counts.get("BB Lebih", 0)
                            st.metric("BB Lebih", f"{bb_lebih_count}", f"{bb_lebih_count/total_data*100:.1f}%")
                        
                        with col4:
                            obesitas_count = status_counts.get("Obesitas", 0)
                            st.metric("Obesitas", f"{obesitas_count}", f"{obesitas_count/total_data*100:.1f}%")
                        
                        st.subheader("Distribusi Status Gizi:")
                        chart_data = pd.DataFrame({
                            'Status': status_counts.index,
                            'Jumlah': status_counts.values
                        })
                        st.bar_chart(chart_data.set_index('Status'))
                        
                        if "Umur_Bulan" in hasil_batch.columns:
                            anak_data = hasil_batch[hasil_batch["Umur_Bulan"].between(61, 68)]
                            remaja_data = hasil_batch[hasil_batch["Umur_Bulan"].between(204, 251)]
                            
                            if len(anak_data) > 0:
                                st.subheader("Hasil Anak (61-68 bulan):")
                                anak_display = anak_data[[col for col in display_cols if col in anak_data.columns]]
                                if "IMT" in anak_display.columns:
                                    anak_display["IMT"] = anak_display["IMT"].round(2)
                                if "BAZ" in anak_display.columns:
                                    anak_display["BAZ"] = anak_display["BAZ"].round(2)
                                anak_display.rename(columns=column_mapping, inplace=True)
                                if "Gender" in anak_display.columns:
                                    anak_display["Gender"] = anak_display["Gender"].map({1: "Laki-laki", 0: "Perempuan"})
                                st.dataframe(anak_display, use_container_width=True)
                            
                            if len(remaja_data) > 0:
                                st.subheader("Hasil Remaja (204-251 bulan):")
                                remaja_display = remaja_data[[col for col in display_cols if col in remaja_data.columns]]
                                if "IMT" in remaja_display.columns:
                                    remaja_display["IMT"] = remaja_display["IMT"].round(2)
                                if "BAZ" in remaja_display.columns:
                                    remaja_display["BAZ"] = remaja_display["BAZ"].round(2)
                                remaja_display.rename(columns=column_mapping, inplace=True)
                                if "Gender" in remaja_display.columns:
                                    remaja_display["Gender"] = remaja_display["Gender"].map({1: "Laki-laki", 0: "Perempuan"})
                                st.dataframe(remaja_display, use_container_width=True)
                        
                        st.subheader("Download Hasil:")
                        
                        download_data = hasil_batch.copy()
                        if "IMT" in download_data.columns:
                            download_data["IMT"] = download_data["IMT"].round(2)
                        if "BAZ" in download_data.columns:
                            download_data["BAZ"] = download_data["BAZ"].round(2)
                        
                        csv = download_data.to_csv(index=False)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.download_button(
                                label="Download Hasil (CSV)",
                                data=csv,
                                file_name=f"hasil_prediksi_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                help="Download hasil prediksi dalam format CSV"
                            )
                        
                        with col2:
                            from io import BytesIO
                            output = BytesIO()
                            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                download_data.to_excel(writer, sheet_name='Hasil_Prediksi', index=False)
                                summary_df = pd.DataFrame({
                                    'Status_Gizi': status_counts.index,
                                    'Jumlah': status_counts.values,
                                    'Persentase': [f"{count/total_data*100:.1f}%" for count in status_counts.values]
                                })
                                summary_df.to_excel(writer, sheet_name='Ringkasan', index=False)
                            
                            st.download_button(
                                label="Download Hasil (Excel)",
                                data=output.getvalue(),
                                file_name=f"hasil_prediksi_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                help="Download hasil prediksi dalam format Excel dengan sheet ringkasan"
                            )
                        
                    except Exception as e:
                        st.error(f"Error dalam prediksi batch: {str(e)}")
                        st.info("Pastikan data sesuai format dan rentang umur yang didukung (61-68 bulan untuk anak, 204-251 bulan untuk remaja)")
        
        except Exception as e:
            st.error(f"Error membaca file: {str(e)}")
            st.info("Pastikan file dalam format CSV atau Excel yang valid")
    
    st.subheader("Template File")
    st.markdown("Download template file untuk memudahkan input data:")
    
    template_data = pd.DataFrame({
        'Jenis_Kelamin': [1, 0, 1, 0, 1, 0], 
        'Tinggi_Badan': [111, 108, 155, 170, 122, 162],  
        'Berat_Badan': [25.8, 15.0, 72, 65, 23, 50],
        'Umur_Bulan': [65, 62, 210, 228, 65, 215]
    })
    
    col1, col2 = st.columns(2)
    with col1:
        csv_template = template_data.to_csv(index=False)
        st.download_button(
            label="Download Template CSV",
            data=csv_template,
            file_name="template_prediksi_batch.csv",
            mime="text/csv",
            help="Download template dalam format CSV"
        )
    
    with col2:
        output_template = BytesIO()
        with pd.ExcelWriter(output_template, engine='openpyxl') as writer:
            template_data.to_excel(writer, sheet_name='Data', index=False)
            instruksi_df = pd.DataFrame({
                'Kolom': ['Jenis_Kelamin', 'Tinggi_Badan', 'Berat_Badan', 'Umur_Bulan'],
                'Keterangan': [
                    '1 = Laki-laki, 0 = Perempuan',
                    'Tinggi badan dalam centimeter (cm)',
                    'Berat badan dalam kilogram (kg)', 
                    'Umur dalam bulan'
                ],
                'Rentang_Supported': [
                    '0 atau 1',
                    '50-250 cm',
                    '1-200 kg',
                    '61-68 (anak) atau 204-251 (remaja)'
                ]
            })
            instruksi_df.to_excel(writer, sheet_name='Instruksi', index=False)
        
        st.download_button(
            label="Download Template Excel",
            data=output_template.getvalue(),
            file_name="template_prediksi_batch.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Download template dalam format Excel dengan instruksi"
        )
    
    with st.expander("Informasi Format Data"):
        st.markdown("""
        **Format Data yang Diperlukan:**
        
        1. **Jenis_Kelamin**: 
           - 1 = Laki-laki
           - 0 = Perempuan
           - Atau bisa menggunakan teks "Laki-laki"/"Perempuan"
        
        2. **Tinggi_Badan**: Dalam centimeter (cm)
        
        3. **Berat_Badan**: Dalam kilogram (kg)
        
        4. **Umur_Bulan**: Dalam bulan
           - Anak: 61-68 bulan (sekitar 5 tahun)
           - Remaja: 204-251 bulan (sekitar 17-20 tahun)
        
        **Catatan:**
        - Data di luar rentang umur yang didukung akan diabaikan
        - Sistem akan otomatis menghitung IMT dan BAZ sesuai kategori umur
        - Hasil prediksi menggunakan model KNN yang telah dilatih
        """)

if model_child is not None and model_teen is not None:
    with st.expander("Informasi Detail Model"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Model Anak (61-68 bulan):**")
            st.write(f"- Tipe Model: {type(model_child).__name__}")
            if hasattr(model_child, 'n_neighbors'):
                st.write(f"- Jumlah Neighbors (K): {model_child.n_neighbors}")
            if hasattr(model_child, 'metric'):
                st.write(f"- Metrik Jarak: {model_child.metric}")
            
            expected_feats_child = _expected_features(model_child)
            if expected_feats_child:
                st.write("**Fitur Input Model Anak:**")
                for i, feat in enumerate(expected_feats_child, 1):
                    st.write(f"{i}. {feat}")
            else:
                st.write("**Fitur Input Model Anak:** Jenis_Kelamin, Tinggi_Badan, Berat_Badan, Umur_Bulan, BAZ")
        
        with col2:
            st.write("**Model Remaja (204-251 bulan):**")
            st.write(f"- Tipe Model: {type(model_teen).__name__}")
            if hasattr(model_teen, 'n_neighbors'):
                st.write(f"- Jumlah Neighbors (K): {model_teen.n_neighbors}")
            if hasattr(model_teen, 'metric'):
                st.write(f"- Metrik Jarak: {model_teen.metric}")
            
            expected_feats_teen = _expected_features(model_teen)
            if expected_feats_teen:
                st.write("**Fitur Input Model Remaja:**")
                for i, feat in enumerate(expected_feats_teen, 1):
                    st.write(f"{i}. {feat}")
            else:
                st.write("**Fitur Input Model Remaja:** Jenis_Kelamin, IMT")
        
        st.markdown("---")
        st.write("**Target Prediksi:** 4 kategori status gizi")
        st.write("- 0: Kurus")
        st.write("- 1: Normal") 
        st.write("- 2: BB Lebih")
        st.write("- 3: Obesitas")

else:
    st.warning("Model KNN tidak tersedia. Pastikan file model ada di direktori yang benar.")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>Aplikasi Klasifikasi Status Gizi dengan AI | Powered by Machine Learning (KNN)</p>
    <p><em>Hasil prediksi AI harus dikonfirmasi dengan tenaga kesehatan profesional</em></p>
</div>
""", unsafe_allow_html=True)

