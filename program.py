import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')
import json
from datetime import datetime
import plotly.graph_objects as go

# SETTING STREAMLIT PAGE
st.set_page_config(
    page_title="Sistem Klasifikasi Cuaca Harian Peringatan Deteksi Dini Potensi Hujan Lebat",
    page_icon="⛈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS CUSTOM STYLING
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .section-header {
        font-size: 1.8rem;
        color: #374151;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: 700;
        border-bottom: 3px solid #3B82F6;
        padding-bottom: 0.5rem;
    }
    
    .alert-box {
        padding: 1.8rem;
        border-radius: 12px;
        margin: 1.2rem 0;
        border-left: 8px solid;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transition: transform 0.3s ease;
    }
    
    .alert-box:hover {
        transform: translateY(-2px);
    }
    
    .alert-normal {
        background: linear-gradient(135deg, #D1FAE5 0%, #A7F3D0 100%);
        border-left-color: #10B981;
    }
    
    .alert-warning {
        background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%);
        border-left-color: #F59E0B;
    }
    
    .alert-danger {
        background: linear-gradient(135deg, #FEE2E2 0%, #FCA5A5 100%);
        border-left-color: #EF4444;
    }
    
    .metric-card {
        background: white;
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.08);
        border: 1px solid #E5E7EB;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        box-shadow: 0 6px 15px rgba(0,0,0,0.12);
    }
    
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.8rem 1.5rem;
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(59, 130, 246, 0.3);
    }
    
    .info-box {
        background: #F0F9FF;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #3B82F6;
        margin: 1rem 0;
    }
    
    .tab-content {
        padding: 1.5rem;
        background: white;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin-top: 1rem;
    }
    
    .success-box {
        background: linear-gradient(135deg, #D1FAE5 0%, #A7F3D0 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #10B981;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #F59E0B;
        margin: 1rem 0;
    }
    
    .error-box {
        background: linear-gradient(135deg, #FEE2E2 0%, #FCA5A5 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #EF4444;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# FUNGSI UTAMA: PREPROCESSING DATA
class WeatherDataPreprocessor:
    def __init__(self):
        self.df = None
        self.feature_columns = []
        self.scaler = StandardScaler()
    
    def load_and_process(self, uploaded_file):
        """Proses file CSV yang diupload"""
        try:
            # Baca file CSV
            self.df = pd.read_csv(uploaded_file, delimiter=';', encoding='utf-8')
            
            # Proses kolom numerik
            numeric_cols = ['Suhu', 'Kelembaban', 'Angin']
            for col in numeric_cols:
                if col in self.df.columns:
                    # Konversi ke string dan hapus karakter non-numerik
                    temp_series = self.df[col].astype(str).str.replace(',', '.', regex=False)
                    
                    # Hapus titik yang berlebihan
                    if temp_series.str.count('\.').max() > 1:
                        temp_series = temp_series.str.replace('.', '', regex=False)
                    
                    # Konversi ke numeric
                    self.df[f'{col}_fixed'] = pd.to_numeric(temp_series, errors='coerce')
                    self.feature_columns.append(f'{col}_fixed')
            
            # Handle missing values
            for col in self.feature_columns:
                median_val = self.df[col].median()
                self.df[col].fillna(median_val, inplace=True)
            
            return True, "Data berhasil diproses"
            
        except Exception as e:
            return False, f"Error memproses data: {str(e)}"
    
    def normalize_features(self):
        """Normalisasi fitur untuk clustering"""
        X_scaled = self.scaler.fit_transform(self.df[self.feature_columns])
        return X_scaled
    
    def get_statistics(self):
        """Ambil statistik deskriptif"""
        stats = {}
        for col in self.feature_columns:
            clean_name = col.replace('_fixed', '')
            stats[clean_name] = {
                'mean': float(self.df[col].mean()),
                'std': float(self.df[col].std()),
                'min': float(self.df[col].min()),
                'max': float(self.df[col].max())
            }
        return stats

# FUNGSI UTAMA: CLUSTERING
class HybridKMeansClustering:
    def __init__(self):
        self.optimal_k = 4  # Default 4 cluster untuk sistem peringatan
        self.kmeans_model = None
        self.cluster_labels = None
    
    def find_optimal_k(self, X_scaled, max_k=10):
        """Cari K optimal secara otomatis"""
        inertias = []
        silhouette_scores = []
        
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            
            inertias.append(kmeans.inertia_)
            if k > 1:
                sil_score = silhouette_score(X_scaled, labels)
                silhouette_scores.append(sil_score)
        
        # Cari elbow point
        if len(inertias) > 1:
            reductions = []
            for i in range(1, len(inertias)):
                reduction = (inertias[i-1] - inertias[i]) / inertias[i-1] * 100
                reductions.append(reduction)
            
            # Cari dimana penurunan melambat
            for i in range(1, len(reductions)-1):
                if reductions[i] < np.mean(reductions[:i]) * 0.5:
                    self.optimal_k = i + 2
                    break
        
        return self.optimal_k
    
    def perform_clustering(self, X_scaled):
        """Lakukan clustering dengan K optimal"""
        self.kmeans_model = KMeans(
            n_clusters=self.optimal_k,
            init='k-means++',
            n_init=15,
            max_iter=300,
            random_state=42
        )
        
        self.cluster_labels = self.kmeans_model.fit_predict(X_scaled)
        return self.cluster_labels
    
    def analyze_clusters(self, df, feature_cols):
        """Analisis karakteristik cluster"""
        df['Cluster'] = self.cluster_labels
        cluster_stats = {}
        
        for cluster_id in range(self.optimal_k):
            cluster_data = df[df['Cluster'] == cluster_id]
            stats = {}
            
            for col in feature_cols:
                clean_name = col.replace('_fixed', '')
                stats[clean_name] = {
                    'mean': float(cluster_data[col].mean()),
                    'std': float(cluster_data[col].std())
                }
            
            cluster_stats[cluster_id] = {
                'size': len(cluster_data),
                'percentage': (len(cluster_data) / len(df)) * 100,
                'stats': stats
            }
        
        return df, cluster_stats

# FUNGSI UTAMA: EARLY WARNING SYSTEM - DIREVISI
class EarlyWarningSystem:
    def __init__(self, kmeans_model, scaler, n_clusters):
        self.model = kmeans_model
        self.scaler = scaler
        self.n_clusters = n_clusters
        
        # Dynamically generate alert definitions based on number of clusters
        self.alert_definitions = self._generate_alert_definitions(n_clusters)
    
    def _generate_alert_definitions(self, n_clusters):
        """Generate alert definitions dynamically based on number of clusters"""
        base_definitions = {
            0: {'name': 'NORMAL', 'icon': '✅', 'color': '#10B981'},
            1: {'name': 'WASPADA', 'icon': '⚠️', 'color': '#F59E0B'},
            2: {'name': 'SIAGA', 'icon': '🚧', 'color': '#F97316'},
            3: {'name': 'AWAS', 'icon': '🚨', 'color': '#EF4444'}
        }
        
        definitions = {}
        for i in range(n_clusters):
            if i in base_definitions:
                definitions[i] = {
                    'level': i + 1,
                    'name': base_definitions[i]['name'],
                    'description': self._get_description(i, n_clusters),
                    'action': self._get_action(i, n_clusters),
                    'icon': base_definitions[i]['icon'],
                    'color': base_definitions[i]['color']
                }
            else:
                # For clusters beyond 4, use default
                definitions[i] = {
                    'level': min(i + 1, 4),
                    'name': f'CLUSTER_{i}',
                    'description': f'Kondisi cuaca cluster {i}',
                    'action': 'Pantau perkembangan cuaca',
                    'icon': '📊',
                    'color': '#6B7280'
                }
        
        return definitions
    
    def _get_description(self, cluster_id, n_clusters):
        """Get description based on cluster and total clusters"""
        if n_clusters == 4:
            descriptions = [
                'Kondisi cuaca stabil dan aman',
                'Kondisi mulai tidak stabil, perlu pemantauan ketat',
                'Potensi cuaca ekstrem dalam 12-24 jam ke depan',
                'Potensi cuaca ekstrem tinggi dalam 6-12 jam ke depan'
            ]
            return descriptions[cluster_id] if cluster_id < 4 else 'Kondisi cuaca tidak terdefinisi'
        else:
            # For other number of clusters
            risk_level = cluster_id / (n_clusters - 1)  # 0 to 1 scale
            if risk_level < 0.25:
                return 'Kondisi cuaca sangat stabil'
            elif risk_level < 0.5:
                return 'Kondisi cuaca relatif stabil'
            elif risk_level < 0.75:
                return 'Kondisi cuaca mulai tidak stabil'
            else:
                return 'Kondisi cuaca berpotensi ekstrem'
    
    def _get_action(self, cluster_id, n_clusters):
        """Get action recommendation based on cluster"""
        if n_clusters == 4:
            actions = [
                'Monitoring rutin, tidak ada tindakan khusus diperlukan',
                'Tingkatkan frekuensi monitoring, siapkan rencana darurat',
                'Aktifkan posko siaga, siapkan sumber daya tanggap darurat',
                'Evakuasi daerah rawan, aktifkan semua sistem tanggap darurat'
            ]
            return actions[cluster_id] if cluster_id < 4 else 'Pantau perkembangan cuaca'
        else:
            risk_level = cluster_id / (n_clusters - 1)
            if risk_level < 0.25:
                return 'Monitoring normal'
            elif risk_level < 0.5:
                return 'Perhatikan perkembangan cuaca'
            elif risk_level < 0.75:
                return 'Siapkan rencana darurat'
            else:
                return 'Waspada dan siap bertindak'
    
    def predict_alert_level(self, temperature, humidity, wind_speed):
        """Prediksi level peringatan berdasarkan parameter cuaca"""
        try:
            # Create input array with correct shape and order
            input_data = np.array([[temperature, humidity, wind_speed]])
            
            # Scale input using the same scaler from training
            input_scaled = self.scaler.transform(input_data)
            
            # Predict cluster
            cluster = self.model.predict(input_scaled)[0]
            
            # Get alert info safely
            alert_info = self.alert_definitions.get(cluster, self.alert_definitions[0])
            
            return {
                'cluster': int(cluster),
                'alert_level': alert_info['level'],
                'alert_name': alert_info['name'],
                'description': alert_info['description'],
                'recommended_action': alert_info['action'],
                'icon': alert_info['icon'],
                'color': alert_info['color'],
                'input_values': {
                    'temperature': temperature,
                    'humidity': humidity,
                    'wind_speed': wind_speed
                }
            }
            
        except Exception as e:
            st.error(f"Error dalam prediksi: {str(e)}")
            # Return default prediction
            return {
                'cluster': 0,
                'alert_level': 1,
                'alert_name': 'NORMAL',
                'description': 'Sistem prediksi sedang diproses',
                'recommended_action': 'Tunggu hingga sistem siap',
                'icon': '🔄',
                'color': '#6B7280',
                'input_values': {
                    'temperature': temperature,
                    'humidity': humidity,
                    'wind_speed': wind_speed
                }
            }

# DASHBOARD UTAMA - DIREVISI
def main():
    # HEADER UTAMA
    st.markdown('<h1 class="main-header">SISTEM PERINGATAN DINI CUACA POTENSI HUJAN LEBAT</h1>', unsafe_allow_html=True)
    
    # SIDEBAR UNTUK KONFIGURASI
    with st.sidebar:
        st.markdown("### ⚙️ **KONFIGURASI SISTEM**")
        st.markdown("---")
        
        # Upload file data
        st.markdown("#### 📁 **UPLOAD DATA CUACA**")
        uploaded_file = st.file_uploader("Pilih file CSV data cuaca", type=['csv'])
        
        st.markdown("---")
        st.markdown("#### 🎛️ **PENGATURAN CLUSTERING**")
        
        # Pilihan metode pencarian K optimal
        k_method = st.radio(
            "Metode penentuan jumlah cluster:",
            ["Auto Hybrid", "Manual 4 Cluster", "Manual Cluster"]
        )
        
        manual_k = 4
        if k_method == "Manual Cluster":
            manual_k = st.slider("Jumlah cluster (K):", 2, 10, 4)
        
        st.markdown("---")
        
        # Tombol untuk memproses data
        process_data = st.button("**PROSES DATA**", type="primary", use_container_width=True)
        
        # Status sistem
        st.markdown("---")
        st.markdown("#### 📊 **STATUS SISTEM**")
        
        if 'data_processed' in st.session_state and st.session_state.data_processed:
            st.success("✅ Data telah diproses")
        else:
            st.warning("⏳ Menunggu data")
        
        if 'clustering_done' in st.session_state and st.session_state.clustering_done:
            st.success("✅ Clustering selesai")
        else:
            st.warning("⏳ Menunggu clustering")
        
        # Reset button
        if st.button("🔄 **RESET DATA**", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        
        # Informasi tambahan
        with st.expander("ℹ️ **Informasi Sistem**"):
            st.markdown("""
            **Sistem Peringatan Dini Cuaca Potensi Hujan Lebat**
            
            Menggunakan algoritma **K-Means Clustering** dengan **Hybrid Auto-K Optimization** untuk mengelompokkan pola cuaca menjadi level peringatan.
            
            **Parameter yang dianalisis:**
            - 🌡️ Suhu (°C)
            - 💧 Kelembaban (%)
            - 💨 Kecepatan Angin (km/jam)
            
            **Update terakhir:** {}
            """.format(datetime.now().strftime("%d/%m/%Y %H:%M")))
    
    # KOLOM UTAMA - TERDIRI DARI 2 TAB
    tab_main, tab_analysis = st.tabs(["🎯 DASHBOARD PREDIKSI", "📊 ANALISIS DATA"])
    
    with tab_main:
        # Bagian 1: Upload dan Processing Data
        st.markdown('<h2 class="section-header">HASIL PREPROCESSING DATA</h2>', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            # Inisialisasi preprocessor
            preprocessor = WeatherDataPreprocessor()
            
            # Proses data ketika tombol ditekan atau data belum diproses
            if process_data or ('data_processed' not in st.session_state):
                with st.spinner("🔄 Memproses data..."):
                    success, message = preprocessor.load_and_process(uploaded_file)
                    
                    if success:
                        st.session_state.preprocessor = preprocessor
                        st.session_state.data_processed = True
                        
                        # Tampilkan preview data
                        st.markdown("### 📋 Preview Data")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.dataframe(preprocessor.df.head(), use_container_width=True)
                        
                        with col2:
                            st.markdown("#### 📊 Statistik Data")
                            stats = preprocessor.get_statistics()
                            
                            for param, values in stats.items():
                                st.metric(
                                    f"{param}",
                                    f"{values['mean']:.1f}",
                                    f"±{values['std']:.1f}",
                                    help=f"Min: {values['min']:.1f}, Max: {values['max']:.1f}"
                                )
                        
                        st.success(f"✅ {message} - {len(preprocessor.df)} observasi ditemukan")
                    else:
                        st.error(f"❌ {message}")
            elif 'data_processed' in st.session_state and st.session_state.data_processed:
                if 'preprocessor' in st.session_state:
                    preprocessor = st.session_state.preprocessor
                    st.info("✅ Data sudah diproses sebelumnya")
                else:
                    st.warning("⚠️ Data processor tidak ditemukan. Silakan upload ulang.")
        
        # Bagian 2: Clustering
        if 'data_processed' in st.session_state and st.session_state.data_processed and 'preprocessor' in st.session_state:
            st.markdown('<h2 class="section-header">🔢 ANALISIS CLUSTERING</h2>', unsafe_allow_html=True)
            
            preprocessor = st.session_state.preprocessor
            
            if process_data or ('clustering_done' not in st.session_state):
                with st.spinner("🔄 Melakukan clustering..."):
                    try:
                        # Normalisasi data
                        X_scaled = preprocessor.normalize_features()
                        
                        # Clustering
                        clustering = HybridKMeansClustering()
                        
                        if k_method == "Auto Hybrid (Rekomendasi)":
                            optimal_k = clustering.find_optimal_k(X_scaled)
                        elif k_method == "Manual 4 Cluster":
                            optimal_k = 4
                            clustering.optimal_k = 4
                        else:
                            optimal_k = manual_k
                            clustering.optimal_k = manual_k
                        
                        # Perform clustering
                        cluster_labels = clustering.perform_clustering(X_scaled)
                        
                        # Analisis cluster
                        df_with_clusters, cluster_stats = clustering.analyze_clusters(
                            preprocessor.df.copy(), preprocessor.feature_columns
                        )
                        
                        # Create EWS dengan jumlah cluster yang benar
                        ews_system = EarlyWarningSystem(
                            clustering.kmeans_model, 
                            preprocessor.scaler,
                            clustering.optimal_k
                        )
                        
                        # Simpan ke session state
                        st.session_state.clustering = clustering
                        st.session_state.df_with_clusters = df_with_clusters
                        st.session_state.cluster_stats = cluster_stats
                        st.session_state.ews_system = ews_system
                        st.session_state.clustering_done = True
                        
                        # Tampilkan hasil clustering
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### 📈 Hasil Clustering")
                            st.metric("Jumlah Cluster Optimal", optimal_k)
                            st.metric("Total Observasi", len(df_with_clusters))
                            
                            # Distribusi cluster
                            cluster_counts = df_with_clusters['Cluster'].value_counts().sort_index()
                            for cluster_id, count in cluster_counts.items():
                                percentage = (count / len(df_with_clusters)) * 100
                                st.metric(f"Cluster {cluster_id}", f"{count}", f"{percentage:.1f}%")
                        
                        with col2:
                            st.markdown("#### 📊 Visualisasi Cluster")
                            fig, ax = plt.subplots(figsize=(8, 6))
                            
                            # Colors for clusters
                            colors = ['#10B981', '#F59E0B', '#F97316', '#EF4444', '#8B5CF6', '#EC4899', '#14B8A6', '#F43F5E']
                            cluster_counts.plot(kind='bar', color=colors[:len(cluster_counts)], ax=ax)
                            
                            ax.set_xlabel('Cluster ID')
                            ax.set_ylabel('Jumlah Observasi')
                            ax.set_title('Distribusi Cluster')
                            ax.grid(True, alpha=0.3, axis='y')
                            
                            # Tambahkan label jumlah
                            for i, (_, count) in enumerate(cluster_counts.items()):
                                ax.text(i, count + 0.5, str(count), ha='center', va='bottom', fontweight='bold')
                            
                            st.pyplot(fig)
                        
                        st.success(f"✅ Clustering selesai! {optimal_k} cluster berhasil dibuat")
                        
                    except Exception as e:
                        st.error(f"Error dalam clustering : {str(e)}")
            
            elif 'clustering_done' in st.session_state and st.session_state.clustering_done:
                st.info("✅ Clustering sudah dilakukan sebelumnya")
        
        # Bagian 3: Prediksi Real-time
        if 'clustering_done' in st.session_state and st.session_state.clustering_done and 'ews_system' in st.session_state:
            st.markdown('<h2 class="section-header">🎯 PREDIKSI REAL-TIME</h2>', unsafe_allow_html=True)
            
            # Input parameter cuaca
            col_input1, col_input2 = st.columns(2)
            
            with col_input1:
                st.markdown("#### 🌡️ Parameter Cuaca Saat Ini")
                temperature = st.number_input("Suhu (°C)", min_value=-20.0, max_value=60.0, value=28.5, step=0.1,
                                            help="Suhu udara dalam derajat Celsius")
                humidity = st.number_input("Kelembaban (%)", min_value=0.0, max_value=100.0, value=75.0, step=0.1,
                                         help="Tingkat kelembaban relatif")
                wind_speed = st.number_input("Kecepatan Angin (km/jam)", min_value=0.0, max_value=200.0, value=15.0, step=0.1,
                                           help="Kecepatan angin maksimum")
                
                predict_button = st.button("🔍 **PREDIKSI SEKARANG**", type="primary", use_container_width=True)
                
                if predict_button:
                    st.session_state.do_prediction = True
                    st.session_state.prediction_input = {
                        'temperature': temperature,
                        'humidity': humidity,
                        'wind_speed': wind_speed
                    }
            
            with col_input2:
                st.markdown("#### 📊 Kondisi Saat Ini")
                
                col_temp, col_hum, col_wind = st.columns(3)
                with col_temp:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    temp_status = "Normal"
                    if temperature >= 35:
                        temp_status = "Tinggi"
                    elif temperature >= 30:
                        temp_status = "Sedang"
                    elif temperature < 10:
                        temp_status = "Rendah"
                    st.metric("🌡️ Suhu", f"{temperature:.1f}°C", temp_status)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col_hum:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    hum_status = "Normal"
                    if humidity >= 85:
                        hum_status = "Sangat Tinggi"
                    elif humidity >= 75:
                        hum_status = "Tinggi"
                    elif humidity < 30:
                        hum_status = "Rendah"
                    st.metric("💧 Kelembaban", f"{humidity:.1f}%", hum_status)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col_wind:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    wind_status = "Normal"
                    if wind_speed >= 40:
                        wind_status = "Sangat Kencang"
                    elif wind_speed >= 25:
                        wind_status = "Kencang"
                    elif wind_speed < 5:
                        wind_status = "Tenang"
                    st.metric("💨 Angin", f"{wind_speed:.1f} km/jam", wind_status)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Parameter Kritis
                st.markdown("#### ⚠️ Parameter Kritis")
                critical_params = []
                if temperature >= 35 or temperature <= 5:
                    critical_params.append(f"🌡️ Suhu {'ekstrem tinggi' if temperature >= 35 else 'ekstrem rendah'}")
                if humidity >= 85 or humidity <= 20:
                    critical_params.append(f"💧 Kelembaban {'ekstrem tinggi' if humidity >= 85 else 'ekstrem rendah'}")
                if wind_speed >= 40:
                    critical_params.append("💨 Angin ekstrem kencang")
                
                if critical_params:
                    for param in critical_params:
                        st.warning(param)
                else:
                    st.success("✅ Semua parameter dalam batas normal")
            
            # Hasil prediksi
            if 'do_prediction' in st.session_state and st.session_state.do_prediction:
                ews_system = st.session_state.ews_system
                
                # Use stored input values
                if 'prediction_input' in st.session_state:
                    temp = st.session_state.prediction_input['temperature']
                    hum = st.session_state.prediction_input['humidity']
                    wind = st.session_state.prediction_input['wind_speed']
                else:
                    temp, hum, wind = temperature, humidity, wind_speed
                
                with st.spinner("🔄 Melakukan prediksi..."):
                    prediction = ews_system.predict_alert_level(temp, hum, wind)
                    
                    if prediction:
                        st.markdown("#### 🚨 **HASIL PREDIKSI**")
                        
                        # Tentukan kelas alert
                        alert_level = prediction['alert_level']
                        max_level = max(1, ews_system.n_clusters)  # Dynamic max level
                        
                        # Dynamic alert classes
                        if max_level == 4:
                            alert_classes = {
                                1: ("alert-normal", "✅"),
                                2: ("alert-warning", "⚠️"),
                                3: ("alert-danger", "🚧"),
                                4: ("alert-danger", "🚨")
                            }
                        else:
                            # For other numbers of clusters
                            alert_classes = {}
                            for i in range(1, max_level + 1):
                                if i == 1:
                                    alert_classes[i] = ("alert-normal", "✅")
                                elif i <= max_level // 2:
                                    alert_classes[i] = ("alert-warning", "⚠️")
                                else:
                                    alert_classes[i] = ("alert-danger", "🚨")
                        
                        alert_class, alert_icon = alert_classes.get(
                            min(alert_level, max_level), 
                            ("alert-normal", "✅")
                        )
                        
                        # Tampilkan box alert
                        st.markdown(
                            f'''
                            <div class="alert-box {alert_class}">
                                <div style="display: flex; align-items: center; margin-bottom: 15px;">
                                    <span style="font-size: 2.5rem; margin-right: 15px;">{alert_icon}</span>
                                    <div>
                                        <h2 style="margin: 0; color: #1F2937;">LEVEL PERINGATAN: {prediction['alert_name']}</h2>
                                        <p style="margin: 5px 0 0 0; color: #6B7280;">Level {prediction['alert_level']}/{max_level}</p>
                                    </div>
                                </div>
                                <div style="margin-top: 15px;">
                                    <p style="margin: 10px 0;"><strong>📝 Deskripsi:</strong> {prediction['description']}</p>
                                    <p style="margin: 10px 0;"><strong>🎯 Aksi yang Disarankan:</strong> {prediction['recommended_action']}</p>
                                    <p style="margin: 10px 0;"><strong>🔢 Cluster Prediksi:</strong> Cluster {prediction['cluster']}</p>
                                    <p style="margin: 10px 0;"><strong>🌡️ Input:</strong> Suhu={temp}°C, Kelembaban={hum}%, Angin={wind} km/jam</p>
                                </div>
                            </div>
                            ''', 
                            unsafe_allow_html=True
                        )
                        
                        # Gauge chart
                        col_gauge, col_info = st.columns([2, 1])
                        
                        with col_gauge:
                            fig = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=prediction['alert_level'],
                                domain={'x': [0, 1], 'y': [0, 1]},
                                title={
                                    'text': "LEVEL PERINGATAN",
                                    'font': {'size': 20, 'color': "darkblue"}
                                },
                                number={
                                    'font': {'size': 40},
                                    'suffix': f"/{max_level}"
                                },
                                gauge={
                                    'axis': {'range': [1, max_level], 'tickwidth': 2, 'tickcolor': "darkblue"},
                                    'bar': {'color': prediction['color'], 'thickness': 0.25},
                                    'bgcolor': "white",
                                    'borderwidth': 2,
                                    'bordercolor': "gray",
                                    'steps': [
                                        {'range': [1, 1.99], 'color': "#10B981"},
                                        {'range': [2, 2.99], 'color': "#F59E0B"},
                                        {'range': [3, max_level], 'color': "#EF4444"}
                                    ],
                                    'threshold': {
                                        'line': {'color': "black", 'width': 4},
                                        'thickness': 0.75,
                                        'value': prediction['alert_level']
                                    }
                                }
                            ))
                            
                            fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col_info:
                            st.markdown("#### 📋 Informasi Level")
                            
                            # Buat string skala
                            scale_lines = []
                            for i in range(ews_system.n_clusters):
                                alert_name = ews_system.alert_definitions.get(i, {'name': f'Level {i+1}'})['name']
                                if i == 0:
                                    scale_lines.append(f"🟢 **{alert_name}**")
                                elif i == ews_system.n_clusters - 1:
                                    scale_lines.append(f"🔴 **{alert_name}**")
                                elif i < ews_system.n_clusters / 2:
                                    scale_lines.append(f"🟡 **{alert_name}**")
                                else:
                                    scale_lines.append(f"🟠 **{alert_name}**")
                            
                            # Gabungkan dengan <br> untuk line break di HTML
                            scale_html = "<br>".join(scale_lines)
                            
                            # Tampilkan dengan HTML
                            st.markdown(
                                f'<div style="background: #F8FAFC; padding: 1rem; border-radius: 10px;">'
                                f'<strong>Skala Peringatan ({ews_system.n_clusters} level):</strong><br><br>'
                                f'{scale_html}'
                                f'</div>',
                                unsafe_allow_html=True
                            )
    
    with tab_analysis:
        if 'clustering_done' in st.session_state and st.session_state.clustering_done:
            st.markdown('<h2 class="section-header">📊 ANALISIS DATA DETAIL</h2>', unsafe_allow_html=True)
            
            df_with_clusters = st.session_state.df_with_clusters
            cluster_stats = st.session_state.cluster_stats
            
            # Tabs untuk berbagai analisis
            tab1, tab2, tab3 = st.tabs(["📈 Distribusi Cluster", "📋 Karakteristik", "📊 Statistik"])
            
            with tab1:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Pie chart distribusi
                    cluster_counts = df_with_clusters['Cluster'].value_counts().sort_index()
                    colors = ['#10B981', '#F59E0B', '#F97316', '#EF4444', '#8B5CF6', '#EC4899', '#14B8A6', '#F43F5E']
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.pie(cluster_counts.values, 
                          labels=[f'Cluster {i}' for i in cluster_counts.index],
                          colors=colors[:len(cluster_counts)],
                          autopct='%1.1f%%',
                          startangle=90)
                    ax.set_title('Distribusi Cluster')
                    st.pyplot(fig)
                
                with col2:
                    # Tabel distribusi
                    dist_df = pd.DataFrame({
                        'Cluster': cluster_counts.index,
                        'Jumlah': cluster_counts.values,
                        'Persentase': (cluster_counts.values / len(df_with_clusters) * 100).round(1)
                    })
                    
                    ews_system = st.session_state.ews_system
                    # FIXED: Use get() method to avoid KeyError
                    dist_df['Level'] = [
                        ews_system.alert_definitions.get(i, {'name': f'CLUSTER_{i}'})['name'] 
                        for i in dist_df['Cluster']
                    ]
                    
                    st.dataframe(dist_df, use_container_width=True)
                    
                    # Summary
                    st.markdown("#### 📊 Ringkasan")
                    st.write(f"**Total Data:** {len(df_with_clusters):,} observasi")
                    st.write(f"**Jumlah Cluster:** {len(cluster_counts)}")
                    st.write(f"**Cluster Terbanyak:** Cluster {cluster_counts.idxmax()} ({cluster_counts.max()} data)")
                    st.write(f"**Cluster Tersedikit:** Cluster {cluster_counts.idxmin()} ({cluster_counts.min()} data)")
            
            with tab2:
                st.markdown("#### 📋 Karakteristik Rata-rata per Cluster")
                
                # Buat tabel karakteristik
                stats_data = []
                ews_system = st.session_state.ews_system
                
                for cluster_id in sorted(cluster_stats.keys()):
                    stats = cluster_stats[cluster_id]['stats']
                    
                    # Get level name safely
                    level_name = ews_system.alert_definitions.get(
                        cluster_id, 
                        {'name': f'CLUSTER_{cluster_id}'}
                    )['name']
                    
                    stats_data.append({
                        'Cluster': cluster_id,
                        'Level': level_name,
                        'Suhu_Rata': f"{stats['Suhu']['mean']:.1f}°C",
                        'Kelembaban_Rata': f"{stats['Kelembaban']['mean']:.1f}%",
                        'Angin_Rata': f"{stats['Angin']['mean']:.1f} km/jam",
                        'Jumlah': cluster_stats[cluster_id]['size'],
                        'Persentase': f"{cluster_stats[cluster_id]['percentage']:.1f}%"
                    })
                
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df, use_container_width=True)
            
            with tab3:
                st.markdown("#### 📊 Statistik Deskriptif Lengkap")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("##### Statistik Numerik")
                    
                    # Statistik deskriptif
                    desc_df = df_with_clusters[['Suhu_fixed', 'Kelembaban_fixed', 'Angin_fixed']].describe()
                    st.dataframe(
                        desc_df.style.format("{:.1f}"),
                        use_container_width=True
                    )
                
                with col2:
                    st.markdown("##### Ringkasan Dataset")
                    
                    st.metric("Total Observasi", len(df_with_clusters))
                    st.metric("Jumlah Cluster", len(st.session_state.cluster_stats))
                    
                    preprocessor = st.session_state.preprocessor
                    stats = preprocessor.get_statistics()
                    
                    for param, values in stats.items():
                        st.metric(
                            f"{param} (Rata-rata)",
                            f"{values['mean']:.1f}",
                            f"±{values['std']:.1f}"
                        )
        else:
            st.warning("⚠️ Silakan proses data dan clustering terlebih dahulu di tab Dashboard Prediksi")
    
    # FOOTER
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #6B7280; padding: 2rem 0;">
        <p style="margin: 0.5rem 0;"><strong>SISTEM KLASIFIKASI CUACA HARIAN PERINGATAN DETEKSI DINI POTENSI HUJAN LEBAT</strong></p>
    </div>
    """, unsafe_allow_html=True)

def _generate_scale_info(self, ews_system):
    """Generate scale information dynamically"""
    scale_info = ""
    for i in range(ews_system.n_clusters):
        alert = ews_system.alert_definitions.get(i, {'name': f'Level {i+1}', 'icon': '📊'})
        icon = alert['icon']
        name = alert['name']
        
        if i == 0:
            color = "🟢"
        elif i == ews_system.n_clusters - 1:
            color = "🔴"
        elif i < ews_system.n_clusters / 2:
            color = "🟡"
        else:
            color = "🟠"
            
        scale_info += f"{color} **{name}**  \n"
    
    return scale_info

# Attach helper method to main
main._generate_scale_info = _generate_scale_info.__get__(main)

# JALANKAN APLIKASI
if __name__ == "__main__":
    # Inisialisasi session state
    if 'data_processed' not in st.session_state:
        st.session_state.data_processed = False
    if 'clustering_done' not in st.session_state:
        st.session_state.clustering_done = False
    if 'do_prediction' not in st.session_state:
        st.session_state.do_prediction = False
    
    # Jalankan aplikasi utama
    main()