import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
import joblib


try:
    model = load_model('model_status_sakit.h5')
    preprocessor = joblib.load('preprocessor.pkl')
except (OSError, FileNotFoundError):
    st.error("Gagal memuat model atau preprocessor. Pastikan file 'model_status_sakit.h5' dan 'preprocessor.pkl' ada.")
    st.stop()

#--- Judul Aplikasi ---
st.subheader("Aplikasi Prediksi Penyakit Jantung bawaan")
st.write("Masukkan data untuk memprediksi Penyakit Jantung bawaan.")
b1kol1,b1kol2,b1kol3,b1kol4 = st.columns(4) 
with b1kol1:
    st.markdown("Usia(thn)    :")
    st.markdown("   ")
    st.markdown("Cepat Lelah  :")
    st.markdown("   ")
    st.markdown("Sesak Nafas  :")
    st.markdown("   ")
    st.markdown("Sering pusing:")
    st.markdown("   ")
    st.markdown("Pilek        :")
    st.markdown("   ")
    st.markdown("Sering Lemas :")
    st.markdown("   ")
    st.markdown("Sering Kejang:")
    st.markdown("   ")
    st.markdown("Gagal Tumbuh :")


with b1kol2:
    umur = st.number_input("Umur", min_value=0, max_value=30, value=5, step=1,label_visibility="collapsed")
    cepat_lelah1 = st.selectbox("Cepat Lelah", options=['Tidak', 'Ya'],label_visibility="collapsed")
    sesak_nafas1 = st.selectbox("Sesak Nafas", options=['Tidak', 'Ya'],label_visibility="collapsed")
    sering_pusing1 = st.selectbox("Sering Pusing", options=['Tidak', 'Ya'],label_visibility="collapsed")
    pilek1 = st.selectbox("Pilek", options=['Tidak', 'Ya'],label_visibility="collapsed")
    sering_lemas1 = st.selectbox("Sering Lemas", options=['Tidak', 'Ya'],label_visibility="collapsed")
    sering_kejang1 = st.selectbox("Sering Kejang", options=['Tidak', 'Ya'],label_visibility="collapsed")
    gagal_tumbuh1 = st.selectbox("Gagal Tumbuh", options=['Tidak', 'Ya'],label_visibility="collapsed")

with b1kol3:
    st.markdown("Jenis Kelamin:")
    st.markdown("  ")
    st.markdown("Demam                  :")
    # st.markdown("   ")
    st.markdown("Kebiruan(pada bibir/kuku):")
    # st.markdown("   ")
    st.markdown("Sering Batuk         :")
    st.markdown("   ")
    st.markdown("Cuping Hidung        :")
    st.markdown("   ")
    st.markdown("Berdebar-debar       :")
    st.markdown("   ")
    st.markdown("Retraksi             :")
    st.markdown("   ")
    st.markdown("Infeksi Paru Berulang:")
with b1kol4:
    jenis_kelamin1 = st.selectbox("Jenis Kelamin", options=['Laki-laki', 'Perempuan'],label_visibility="collapsed")
    demam1 = st.selectbox("Demam", options=['Tidak', 'Ya'],label_visibility="collapsed")
    kebiruan1 = st.selectbox("Kebiruan", options=['Tidak', 'Ya'],label_visibility="collapsed")
    sering_batuk1 = st.selectbox("Sering Batuk", options=['Tidak', 'Ya'],label_visibility="collapsed")
    cuping_hidung1 = st.selectbox("Cuping Hidung", options=['Tidak', 'Ya'],label_visibility="collapsed")
    berdebar_debar1 = st.selectbox("Berdebar-debar", options=['Tidak', 'Ya'],label_visibility="collapsed")
    retraksi1 = st.selectbox("Retraksi", options=['Tidak', 'Ya'],label_visibility="collapsed")
    infeksi_paru_berulang1 = st.selectbox("Infeksi Paru Berulang", options=['Tidak', 'Ya'],label_visibility="collapsed")

# Input kategorikal

# Tombol Prediksi
if st.button("Uji Data"):
    # 1. Mengumpulkan data dari form ke dalam DataFrame
    if jenis_kelamin1 == 'Laki-laki': jenis_kelamin = 1
    else: jenis_kelamin = 0
    if cepat_lelah1 == 'Tidak': cepat_lelah = 0
    else: cepat_lelah = 1        
    if demam1 == 'Tidak': demam = 0
    else: demam = 1        
    if sesak_nafas1 == 'Tidak': sesak_nafas = 0
    else: sesak_nafas = 1        
    if kebiruan1 == 'Tidak': kebiruan = 0
    else: kebiruan = 1     
    if sering_pusing1 == 'Tidak': sering_pusing = 0
    else: sering_pusing = 1        
    if sering_batuk1 == 'Tidak': sering_batuk = 0
    else: sering_batuk = 1        
    if pilek1 == 'Tidak': pilek = 0
    else: pilek = 1        
    if cuping_hidung1 == 'Tidak': cuping_hidung = 0
    else: cuping_hidung = 1        

    if sering_lemas1 == 'Tidak': sering_lemas = 0
    else: sering_lemas = 1        
    if berdebar_debar1 == 'Tidak': berdebar_debar = 0
    else: berdebar_debar = 1     
    if sering_kejang1 == 'Tidak': sering_kejang = 0
    else: sering_kejang = 1        
    if retraksi1 == 'Tidak': retraksi = 0
    else: retraksi = 1        
    if gagal_tumbuh1 == 'Tidak': gagal_tumbuh = 0
    else: gagal_tumbuh = 1        
    if infeksi_paru_berulang1 == 'Tidak': infeksi_paru_berulang = 0
    else: infeksi_paru_berulang = 1        
    data_baru = pd.DataFrame({
        'jenis_kelamin': [jenis_kelamin],
        'umur': [umur],
        'cepat_lelah': [cepat_lelah],
        'demam': [demam],
        'sesak_nafas': [sesak_nafas],
        'kebiruan': [kebiruan],
        'sering_pusing': [sering_batuk],
        'sering_batuk': [sering_batuk],
        'pilek': [pilek],
        'cuping_hidung': [cuping_hidung],
        'sering_lemas': [sering_lemas],
        'berdebar-debar': [berdebar_debar],
        'sering_kejang': [sering_kejang],
        'retraksi': [retraksi],
        'gagal_tumbuh': [gagal_tumbuh],
        'infeksi_paru_berulang': [infeksi_paru_berulang]
    })
    
    try:
        data_baru_processed = preprocessor.transform(data_baru)
    except NameError:
        st.error("Variabel 'preprocessor' tidak ditemukan. Pastikan Anda sudah memuat preprocessor.")
        st.stop()
        
    # 3. Melakukan prediksi dengan model H5
    prediksi_prob = model.predict(data_baru_processed)
    prediksi_class = np.argmax(prediksi_prob, axis=1)


    # Mengambil probabilitas kelas 0
    prob_kelas_0 = prediksi_prob[0][0]  # atau prediksi_prob[0, 0]

    # Mengambil probabilitas kelas 1
    prob_kelas_1 = prediksi_prob[0][1]  # atau prediksi_prob[0, 1]

  
    if prediksi_class[0] == 0 :
       st.write(f"Kemungkinan responden TIDAK memiliki penyakit Jantung Bawaan, Probabilitasnya {prob_kelas_0*100:.2f}%")  
     
    else :
       st.write(f"Kemungkinan responden MEMILIKI penyakit Jantung Bawaan,Probabilitasnya {prob_kelas_1*100:.2f}% ")  
    st.write("Lebih jelasnya Silahkan konsultasi pada dokter Spesialis Jantung Anak.")        
