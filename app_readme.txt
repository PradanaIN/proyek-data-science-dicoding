Cara menjalankan streamlit app.py di local:

1. Pastikan file ada di folder proyek

| Nama File          | Fungsi                                     |
| ------------------ | ------------------------------------------ |
| `app.py`           | File aplikasi utama Streamlit              |
| `rf_model.pkl`     | Model machine learning (Random Forest)     |
| `scaler.pkl`       | Scaler untuk transformasi fitur            |
| `columns.json`     | Daftar fitur hasil encoding yang digunakan |
| `requirements.txt` | Daftar library Python yang dibutuhkan      |

2. install dependensi : pip install -r requirements.txt
3. Jalankan aplikasi streamlit : streamlit run app.py

Cara menggunakan aplikasi:

1. Masukkan informasi mahasiswa seperti:
   - Nilai masuk universitas
   - Rata-rata nilai semester
   - Jumlah evaluasi
   - Program studi
   - Pekerjaan ayah
2. Klik tombol "🔍 Prediksi Status"
3. Aplikasi akan menampilkan hasil prediksi status mahasiswa:
   - Lulus
   - Masih Kuliah
   - Dropout
   - Juga ditampilkan probabilitas (%) untuk masing-masing kategori.


Streamlit App juga dapat diakses secara remote di Streamlit Community Cloud.
Dapat diakses melalui : https://proyek-data-science-dicoding.streamlit.app/
