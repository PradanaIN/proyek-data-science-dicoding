# Laporan Proyek Data Science - Novanni Indi Pradana

## 1. Project Overview

Jaya Jaya Maju adalah perusahaan multinasional dengan lebih dari 1000 karyawan yang tersebar di seluruh penjuru negeri. Meskipun skalanya besar, perusahaan mengalami tantangan dalam mengelola sumber daya manusia, salah satunya adalah tingginya attrition rate (>10%), yaitu tingkat karyawan yang keluar dari perusahaan.

Melalui proyek ini, tim Data Science diminta untuk:

- Mengidentifikasi faktor-faktor yang berpengaruh terhadap tingginya attrition rate.
- Mengembangkan model prediksi karyawan yang berpotensi keluar.
- Menyediakan dashboard bisnis untuk monitoring faktor-faktor utama.

## 2. Business Understanding

### Problem Statements:

Tingginya attrition rate menunjukkan potensi masalah dalam kepuasan kerja, kompensasi, atau keseimbangan kerja-hidup. Hal ini berisiko menurunkan produktivitas dan meningkatkan biaya rekrutmen.

### Goals:

Tujuan dari proyek ini adalah:

- Menemukan pola yang memengaruhi attrition.
- Membuat model prediktif untuk mendeteksi potensi karyawan keluar.
- Menyediakan dashboard untuk membantu tim HR dalam mengambil keputusan berbasis data.

### Solution Statements:

Untuk menjawab permasalahan tersebut, solusi yang akan dikembangkan meliputi:

- Pembuatan model machine learning klasifikasi untuk memprediksi potensi karyawan keluar berdasarkan data historis.
- Analisis faktor penyebab menggunakan eksplorasi data dan visualisasi interaktif.
- Pembuatan dashboard bisnis yang dapat diakses HR untuk monitoring dan pengambilan keputusan berbasis data.

### Cakupan Proyek

Proyek ini mencakup:

- Preprocessing data: Membersihkan, mengubah, dan menyeimbangkan dataset.
- Eksplorasi Data (EDA): Menganalisis distribusi data dan hubungan antar fitur.
- Modeling: Membangun dan membandingkan dua model klasifikasi (Logistic Regression dan Random Forest).
- Evaluasi: Menilai performa model dengan berbagai metrik klasifikasi (precision, recall, f1-score, ROC-AUC).
- Dashboard: Menyajikan insight dari data dan model dalam bentuk visual menggunakan Looker Studio.
- Aplikasi Prediksi: Membangun aplikasi berbasis Streamlit untuk memprediksi kemungkinan attrition.

### Persiapan Proyek

#### Sumber Data:

Dataset yang digunakan berasal dari proyek "Belajar Penerapan Data Science" dari platform Dicoding. Dataset ini berisi 1.470 entri karyawan dengan 35 kolom fitur yang merepresentasikan berbagai aspek seperti:

- Data demografi (umur, jenis kelamin, status pernikahan)
- Informasi pekerjaan (department, job role, years at company)
- Kepuasan dan performa kerja (job satisfaction, performance rating)
- Status Attrition (target variabel: apakah karyawan resign atau tidak)

Dataset dapat diakses melalui link berikut:
https://github.com/dicodingacademy/dicoding_dataset/tree/main/employee

#### Setup Environment Proyek:

Opsi 1: Menjalankan Proyek secara Lokal dengan Anaconda

1. Buat environment baru:

```
conda create --name attrition-predictor python=3.9
```

2. Aktifkan environment:

```
conda activate attrition-predictor
```

3. Install dependencies:

```
pip install -r requirements.txt
```

Pastikan file requirements.txt sudah tersedia di direktori utama proyek. File ini berisi semua library yang dibutuhkan.

Opsi 2: Menjalankan Proyek secara Online dengan Google Colab

1. Buka Google Colab: https://colab.research.google.com
2. Unggah file notebook .ipynb proyek ke Colab.
3. Jalankan sel pada notebook

Google Colab sudah memiliki banyak library bawaan seperti pandas, numpy, matplotlib, seaborn, dan scikit-learn, jadi kita hanya perlu menambahkan library yang tidak tersedia secara default, misalnya imbalanced-learn dan joblib.

## 3. Data Understanding

Dataset yang digunakan adalah data internal perusahaan terkait karyawan aktif maupun yang sudah keluar. Data ini berisi berbagai atribut seperti:

- Data demografis (usia, jenis kelamin, status pernikahan),
- Karakteristik pekerjaan (departemen, jabatan, masa kerja),
- Kepuasan kerja, penilaian performa, dan lainnya.

Ringkasan Data:

- Jumlah baris awal: 1.470
- Jumlah kolom: 35
- Target kolom: Attrition (0 = bertahan, 1 = keluar)

Cuplikan dataset:
| Age | Attrition | BusinessTravel | DailyRate | ... |
|-----|-----------|----------------|-----------|-----|
| 41 | 0 | Travel_Rarely | 1102 | ... |
| 49 | 1 | Travel_Frequently | 279 | ... |

### Exploratory Data Analysis (EDA)

Tahapan eksplorasi data bertujuan untuk memahami struktur, distribusi, serta hubungan antar variabel dalam dataset. EDA dilakukan dalam beberapa sub-bagian berikut:

#### 1. Univariate Analysis

#### Duplikasi dan Missing Values

```
df.duplicated().sum()
```

Tidak ditemukan data duplikat (0 baris).

```
print(df.isnull().sum())
```

Terdapat beberapa nilai kosong (missing values) pada kolom Attrition sebanyak 412 yang perlu ditangani sebelum proses modeling.

Distribusi target Attrition:

- Tidak resign: 83%
- Resign: 17%
- Distribusi Age, MonthlyIncome, YearsAtCompany: terlihat normal hingga skewed.
- Kolom kategorikal seperti Gender, MaritalStatus, dan OverTime terdiri dari beberapa kategori yang relevan.

#### Outlier

```
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

plt.figure(figsize=(15, 8))
df[numeric_cols].boxplot(rot=90)
plt.title("Boxplot Seluruh Fitur Numerik untuk Deteksi Outlier")
plt.show()

Q1 = df[numeric_cols].quantile(0.25)
Q3 = df[numeric_cols].quantile(0.75)
IQR = Q3 - Q1
```

Jumlah data dengan outlier: 691

#### Fitur Numerikal

Visualisasi menggunakan histogram dan KDE (Kernel Density Estimation) dilakukan untuk melihat distribusi fitur seperti Age, MonthlyIncome, YearsAtCompany, dll. Hal ini membantu dalam mengidentifikasi data skewness, outlier, dan range nilai.

- Korelasi kuat terlihat antara MonthlyIncome dengan JobLevel dan TotalWorkingYears.
- Korelasi lemah terhadap target, mengindikasikan perlunya model non-linear.

#### Fitur Kategorikal

Distribusi dari fitur kategorikal seperti Gender, Department, JobRole, dll ditampilkan dalam bentuk countplot. Tujuannya adalah untuk melihat sebaran data dalam tiap kategori.

- Karyawan yang bekerja lembur (OverTime = Yes) cenderung memiliki attrition lebih tinggi.
- Status Single lebih banyak resign dibanding Married.

### 2. Multivariate Analysis

Multivariate analysis berfokus pada hubungan antar fitur, terutama dalam kaitannya dengan target variabel (Attrition).

Gabungan antara JobRole, OverTime, dan JobSatisfaction menunjukkan tren menarik bahwa karyawan yang sering lembur di posisi tertentu dan memiliki kepuasan kerja rendah lebih cenderung resign.

#### Korelasi antar fitur numerikal

Heatmap korelasi digunakan untuk melihat hubungan antar variabel numerik. Ini membantu dalam mengidentifikasi fitur yang saling berkorelasi tinggi yang mungkin menyebabkan multikolinearitas.

#### Hubungan antara fitur dan Attrition

Boxplot digunakan untuk fitur numerikal terhadap Attrition, untuk melihat apakah terdapat perbedaan distribusi antara karyawan yang bertahan dan yang resign.

### 3. Analisis Fitur Kategorikal terhadap Attrition

Untuk fitur kategorikal yang telah diencoding (one-hot encoding), digunakan barplot untuk melihat proporsi Attrition terhadap setiap kategori. Ini memberikan insight tentang kategori mana yang memiliki proporsi attrition lebih tinggi, seperti misalnya OverTime_Yes, MaritalStatus_Single, atau JobRole_Sales Executive.

## 4. Data Preparation

### Menghapus Missing Value

Kolom target `Attrition` memiliki 412 nilai kosong. Data tersebut dihapus agar tidak mengganggu proses klasifikasi.

```python
# Menghapus baris dengan missing value pada kolom target
df = df.dropna(subset=['Attrition'])
```

### Menangani Outlier

Outlier ditangani dengan winsorizing.

```
# Hitung kuartil
Q1 = df[numeric_cols].quantile(0.25)
Q3 = df[numeric_cols].quantile(0.75)
IQR = Q3 - Q1

# Winsorizing: batasi nilai ekstrem
for col in numeric_cols:
    lower_bound = Q1[col] - 1.5 * IQR[col]
    upper_bound = Q3[col] + 1.5 * IQR[col]
    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
```

```
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outlier_count = df[(df[col] < lower) | (df[col] > upper)].shape[0]
    print(f'{col}: {outlier_count} outlier(s) remaining')
```

0 outlier(s) remaining. Data Sudah bersih dari outlier.

### Drop Kolom Non-informatif/Irrelevant

Beberapa kolom seperti ID unik atau nilai konstan dihapus karena tidak memiliki nilai prediktif.

```python
drop_cols = ['EmployeeId', 'EmployeeCount', 'Over18', 'StandardHours']
cols_to_drop = [col for col in drop_cols if col in df.columns]
df.drop(columns=cols_to_drop, inplace=True)
```

### Encoding Kolom Target

Attrition: 'Yes' → 1, 'No' → 0

```python
le = LabelEncoder()
df['Attrition'] = le.fit_transform(df['Attrition'])  # 'Yes' = 1, 'No' = 0
```

### Encoding Fitur Kategorikal

Variabel kategorikal dikonversi menjadi variabel numerik menggunakan one-hot encoding.

```python
categorical_cols = df.select_dtypes(include='object').columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
```

### Feature Scaling

Target variabel: Attrition

Split fitur (X) dan target (Y)

```
X = df.drop('Attrition', axis=1)
y = df['Attrition']
```

Normalisasi fitur:

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Split Dataset

Data dibagi menjadi training dan testing set dengan rasio 80:20.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)
```

### Distribusi Data Terpilih

- Jumlah fitur: 44
- Jumlah data latih: 846
- Jumlah data uji: 212
- Distribusi target (train): Attrition
  - 0 (Tidak keluar) = 703 = 83%
  - 1 (Keluar) = 143 = 17%

## 5. Modeling

Tahapan modeling bertujuan membangun model machine learning untuk memprediksi kemungkinan seorang karyawan keluar dari perusahaan (attrition). Ini diharapkan membantu departemen HR dalam mengantisipasi risiko attrition lebih awal.

Target (Attrition) memiliki ketidakseimbangan kelas, di mana karyawan yang keluar jauh lebih sedikit. Untuk mengatasi hal ini, digunakan SMOTE (Synthetic Minority Over-sampling Technique) pada data latih untuk menyeimbangkan proporsi kelas.

### Model yang digunakan:

1. **Logistic Regression**

- Sederhana dan cepat.
- Memberikan probabilitas dan interpretasi koefisien fitur.
- Cocok untuk klasifikasi biner.

2. **Random Forest**

- Mampu menangani non-linearitas.
- Lebih akurat dalam banyak kasus klasifikasi.
- Menyediakan feature importance.

### Proses Modeling

#### 1. Pemisahan Fitur dan Target:

```python
X = df.drop('Attrition', axis=1)
y = df['Attrition']
```

#### 2. Standardisasi Fitur

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

#### 3. Split Data Train dan Test

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)
```

#### 4. Penanganan Imbalanced Data dengan SMOTE

```python
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
```

#### 5. Training Model Logistic Regression

```pyhton
lr = LogisticRegression()
lr.fit(X_train_smote, y_train_smote)
```

#### 6. Training Model Random Forest

```python
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_smote, y_train_smote)
```

## 6. Evaluasi Model

Evaluasi dilakukan untuk mengukur performa dua model klasifikasi **Logistic Regression** dan **Random Forest Classifier** dalam memprediksi karyawan yang berpotensi keluar dari perusahaan. Fokus utama ada pada kemampuan model mendeteksi kelas minoritas (attrition = 1), karena kesalahan prediksi pada kasus ini bisa berdampak langsung terhadap pengambilan keputusan strategis HR.

### Metode Evaluasi

- Accuracy: Proporsi prediksi yang benar dari keseluruhan data.
- Precision: Proporsi karyawan yang diprediksi keluar dan benar-benar keluar.
- Recall: Proporsi karyawan yang keluar yang berhasil terdeteksi model.
- F1-score: rata-rata precision dan recall.
- ROC AUC Score: Kemampuan model membedakan antara kelas 0 dan 1 secara keseluruhan.

### Hasil Evaluasi

1. Logistic Regression
   | Metrik | Nilai |
   | ------------------- | ---------- |
   | Accuracy | 72% |
   | Precision (class 1) | 0.35 |
   | Recall (class 1) | 0.75 |
   | F1-score (class 1) | 0.47 |
   | ROC AUC Score | **0.8147** |

Model mampu mendeteksi sebagian besar kasus karyawan keluar karena nilai recall tinggi meskipun precision-nya rendah.

2. Random Forest (SMOTE)
   | Metrik | Nilai |
   | ------------------- | ---------- |
   | Accuracy | **87%** |
   | Precision (class 1) | 0.75 |
   | Recall (class 1) | 0.33 |
   | F1-score (class 1) | 0.476 |
   | ROC AUC Score | **0.8093** |

Model lebih akurat secara keseluruhan, tetapi lebih lemah dalam mendeteksi karyawan yang benar-benar keluar karena recall rendah.

## 7. Kesimpulan

Berdasarkan hasil analisis dan pemodelan terhadap data karyawan perusahaan Jaya Jaya Maju, didapatkan bahwa:

1. Tingkat Attrition pada perusahaan mencapai lebih dari 10%, menunjukkan tingkat turnover yang cukup tinggi dan memerlukan perhatian khusus dari manajemen HR.
2. Beberapa faktor utama yang berkontribusi terhadap tingginya attrition berdasarkan feature importance dari model Random Forest adalah:

   - OverTime → Karyawan yang sering lembur memiliki kemungkinan resign lebih tinggi.
   - StockOptionLevel → Karyawan dengan insentif saham lebih rendah cenderung keluar.
   - JobInvolvement, JobSatisfaction, dan WorkLifeBalance → Tingkat keterlibatan, kepuasan kerja, dan keseimbangan hidup-kerja yang rendah berasosiasi dengan peningkatan risiko attrition.
   - MonthlyIncome → Karyawan dengan pendapatan bulanan rendah memiliki kecenderungan lebih tinggi untuk keluar.

3. Sehingga didapatkan bahwa karakteristik umum karyawan yang cenderung resign berdasarkan analisis data dan model prediktif antara lain:

   - Sering lembur (OverTime = Yes) → Beban kerja yang tinggi berkontribusi besar terhadap keinginan resign.
   - Pendapatan bulanan rendah (MonthlyIncome rendah) → Karyawan dengan kompensasi finansial lebih rendah lebih rentan untuk mencari peluang kerja baru.
   - Kepuasan kerja dan keterlibatan kerja rendah (JobSatisfaction & JobInvolvement rendah) → Ketidakpuasan terhadap pekerjaan menjadi faktor utama penyebab resign.
   - WorkLifeBalance rendah → Karyawan yang merasa keseimbangan hidup dan kerja terganggu menunjukkan kecenderungan keluar dari perusahaan.
   - Insentif saham rendah (StockOptionLevel = 0) → Karyawan yang tidak diberikan insentif jangka panjang kurang memiliki ikatan terhadap perusahaan.

4. Model prediktif yang dibangun dengan pendekatan Random Forest dan SMOTE memberikan hasil terbaik:
   - Accuracy: 87%
   - Recall untuk attrition (kelas 1): 33%
   - ROC AUC Score: 0.8093
   - Meskipun recall untuk kelas minoritas (attrition) masih terbatas, model ini cukup baik dalam mengidentifikasi potensi karyawan yang berisiko keluar.

## 8. Rekomendasi Kepada HR

1. Mengevaluasi dan mengurangi jam lembur

   Lembur yang berlebihan sangat berkorelasi dengan niat resign. Perusahaan dapat mengkaji ulang workload dan distribusi tugas.

2. Meningkatkan Keterlibatan dan Kepuasan Karyawan

   Program engagement, training, atau penyesuaian job role dapat membantu meningkatkan keterlibatan dan kepuasan kerja.

3. Mengkaji skema insentif dan gaji

   Memberikan kompensasi kompetitif, termasuk opsi saham (stock option), dapat meningkatkan loyalitas karyawan.

4. Fokus pada Karyawan Muda dan Single

   Karyawan dengan karakteristik ini menunjukkan kecenderungan keluar yang lebih tinggi, sehingga perlu pendekatan personalisasi atau mentoring yang lebih intensif.

5. Membangun Sistem Monitoring

   Dengan adanya dashboard interaktif (seperti Looker Studio / Metabase) dapat digunakan untuk terus memantau faktor-faktor yang mempengaruhi attrition.

## 9. Business Dashboard

Dashboard ini dibuat menggunakan Looker Studio untuk memberikan insight visual mengenai faktor-faktor yang mempengaruhi tingginya tingkat attrition karyawan. Beberapa faktor yang divisualisasikan antara lain OverTime, Joblevel, dan MonthlyIncome.

[Klik di sini untuk melihat dashboard](https://lookerstudio.google.com/reporting/59e90320-a706-4c7a-8070-ed3270acf3b0)

Visualisasi yang disediakan:

- Attrition distribution berdasarkan fitur penting (OverTime, Joblevel, Age, MonthlIncome, & MaritalStatus)
- Perbandingan Monthly Income at Company untuk karyawan yang bertahan vs keluar

---
