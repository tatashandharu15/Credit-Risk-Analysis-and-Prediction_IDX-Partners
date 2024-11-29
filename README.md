# Credit Risk Analysis and Prediction by Using Machine Learning

## 1. Goals & Objectives
**Goals:**
1. Mengurangi persentase pinjaman buruk menjadi kurang dari 2,5% (rata-rata persentase pinjaman bermasalah di Indonesia).
2. Mengetahui faktor-faktor yang dapat memprediksi apakah sebuah pinjaman baik atau buruk.

**Objectives:**
1. Menganalisis data historis tentang pinjaman baik dan buruk untuk menemukan wawasan dan pola.
2. Membuat model klasifikasi machine learning untuk memprediksi apakah sebuah pinjaman baik atau buruk.

## 2. Exploratory Data Analysis
<p align="center"><img src="images/Percentage of Loan Approved.png" alt="Percentage of Loan Approved" width = 40%></p>

### 2.1. Univariate Analysis
<p align="center"><img src="images/Univariate%20Analysis.png" alt="Univariate Analysis"></p>

#### Observation:
- Semakin **panjang** jangka waktu pinjaman, semakin **tinggi** kemungkinan pinjaman buruk.
- **Grade A** memiliki kemungkinan **terendah** untuk pinjaman buruk dan **Grade G** memiliki kemungkinan **tertinggi** untuk pinjaman buruk.
- Setiap **emp_length** memiliki rasio pinjaman buruk yang cukup mirip dengan yang **terendah** pada **10+ tahun** dan yang **tertinggi** pada **< 1 tahun**.
- **MORTGAGE** home_ownership memiliki kemungkinan pinjaman buruk yang **lebih rendah** dibandingkan dengan **OWN** dan **RENT**.
- Pendapatan dengan status **Verified** sebenarnya memiliki rasio pinjaman buruk yang **tertinggi**.
- Kemungkinan pinjaman buruk **terendah** terjadi ketika pinjaman digunakan untuk **kartu kredit** dan **tertinggi** untuk **usaha kecil**.

### 2.2. Bivariate Analysis
<p align="center"><img src="images/Bivariate%20Analysis.png" alt="Bivariate Analysis"></p>

#### Observation:
- Semakin **panjang** jangka waktu pinjaman, semakin **tinggi** jumlah dana yang didanai.
- **Grade B** memiliki jumlah dana yang **terendah** dan **Grade G** memiliki jumlah dana yang **tertinggi**.
- Semakin **panjang** emp_length, semakin **tinggi** jumlah dana yang didanai.
- Jumlah dana yang **tertinggi** adalah ketika home_ownership adalah **MORTGAGE** dibandingkan dengan **OWN** atau **RENT**.
- Pendapatan dengan status **Verified** memiliki jumlah dana yang **tertinggi** dan status **Not Verified** memiliki jumlah dana yang **terendah**.
- Jumlah dana yang **tertinggi** adalah ketika pinjaman digunakan untuk **usaha kecil** dan yang **terendah** adalah untuk **liburan**.

## 3. Data Preprocessing
- Mengisi nilai null dengan **< 1 tahun** untuk kolom **emp_length** karena diasumsikan mereka tidak memiliki pengalaman kerja, **mode** untuk kolom kategorikal, **median** untuk kolom yang memiliki distribusi miring, **mean** untuk kolom yang memiliki distribusi simetris, dan **menghapus** kolom yang memiliki terlalu banyak nilai yang hilang.
- Dataset **tidak memiliki** data duplikat.
- Feature engineering untuk fitur yang terkait dengan **tanggal**.
- Melakukan **Label Encoding** pada fitur yang memiliki 2 nilai unik atau data ordinal dan dengan **OHE** pada fitur yang memiliki data nominal.
- Pemilihan fitur dengan **Mutual Information** dan **Pearson Correlation**.
- Membagi data menjadi proporsi 70:30, **70% untuk pelatihan** dan **30% untuk pengujian**.
- Melakukan proses **standarisasi** pada fitur yang digunakan dalam data pelatihan dan pengujian.

## 4. Modeling
Kami akan memilih **precision** sebagai metrik utama karena kami ingin meminimalkan **false positive**, yaitu orang yang diprediksi mampu membayar pinjaman tetapi ternyata tidak bisa. Hal ini karena kerugian dari **memberikan pinjaman** kepada orang yang **tidak mampu membayar** jauh lebih besar dibandingkan **tidak memberikan pinjaman** kepada orang yang **mampu membayar**.

### 4.1. Model Training & Validation
| No | Model | Acc (Train) | Acc (Test) | Prec (Train) | Prec (Test) | Recall (Train) | Recall (Test) | ROC AUC (Train) | ROC AUC (Test) | Time Elapsed |
| :- | :- | :- | :- | :- | :- | :- | :- | :- | :- | :- |
| 1 | Decision Tree | 0.99999 | 0.98517 | 1.00000 | 0.99139 | 0.99999 | 0.99190 | 0.99999 | 0.96201 | 6.576643 |
| 2 | Random Forest | 0.99999 | 0.98384 | 0.99999 | 0.98272 | 1.00000 | 0.99936 | 0.99995 | 0.93044 | 133.361791 |
| 3 | Gradient Boosting | 0.97977 | 0.97889 | 0.97808 | 0.97718 | 0.99963 | 0.99955 | 0.91066 | 0.90781 | 124.221854 |
| 4 | Extra Trees | 0.99999 | 0.97783 | 1.00000 | 0.97627 | 0.99999 | 0.99931 | 0.99999 | 0.90394 | 82.568233 |
| 5 | Logistic Regression | 0.97430 | 0.97360 | 0.97372 | 0.97315 | 0.99800 | 0.99778 | 0.89184 | 0.89041 | 2.956187 |
| 6 | Ada Boost | 0.97089 | 0.97084 | 0.96915 | 0.96906 | 0.99904 | 0.99903 | 0.87297 | 0.87383 | 37.106158 |

#### Observation:
Dari hasil di atas, dapat dilihat bahwa **Decision Tree** adalah **model terbaik** karena memiliki Prec (Test) tertinggi dan **model terburuk** adalah **Ada Boost** karena memiliki Prec (Test) terendah dibandingkan dengan model lainnya.

### 4.2. Hyperparameter Tuning
| No | Model | Acc (Train) | Acc (Test) | Prec (Train) | Prec (Test) | Recall (Train) | Recall (Test) | ROC AUC (Train) | ROC AUC (Test) | Time Elapsed |
| :- | :- | :- | :- | :- | :- | :- | :- | :- | :- | :- |
| 1 | Gradient Boosting | 0.99226 | 0.98901 | 0.99150 | 0.98831 | 0.99986 | 0.99944 | 0.96583 | 0.95312 | 1764.521223 |
| 2 | Random Forest | 0.99988 | 0.98419 | 0.99987 | 0.98301 | 1.00000 | 0.99946 | 0.99948 | 0.93163 | 644.425480 |
| 3 | Extra Trees | 0.99604 | 0.97599 | 0.99557 | 0.97413 | 1.00000 | 0.99949 | 0.98228 | 0.89514 | 277.672801 |
| 4 | Ada Boost | 0.97570 | 0.97542 | 0.97401 | 0.97365 | 0.99932 | 0.99934 | 0.89354 | 0.89310 | 1691.129956 |
| 5 | Logistic Regression | 0.97375 | 0.97306 | 0.97308 | 0.97252 | 0.99807 | 0.99783 | 0.88916 | 0.88781 | 64.776840 |
| 6 | Decision Tree | 0.97790 | 0.95494 | 0.98354 | 0.96942 | 0.99173 | 0.98014 | 0.92982 | 0.86823 | 11.585753 |

#### Observation:
Setelah hyperparameter tuning, terdapat **perubahan sedikit** pada kinerja model, dapat dilihat bahwa **Gradient Boosting** sekarang menjadi **model terbaik** karena memiliki Prec (Test) tertinggi dibandingkan dengan model lainnya.

### 4.3. Feature Importances
<p align="center"><img src="images/Feature%20Importances.png" alt="Feature Importances" width=70%></p>

#### Observation:
Berdasarkan **feature importances** dari model Gradient Boosting, 10 fitur teratas yang memiliki **kontribusi tertinggi** dalam membuat prediksi yang akurat adalah fitur **recoveries**, **total_rec_prncp**, **loan_duration**, **out_prncp**, **credit_report_age**, **total_rec_int**, **installment**, **total_rec_late_fee**, **grade**, dan **term**.

### 4.4. SHAP Values
<p align="center"><img src="images/SHAP%20Values.png" alt="SHAP Values" width=70%></p>

#### Observation:
Kemudian, dari **SHAP values** kita dapat melihat dampak setiap fitur terhadap output model. Fitur-fitur yang memiliki nilai lebih tinggi cenderung berhubungan dengan **kredit baik**, yaitu **total_rec_prncp**, **loan_duration**, **term**, dan **grade**. Sementara itu, fitur-fitur yang memiliki nilai lebih tinggi cenderung berhubungan dengan **kredit buruk**, yaitu **credit_report_age**, **installment**, **recoveries**, **out_prncp**, **total_rec_int**, dan **total_rec_late_fee**.

### 4.5. Confusion Matrix
<p align="center"><img src="images/Confusion%20Matrix.png" alt="Confusion Matrix" width = 70%></p>

Dengan menggunakan hasil *hyperparameter tuning* untuk model Gradient Boosting, kami melatih model kembali untuk mendapatkan **confusion matrix** seperti yang ditunjukkan di atas, dengan hasil sebagai berikut:

- **True Positive**: Memprediksi pinjaman disetujui dan ternyata benar sebanyak 124.057 kali.
- **True Negative**: Memprediksi pinjaman tidak disetujui dan ternyata benar sebanyak 14.289 kali.
- **False Positive**: Memprediksi pinjaman disetujui dan ternyata salah sebanyak 1.461 kali.
- **False Negative**: Memprediksi pinjaman tidak disetujui dan ternyata salah sebanyak 79 kali.

## 5. Business Simulation
**Before Using Machine Learning Model:**
- Good Loans = 0.888 * 466,285 = 414,061
- Bad Loans = 0.112 * 466,285 = 52,224

**After Using Machine Learning Model:**
- Good Loans = 0.988 * 466,285 = 460,690
- Bad Loans = 0.012 * 466,285 = 5,595

**Percentage:**
- Good Loans = ((460,690 - 414,061) / 414,061) * 100% = +11.26%
- Bad Loans = ((5,595 - 52,224) / 52,224) * 100% = -89.29%

#### Conclusion:
Setelah menggunakan machine learning, jumlah **pinjaman baik meningkat sebesar 11,26%** menjadi 98,8% atau jumlah **pinjaman buruk menurun sebesar 89,29%** menjadi 1,2%.
