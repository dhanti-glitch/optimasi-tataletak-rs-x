# 🏥 Optimasi Tata Letak Departemen Rumah Sakit dengan Algoritma Genetika

Dashboard interaktif untuk menyelesaikan permasalahan **Quadratic Assignment Problem (QAP)** pada tata letak departemen Rumah Sakit "X" menggunakan **Algoritma Genetika**.

> Tugas Akhir Mata Kuliah Optimasi — Kelompok 8  
> Program Studi S-1 Matematika, Fakultas Sains dan Teknologi, Universitas Airlangga, 2026

---

## 📌 Fitur

- Pilihan dataset: **Data Kecil (n=5)** dan **Data Sedang (n=12)**
- Parameter GA yang bisa diatur langsung dari sidebar
- Visualisasi **kurva konvergensi** real-time per generasi
- **Visualisasi layout 2D** hasil tata letak optimal
- Heatmap **matriks aliran** dan **matriks jarak**
- Menampilkan nilai objektif, fitness, dan persentase perbaikan

---

## ⚙️ Operator Algoritma Genetika

| Operator | Metode |
|---|---|
| Seleksi | Elitisme |
| Crossover | Order Crossover (OX) |
| Mutasi | Reciprocal Exchange Mutation |

**Fungsi Objektif:**
```
Z = Σ F[a][b] × D[π(a)][π(b)]
```
Meminimalkan total biaya perpindahan = frekuensi aliran × jarak antar lokasi.

---

## 🚀 Cara Menjalankan

### 1. Clone repository
```bash
git clone https://github.com/username/nama-repo.git
cd nama-repo
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Jalankan dashboard
```bash
streamlit run dashboard_qap.py
```

### 4. Buka di browser
Streamlit akan otomatis membuka browser. Jika tidak, buka manual:
```
http://localhost:8501
```

---

## 📁 Struktur Repository

```
├── dashboard_qap.py     # File utama aplikasi Streamlit
├── requirements.txt     # Daftar dependencies Python
└── README.md            # Dokumentasi ini
```

---

## 👥 Anggota Kelompok 8

| Nama | NIM |
|---|---|
| Rizki Ramadhanti | 181231008 |
| Garnisha Citra Lituhayu | 181231017 |
| Paskah Verjinia Panjaitan | 181231047 |
| Nur Aisyah Rusdiana | 181231054 |
| Altisya Naila Ikaputri Filyanto | 181231058 |

**Dosen Pengampu:** Dr. Herry Suprajitno, M.Si. dan Muhammadun, S.Si., M.Si.
