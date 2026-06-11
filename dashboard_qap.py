import streamlit as st
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time

st.set_page_config(page_title="Optimasi Tata Letak RS - Algoritma Genetika", layout="wide")

# ─── DATA ────────────────────────────────────────────────────────────────────

DATA = {
    "Kecil (n=5)": {
        "n": 5,
        "departments": ["Manual Charging (F3)", "Cardiology (F6)", "Internal Medicine (F9)",
                         "Endocrinology (F12)", "Ultrasound Dept (F17)"],
        "flow": np.array([
            [0, 3213, 566, 404, 13538],
            [3213, 0, 0, 0, 0],
            [566, 0, 0, 0, 0],
            [404, 0, 0, 0, 0],
            [13538, 0, 0, 0, 0],
        ], dtype=float),
        "dist": np.array([
            [0, 47, 126, 110, 73],
            [47, 0, 162, 148, 111],
            [126, 162, 0, 45, 69],
            [110, 148, 45, 0, 46],
            [73, 111, 69, 46, 0],
        ], dtype=float),
    },
    "Sedang (n=12)": {
        "n": 12,
        "departments": ["Pediatrics (F1)", "Obs-Gyn (F2)", "Manual Charging (F3)",
                         "Cardiology (F6)", "Neurology (F7)", "Hepatobiliary Surg (F8)",
                         "Internal Medicine (F9)", "Ophthalmology (F10)", "Rheumatology (F11)",
                         "Endocrinology (F12)", "Nephrology (F13)", "Ultrasound Dept (F17)"],
        "flow": np.array([
            [0,76687,0,819,135,1368,819,5630,0,3432,9082,13732],
            [76687,0,4095,1205,519,2746,1097,5712,0,0,268,0],
            [0,4095,0,3213,2071,4225,566,0,0,404,9372,13538],
            [819,1205,3213,0,0,0,0,926,161,0,0,0],
            [135,519,2071,0,0,0,196,1538,196,0,0,0],
            [1368,2746,4225,0,0,0,0,0,301,0,0,0],
            [819,1097,566,0,196,0,0,1954,418,0,0,0],
            [5630,5712,0,926,1538,0,1954,0,0,282,0,0],
            [0,0,0,161,196,301,418,0,0,1686,0,226],
            [3432,0,404,0,0,0,0,282,1686,0,0,0],
            [9082,0,9372,0,0,0,0,0,0,0,0,0],
            [13732,0,13538,0,0,0,0,0,226,0,0,0],
        ], dtype=float),
        "dist": np.array([
            [0,12,36,44,110,126,94,63,130,102,65,126],
            [12,0,24,75,108,70,124,86,93,106,58,70],
            [36,24,0,47,110,73,126,71,95,110,46,73],
            [44,75,47,0,148,111,162,52,96,148,49,111],
            [110,108,110,148,0,46,46,136,47,30,108,46],
            [126,70,73,111,46,0,69,141,63,46,119,27],
            [94,124,126,162,46,69,0,102,34,45,84,69],
            [63,86,71,52,136,141,102,0,64,118,29,141],
            [130,93,95,96,47,63,34,64,0,47,56,63],
            [102,106,110,148,30,46,45,118,47,0,100,46],
            [65,58,46,49,108,119,84,29,56,100,0,119],
            [126,70,73,111,46,27,69,141,63,46,119,0],
        ], dtype=float),
    },
}

# ─── ALGORITMA GENETIKA ──────────────────────────────────────────────────────

def calc_objective(perm, flow, dist):
    n = len(perm)
    z = 0.0
    for a in range(n):
        for b in range(n):
            z += flow[a][b] * dist[perm[a]][perm[b]]
    return z

def fitness(z):
    return 1.0 / (z + 1)

def order_crossover(p1, p2):
    n = len(p1)
    c1, c2 = sorted(random.sample(range(n), 2))
    child = [None] * n
    child[c1:c2] = p1[c1:c2]
    remaining = [g for g in p2 if g not in child]
    idx = 0
    for i in range(n):
        if child[i] is None:
            child[i] = remaining[idx]; idx += 1
    return child

def reciprocal_mutation(chrom):
    c = chrom[:]
    i, j = random.sample(range(len(c)), 2)
    c[i], c[j] = c[j], c[i]
    return c

def run_ga(flow, dist, n, popsize, elite_count, pc, pm, max_gen, progress_cb=None):
    # Init populasi
    pop = [random.sample(range(n), n) for _ in range(popsize)]
    history = []
    best_overall = None
    best_z_overall = float('inf')

    for gen in range(max_gen):
        # Evaluasi
        zvals = [calc_objective(c, flow, dist) for c in pop]
        fits  = [fitness(z) for z in zvals]

        # Urutkan
        order = sorted(range(popsize), key=lambda i: fits[i], reverse=True)
        pop   = [pop[i] for i in order]
        zvals = [zvals[i] for i in order]
        fits  = [fits[i] for i in order]

        best_z = zvals[0]
        if best_z < best_z_overall:
            best_z_overall = best_z
            best_overall   = pop[0][:]

        history.append(best_z)

        if progress_cb:
            progress_cb(gen + 1, max_gen, best_z)

        # Elite
        new_pop = [pop[i][:] for i in range(elite_count)]

        # Crossover + mutasi
        while len(new_pop) < popsize:
            p1, p2 = random.choices(pop[:max(2, popsize//2)], k=2)
            child = order_crossover(p1, p2) if random.random() < pc else p1[:]
            if random.random() < pm:
                child = reciprocal_mutation(child)
            new_pop.append(child)

        pop = new_pop

    return best_overall, best_z_overall, history

# ─── UI ──────────────────────────────────────────────────────────────────────

st.title("🏥 Optimasi Tata Letak Departemen Rumah Sakit")
st.caption("Menggunakan Algoritma Genetika (QAP) — Kelompok 8, Universitas Airlangga 2026")

st.sidebar.header("⚙️ Pengaturan")

data_choice = st.sidebar.selectbox("Pilih Ukuran Data", list(DATA.keys()))
data = DATA[data_choice]
n = data["n"]

st.sidebar.markdown("---")
st.sidebar.subheader("Parameter GA")
popsize     = st.sidebar.slider("Ukuran Populasi",    10, 200, 50,  10)
elite_count = st.sidebar.slider("Jumlah Elite",        1,  10,  2,   1)
pc          = st.sidebar.slider("Prob. Crossover",   0.5, 1.0, 0.8, 0.05)
pm          = st.sidebar.slider("Prob. Mutasi",      0.0, 0.5, 0.2, 0.05)
max_gen     = st.sidebar.slider("Maks. Generasi",    10, 500, 100,  10)

run_btn = st.sidebar.button("🚀 Jalankan Optimasi", type="primary", use_container_width=True)

# ─── Tabs ────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["📊 Hasil Optimasi", "🔢 Data Matrix", "📖 Tentang"])

with tab2:
    col1, col2 = st.columns(2)
    dept_labels = [f"D{i+1}" for i in range(n)]
    with col1:
        st.subheader("Matriks Aliran (Flow)")
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(data["flow"], cmap="YlOrRd")
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(dept_labels, fontsize=8)
        ax.set_yticklabels(dept_labels, fontsize=8)
        plt.colorbar(im, ax=ax)
        ax.set_title("Flow Matrix")
        st.pyplot(fig); plt.close()
    with col2:
        st.subheader("Matriks Jarak (Distance)")
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(data["dist"], cmap="Blues")
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(dept_labels, fontsize=8)
        ax.set_yticklabels(dept_labels, fontsize=8)
        plt.colorbar(im, ax=ax)
        ax.set_title("Distance Matrix (meter)")
        st.pyplot(fig); plt.close()

    st.subheader("Daftar Departemen")
    for i, d in enumerate(data["departments"]):
        st.markdown(f"**D{i+1}** — {d}")

with tab3:
    st.markdown("""
    ### Tentang Program Ini
    Program ini mengimplementasikan **Algoritma Genetika** untuk menyelesaikan
    **Quadratic Assignment Problem (QAP)** dalam konteks optimasi tata letak
    departemen Rumah Sakit "X".

    #### Operator yang digunakan:
    - **Seleksi** : Elitisme — individu terbaik dipertahankan langsung
    - **Crossover** : Order Crossover (OX) — menjaga validitas permutasi
    - **Mutasi** : Reciprocal Exchange Mutation — menukar dua gen secara acak

    #### Fungsi Objektif:
    ```
    Z = Σ F[a][b] × D[π(a)][π(b)]
    ```
    Minimasi total biaya = (frekuensi aliran) × (jarak antar lokasi)

    #### Referensi:
    Laporan Kelompok 8, Program Studi S-1 Matematika,  
    Fakultas Sains dan Teknologi, Universitas Airlangga, 2026.
    """)

with tab1:
    if not run_btn:
        st.info("👈 Atur parameter di sidebar, lalu klik **Jalankan Optimasi**.")
    else:
        prog_bar  = st.progress(0)
        prog_text = st.empty()
        history_placeholder = st.empty()

        hist_so_far = []

        def update_progress(g, total, best_z):
            pct = int(g / total * 100)
            prog_bar.progress(pct)
            prog_text.text(f"Generasi {g}/{total} — Objektif terbaik: {best_z:,.0f}")
            hist_so_far.append(best_z)
            if g % max(1, total // 20) == 0 or g == total:
                fig, ax = plt.subplots(figsize=(8, 3))
                ax.plot(hist_so_far, color="#E94F37", linewidth=1.5)
                ax.set_xlabel("Generasi"); ax.set_ylabel("Nilai Objektif")
                ax.set_title("Konvergensi Algoritma Genetika")
                ax.grid(True, alpha=0.3)
                history_placeholder.pyplot(fig); plt.close()

        t0 = time.time()
        best_perm, best_z, history = run_ga(
            data["flow"], data["dist"], n,
            popsize, elite_count, pc, pm, max_gen,
            progress_cb=update_progress
        )
        elapsed = time.time() - t0

        prog_bar.progress(100)
        prog_text.text(f"✅ Selesai dalam {elapsed:.2f} detik")

        st.markdown("---")

        # ── Metrics ──
        col1, col2, col3 = st.columns(3)
        col1.metric("🏆 Nilai Objektif Terbaik", f"{best_z:,.0f}")
        col2.metric("💪 Fitness Terbaik", f"{1/(best_z+1):.8f}")
        col3.metric("⏱ Waktu Komputasi", f"{elapsed:.2f} s")

        # ── Tabel hasil penugasan ──
        st.subheader("📋 Tata Letak Optimal")
        cols = st.columns(2)
        with cols[0]:
            st.markdown("**Penugasan Departemen → Lokasi**")
            for i, dept_idx in enumerate(best_perm):
                st.markdown(f"- **Lokasi {i+1}** ← {data['departments'][dept_idx]}")
        with cols[1]:
            st.markdown("**Urutan berdasarkan Departemen**")
            assignment = [''] * n
            for loc, dept in enumerate(best_perm):
                assignment[dept] = loc + 1
            for i, dept in enumerate(data['departments']):
                st.markdown(f"- **{dept}** → Lokasi {assignment[i]}")

        # ── Layout visual ──
        st.subheader("🗺️ Visualisasi Tata Letak")
        cols_layout = int(np.ceil(np.sqrt(n)))
        rows_layout = int(np.ceil(n / cols_layout))

        cmap = plt.cm.get_cmap("tab20", n)
        fig, ax = plt.subplots(figsize=(cols_layout * 2.5, rows_layout * 2.2))
        ax.set_xlim(0, cols_layout); ax.set_ylim(0, rows_layout)
        ax.axis("off")
        ax.set_title(f"Tata Letak Optimal — Nilai Objektif: {best_z:,.0f}", fontsize=13, pad=12)

        for loc in range(n):
            col_pos = loc % cols_layout
            row_pos = rows_layout - 1 - (loc // cols_layout)
            dept_idx = best_perm[loc]
            color = cmap(dept_idx)
            rect = mpatches.FancyBboxPatch(
                (col_pos + 0.05, row_pos + 0.05), 0.9, 0.9,
                boxstyle="round,pad=0.05",
                facecolor=color, edgecolor="white", linewidth=2
            )
            ax.add_patch(rect)
            short = data['departments'][dept_idx].split('(')[1].rstrip(')')
            full  = data['departments'][dept_idx].split('(')[0].strip()
            ax.text(col_pos + 0.5, row_pos + 0.62, f"Lok {loc+1}", ha='center',
                    fontsize=9, fontweight='bold', color='white')
            ax.text(col_pos + 0.5, row_pos + 0.42, short, ha='center',
                    fontsize=8, color='white')
            ax.text(col_pos + 0.5, row_pos + 0.22, full[:14], ha='center',
                    fontsize=6.5, color='white', alpha=0.9)

        st.pyplot(fig); plt.close()

        # ── Konvergensi final ──
        st.subheader("📈 Kurva Konvergensi")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(history, color="#E94F37", linewidth=2, label="Nilai Objektif Terbaik")
        ax.fill_between(range(len(history)), history, alpha=0.15, color="#E94F37")
        ax.set_xlabel("Generasi", fontsize=11)
        ax.set_ylabel("Nilai Objektif", fontsize=11)
        ax.set_title("Konvergensi Algoritma Genetika", fontsize=13)
        ax.legend(); ax.grid(True, alpha=0.3)
        st.pyplot(fig); plt.close()

        # ── Improvement ──
        improvement = (history[0] - history[-1]) / history[0] * 100
        st.success(f"✅ Perbaikan dari generasi awal ke akhir: **{improvement:.1f}%**  "
                   f"(dari {history[0]:,.0f} → {history[-1]:,.0f})")
