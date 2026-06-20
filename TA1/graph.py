"""
Script ini akan mengconvert data csv menjadi grafik
  - pandas     : membaca file CSV menjadi tabel data
  - matplotlib : menggambar grafik dari tabel data tersebut

Menghasilkan dua  gambar:
  1. data_grafik.png  -> tampilan data (Gambar 1)
  2. diagnostik.png   -> grafik pemeriksaan model (Gambar 2)
==============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # mode simpan ke file (tanpa jendela tampilan)
import matplotlib.pyplot as plt
from scipy import stats

# 1. Baca data dari file CSV menjadi tabel
df = pd.read_csv("data_regresi.csv")

# 2. Pengaturan warna dan ukuran huruf
plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})
MERAH = "#C0392B"
GELAP = "#2C3E50"

# ----------------------------------------------------------------------------
# GAMBAR 1 : Tampilan data
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2))

# (a) Jumlah review vs penjualan + garis tren
x = df["jumlah_review"].values
y = df["penjualan"].values
ax[0].scatter(x, y, s=28, alpha=0.6, color=GELAP, edgecolor="white", linewidth=0.5)
b1, b0 = np.polyfit(x, y, 1)            # garis tren sederhana
xs = np.linspace(x.min(), x.max(), 100)
ax[0].plot(xs, b0 + b1 * xs, color=MERAH, linewidth=2, label=f"tren: y={b1:.2f}x+{b0:.0f}")
ax[0].set_xlabel("Jumlah Review"); ax[0].set_ylabel("Volume Penjualan")
ax[0].set_title("(a) Review vs Penjualan", fontweight="bold")
ax[0].legend(frameon=False, fontsize=8); ax[0].grid(alpha=0.25)

# (b) Rating vs penjualan + garis tren
x = df["rating"].values
ax[1].scatter(x, y, s=28, alpha=0.6, color=GELAP, edgecolor="white", linewidth=0.5)
b1, b0 = np.polyfit(x, y, 1)
xs = np.linspace(x.min(), x.max(), 100)
ax[1].plot(xs, b0 + b1 * xs, color=MERAH, linewidth=2, label=f"tren: y={b1:.1f}x+{b0:.0f}")
ax[1].set_xlabel("Rating Produk"); ax[1].set_ylabel("Volume Penjualan")
ax[1].set_title("(b) Rating vs Penjualan", fontweight="bold")
ax[1].legend(frameon=False, fontsize=8); ax[1].grid(alpha=0.25)

# (c) Sebaran (histogram) penjualan
ax[2].hist(y, bins=20, color=GELAP, alpha=0.75, edgecolor="white")
ax[2].axvline(y.mean(), color=MERAH, linestyle="--", linewidth=2, label=f"rata-rata={y.mean():.0f}")
ax[2].set_xlabel("Volume Penjualan"); ax[2].set_ylabel("Frekuensi")
ax[2].set_title("(c) Sebaran Penjualan", fontweight="bold")
ax[2].legend(frameon=False, fontsize=8); ax[2].grid(alpha=0.25)

plt.tight_layout()
plt.savefig("data_grafik.png", dpi=150, bbox_inches="tight")
print("[OK] data_grafik.png tersimpan")

# ----------------------------------------------------------------------------
# GAMBAR 2 : Grafik pemeriksaan model
# ----------------------------------------------------------------------------
resid = df["residual"].values
fitted = df["penjualan_prediksi"].values
actual = df["penjualan"].values

fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2))

# (a) Penjualan asli vs tebakan
ax[0].scatter(actual, fitted, s=28, alpha=0.6, color=GELAP, edgecolor="white", linewidth=0.5)
lims = [min(actual.min(), fitted.min()), max(actual.max(), fitted.max())]
ax[0].plot(lims, lims, "--", color=MERAH, linewidth=1.6, label="tebakan sempurna")
ax[0].set_xlabel("Penjualan Asli"); ax[0].set_ylabel("Penjualan Tebakan")
ax[0].set_title("(a) Asli vs Tebakan", fontweight="bold")
ax[0].legend(frameon=False, fontsize=8); ax[0].grid(alpha=0.25)

# (b) Sisa kesalahan
ax[1].scatter(fitted, resid, s=28, alpha=0.6, color=GELAP, edgecolor="white", linewidth=0.5)
ax[1].axhline(0, color=MERAH, linestyle="--", linewidth=1.6)
ax[1].set_xlabel("Nilai Tebakan"); ax[1].set_ylabel("Sisa Kesalahan")
ax[1].set_title("(b) Sisa Kesalahan", fontweight="bold")
ax[1].grid(alpha=0.25)

# (c) Kewajaran sisa kesalahan (Q-Q plot)
(osm, osr), (slope, intercept, r) = stats.probplot(resid, dist="norm")
ax[2].scatter(osm, osr, s=28, alpha=0.6, color=GELAP, edgecolor="white", linewidth=0.5)
ax[2].plot(osm, slope * osm + intercept, "--", color=MERAH, linewidth=1.6)
ax[2].set_xlabel("Nilai Teoretis"); ax[2].set_ylabel("Sisa Kesalahan")
ax[2].set_title("(c) Kewajaran Sisa Kesalahan", fontweight="bold")
ax[2].grid(alpha=0.25)

plt.tight_layout()
plt.savefig("diagnostik.png", dpi=150, bbox_inches="tight")
print("[OK] diagnostik.png tersimpan")
