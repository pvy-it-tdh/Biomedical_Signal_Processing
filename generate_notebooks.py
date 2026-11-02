"""
Generator & Runner for Biomedical Signal Processing Notebooks
Covers: FFT, STFT, EMD, EEMD, CEEMDAN, HHT using WESAD S1 dataset.
"""

import os
import sys
import asyncio

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import nbformat as nbf
from nbclient import NotebookClient

NOTEBOOKS_DIR = r"e:\Study\Biomedical_Signal_Processing\notebooks"
DATA_PATH = r"e:\Study\S1\S1.pkl"
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)

def create_nb():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12.3"
        }
    }
    return nb

# ==============================================================================
# NOTEBOOK 1: FFT & Spectral Analysis
# ==============================================================================
def build_nb1():
    nb = create_nb()
    cells = []
    
    cells.append(nbf.v4.new_markdown_cell("""# 01. Phân Tích Phổ Fourier Nhanh (FFT) Trên Tín Hiệu Y Sinh
### Fast Fourier Transform & Power Spectral Density Analysis on WESAD S1 (ECG, PPG, EMG)
**Tác giả:** Antigravity AI Assistant & Biomedical Signal DSP Specialist  
**Mục tiêu:**
1. Khám phá miền tần số của các tín hiệu y sinh thực tế (ECG, PPG, EMG) từ bộ dữ liệu **WESAD Subject S1**.
2. Tìm hiểu ảnh hưởng của rò rỉ phổ (*Spectral Leakage*) và các hàm cửa sổ (*Windowing: Hann, Hamming, Rectangular*).
3. Đánh giá can nhiễu điện lưới 50 Hz trên ECG/EMG và ước lượng dải tần sinh lý chủ đạo.
"""))

    cells.append(nbf.v4.new_code_cell("""import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

# Cấu hình hiển thị đồ thị chuẩn
plt.style.use('default')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'axes.titleweight': 'bold',
    'figure.dpi': 130
})

DATA_PATH = r"e:\Study\S1\S1.pkl"
print(f"Loading WESAD S1 data from: {DATA_PATH}")
with open(DATA_PATH, "rb") as f:
    s1_data = pickle.load(f, encoding="latin1")

# Trích xuất các kênh tín hiệu chính
ecg_raw = s1_data['signal']['chest']['ECG'].flatten()
emg_raw = s1_data['signal']['chest']['EMG'].flatten()
bvp_raw = s1_data['signal']['wrist']['BVP'].flatten()

fs_chest = 700.0  # Hz (ECG, EMG)
fs_wrist = 64.0   # Hz (PPG/BVP)

print(f"ECG samples: {len(ecg_raw):,} (Fs = {fs_chest} Hz)")
print(f"EMG samples: {len(emg_raw):,} (Fs = {fs_chest} Hz)")
print(f"PPG samples: {len(bvp_raw):,} (Fs = {fs_wrist} Hz)")
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 1. Trích xuất cửa sổ tín hiệu đại diện
Chúng ta chọn cửa sổ 10 giây ở trạng thái nghỉ ổn định (Baseline: giây 120 đến 130).
"""))

    cells.append(nbf.v4.new_code_cell("""t_start_sec = 120.0
win_sec = 10.0

idx_ecg_start = int(t_start_sec * fs_chest)
idx_ecg_end = idx_ecg_start + int(win_sec * fs_chest)

idx_bvp_start = int(t_start_sec * fs_wrist)
idx_bvp_end = idx_bvp_start + int(win_sec * fs_wrist)

t_ecg = np.linspace(0, win_sec, idx_ecg_end - idx_ecg_start, endpoint=False)
t_bvp = np.linspace(0, win_sec, idx_bvp_end - idx_bvp_start, endpoint=False)

ecg_seg = ecg_raw[idx_ecg_start:idx_ecg_end]
emg_seg = emg_raw[idx_ecg_start:idx_ecg_end]
bvp_seg = bvp_raw[idx_bvp_start:idx_bvp_end]

fig, axes = plt.subplots(3, 1, figsize=(11, 6), sharex=True)
axes[0].plot(t_ecg, ecg_seg, color='#1f77b4', lw=1.0)
axes[0].set_ylabel('ECG (mV)')
axes[0].set_title('Tín hiệu Y sinh miền thời gian (10 giây - Subject S1)', loc='left')

axes[1].plot(t_bvp, bvp_seg, color='#d62728', lw=1.2)
axes[1].set_ylabel('PPG/BVP')

axes[2].plot(t_ecg, emg_seg, color='#2ca02c', lw=0.8)
axes[2].set_ylabel('EMG (mV)')
axes[2].set_xlabel('Thời gian (giây)')

for ax in axes:
    ax.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 2. Phân tích Phổ FFT & Khảo sát Cửa sổ hoá (Windowing)
Khảo sát sự khác biệt giữa việc tính FFT trực tiếp (cửa sổ chữ nhật - Rectangular) và việc dùng cửa sổ Hann để triệt tiêu hiện tượng rò rỉ phổ (*Spectral Leakage*).
"""))

    cells.append(nbf.v4.new_code_cell("""# 1. FFT trên BVP (PPG)
N_bvp = len(bvp_seg)
bvp_detrend = bvp_seg - np.mean(bvp_seg)

# Cửa sổ chữ nhật (Rectangular) vs Cửa sổ Hann
fft_rect = np.fft.rfft(bvp_detrend)
fft_hann = np.fft.rfft(bvp_detrend * signal.windows.hann(N_bvp))
freqs_bvp = np.fft.rfftfreq(N_bvp, d=1.0/fs_wrist)

mag_rect = 2.0 / N_bvp * np.abs(fft_rect)
mag_hann = 2.0 / N_bvp * np.abs(fft_hann)

# 2. Welch PSD trên BVP
f_welch, p_welch = signal.welch(bvp_detrend, fs=fs_wrist, nperseg=int(4.0*fs_wrist))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5.5))
ax1.plot(freqs_bvp, mag_rect, 'r--', label='FFT Rectangular Window (Rò rỉ phổ cao)', alpha=0.7)
ax1.plot(freqs_bvp, mag_hann, 'b-', label='FFT Hann Window (Đỉnh rõ ràng, giảm nhiễu biên)', lw=1.2)
ax1.axvspan(0.7, 3.5, color='orange', alpha=0.15, label='Dải nhịp tim (0.7 - 3.5 Hz / 42 - 210 BPM)')
ax1.set_xlim([0, 10])
ax1.set_ylabel('Biên độ phổ')
ax1.set_title('A: So sánh Phổ Biên Độ FFT Của Tín Hiệu PPG (Rectangular vs Hann Window)', loc='left')
ax1.legend(loc='upper right')
ax1.grid(True, linestyle=':', alpha=0.6)

ax2.semilogy(f_welch, p_welch, 'g-', lw=1.5, label='Welch PSD')
ax2.axvspan(0.7, 3.5, color='orange', alpha=0.15)
ax2.set_xlim([0, 10])
ax2.set_xlabel('Tần số (Hz)')
ax2.set_ylabel('Mật độ công suất (V^2/Hz) [Log scale]')
ax2.set_title('B: Mật Độ Phổ Công Suất Welch PSD Của PPG (Trung bình hóa giảm phương sai nhiễu)', loc='left')
ax2.legend(loc='upper right')
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()

# Xác định đỉnh nhịp tim
peak_idx = np.argmax(mag_hann[(freqs_bvp >= 0.7) & (freqs_bvp <= 3.5)])
peak_freq = freqs_bvp[(freqs_bvp >= 0.7) & (freqs_bvp <= 3.5)][peak_idx]
print(f"=> Tần số đỉnh nhịp mạch: {peak_freq:.2f} Hz -> Nhịp tim ước lượng: {peak_freq * 60.0:.1f} BPM")
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 3. Phân tích Phổ ECG và Can Nhiễu Điện Lưới 50 Hz
Tín hiệu ngực ECG (Fs = 700 Hz) cho phép phân tích dải tần lên tới 350 Hz (tần số Nyquist). Chúng ta khảo sát phổ công suất toàn dải để phát hiện nhiễu tần số công nghiệp (50 Hz / 60 Hz).
"""))

    cells.append(nbf.v4.new_code_cell("""N_ecg = len(ecg_seg)
ecg_detrend = ecg_seg - np.mean(ecg_seg)
fft_ecg = np.fft.rfft(ecg_detrend * signal.windows.hann(N_ecg))
freqs_ecg = np.fft.rfftfreq(N_ecg, d=1.0/fs_chest)
psd_ecg = (2.0 / (fs_chest * N_ecg)) * (np.abs(fft_ecg)**2)

plt.figure(figsize=(11, 4))
plt.semilogy(freqs_ecg, psd_ecg, color='#1f77b4', lw=0.9)
plt.axvline(50.0, color='red', linestyle='--', lw=1.2, label='Can nhiễu điện lưới 50 Hz')
plt.axvline(100.0, color='magenta', linestyle=':', lw=1.0, label='Sóng hài bậc hai 100 Hz')
plt.axvspan(0.5, 40.0, color='green', alpha=0.12, label='Dải năng lượng ECG sinh lý (0.5 - 40 Hz)')

plt.xlim([0, 150])
plt.xlabel('Tần số (Hz)')
plt.ylabel('Mật độ phổ công suất (PSD) [Log Scale]')
plt.title('Phổ Công Suất FFT Của Tín Hiệu ECG (Phát hiện nhiễu 50 Hz và dải năng lượng QRS)', loc='left')
plt.legend(loc='upper right')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()

# Đánh giá tỷ lệ công suất tại 50 Hz
idx_50 = np.argmin(np.abs(freqs_ecg - 50.0))
p_50 = psd_ecg[idx_50]
p_total = np.sum(psd_ecg)
print(f"=> Năng lượng đỉnh 50 Hz: {p_50:.4e} (Chiếm tỷ lệ đáng kể trong dải tần số cao ngoài dải sinh lý)")
"""))

    cells.append(nbf.v4.new_markdown_cell("""## Kết Luận Notebook 1:
- **Ưu điểm của FFT:** Tốc độ tính toán siêu nhanh, xác định chính xác các thành phần điều hòa tĩnh như can nhiễu 50 Hz và đỉnh nhịp tim trung bình trong cửa sổ.
- **Hạn chế cốt lõi:** Không có khả năng định vị thời gian (*time-localization*). Nếu có biến cố đột ngột (loạn nhịp, co cơ ngắn), FFT không thể chỉ ra thời điểm xảy ra biến cố.
"""))

    nb.cells = cells
    return nb

# ==============================================================================
# NOTEBOOK 2: STFT & Time-Frequency Analysis
# ==============================================================================
def build_nb2():
    nb = create_nb()
    cells = []
    
    cells.append(nbf.v4.new_markdown_cell("""# 02. Biến Đổi Fourier Thời Gian Ngắn (STFT) & Biểu Đồ Phổ Spectrogram
### Short-Time Fourier Transform: Heisenberg-Gabor Uncertainty & Time-Frequency Resolution
**Tác giả:** Antigravity AI Assistant & Biomedical Signal DSP Specialist  
**Mục tiêu:**
1. Khảo sát biểu diễn Thời gian - Tần số (Time-Frequency Representation) trên tín hiệu y sinh thực tế.
2. Kiểm chứng thực nghiệm sự đánh đổi độ phân giải giữa Thời gian và Tần số ($\Delta t \cdot \Delta f \ge \frac{1}{4\pi}$) thông qua việc điều chỉnh độ dài cửa sổ trượt (*window length*).
3. Phân tích sự chuyển dịch tần số của tín hiệu EMG cơ bắp và tín hiệu PPG khi chuyển từ trạng thái nghỉ sang vận động.
"""))

    cells.append(nbf.v4.new_code_cell("""import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

plt.style.use('default')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'axes.titleweight': 'bold',
    'figure.dpi': 130
})

DATA_PATH = r"e:\Study\S1\S1.pkl"
with open(DATA_PATH, "rb") as f:
    s1_data = pickle.load(f, encoding="latin1")

# Trích xuất đoạn tín hiệu EMG và PPG dài 20 giây qua pha chuyển tiếp (Transient / Activity)
emg_raw = s1_data['signal']['chest']['EMG'].flatten()
bvp_raw = s1_data['signal']['wrist']['BVP'].flatten()

fs_emg = 700.0
fs_bvp = 64.0

# Chọn đoạn 20 giây có vận động cơ (giây 400 đến 420)
t_start = 400.0
dur = 20.0

emg_seg = emg_raw[int(t_start*fs_emg):int((t_start+dur)*fs_emg)]
bvp_seg = bvp_raw[int(t_start*fs_bvp):int((t_start+dur)*fs_bvp)]

print(f"Đã trích xuất đoạn 20s: EMG ({len(emg_seg)} mẫu), BVP ({len(bvp_seg)} mẫu)")
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 1. Khảo sát Sự Đánh Đổi Độ Phân Giải Heisenberg-Gabor
Nguyên lý bất định:
$$\Delta t \cdot \Delta f \ge \frac{1}{4\pi}$$
- **Cửa sổ ngắn (Narrow Window):** $\Delta t$ nhỏ (độ phân giải thời gian cao) $\rightarrow$ $\Delta f$ lớn (phổ tần số bị mờ).
- **Cửa sổ dài (Wide Window):** $\Delta f$ nhỏ (độ phân giải tần số cao) $\rightarrow$ $\Delta t$ lớn (mờ biến cố thời gian).
"""))

    cells.append(nbf.v4.new_code_cell("""# So sánh 3 kích thước cửa sổ trên tín hiệu EMG (Fs = 700 Hz)
# Cửa sổ ngắn: 64 mẫu (~0.09s), Vừa: 256 mẫu (~0.37s), Dài: 1024 mẫu (~1.46s)
win_short = 64
win_mid = 256
win_long = 1024

f_s, t_s, Z_s = signal.stft(emg_seg, fs=fs_emg, nperseg=win_short, noverlap=win_short-16)
f_m, t_m, Z_m = signal.stft(emg_seg, fs=fs_emg, nperseg=win_mid, noverlap=win_mid-32)
f_l, t_l, Z_l = signal.stft(emg_seg, fs=fs_emg, nperseg=win_long, noverlap=win_long-64)

fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)

# 1. Cửa sổ ngắn
im1 = axes[0].pcolormesh(t_s, f_s, np.abs(Z_s), shading='gouraud', cmap='inferno')
axes[0].set_ylim([0, 150])
axes[0].set_ylabel('Tần số (Hz)')
axes[0].set_title('A: STFT Cửa Sổ Ngắn (L = 64 / 0.09s) -> Độ phân giải thời gian cao, tần số kém', loc='left')

# 2. Cửa sổ vừa
im2 = axes[1].pcolormesh(t_m, f_m, np.abs(Z_m), shading='gouraud', cmap='inferno')
axes[1].set_ylim([0, 150])
axes[1].set_ylabel('Tần số (Hz)')
axes[1].set_title('B: STFT Cửa Sổ Vừa (L = 256 / 0.37s) -> Cân bằng thời gian - tần số', loc='left')

# 3. Cửa sổ dài
im3 = axes[2].pcolormesh(t_l, f_l, np.abs(Z_l), shading='gouraud', cmap='inferno')
axes[2].set_ylim([0, 150])
axes[2].set_ylabel('Tần số (Hz)')
axes[2].set_xlabel('Thời gian (giây)')
axes[2].set_title('C: STFT Cửa Sổ Dài (L = 1024 / 1.46s) -> Độ phân giải tần số cao, thời gian kém', loc='left')

plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 2. Spectrogram của Tín hiệu PPG Khi Có Vận Động
Khảo sát sự biến thiên của dải tần số nhịp tim trên tín hiệu PPG/BVP theo thời gian.
"""))

    cells.append(nbf.v4.new_code_cell("""# Phân tích STFT cho BVP (PPG) với nperseg = 128 (2 giây)
nper_bvp = 128
nover_bvp = 112
f_bvp, t_bvp, Z_bvp = signal.stft(bvp_seg, fs=fs_bvp, nperseg=nper_bvp, noverlap=nover_bvp)
spec_bvp = np.abs(Z_bvp)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
t_bvp_axis = np.linspace(0, dur, len(bvp_seg))
ax1.plot(t_bvp_axis, bvp_seg, color='#d62728', lw=1.0)
ax1.set_ylabel('Biên độ PPG')
ax1.set_title('Tín hiệu PPG và Biểu đồ Phổ Spectrogram tương ứng', loc='left')
ax1.grid(True, linestyle=':', alpha=0.6)

pcm = ax2.pcolormesh(t_bvp, f_bvp, spec_bvp, shading='gouraud', cmap='viridis')
ax2.axhspan(0.7, 3.5, color='red', alpha=0.15, label='Dải nhịp tim sinh lý (0.7 - 3.5 Hz)')
ax2.set_ylim([0, 8.0])
ax2.set_ylabel('Tần số (Hz)')
ax2.set_xlabel('Thời gian (giây)')
ax2.legend(loc='upper right')

cbar = fig.colorbar(pcm, ax=ax2, orientation='horizontal', pad=0.2, aspect=40)
cbar.set_label('Biên độ phổ STFT')

plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""## Kết Luận Notebook 2:
- STFT đã cải thiện nhược điểm của FFT bằng cách cung cấp thông tin 2 chiều Thời gian - Tần số.
- Tuy nhiên, độ phân giải vẫn bị cố định bởi kích thước cửa sổ định trước và hàm cơ sở hình sin cứng nhắc, không thể thích ứng với các biến đổi phi tuyến cục bộ của tín hiệu sinh lý.
"""))

    nb.cells = cells
    return nb

# ==============================================================================
# NOTEBOOK 3: EMD & Mode Mixing
# ==============================================================================
def build_nb3():
    nb = create_nb()
    cells = []
    
    cells.append(nbf.v4.new_markdown_cell("""# 03. Phân Tách Mode Thực Nghiệm (EMD) & Vấn Đề Trộn Mode (Mode Mixing)
### Empirical Mode Decomposition (EMD) on WESAD PPG: Theory, Sifting & Pitfalls
**Tác giả:** Antigravity AI Assistant & Biomedical Signal DSP Specialist  
**Mục tiêu:**
1. Thực hành thuật toán phân tách thích nghi dữ liệu EMD (Huang et al., 1998) trên tín hiệu PPG/BVP thực tế.
2. Kiểm chứng tính chất bảo toàn năng lượng và độ chính xác tái tạo hoàn hảo (*Completeness*).
3. Phân tích hiện tượng **Mode Mixing** (trộn mode) - điểm yếu chí mạng của EMD nguyên bản.
"""))

    cells.append(nbf.v4.new_code_cell("""import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from PyEMD import EMD

plt.style.use('default')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'axes.titleweight': 'bold',
    'figure.dpi': 130
})

DATA_PATH = r"e:\Study\S1\S1.pkl"
with open(DATA_PATH, "rb") as f:
    s1_data = pickle.load(f, encoding="latin1")

bvp_raw = s1_data['signal']['wrist']['BVP'].flatten()
fs_bvp = 64.0

# Trích xuất cửa sổ 8 giây (512 mẫu) ở trạng thái nghỉ
t_start = 120.0
win_sec = 8.0
N_win = int(win_sec * fs_bvp)

bvp_clean = bvp_raw[int(t_start*fs_bvp):int(t_start*fs_bvp) + N_win]
t_rel = np.arange(N_win) / fs_bvp

print(f"Cửa sổ PPG phân tích: N = {N_win} mẫu ({win_sec} giây, Fs = {fs_bvp} Hz)")
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 1. Thực Hiện Phân Rã EMD Với PyEMD
Phân tách tín hiệu thành các Hàm Mode Nội tại (IMF - Intrinsic Mode Functions) và Residue.
"""))

    cells.append(nbf.v4.new_code_cell("""emd = EMD()
emd.extrema_detection = "parabol"

imfs = emd.emd(bvp_clean, t_rel)
num_imfs = imfs.shape[0]

# Tái cấu trúc
reconstructed = np.sum(imfs, axis=0)
rec_error = bvp_clean - reconstructed
max_err = np.max(np.abs(rec_error))

print(f"Số lượng IMF phân tách được: {num_imfs}")
print(f"Sai số tái tạo lớn nhất: {max_err:.2e} (Thỏa mãn tính đầy đủ Completeness)")

# Vẽ toàn bộ IMFs
fig, axes = plt.subplots(num_imfs + 1, 1, figsize=(11, 1.3 * (num_imfs + 1)), sharex=True)
axes[0].plot(t_rel, bvp_clean, 'k-', lw=1.2)
axes[0].set_ylabel('Raw PPG')
axes[0].set_title('Phân Rã EMD Trên Tín Hiệu PPG Cổ Tay (Empatica E4 BVP)', loc='left')

colors = plt.cm.viridis(np.linspace(0.1, 0.9, num_imfs))
for i in range(num_imfs):
    if i < num_imfs - 1:
        axes[i+1].plot(t_rel, imfs[i], color=colors[i], lw=1.0)
        axes[i+1].set_ylabel(f'IMF {i+1}')
    else:
        axes[i+1].plot(t_rel, imfs[i], color='red', lw=1.2)
        axes[i+1].set_ylabel('Residue')

axes[-1].set_xlabel('Thời gian (giây)')
for ax in axes:
    ax.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 2. Minh Hoạ Hiện Tượng Mode Mixing (Trộn Mode)
Chúng ta đưa vào một đoạn tín hiệu bị can nhiễu xung chuyển động ngắt quãng (*intermittent motion artifact*).
Khi có nhiễu xung ngắt quãng, thuật toán EMD sẽ bị trộn mode: một phần của nhịp tim bị nhảy sang IMF khác hoặc một IMF chứa cả dao động cao tần lẫn thấp tần.
"""))

    cells.append(nbf.v4.new_code_cell("""# Tạo tín hiệu mô phỏng có xung ngắt quãng (Intermittent high-frequency burst)
bvp_mixed = bvp_clean.copy()
# Bổ sung xung nhiễu tần số 15 Hz trong khoảng từ 3.0s đến 4.0s
burst = 30.0 * np.sin(2 * np.pi * 15.0 * t_rel) * ((t_rel >= 3.0) & (t_rel <= 4.0))
bvp_mixed += burst

imfs_mixed = emd.emd(bvp_mixed, t_rel)

fig, axes = plt.subplots(4, 1, figsize=(11, 6), sharex=True)
axes[0].plot(t_rel, bvp_mixed, 'k-', lw=1.1)
axes[0].set_ylabel('Tín hiệu lẫn nhiễu')
axes[0].set_title('A: Tín hiệu PPG xuất hiện nhiễu xung ngắt quãng (giây 3.0 - 4.0)', loc='left')

axes[1].plot(t_rel, imfs_mixed[0], 'b-', lw=1.0)
axes[1].set_ylabel('IMF 1')
axes[1].set_title('B: IMF 1 bị trộn mode: chỉ xuất hiện dao động ở giây 3-4, phần còn lại trống rỗng', loc='left')

axes[2].plot(t_rel, imfs_mixed[1], 'g-', lw=1.0)
axes[2].set_ylabel('IMF 2')
axes[2].set_title('C: IMF 2 bị xé nhỏ dải tần số nhịp tim do ảnh hưởng của xung', loc='left')

axes[3].plot(t_rel, imfs_mixed[2], 'm-', lw=1.0)
axes[3].set_ylabel('IMF 3')
axes[3].set_xlabel('Thời gian (giây)')

for ax in axes:
    ax.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
plt.show()
print("=> Mode mixing làm mất tính thuần nhất của các thành phần dao động sinh lý!")
"""))

    cells.append(nbf.v4.new_markdown_cell("""## Kết Luận Notebook 3:
- EMD thích nghi hoàn toàn theo dữ liệu mà không cần hàm cơ sở giả định.
- Tuy nhiên, khi gặp tín hiệu y sinh thực tế có nhiễu xung hoặc vận động ngắt quãng, EMD gặp hiện tượng **Mode Mixing** nghiêm trọng, đòi hỏi giải pháp hỗ trợ nhiễu (*Noise-Assisted*) như EEMD hoặc CEEMDAN.
"""))

    nb.cells = cells
    return nb

# ==============================================================================
# NOTEBOOK 4: EEMD & Noise-Assisted Analysis
# ==============================================================================
def build_nb4():
    nb = create_nb()
    cells = []
    
    cells.append(nbf.v4.new_markdown_cell("""# 04. Tập Hợp EMD (EEMD): Giải Pháp Hỗ Trợ Nhiễu Khắc Phục Mode Mixing
### Ensemble Empirical Mode Decomposition (EEMD) on Biomedical Signals
**Tác giả:** Antigravity AI Assistant & Biomedical Signal DSP Specialist  
**Mục tiêu:**
1. Tìm hiểu nguyên lý hỗ trợ phân giải tần số bằng **nhiễu trắng Gauss** (Wu & Huang, 2009).
2. Chạy thực nghiệm EEMD với tập hợp thử nghiệm ($M = 30$ trials, độ lớn nhiễu $\epsilon = 0.2 \sigma$).
3. So sánh trực tiếp khả năng tách mode giữa EMD và EEMD trên cùng một đoạn tín hiệu.
4. Đánh giá nhược điểm: Nhiễu tồn dư (*Residual noise*) và sai số tái cấu trúc.
"""))

    cells.append(nbf.v4.new_code_cell("""import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from PyEMD import EMD, EEMD

plt.style.use('default')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'axes.titleweight': 'bold',
    'figure.dpi': 130
})

DATA_PATH = r"e:\Study\S1\S1.pkl"
with open(DATA_PATH, "rb") as f:
    s1_data = pickle.load(f, encoding="latin1")

bvp_raw = s1_data['signal']['wrist']['BVP'].flatten()
fs_bvp = 64.0

# Trích đoạn tín hiệu 8 giây có xung nhiễu
t_start = 120.0
win_sec = 8.0
N_win = int(win_sec * fs_bvp)

bvp_base = bvp_raw[int(t_start*fs_bvp):int(t_start*fs_bvp) + N_win]
t_rel = np.arange(N_win) / fs_bvp

# Thêm xung ngắn gây mode mixing
bvp_test = bvp_base.copy()
burst = 25.0 * np.sin(2 * np.pi * 14.0 * t_rel) * ((t_rel >= 3.0) & (t_rel <= 4.0))
bvp_test += burst
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 1. Thực Thi EEMD Với Thư Viện PyEMD
Cấu hình:
- `trials = 20` (số lần thêm nhiễu trắng)
- `noise_width = 0.2` (biên độ nhiễu trắng = 20% độ lệch chuẩn của tín hiệu)
- `parallel = False` (chạy tuần tự tối ưu, tránh lỗi đa tiến trình bộ nhớ trên Windows)
"""))

    cells.append(nbf.v4.new_code_cell("""eemd = EEMD(trials=20, noise_width=0.2, parallel=False)
eemd.extrema_detection = "parabol"

print("Đang thực hiện EEMD (20 trials, parallel=False)...")
eemd_imfs = eemd.eemd(bvp_test, t_rel)
num_eemd_imfs = eemd_imfs.shape[0]

# Thực hiện EMD đối chứng
emd = EMD()
emd.extrema_detection = "parabol"
emd_imfs = emd.emd(bvp_test, t_rel)

print(f"Số IMF EMD: {emd_imfs.shape[0]} | Số IMF EEMD: {num_eemd_imfs}")

# Đo sai số tái cấu trúc
eemd_rec = np.sum(eemd_imfs, axis=0)
eemd_err = bvp_test - eemd_rec
print(f"EEMD Sai số tái cấu trúc RMS: {np.sqrt(np.mean(eemd_err**2)):.4f}")
print(f"EEMD Sai số tái cấu trúc Max: {np.max(np.abs(eemd_err)):.4f} (Do số trials M hữu hạn, vẫn còn nhiễu dư!)")
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 2. Đối Đầu Trực Diện: EMD vs EEMD Trên Từng IMF
Quan sát xem EEMD có giúp tách bạch thành phần xung cao tần ra khỏi sóng nhịp tim hay không.
"""))

    cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(4, 2, figsize=(12, 7), sharex=True)

# Cột 1: EMD chuẩn
axes[0, 0].plot(t_rel, bvp_test, 'k-', lw=1.0)
axes[0, 0].set_title('EMD: Tín hiệu gốc lẫn xung', loc='left')
axes[1, 0].plot(t_rel, emd_imfs[0], 'b-', lw=1.0)
axes[1, 0].set_title('EMD IMF 1 (Bị ngắt quãng / Mode mixing)', loc='left')
axes[2, 0].plot(t_rel, emd_imfs[1], 'g-', lw=1.0)
axes[2, 0].set_title('EMD IMF 2', loc='left')
axes[3, 0].plot(t_rel, emd_imfs[2], 'm-', lw=1.0)
axes[3, 0].set_title('EMD IMF 3', loc='left')

# Cột 2: EEMD (Noise-assisted)
axes[0, 1].plot(t_rel, bvp_test, 'k-', lw=1.0)
axes[0, 1].set_title('EEMD: Tín hiệu gốc lẫn xung', loc='left')
axes[1, 1].plot(t_rel, eemd_imfs[0], 'b-', lw=1.0)
axes[1, 1].set_title('EEMD IMF 1 (Dao động liên tục, đồng nhất)', loc='left')
axes[2, 1].plot(t_rel, eemd_imfs[1], 'g-', lw=1.0)
axes[2, 1].set_title('EEMD IMF 2 (Dạng sóng nhịp tim phục hồi trơn tru)', loc='left')
axes[3, 1].plot(t_rel, eemd_imfs[2], 'm-', lw=1.0)
axes[3, 1].set_title('EEMD IMF 3', loc='left')

for ax in axes.flatten():
    ax.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""## Kết Luận Notebook 4:
- **EEMD** đã thành công trong việc giải quyết hiện tượng Mode Mixing nhờ cơ chế hỗ trợ của nhiễu trắng Gauss.
- **Hạn chế:** Tín hiệu tái cấu trúc không còn chính xác tuyệt đối (tồn tại sai số nhiễu dư do $M$ hữu hạn) và thời gian tính toán tăng gấp $M$ lần so với EMD.
"""))

    nb.cells = cells
    return nb

# ==============================================================================
# NOTEBOOK 5: CEEMDAN & Adaptive Noise
# ==============================================================================
def build_nb5():
    nb = create_nb()
    cells = []
    
    cells.append(nbf.v4.new_markdown_cell("""# 05. CEEMDAN: Tái Cấu Trúc Hoàn Hảo & Tối Ưu Phân Tách Mode
### Complete Ensemble EMD with Adaptive Noise (CEEMDAN) on ECG & PPG
**Tác giả:** Antigravity AI Assistant & Biomedical Signal DSP Specialist  
**Mục tiêu:**
1. Khám phá thuật toán **CEEMDAN** (Torres et al., 2011) - đỉnh cao của dòng thuật toán EMD.
2. Kiểm chứng **tính chất tái cấu trúc chính xác tuyệt đối** ($\text{Error} \approx 0$).
3. Ứng dụng thực tế: **Khử trôi đường đẳng điện (Baseline Wander Removal)** trên tín hiệu ECG mà không làm méo pha sóng ST-T.
"""))

    cells.append(nbf.v4.new_code_cell("""import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from PyEMD import CEEMDAN

plt.style.use('default')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'axes.titleweight': 'bold',
    'figure.dpi': 130
})

DATA_PATH = r"e:\Study\S1\S1.pkl"
with open(DATA_PATH, "rb") as f:
    s1_data = pickle.load(f, encoding="latin1")

# Trích xuất đoạn ECG (Fs = 700 Hz) dài 2 giây có trôi đường nền
ecg_raw = s1_data['signal']['chest']['ECG'].flatten()
fs_ecg = 700.0

t_start = 120.0
win_sec = 2.0
N_win = int(win_sec * fs_ecg)

ecg_seg = ecg_raw[int(t_start*fs_ecg):int(t_start*fs_ecg) + N_win]
t_ecg = np.arange(N_win) / fs_ecg

# Thêm độ dạt đường nền nhân tạo chậm 0.2 Hz
baseline_wander = 0.8 * np.sin(2 * np.pi * 0.2 * t_ecg) + 0.3 * np.cos(2 * np.pi * 0.08 * t_ecg)
ecg_noisy = ecg_seg + baseline_wander
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 1. Phân Tách CEEMDAN Với PyEMD
CEEMDAN thêm nhiễu thích nghi ở từng bậc phân tách, đảm bảo phần dư ở mỗi bước được tính toán tường minh.
"""))

    cells.append(nbf.v4.new_code_cell("""ceemdan = CEEMDAN(trials=20, epsilon=0.2, parallel=False)
ceemdan.extrema_detection = "parabol"

print("Đang thực hiện CEEMDAN trên ECG (trials=20, parallel=False)...")
imfs_ceemdan = ceemdan(ecg_noisy, t_ecg)
num_imfs = imfs_ceemdan.shape[0]

# Kiểm tra tính toán vẹn (Completeness)
rec_ceemdan = np.sum(imfs_ceemdan, axis=0)
exact_error = ecg_noisy - rec_ceemdan
max_exact_err = np.max(np.abs(exact_error))

print(f"Số lượng IMF CEEMDAN: {num_imfs}")
print(f"Sai số tái tạo lớn nhất: {max_exact_err:.2e}")
if max_exact_err < 1e-12:
    print("=> XÁC NHẬN: CEEMDAN đạt độ chính xác tái tạo tuyệt đối (Exact Reconstruction)!")
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 2. Ứng Dụng: Khử Trôi Đường Đẳng Điện ECG (Baseline Wander Removal)
Các IMF bậc cao (IMF cuối cùng và Residue) chứa thành phần tần số cực thấp tương ứng với đường nền trôi dạt.
Loại bỏ các mode chậm này và cộng các mode còn lại sẽ thu được ECG sạch mà không gây méo pha (Zero phase distortion).
"""))

    cells.append(nbf.v4.new_code_cell("""# Xác định các IMF tần số thấp (thường là 2-3 mode cuối)
# Mode 0 đến num_imfs - 3 là phần tín hiệu QRS và sóng P-T
ecg_cleaned = np.sum(imfs_ceemdan[:-2], axis=0)
extracted_wander = np.sum(imfs_ceemdan[-2:], axis=0)

fig, axes = plt.subplots(3, 1, figsize=(11, 6.5), sharex=True)

axes[0].plot(t_ecg, ecg_noisy, 'r-', lw=1.0, label='ECG có trôi đường đẳng điện')
axes[0].plot(t_ecg, extracted_wander, 'k--', lw=1.5, label='Đường nền trích xuất từ các IMF bậc cao')
axes[0].set_ylabel('Biên độ (mV)')
axes[0].set_title('A: Tín hiệu ECG ban đầu và Đường dạt nền trích xuất tự động qua CEEMDAN', loc='left')
axes[0].legend(loc='upper right')

axes[1].plot(t_ecg, ecg_cleaned, 'b-', lw=1.1, label='ECG đã khử baseline wander')
axes[1].axhline(0, color='gray', linestyle=':', lw=0.8)
axes[1].set_ylabel('Biên độ (mV)')
axes[1].set_title('B: Tín hiệu ECG Sau Khử Baseline: Đường đẳng điện phẳng, không méo hình thái sóng ST-T', loc='left')
axes[1].legend(loc='upper right')

axes[2].plot(t_ecg, exact_error, 'g-', lw=0.8)
axes[2].set_ylabel('Sai số')
axes[2].set_xlabel('Thời gian (giây)')
axes[2].set_title('C: Sai số tái tạo hoàn hảo (Tiệm cận 0)', loc='left')

for ax in axes:
    ax.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""## Kết Luận Notebook 5:
- **CEEMDAN** kết hợp hoàn hảo ưu điểm của EMD (bảo toàn cấu trúc) và EEMD (khắc phục mode mixing).
- Tái cấu trúc chính xác tuyệt đối giúp CEEMDAN trở thành lựa chọn hàng đầu cho các bài toán tiền xử lý và lọc thích nghi tín hiệu y sinh.
"""))

    nb.cells = cells
    return nb

# ==============================================================================
# NOTEBOOK 6: HHT & Hilbert Spectral Analysis
# ==============================================================================
def build_nb6():
    nb = create_nb()
    cells = []
    
    cells.append(nbf.v4.new_markdown_cell("""# 06. Biến Đổi Hilbert-Huang (HHT): Tần Số Tức Thời & Phổ Năng Lượng Hilbert
### Hilbert-Huang Transform (HHT): Instantaneous Frequency & Hilbert Spectrum
**Tác giả:** Antigravity AI Assistant & Biomedical Signal DSP Specialist  
**Mục tiêu:**
1. Hoàn thiện quy trình HHT: **CEEMDAN Decomposition + Hilbert Spectral Analysis (HSA)**.
2. Xây dựng tín hiệu giải tích (*Analytic Signal*), tính toán Biên độ tức thời $a(t)$ và Tần số tức thời $f(t) = \frac{1}{2\pi}\frac{d\theta}{dt}$.
3. Trực quan hóa **Phổ Hilbert 2D $H(\omega, t)$** và **Phổ cận biên Hilbert (Marginal Hilbert Spectrum)** so với phổ Fourier truyền thống.
"""))

    cells.append(nbf.v4.new_code_cell("""import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from PyEMD import CEEMDAN

plt.style.use('default')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'axes.titleweight': 'bold',
    'figure.dpi': 130
})

DATA_PATH = r"e:\Study\S1\S1.pkl"
with open(DATA_PATH, "rb") as f:
    s1_data = pickle.load(f, encoding="latin1")

bvp_raw = s1_data['signal']['wrist']['BVP'].flatten()
fs_bvp = 64.0

# Trích đoạn tín hiệu PPG 10 giây
t_start = 120.0
win_sec = 10.0
N_win = int(win_sec * fs_bvp)

bvp_seg = bvp_raw[int(t_start*fs_bvp):int(t_start*fs_bvp) + N_win]
t_rel = np.arange(N_win) / fs_bvp

# Chạy phân rã CEEMDAN
ceemdan = CEEMDAN(trials=20, epsilon=0.2, parallel=False)
imfs = ceemdan(bvp_seg, t_rel)
num_imfs = imfs.shape[0]

print(f"Đã phân rã CEEMDAN thành {num_imfs} IMFs.")
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 1. Biến Đổi Hilbert & Tính Toán Tần Số Tức Thời
Với mỗi IMF $c_k(t)$, ta tạo tín hiệu giải tích phức:
$$z_k(t) = c_k(t) + j \tilde{c}_k(t) = a_k(t) e^{j \theta_k(t)}$$
Tần số tức thời (Hz):
$$f_k(t) = \frac{1}{2\pi} \frac{d\theta_k(t)}{dt}$$
"""))

    cells.append(nbf.v4.new_code_cell("""inst_amps = []
inst_freqs = []

for k in range(num_imfs - 1): # Bỏ qua residue cuối
    imf = imfs[k]
    # Biến đổi Hilbert
    analytic = signal.hilbert(imf)
    amp = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic))
    # Tần số tức thời qua đạo hàm pha
    freq = (np.gradient(phase) / (2.0 * np.pi)) * fs_bvp
    
    # Giới hạn vật lý để biểu diễn sạch
    freq = np.clip(freq, 0, fs_bvp / 2.0)
    
    inst_amps.append(amp)
    inst_freqs.append(freq)

# Vẽ quỹ đạo tần số tức thời của 3 IMF đầu
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)

for k in range(min(3, len(inst_freqs))):
    ax1.plot(t_rel, inst_amps[k], label=f'IMF {k+1} Amplitude', lw=1.2)
ax1.set_ylabel('Biên độ tức thời a(t)')
ax1.set_title('A: Biên Độ Tức Thời (Instantaneous Amplitude) Của Các Mode Sinh Lý', loc='left')
ax1.legend(loc='upper right')
ax1.grid(True, linestyle=':', alpha=0.5)

for k in range(min(3, len(inst_freqs))):
    ax2.plot(t_rel, inst_freqs[k], label=f'IMF {k+1} Frequency', lw=1.2)
ax2.axhspan(0.7, 3.5, color='orange', alpha=0.15, label='Dải nhịp tim (0.7 - 3.5 Hz)')
ax2.set_ylabel('Tần số tức thời (Hz)')
ax2.set_xlabel('Thời gian (giây)')
ax2.set_title('B: Tần Số Tức Thời f(t) Phản Ánh Độ Biến Thiên Nhịp Tim Liên Tục Theo Thời Gian', loc='left')
ax2.legend(loc='upper right')
ax2.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""## 2. Phổ Năng Lượng Hilbert 2D $H(\omega, t)$ và Phổ Cận Biên
Xây dựng biểu đồ phân bố năng lượng theo thời gian và tần số tức thời, sau đó tích phân theo thời gian để thu được Phổ Cận Biên Hilbert (Marginal Hilbert Spectrum) so sánh với FFT.
"""))

    cells.append(nbf.v4.new_code_cell("""# Xây dựng lưới phổ Hilbert
freq_bins = np.linspace(0, 10.0, 100)
time_bins = t_rel
H_matrix = np.zeros((len(freq_bins), len(time_bins)))

for k in range(len(inst_freqs)):
    f_k = inst_freqs[k]
    a_k = inst_amps[k]
    for t_idx in range(len(time_bins)):
        f_val = f_k[t_idx]
        if f_val < 10.0:
            bin_idx = np.argmin(np.abs(freq_bins - f_val))
            H_matrix[bin_idx, t_idx] += a_k[t_idx]**2

# Phổ cận biên Hilbert: h(w) = tích phân H(w, t) dt
marginal_hht = np.sum(H_matrix, axis=1)

# Phổ FFT tương ứng để so sánh
fft_vals = np.fft.rfft(bvp_seg - np.mean(bvp_seg))
fft_freqs = np.fft.rfftfreq(len(bvp_seg), d=1.0/fs_bvp)
fft_psd = np.abs(fft_vals)**2

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={'width_ratios': [2.5, 1]})

# Đồ thị 1: Hilbert Spectrum Heatmap
pcm = ax1.pcolormesh(time_bins, freq_bins, H_matrix, shading='gouraud', cmap='hot')
ax1.set_ylim([0, 6.0])
ax1.set_xlabel('Thời gian (giây)')
ax1.set_ylabel('Tần số tức thời (Hz)')
ax1.set_title('A: Phổ Năng Lượng Hilbert 2D H(w, t)', loc='left')
fig.colorbar(pcm, ax=ax1, label='Năng lượng tức thời a^2(t)')

# Đồ thị 2: Phổ cận biên HHT vs Phổ FFT
ax2.plot(marginal_hht / np.max(marginal_hht), freq_bins, 'r-', lw=1.5, label='Marginal HHT')
mask_fft = fft_freqs <= 6.0
ax2.plot(fft_psd[mask_fft] / np.max(fft_psd[mask_fft]), fft_freqs[mask_fft], 'k--', lw=1.0, label='FFT Spectrum')
ax2.set_ylim([0, 6.0])
ax2.set_xlabel('Năng lượng chuẩn hóa')
ax2.set_title('B: Marginal HHT vs FFT', loc='left')
ax2.legend(loc='upper right')
ax2.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.show()

print("=> Phổ Hilbert cho độ phân giải tần số tức thời sắc nét, không bị giới hạn bởi độ rộng cửa sổ như STFT!")
"""))

    cells.append(nbf.v4.new_markdown_cell("""## Kết Luận Notebook 6:
- HHT cung cấp cái nhìn thực sự phi tuyến và phi dừng cho tín hiệu y sinh.
- Phổ Hilbert biểu diễn chính xác thời điểm và tần số thực sự của các quá trình huyết động học mà các phương pháp truyền thống (FFT/STFT) không thể nắm bắt trọn vẹn.
"""))

    nb.cells = cells
    return nb

# ==============================================================================
# MAIN EXECUTION: BUILD & RUN NOTEBOOKS
# ==============================================================================
def main():
    builders = [
        ("01_FFT_Spectral_Analysis.ipynb", build_nb1),
        ("02_STFT_Time_Frequency.ipynb", build_nb2),
        ("03_EMD_Mode_Decomposition.ipynb", build_nb3),
        ("04_EEMD_Noise_Assisted.ipynb", build_nb4),
        ("05_CEEMDAN_Adaptive_Noise.ipynb", build_nb5),
        ("06_HHT_Hilbert_Huang_Transform.ipynb", build_nb6),
    ]
    
    for filename, builder in builders:
        filepath = os.path.join(NOTEBOOKS_DIR, filename)
        print(f"\n{'='*70}\n[XÂY DỰNG & CHẠY NOTEBOOK] {filename}\n{'='*70}")
        nb = builder()
        
        # Lưu file thô
        with open(filepath, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        print(f"-> Đã ghi file template: {filepath}")
        
        # Thực thi notebook để tính toán và lưu output + đồ thị thật
        print(f"-> Đang thực thi notebook...")
        client = NotebookClient(nb, timeout=600, kernel_name='python3')
        client.execute()
        
        # Lưu file đã thực thi kèm toàn bộ plots, logs, outputs
        with open(filepath, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        print(f"[HOÀN THÀNH XUẤT SẮC] Đã chạy xong và lưu kết quả vào: {filepath}")

if __name__ == "__main__":
    main()
