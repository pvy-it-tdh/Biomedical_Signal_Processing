# Cẩm Nang Xử Lý Tín Hiệu Y Sinh: Phân Tích Phổ & Thời Gian - Tần Số
*(Biomedical Signal Processing: Spectral & Time-Frequency Analysis)*

Tài liệu này tổng hợp toàn diện cơ sở lý thuyết, công thức toán học, thuật toán, ưu/nhược điểm và hướng dẫn ứng dụng thực tế trong xử lý tín hiệu y sinh (ECG, EEG, EMG, PPG) cho các phương pháp:
1. **FFT** (Fast Fourier Transform)
2. **STFT** (Short-Time Fourier Transform)
3. **EMD** (Empirical Mode Decomposition)
4. **EEMD** (Ensemble Empirical Mode Decomposition)
5. **CEEMDAN** (Complete Ensemble Empirical Mode Decomposition with Adaptive Noise)
6. **HHT** (Hilbert-Huang Transform)

---

## Mục lục
1. [Bản chất tín hiệu Y sinh và Thách thức](#1-bản-chất-tín-hiệu-y-sinh-và-thách-thức)
2. [Biến đổi Fourier nhanh (FFT - Fast Fourier Transform)](#2-biến-đổi-fourier-nhanh-fft)
3. [Biến đổi Fourier thời gian ngắn (STFT - Short-Time Fourier Transform)](#3-biến-đổi-fourier-thời-gian-ngắn-stft)
4. [Phân tách Mode thực nghiệm (EMD - Empirical Mode Decomposition)](#4-phân-tách-mode-thực-nghiệm-emd)
5. [Tập hợp EMD (EEMD - Ensemble Empirical Mode Decomposition)](#5-tập-hợp-emd-eemd)
6. [CEEMDAN (Complete EEMD with Adaptive Noise)](#6-ceemdan)
7. [Biến đổi Hilbert-Huang (HHT - Hilbert-Huang Transform)](#7-biến-đổi-hilbert-huang-hht)
8. [Bảng so sánh tổng hợp các phương pháp](#8-bảng-so-sánh-tổng-hợp-các-phương-pháp)
9. [Khuyến nghị triển khai trên tín hiệu ECG, EEG, EMG, PPG](#9-khuyến-nghị-triển-khai-trên-tín-hiệu-ecg-eeg-emg-ppg)
10. [Mã nguồn Python minh họa](#10-mã-nguồn-python-minh-họa)

---

## 1. Bản chất tín hiệu Y sinh và Thách thức

Hầu hết tín hiệu sinh lý học (Physiological signals) xuất phát từ các hệ thống sinh học phức tạp, có cơ chế điều hòa phi tuyến và biến đổi theo trạng thái sinh lý/bệnh lý:
* **Tính phi dừng (Non-stationarity):** Đặc tính thống kê (kỳ vọng, phương sai, tần số thành phần) biến thiên theo thời gian. Ví dụ: nhịp tim thay đổi, co thắt cơ bắp, sóng co giật động kinh (epileptic spikes).
* **Tính phi tuyến (Non-linearity):** Sự truyền dẫn qua mô, sự đáp ứng thụ thể hay cơ chế feedback thần kinh đều có bản chất phi tuyến.
* **Tỷ số tín hiệu trên nhiễu (SNR) thấp:** Thường lẫn nhiễu trôi đường đẳng điện (baseline wander), nhiễu điện cơ (EMG noise), nhiễu chuyển động (motion artifacts), và can nhiễu tần số điện lưới (50/60 Hz).

Do đó, các công cụ phân tích tín hiệu cần tiến hóa từ **phân tích miền tần số tĩnh (FFT)** $\rightarrow$ **phân tích thời gian - tần số tuyến tính (STFT)** $\rightarrow$ **phân tích tự thích nghi dữ liệu, phi tuyến & phi dừng (EMD $\rightarrow$ EEMD $\rightarrow$ CEEMDAN $\rightarrow$ HHT)**.

---

## 2. Biến đổi Fourier nhanh (FFT)

### 2.1. Cơ sở lý thuyết
Biến đổi Fourier rời rạc (DFT - Discrete Fourier Transform) chiếu tín hiệu từ miền thời gian sang một hệ cơ sở trực giao xác định trước: các sóng hàm điều hòa (sin/cosine) có biên độ và tần số cố định, kéo dài vô hạn trong thời gian. FFT (thuật toán Cooley-Tukey) là thuật toán tối ưu hóa tính toán DFT với độ phức tạp giảm từ $\mathcal{O}(N^2)$ xuống $\mathcal{O}(N \log N)$.

### 2.2. Công thức toán học
Với tín hiệu rời rạc $x[n]$ có độ dài $N$ mẫu ($n = 0, 1, \dots, N-1$), DFT được định nghĩa:
$$X[k] = \sum_{n=0}^{N-1} x[n] \, e^{-j \frac{2\pi}{N} k n}, \quad k = 0, 1, \dots, N-1$$

Tần số vật lý tương ứng với chỉ số $k$:
$$f_k = \frac{k \cdot f_s}{N}$$
*(trong đó $f_s$ là tần số lấy mẫu - sampling rate).*

Mật độ phổ công suất (PSD - Power Spectral Density):
$$P[k] = \frac{1}{N} |X[k]|^2$$

### 2.3. Ưu điểm & Hạn chế
* **Ưu điểm:**
  - Tốc độ tính toán cực nhanh, tối ưu hóa phần cứng tốt.
  - Thích hợp phân tích các thành phần tuần hoàn tĩnh, xác định phổ công suất toàn cục (ví dụ: phát hiện nhiễu 50Hz, phân tích dải tần trung bình).
* **Hạn chế:**
  - **Mất hoàn toàn thông tin định vị thời gian:** Ta chỉ biết tần số $f$ xuất hiện trong tín hiệu nhưng không biết nó xuất hiện ở thời điểm nào.
  - **Giả định tính dừng (Stationarity):** Coi tín hiệu lặp lại tuần hoàn vô hạn. Khi áp dụng cho tín hiệu y sinh phi dừng (như ECG có loạn nhịp, EEG có gai nhọn), FFT tạo ra phổ giả và che lấp các biến đổi cục bộ.
  - **Rò rỉ phổ (Spectral leakage):** Do tín hiệu thực tế bị cắt cụt (truncation), cần dùng cửa sổ (windowing như Hann, Hamming) để giảm thiểu.

---

## 3. Biến đổi Fourier thời gian ngắn (STFT)

### 3.1. Cơ sở lý thuyết
Để khắc phục nhược điểm mất thông tin thời gian của FFT, Dennis Gabor (1946) đề xuất STFT: chia tín hiệu thành các đoạn ngắn bằng một hàm cửa sổ trượt $w(t)$ theo thời gian, giả định rằng trong phạm vi cửa sổ hẹp, tín hiệu là **gần như dừng (quasi-stationary)**. Sau đó áp dụng Fourier Transform cho từng đoạn.

### 3.2. Công thức toán học
Dạng liên tục:
$$\text{STFT}_x(\tau, \omega) = \int_{-\infty}^{\infty} x(t) w(t - \tau) e^{-j \omega t} \, dt$$

Dạng rời rạc:
$$X[m, k] = \sum_{n=0}^{L-1} x[n + mR] w[n] \, e^{-j \frac{2\pi}{L} k n}$$
*(với $w[n]$ là cửa sổ độ dài $L$, $R$ là độ dời bước nhảy - hop size, $m$ là chỉ số thời gian, $k$ là chỉ số tần số).*

Biểu đồ Spectrogram (phổ thời gian - tần số):
$$S[m, k] = |X[m, k]|^2$$

### 3.3. Nguyên lý bất định Heisenberg-Gabor
Độ phân giải thời gian ($\Delta t$) và độ phân giải tần số ($\Delta f$) bị ràng buộc bởi nguyên lý bất định:
$$\Delta t \cdot \Delta f \ge \frac{1}{4\pi}$$
* Nếu chọn cửa sổ ngắn (narrow window): $\Delta t$ nhỏ (độ phân giải thời gian cao) $\rightarrow$ $\Delta f$ lớn (độ phân giải tần số kém).
* Nếu chọn cửa sổ dài (wide window): $\Delta f$ nhỏ (độ phân giải tần số cao) $\rightarrow$ $\Delta t$ lớn (độ phân giải thời gian kém).

### 3.4. Ưu điểm & Hạn chế
* **Ưu điểm:** Cung cấp biểu diễn 2D Thời gian - Tần số (Spectrogram), dễ diễn giải, chuẩn hóa tốt.
* **Hạn chế:** Kích thước cửa sổ cố định cho toàn bộ dải tần (không thể vừa bắt nhịp nhanh tần số cao vừa phân giải chi tiết tần số thấp). Vẫn sử dụng cơ sở hàm điều hòa sin/cos cố định, không thích ứng theo hình thái sóng y sinh.

---

## 4. Phân tách Mode thực nghiệm (EMD)

### 4.1. Cơ sở lý thuyết
Norden E. Huang et al. (1998) đề xuất EMD như một phương pháp đột phá: **hoàn toàn tự thích nghi (data-driven)**, không dựa vào hàm cơ sở định sẵn (như Fourier hay Wavelet). EMD phân tách tín hiệu phức tạp thành một tập hợp hữu hạn các hàm thành phần gọi là **Hàm Mode nội tại (IMF - Intrinsic Mode Function)** và một hàm xu hướng dư (Residue).

### 4.2. Định nghĩa hàm IMF
Một hàm $c(t)$ được coi là một IMF nếu thỏa mãn 2 điều kiện ngặt nghèo:
1. Số điểm cực trị (cực đại + cực tiểu) và số lần đi qua điểm 0 (zero-crossings) bằng nhau hoặc sai khác nhau tối đa là 1 trên toàn bộ miền dữ liệu.
2. Tại bất kỳ điểm nào, giá trị trung bình của đường bao trên (tạo bởi các cực đại) và đường bao dưới (tạo bởi các cực tiểu) phải bằng 0:
   $$\frac{e_{\text{upper}}(t) + e_{\text{lower}}(t)}{2} = 0$$

### 4.3. Thuật toán Sifting Process (Sàng lọc)
Cho tín hiệu $x(t)$:
1. Xác định tất cả các cực đại và cực tiểu cục bộ của tín hiệu $h_0(t) = x(t)$.
2. Nội suy spline bậc 3 (cubic spline) nối các cực đại tạo thành đường bao trên $e_{\text{upper}}(t)$, nối các cực tiểu tạo thành đường bao dưới $e_{\text{lower}}(t)$.
3. Tính đường trung bình: $m_1(t) = \frac{e_{\text{upper}}(t) + e_{\text{lower}}(t)}{2}$.
4. Tách thành phần chi tiết: $h_1(t) = h_0(t) - m_1(t)$.
5. Kiểm tra $h_1(t)$ có thỏa mãn điều kiện IMF không:
   - Nếu **không**: lặp lại bước 1-4 trên $h_1(t)$ cho đến khi đạt tiêu chuẩn dừng (dựa trên sai số Cauchy hoặc số lần lặp). Kết quả là $\text{IMF}_1(t) = h_1(t)$.
6. Tính phần dư: $r_1(t) = x(t) - \text{IMF}_1(t)$.
7. Coi $r_1(t)$ là tín hiệu ban đầu và lặp lại toàn bộ quy trình để tìm $\text{IMF}_2, \text{IMF}_3, \dots, \text{IMF}_K$.
8. Dừng khi phần dư $r_K(t)$ trở thành một hàm đơn điệu hoặc không còn đủ ít nhất 2 cực trị.

Tín hiệu tái cấu trúc:
$$x(t) = \sum_{k=1}^{K} \text{IMF}_k(t) + r_K(t)$$

### 4.4. Hạn chế cốt lõi của EMD: Trộn Mode (Mode Mixing)
* **Hiện tượng:** Một IMF chứa các thành phần tần số cách biệt nhau rất xa ở các thời điểm khác nhau; hoặc ngược lại, cùng một thành phần tần số bị chia rẽ sang nhiều IMF khác nhau.
* **Nguyên nhân:** Xuất hiện nhiễu xung, biên độ đổi đột ngột (intermittent signals), hoặc mật độ điểm cực trị biến động mạnh, làm sai lệch việc nội suy spline.

---

## 5. Tập hợp EMD (EEMD)

### 5.1. Cơ sở lý thuyết
Wu & Huang (2009) giới thiệu EEMD để giải quyết triệt để vấn đề **Mode Mixing** của EMD.
Nguyên lý: **Nhiễu trắng Gauss (Gaussian White Noise)** có phổ phân bố đồng đều trên toàn dải tần. Khi cộng nhiễu trắng vào tín hiệu, nhiễu sẽ đóng vai trò lấp đầy các khoảng trống tỉ lệ thời gian - tần số, giúp quá trình nội suy đường bao của EMD diễn ra liên tục, đưa các thành phần tần số về đúng thang đo tự nhiên.

### 5.2. Thuật toán EEMD
1. Thêm một chuỗi nhiễu trắng độc lập $w_i(t) \sim \mathcal{N}(0, \sigma^2)$ vào tín hiệu ban đầu:
   $$x_i(t) = x(t) + \epsilon \cdot w_i(t), \quad i = 1, 2, \dots, M$$
   *(thông thường chọn biên độ nhiễu $\epsilon \approx 0.1 - 0.2 \times \text{std}(x)$, số lần thử $M \approx 100 - 500$).*
2. Thực hiện EMD chuẩn trên từng tín hiệu hỗn hợp $x_i(t)$, thu được tập hợp các $\text{IMF}_{k, i}(t)$.
3. Tính trung bình ensemble cho từng IMF:
   $$\overline{\text{IMF}}_k(t) = \frac{1}{M} \sum_{i=1}^M \text{IMF}_{k, i}(t)$$

Nhờ tính chất triệt tiêu thống kê của nhiễu trắng ($1/\sqrt{M}$), khi $M$ đủ lớn, thành phần nhiễu cộng thêm sẽ tiến về 0.

### 5.3. Hạn chế của EEMD
* **Không tái cấu trúc hoàn hảo (Incomplete reconstruction):** Do số lượng ensemble $M$ là hữu hạn, phần nhiễu dư (residual noise) vẫn tồn tại: $\sum \overline{\text{IMF}}_k(t) + r(t) \neq x(t)$.
* **Số lượng IMF không đồng nhất:** Các phép thử khác nhau có thể sinh ra số lượng IMF khác nhau, gây khó khăn khi tính trung bình.
* **Chi phí tính toán cao:** Chạy EMD lặp lại hàng trăm lần.

---

## 6. CEEMDAN

### 6.1. Cơ sở lý thuyết
Torres et al. (2011) đề xuất **CEEMDAN** (Complete Ensemble Empirical Mode Decomposition with Adaptive Noise). Đây là biến thể cải tiến vượt bậc so với EEMD:
* Đảm bảo **tái cấu trúc chính xác tuyệt đối** (sai số số học cỡ $10^{-16}$).
* Giảm đáng kể số lần sàng lọc (computational cost giảm $\sim 50\%$).
* Thêm nhiễu thích nghi có kiểm soát tại **từng bước phân tách** thay vì chỉ thêm ở đầu vào.

### 6.2. Thuật toán CEEMDAN
Gọi $E_j(\cdot)$ là toán tử EMD trích xuất IMF thứ $j$ của một tín hiệu. $w^{(i)}(t)$ là các bản thực hiện của nhiễu trắng chuẩn hóa $\mathcal{N}(0, 1)$.

1. **Tìm $\widetilde{\text{IMF}}_1(t)$:**
   Tạo $x^{(i)}(t) = x(t) + \beta_0 w^{(i)}(t)$.
   $$\widetilde{\text{IMF}}_1(t) = \frac{1}{M} \sum_{i=1}^{M} E_1(x^{(i)}(t))$$
   Phần dư thứ nhất: $r_1(t) = x(t) - \widetilde{\text{IMF}}_1(t)$.

2. **Tìm $\widetilde{\text{IMF}}_2(t)$:**
   Cộng nhiễu được lọc qua toán tử EMD vào phần dư:
   $$\widetilde{\text{IMF}}_2(t) = \frac{1}{M} \sum_{i=1}^{M} E_1 \left( r_1(t) + \beta_1 E_1(w^{(i)}(t)) \right)$$
   Phần dư thứ hai: $r_2(t) = r_1(t) - \widetilde{\text{IMF}}_2(t)$.

3. **Tổng quát cho mode thứ $k$ ($k = 3, \dots, K$):**
   $$\widetilde{\text{IMF}}_k(t) = \frac{1}{M} \sum_{i=1}^{M} E_1 \left( r_{k-1}(t) + \beta_{k-1} E_{k-1}(w^{(i)}(t)) \right)$$
   Phần dư: $r_k(t) = r_{k-1}(t) - \widetilde{\text{IMF}}_k(t)$.

4. Lặp lại cho đến khi $r_k(t)$ không còn phân tách được nữa.

Tín hiệu tái cấu trúc (chính xác tuyệt đối, không còn sai số nhiễu dư):
$$x(t) = \sum_{k=1}^{K} \widetilde{\text{IMF}}_k(t) + r_K(t)$$

---

## 7. Biến đổi Hilbert-Huang (HHT)

### 7.1. Cơ sở lý thuyết
HHT không phải là một thuật toán đơn lẻ, mà là một **quy trình 2 giai đoạn** hoàn chỉnh:
$$\text{HHT} = \text{Mode Decomposition (EMD/EEMD/CEEMDAN)} + \text{Hilbert Spectral Analysis (HSA)}$$

Mục đích: Sau khi tín hiệu được chia thành các IMF đơn thành phần (mono-component), áp dụng biến đổi Hilbert lên từng IMF để tính **Biên độ tức thời (Instantaneous Amplitude)** và **Tần số tức thời (Instantaneous Frequency)**.

### 7.2. Toán học biến đổi Hilbert & Tín hiệu giải tích (Analytic Signal)
Với mỗi IMF $c_k(t)$, biến đổi Hilbert được định nghĩa:
$$\mathcal{H}\{c_k(t)\} = \tilde{c}_k(t) = \frac{1}{\pi} \, \text{P.V.} \int_{-\infty}^{\infty} \frac{c_k(\tau)}{t - \tau} \, d\tau$$
*(P.V. là giá trị chính Cauchy - Cauchy Principal Value).*

Tạo tín hiệu giải tích phức:
$$z_k(t) = c_k(t) + j \tilde{c}_k(t) = a_k(t) \, e^{j \theta_k(t)}$$

Trong đó:
* **Biên độ tức thời:**
  $$a_k(t) = \sqrt{c_k^2(t) + \tilde{c}_k^2(t)}$$
* **Pha tức thời:**
  $$\theta_k(t) = \arctan\left(\frac{\tilde{c}_k(t)}{c_k(t)}\right)$$
* **Tần số tức thời (rad/s và Hz):**
  $$\omega_k(t) = \frac{d\theta_k(t)}{dt} \quad \Longrightarrow \quad f_k(t) = \frac{1}{2\pi} \frac{d\theta_k(t)}{dt}$$

### 7.3. Phổ Hilbert $H(\omega, t)$ và Phổ biên độ Hilbert
Biểu diễn phổ biên độ phân bố liên tục theo cả thời gian và tần số tức thời:
$$H(\omega, t) = \text{Re} \sum_{k=1}^K a_k(t) \, e^{j \int \omega_k(t) dt}$$

Phổ cận biên Hilbert (Marginal Hilbert Spectrum) - đo tổng năng lượng tích lũy tại mỗi tần số:
$$h(\omega) = \int_0^T H(\omega, t) \, dt$$

---

## 8. Bảng so sánh tổng hợp các phương pháp

| Đặc tính / Phương pháp | FFT | STFT | EMD | EEMD | CEEMDAN | HHT (toàn diện) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Bản chất hàm cơ sở** | Tiên nghiệm (Sin/Cos) | Tiên nghiệm (Windowed Sin/Cos) | Tự thích nghi (Data-driven) | Tự thích nghi | Tự thích nghi | Tự thích nghi |
| **Giả định tín hiệu** | Tuyến tính, Dừng | Tuyến tính, Dừng cục bộ | Phi tuyến, Phi dừng | Phi tuyến, Phi dừng | Phi tuyến, Phi dừng | Phi tuyến, Phi dừng |
| **Độ phân giải Thời gian - Tần số** | Không có miền thời gian | Giới hạn bởi Heisenberg ($\Delta t \cdot \Delta f \ge \frac{1}{4\pi}$) | Cục bộ theo từng IMF | Cục bộ theo từng IMF | Cục bộ chính xác theo từng IMF | Tần số tức thời liên tục theo thời gian |
| **Vấn đề Mode Mixing** | Không áp dụng | Không áp dụng | **Nghiêm trọng** | Đã khắc phục phần lớn | **Khắc phục triệt để** | Kế thừa từ thuật toán phân tách |
| **Khả năng tái cấu trúc** | Hoàn hảo | Tùy thuộc overlap-add | Hoàn hảo | **Sai số dư nhiễu** | **Hoàn hảo (Exact)** | Hoàn hảo |
| **Độ phức tạp tính toán** | $\mathcal{O}(N \log N)$ (Rất nhanh) | $\mathcal{O}(M \cdot N \log N)$ (Nhanh) | Trung bình | Rất cao ($M \times \text{EMD}$) | Cao (nhưng ít hơn EEMD $\sim 50\%$) | Bằng CEEMDAN + $\mathcal{O}(N \log N)$ |
| **Khả năng chạy Real-time** | Rất cao | Rất cao (Causal sliding buffer) | Khó (hiệu ứng biên) | Rất khó | Rất khó | Chủ yếu Offline / Batch |

---

## 9. Khuyến nghị triển khai trên tín hiệu ECG, EEG, EMG, PPG

### 9.1. Điện tâm đồ (ECG)
* **Khử trôi đường đẳng điện (Baseline Wander < 0.5 Hz):** 
  - Phân tách CEEMDAN: các IMF bậc cao (IMF cuối) và phần dư $r(t)$ chứa thành phần tần số cực thấp. Loại bỏ các IMF này rồi cộng lại tái cấu trúc tín hiệu giúp khử baseline wander mà **không gây méo pha** sóng ST và sóng T (vấn đề mà bộ lọc IIR thông thường dễ mắc phải).
* **Phát hiện phức bộ QRS & R-peak:** Các IMF bậc thấp (IMF 1-3) tập trung năng lượng dải 10-25 Hz tương ứng với đỉnh R sắc nhọn.

### 9.2. Điện não đồ (EEG)
* **Phân rã nhịp sinh lý (Delta, Theta, Alpha, Beta, Gamma):**
  - EMD/CEEMDAN tự động tách các nhịp tự nhiên của não mà không cần gán cứng dải cắt tần số sắc nhọn như FIR/IIR.
* **Lọc nhiễu chớp mắt (EOG Artifact) & co cơ (EMG Artifact):**
  - Nhiễu chớp mắt thường tập trung ở IMF biên độ lớn tần số thấp tại các kênh trán (Fp1, Fp2). Kết hợp CEEMDAN + ICA (Independent Component Analysis) là pipeline khử nhiễu tiêu chuẩn vàng trong EEG.

### 9.3. Điện cơ (EMG)
* **Phân tích mỏi cơ (Muscle Fatigue):**
  - Tần số tức thời từ HHT phản ánh sự dịch chuyển phổ tần số trung vị (Median Frequency shift) nhạy hơn nhiều so với STFT khi cơ co ngắt quãng hoặc lực co thay đổi liên tục.

### 9.4. Quang thể tích thể tích (PPG)
* **Khử nhiễu chuyển động (Motion Artifact - MA):**
  - Tín hiệu PPG quang học rất nhạy cảm với áp lực tiếp xúc và chuyển động ngón tay. CEEMDAN phân tách nhiễu chuyển động sang các IMF riêng biệt, cho phép trích xuất lại dạng sóng nhịp mạch và tính toán độ bão hòa oxy $SpO_2$ tin cậy.

---

## 10. Mã nguồn Python minh họa

Dưới đây là đoạn mã Python sử dụng thư viện `scipy`, `numpy` và `PyEMD` (hoặc `emd`) minh họa việc áp dụng các phương pháp:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, hilbert

# 1. Giả lập tín hiệu sinh lý phi dừng (non-stationary signal)
# Ví dụ: Sóng tổng hợp có biến đổi tần số theo thời gian + nhiễu
fs = 500  # 500 Hz
t = np.linspace(0, 4, 4 * fs, endpoint=False)

# Thành phần tần số 5 Hz ở 2 giây đầu, chuyển sang 15 Hz ở 2 giây sau
sig = np.sin(2 * np.pi * 5 * t) * (t < 2) + np.sin(2 * np.pi * 15 * t) * (t >= 2)
noise = 0.3 * np.random.normal(size=t.shape)
x = sig + noise

# -------------------------------------------------------------
# A. Fast Fourier Transform (FFT)
# -------------------------------------------------------------
N = len(x)
X_fft = np.fft.rfft(x)
freqs_fft = np.fft.rfftfreq(N, 1/fs)
psd_fft = np.abs(X_fft)**2 / N

# -------------------------------------------------------------
# B. Short-Time Fourier Transform (STFT)
# -------------------------------------------------------------
f_stft, t_stft, Zxx = stft(x, fs=fs, nperseg=128, noverlap=96)
spectrogram = np.abs(Zxx)

# -------------------------------------------------------------
# C. CEEMDAN & Hilbert-Huang Transform (HHT)
# -------------------------------------------------------------
# Lưu ý: Cài đặt qua `pip install EMD-signal`
try:
    from PyEMD import CEEMDAN
    
    ceemdan = CEEMDAN()
    # Phân tách CEEMDAN thu được các IMFs
    imfs = ceemdan(x)
    n_imfs = imfs.shape[0]

    # Tính phổ Hilbert cho từng IMF
    inst_amps = []
    inst_freqs = []
    for k in range(n_imfs):
        analytic_signal = hilbert(imfs[k])
        amp = np.abs(analytic_signal)
        # Pha và tần số tức thời (đạo hàm bậc 1 của pha)
        phase = np.unwrap(np.angle(analytic_signal))
        inst_freq = np.diff(phase) / (2.0 * np.pi) * fs
        
        inst_amps.append(amp)
        inst_freqs.append(inst_freq)
    
    print(f"CEEMDAN phân tách thành công {n_imfs} IMFs.")
except ImportError:
    print("Vui lòng cài đặt thư viện PyEMD: pip install EMD-signal")
```

---

## 11. Hệ Thống Notebook Thực Hành Trên Dữ Liệu WESAD S1

Toàn bộ các phương pháp trên đã được hiện thực hóa và chạy thực nghiệm trực tiếp trên dữ liệu y sinh đa kênh của **WESAD Subject S1** (`ECG`, `PPG/BVP`, `EMG`, `Resp`) tại thư mục [notebooks](file:///e:/Study/Biomedical_Signal_Processing/notebooks):

| STT | Notebook | Phương pháp | Tín hiệu khảo sát | Nội dung thực nghiệm & Đồ thị đã chạy |
| :---: | :--- | :---: | :---: | :--- |
| **01** | [01_FFT_Spectral_Analysis.ipynb](file:///e:/Study/Biomedical_Signal_Processing/notebooks/01_FFT_Spectral_Analysis.ipynb) | **FFT & Welch PSD** | ECG (700Hz), PPG (64Hz), EMG (700Hz) | Khảo sát rò rỉ phổ (*Leakage*), so sánh cửa sổ Hann vs Chữ nhật, phát hiện nhiễu điện lưới 50 Hz trên ECG, ước lượng nhịp tim từ đỉnh phổ PPG. |
| **02** | [02_STFT_Time_Frequency.ipynb](file:///e:/Study/Biomedical_Signal_Processing/notebooks/02_STFT_Time_Frequency.ipynb) | **STFT & Spectrogram** | EMG (700Hz), PPG (64Hz) | Kiểm chứng nguyên lý bất định Heisenberg-Gabor với 3 kích thước cửa sổ trượt, biểu đồ phổ Spectrogram khi cơ co thắt và chuyển trạng thái vận động. |
| **03** | [03_EMD_Mode_Decomposition.ipynb](file:///e:/Study/Biomedical_Signal_Processing/notebooks/03_EMD_Mode_Decomposition.ipynb) | **EMD** | PPG (64Hz) | Phân tách các hàm mode nội tại (IMFs), kiểm chứng tính bảo toàn năng lượng và tái tạo hoàn hảo, minh họa trực quan hiện tượng **Mode Mixing**. |
| **04** | [04_EEMD_Noise_Assisted.ipynb](file:///e:/Study/Biomedical_Signal_Processing/notebooks/04_EEMD_Noise_Assisted.ipynb) | **EEMD** | PPG (64Hz) có nhiễu xung | Bổ sung nhiễu trắng Gauss giải quyết mode mixing, so sánh đối đầu trực diện EMD vs EEMD, đo lường sai số tồn dư (*residual noise*). |
| **05** | [05_CEEMDAN_Adaptive_Noise.ipynb](file:///e:/Study/Biomedical_Signal_Processing/notebooks/05_CEEMDAN_Adaptive_Noise.ipynb) | **CEEMDAN** | ECG (700Hz) | Kiểm chứng tái cấu trúc chính xác tuyệt đối ($\text{Error} \approx 0$), ứng dụng **khử trôi đường đẳng điện (Baseline Wander)** mà không làm biến dạng sóng ST-T. |
| **06** | [06_HHT_Hilbert_Huang_Transform.ipynb](file:///e:/Study/Biomedical_Signal_Processing/notebooks/06_HHT_Hilbert_Huang_Transform.ipynb) | **HHT (CEEMDAN + HSA)** | PPG (64Hz) | Tín hiệu giải tích, tính toán biên độ và tần số tức thời $f(t)$, vẽ biểu đồ nhiệt **Phổ Năng lượng Hilbert 2D $H(\omega, t)$** và so sánh Phổ cận biên với FFT. |

---

## Tài liệu tham khảo cốt lõi
1. **Huang, N. E., et al. (1998).** "The empirical mode decomposition and the Hilbert spectrum for nonlinear and non-stationary time series analysis." *Proceedings of the Royal Society of London. Series A*, 454(1971), 903-995.
2. **Wu, Z., & Huang, N. E. (2009).** "Ensemble empirical mode decomposition: a noise-assisted data analysis method." *Advances in Adaptive Data Analysis*, 1(01), 1-41.
3. **Torres, M. E., et al. (2011).** "A complete ensemble empirical mode decomposition with adaptive noise." *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 4144-4147.
4. **Cohen, M. X. (2014).** *Analyzing Neural Time Series Data: Theory and Practice.* MIT Press.
