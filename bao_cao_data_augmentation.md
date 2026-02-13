# Báo cáo Data Augmentation
## Tăng cường dữ liệu từ ~2,830 lên 25,000 dòng

---

## 1. Tổng quan

| Thông tin | Giá trị |
|---|---|
| **Dữ liệu gốc** | `data3_cleaned.csv` — 2,830 dòng, 7 cột |
| **Mục tiêu** | ~25,000 dòng |
| **Hệ số nhân** | ~8.8x |
| **Lý do** | Dữ liệu 2,830 dòng chưa đủ lớn để train model dự đoán hiệu quả, cần tăng lên ~25,000 dòng nhưng vẫn giữ chất lượng phân phối |

### Cấu trúc dữ liệu

| Cột | Ý nghĩa | Kiểu | Vai trò |
|---|---|---|---|
| `do_am` | Độ ẩm | Integer (90–97) | Feature độc lập |
| `nhiet_do` | Nhiệt độ | Integer (27–30) | Feature độc lập |
| `goc_nghieng` | Góc nghiêng | Categorical (20, 22) | Feature độc lập |
| `ca` | Ca làm việc | Categorical (1, 3) | Feature độc lập |
| `cong_suat_tb` | Công suất trung bình | Continuous | Feature độc lập |
| `san_luong` | Sản lượng | Continuous | **Phụ thuộc** (`= cong_suat_tb × 8`) |
| `tai_tieu_thu` | Tải tiêu thụ | Continuous | **Phụ thuộc** (`= cong_suat_tb × 40`) |

### Ràng buộc bắt buộc

Dữ liệu có **2 mối quan hệ xác định** phải được bảo toàn tuyệt đối trong quá trình augmentation:

```
san_luong     = cong_suat_tb × 8
tai_tieu_thu  = cong_suat_tb × 40
```

→ Mọi phương pháp augmentation chỉ cần sinh giá trị mới cho **5 features độc lập**, sau đó tính lại 2 cột phụ thuộc.

---

## 2. Các phương pháp được triển khai

### 2.1. Noise-based Augmentation

- **Nguyên lý**: Lấy mẫu ngẫu nhiên (with replacement) từ dữ liệu gốc, thêm nhiễu Gaussian có kiểm soát (noise_scale = 15% × std) vào các features liên tục và integer. Giữ nguyên features categorical.
- **Nhiễu được thêm theo nhóm** `(goc_nghieng, ca)` để bảo toàn phân phối điều kiện.
- **Clip giá trị** trong khoảng hợp lý (±5% so với min/max gốc).

### 2.2. KDE Conditional Sampling

- **Nguyên lý**: Sử dụng **Kernel Density Estimation** để học phân phối xác suất của từng feature theo nhóm `(goc_nghieng, ca)`. Sau đó lấy mẫu mới từ phân phối đã học.
- **Bandwidth tối ưu**: `cong_suat_tb = 2.0`, `do_am = 0.8`, `nhiet_do = 0.5` (điều chỉnh theo đặc tính từng feature).
- Tỷ lệ nhóm trong dữ liệu gốc được giữ nguyên khi sinh dữ liệu mới.
- Dữ liệu được sinh **hoàn toàn mới** (không phải copy + noise).

### 2.3. CTGAN (Conditional Tabular GAN)

- **Nguyên lý**: Mạng GAN chuyên dụng cho dữ liệu bảng từ thư viện **SDV**. Generator và Discriminator học phân phối đồng thời của tất cả features.
- **Cấu hình**: 300 epochs, batch_size = 500.
- Sau khi sinh, các cột phụ thuộc được tính lại để đảm bảo quan hệ xác định.

### 2.4. Gaussian Copula

- **Nguyên lý**: Chuyển đổi marginal distributions về dạng Gaussian, sau đó mô hình hóa sự phụ thuộc giữa các features bằng **correlation matrix**. Lấy mẫu từ không gian Gaussian rồi chuyển ngược về phân phối gốc.
- Thuộc thư viện **SDV**, nhanh hơn CTGAN đáng kể.

---

## 3. Kết quả đánh giá

### 3.1. Bảng tổng hợp metrics

| Phương pháp | Tổng dòng | Quan hệ bảo toàn | Mean diff (%) | Std diff (%) | KS p-value | Wasserstein | Corr diff |
|---|---|---|---|---|---|---|---|
| **Noise-based** | 25,000 | ✅ | 0.0706 | 0.1451 | 0.6268 | 0.3700 | 0.0030 |
| **KDE Conditional** | 24,999 | ✅ | **0.0131** | **0.1417** | 0.3193 | **0.3332** | 0.0527 |
| **CTGAN** | 25,000 | ✅ | 0.2367 | 4.2103 | 0.0000 | 1.8269 | 0.0075 |
| **Gaussian Copula** | 25,000 | ✅ | 0.0863 | 13.8595 | 0.0000 | 8.1275 | 0.0051 |

### 3.2. Giải thích các chỉ số

| Chỉ số | Ý nghĩa | Tiêu chí tốt |
|---|---|---|
| **Mean diff (%)** | Sai lệch giá trị trung bình `cong_suat_tb` giữa dữ liệu gốc và synthetic | Càng thấp càng tốt (< 0.5%) |
| **Std diff (%)** | Sai lệch độ lệch chuẩn `cong_suat_tb` | Càng thấp càng tốt (< 5%) |
| **KS p-value** | Kiểm định Kolmogorov-Smirnov — p > 0.05 nghĩa là phân phối synthetic giống gốc | p > 0.05 → PASS |
| **Wasserstein** | Khoảng cách Wasserstein giữa 2 phân phối | Càng thấp càng tốt |
| **Corr diff** | Trung bình sai lệch correlation giữa các features | Càng thấp càng tốt (< 0.1) |

### 3.3. Phân tích chi tiết từng phương pháp

#### Noise-based
- ✅ KS test **PASS** (p = 0.6268) — phân phối rất giống gốc
- ✅ Correlation bảo toàn tốt nhất (diff = 0.0030)
- ✅ Mean và Std sai lệch rất nhỏ
- ⚠️ Dữ liệu mới chỉ là **biến thể nhỏ** của dữ liệu gốc → ít đa dạng, có nguy cơ model overfit vào pattern lặp lại

#### KDE Conditional ⭐
- ✅ KS test **PASS** (p = 0.3193) — phân phối giống gốc
- ✅ **Mean diff thấp nhất** (0.0131%) — bảo toàn giá trị trung bình tốt nhất
- ✅ **Std diff thấp nhất** (0.1417%) — bảo toàn phương sai tốt nhất
- ✅ **Wasserstein thấp nhất** (0.3332) — phân phối tổng thể gần gốc nhất
- ✅ Sinh dữ liệu **hoàn toàn mới** → đa dạng hơn Noise-based
- ⚠️ Correlation diff hơi cao hơn (0.0527) nhưng vẫn ở mức chấp nhận được

#### CTGAN
- ❌ KS test **FAIL** (p = 0.0000) — phân phối khác biệt có ý nghĩa thống kê so với gốc
- ❌ Std diff cao (4.21%) — phương sai bị thay đổi đáng kể
- ❌ Wasserstein cao (1.8269) — gấp ~5.5x so với KDE
- ✅ Correlation bảo toàn tốt (0.0075)
- → Không phù hợp cho dataset này (dataset gốc quá nhỏ cho GAN)

#### Gaussian Copula
- ❌ KS test **FAIL** (p = 0.0000) — phân phối khác biệt rõ rệt
- ❌ **Std diff rất cao** (13.86%) — phương sai bị bóp méo nghiêm trọng
- ❌ **Wasserstein rất cao** (8.1275) — gấp ~24x so với KDE
- → Kém nhất trong 4 phương pháp, không phù hợp

---

## 4. Lựa chọn tối ưu: KDE Conditional Sampling

### Lý do chọn KDE Conditional

| Tiêu chí | KDE Conditional | Lý do vượt trội |
|---|---|---|
| Bảo toàn phân phối | KS p = 0.3193 (PASS) | Chỉ 1 trong 2 phương pháp pass KS test |
| Bảo toàn mean | 0.0131% | **Tốt nhất** — sai lệch gần bằng 0 |
| Bảo toàn std | 0.1417% | **Tốt nhất** — phương sai gần như không đổi |
| Wasserstein distance | 0.3332 | **Thấp nhất** — phân phối gần gốc nhất |
| Đa dạng dữ liệu | Cao | Sinh dữ liệu mới thực sự, không chỉ là copy + noise |
| Quan hệ xác định | ✅ Bảo toàn | `san_luong = cong_suat_tb × 8`, `tai_tieu_thu = cong_suat_tb × 40` |
| Không cần thư viện ngoài | ✅ | Chỉ dùng scipy, sklearn (có sẵn) |

### So sánh trực tiếp KDE vs Noise-based (2 phương pháp pass KS test)

Mặc dù Noise-based có correlation diff thấp hơn (0.003 vs 0.053), **KDE được ưu tiên** vì:

1. **Đa dạng hơn**: KDE sinh dữ liệu mới hoàn toàn từ phân phối đã học, trong khi Noise-based chỉ tạo biến thể nhỏ của dữ liệu gốc. Khi train model, sự đa dạng này giúp model **generalize tốt hơn**.

2. **Mean preservation tốt hơn**: Sai lệch mean 0.0131% (KDE) vs 0.0706% (Noise) — KDE chính xác hơn ~5.4 lần.

3. **Wasserstein thấp hơn**: 0.3332 vs 0.3700 — phân phối tổng thể gần gốc hơn.

4. **Giảm nguy cơ overfit**: Dữ liệu Noise-based có thể khiến model "ghi nhớ" pattern của dữ liệu gốc thay vì học pattern tổng quát.

---

## 5. Dữ liệu đầu ra

| File | Mô tả | Số dòng |
|---|---|---|
| `data3_augmented_25k.csv` | Dataset augmented (sẵn sàng train) | 24,999 |
| `data3_augmented_25k.xlsx` | Phiên bản Excel | 24,999 |
| `data3_augmented_25k_with_source.csv` | Có cột `source` phân biệt original/synthetic | 24,999 |

### Phân bổ dữ liệu

- **Original**: 2,830 dòng (11.3%)
- **Synthetic**: 22,169 dòng (88.7%)

---

## 6. Khuyến nghị khi train model

1. **Validation trên dữ liệu gốc**: Luôn tách riêng dữ liệu gốc (original) làm test set để đánh giá hiệu quả thực tế của model. Sử dụng cột `source` trong file `data3_augmented_25k_with_source.csv` để lọc.

2. **Cross-validation**: Sử dụng stratified k-fold cross-validation, đảm bảo mỗi fold đều có đủ dữ liệu gốc.

3. **Theo dõi overfit**: Nếu model cho accuracy cao trên synthetic nhưng thấp trên original → dấu hiệu overfit vào synthetic data. Khi đó cân nhắc giảm tỷ lệ synthetic hoặc điều chỉnh bandwidth KDE.

4. **Không augment test set**: Test set **chỉ nên chứa dữ liệu gốc** để phản ánh đúng hiệu quả model trong thực tế.

---

*Báo cáo được tạo tự động từ notebook `data_augmentation.ipynb`*  
*Ngày: 13/02/2026*
