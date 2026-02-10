# BẢN TÓM TẮT PROJECT: DỰ BÁO CÔNG SUẤT BĂNG TẢI BẰNG MẠNG NEURON NHÂN TẠO (ANN)

## 1. MỤC TIÊU

Xây dựng mô hình mạng neuron nhân tạo (Artificial Neural Network - ANN) bằng Python để **dự báo công suất đầu ra (tải tiêu thụ)** của hệ thống băng tải dựa trên các thông số vận hành.

---

## 2. DỮ LIỆU ĐẦU VÀO

### Dataset: `data3.xlsx`
- **Kích thước:** 5.220 mẫu (samples), 7 biến (variables)
- **Biến đầu vào (6 features):**

| Biến | Min | Max | Số giá trị unique | Mô tả |
|------|-----|-----|-------------------|-------|
| Độ ẩm | 90% | 97% | 8 | Độ ẩm môi trường |
| Nhiệt độ | 27°C | 30°C | 4 | Nhiệt độ môi trường |
| Góc nghiêng | 20° | 22° | 2 | Góc nghiêng băng tải |
| Ca | 1 | 3 | 2 | Ca làm việc (ngày/đêm) |
| Công suất TB | 224 kW | 310 kW | 62 | Công suất trung bình |
| Sản lượng | 1.792 | 2.480 | 62 | Sản lượng vận chuyển |

- **Biến đầu ra (target):** 
  - **Tải tiêu thụ** (Load Consumption)
  - Range: 8.960 - 12.400 kW
  - Trung bình: ~10.617 kW
  - Độ lệch chuẩn: ~1.224 kW

---

## 3. CÁC BƯỚC XÂY DỰNG MÔ HÌNH

Theo tài liệu `2025 HD cac buoc xay dung mo hinh.docx`:

1. **Phân tích tương quan**
   - Tính hệ số tương quan Pearson giữa các biến
   - Xác định mối quan hệ giữa features và target

2. **Lựa chọn tham số tối ưu**
   - Số lớp mạng (hidden layers)
   - Số nút ẩn (hidden nodes/neurons)
   - Bộ Solver (optimization algorithm)
   - Hàm kích hoạt (activation function)
   - Các thiết lập huấn luyện khác

3. **Thiết lập lưu đồ huấn luyện**
   - Cấu hình kiến trúc mạng
   - Thiết lập hyperparameters

4. **Huấn luyện mô hình**
   - Train trên nhiều cấu hình khác nhau
   - Xuất thời gian huấn luyện và giá trị MAE

5. **So sánh kết quả**
   - So sánh các mô hình theo bảng chỉ số

6. **Kiểm chứng**
   - Test trên tập dữ liệu kiểm tra
   - Xuất kết quả MAPE và biểu đồ

---

## 4. CÁC MÔ HÌNH ĐÃ THỬ NGHIỆM

Tổng cộng **12 cấu hình mô hình** được thử nghiệm:

| Mô hình | Lớp ẩn | Tổng nút | Cấu hình | MAE | RMSE | MAPE (%) | nRMSE | Xếp hạng |
|---------|--------|----------|----------|-----|------|----------|-------|----------|
| **L2N19** ⭐ | **2** | **19** | **8-4-1** | **19.57** | **20.63** | **0.180** | **0.006** | **#1** |
| L3N27 | 3 | 27 | 8-4-8-1 | 26.66 | 29.10 | 0.242 | 0.008 | #2 |
| L2N47 | 2 | 47 | 8-32-1 | 29.09 | 30.43 | 0.287 | 0.009 | #3 |
| L2N31 | 2 | 31 | 8-16-1 | 33.89 | 34.29 | 0.327 | 0.010 | #4 |
| L4N43 | 4 | 43 | 8-4-16-8-1 | 35.94 | 40.38 | 0.324 | 0.012 | #5 |
| L3N39 | 3 | 39 | 8-16-8-1 | 38.12 | 40.02 | 0.376 | 0.012 | #6 |
| L3N35 | 3 | 35 | 8-16-4-1 | 45.22 | 48.83 | 0.412 | 0.014 | #7 |
| L5N131 | 5 | 131 | 8-32-64-16-4-1 | 58.58 | 59.21 | 0.568 | 0.017 | #8 |
| L4N55 | 4 | 55 | 8-16-8-16-1 | 68.35 | 73.53 | 0.623 | 0.021 | #9 |
| L4N67 | 4 | 67 | 8-16-32-4-1 | 93.86 | 94.48 | 0.884 | 0.027 | #10 |
| L5N135 | 5 | 135 | 8-32-64-16-8-1 | 114.42 | 115.10 | 1.080 | 0.033 | #11 |
| L5N79 | 5 | 79 | 8-8-16-32-8-1 | 158.09 | 159.85 | 1.484 | 0.046 | #12 |

### Giải thích cấu hình:
- **8-4-1**: 8 input nodes → 4 hidden nodes (layer 1) → 1 output node
- **8-4-8-1**: 8 input → 4 hidden (L1) → 8 hidden (L2) → 1 output

---

## 5. MÔ HÌNH TỐI ƯU - L2N19 ⭐

### Thông số mô hình:
- **Kiến trúc:** 2 lớp ẩn, 19 nút tổng cộng
- **Cấu hình:** 8-4-1 (8 input → 4 hidden → 1 output)

### Hiệu suất:
- **MAE = 19.57** - Sai số trung bình tuyệt đối thấp nhất
- **RMSE = 20.63** - Sai số bình phương trung bình thấp
- **MAPE = 0.18%** - Sai số phần trăm thấp nhất
- **nRMSE = 0.006** - Biến thiên sai số nhỏ nhất
- **MSE = 425.80**

### Đặc điểm:
- Sai số tuyệt đối nằm trong khoảng **5 - 35** (tương đương **0.05% - 0.33%** so với giá trị trung bình ~10.617)
- Giá trị dự đoán và thực tế **gần như trùng khớp hoàn toàn** trên biểu đồ
- Phân bố sai số theo dạng **Bimodal** (2 đỉnh) do công suất hoạt động chia ca ngày/đêm → **mô hình ổn định**
- Thời gian dự đoán: ~0.297 giây

### Ưu điểm:
✅ Độ chính xác cao nhất trong tất cả các mô hình  
✅ Kiến trúc đơn giản, dễ triển khai  
✅ Thời gian training và prediction nhanh  
✅ Ổn định với cả 2 chế độ hoạt động ngày/đêm  
✅ Không bị overfitting  

---

## 6. PHÂN TÍCH & NHẬN XÉT

### 6.1. Xu hướng theo số lớp ẩn:

| Số lớp | Kết quả quan sát |
|--------|------------------|
| 2 lớp | ✅ **Tốt nhất** - L2N19 đạt hiệu suất cao nhất |
| 3 lớp | ⚠️ Tạm chấp nhận - Sai số tăng nhẹ nhưng vẫn ổn định |
| 4 lớp | ⚠️ Không ổn định - Sai số biến động lớn giữa các cấu hình |
| 5 lớp | ❌ Không tốt - Sai số cao (MAPE 0.5% - 1.5%) |

**Kết luận:** Mô hình càng nhiều lớp ẩn (4-5 lớp) thì sai số càng lớn → Hiện tượng **overfitting** hoặc khó tối ưu hóa với dữ liệu có sẵn.

### 6.2. Phân tích từng nhóm mô hình:

#### **Nhóm 2 lớp (L2):**
- **L2N19** ⭐: Tốt nhất, 2 đỉnh rõ ràng, sai số tập trung 0.125%-0.250%
- **L2N31**: 3 đỉnh (không phù hợp với 2 điều kiện ngày/đêm) → không ổn định
- **L2N47**: 2 đỉnh nhưng dải sai số rộng (0.1%-0.5%) → độ chính xác không cao

#### **Nhóm 3 lớp (L3):**
- **L3N27**: Ổn định với 2 đỉnh nhưng dải sai số trải dài (0.05%-0.4%)
- **L3N35 & L3N39**: Ổn định nhưng sai số lớn, không phù hợp cho dự đoán chính xác cao

#### **Nhóm 4 lớp (L4):**
- **L4N43**: Khá ổn định nhưng dải sai số vẫn lớn
- **L4N55**: Dải sai số 0.3%-0.9%, không tốt
- **L4N67**: Không tạo ra 2 đỉnh rõ ràng → không ổn định

#### **Nhóm 5 lớp (L5):**
- **L5N79**: Độ ổn định CỰC KỲ CAO cho cả 2 chế độ ngày/đêm, nhưng sai số rất lớn (1.3%-1.5%) → **có tiềm năng nếu cải thiện được sai số**
- **L5N131**: 2 đỉnh nhưng dải sai số lớn
- **L5N135**: Ổn định nhưng sai số cao (0.95%-1.25%)

### 6.3. Chỉ số đánh giá mô hình:

**3 chỉ số quan trọng nhất:**

1. **MAE (Mean Absolute Error)**: Độ sai lệch trung bình giữa dự đoán và thực tế
   - Càng nhỏ càng tốt
   - L2N19 có MAE thấp nhất (19.57)

2. **nRMSE (Normalized Root Mean Square Error)**: Độ biến thiên sai số
   - Giá trị càng nhỏ → biến thiên của dữ liệu càng nhỏ → mô hình càng ổn định
   - L2N19: 0.006 (tốt nhất)

3. **MAPE (Mean Absolute Percentage Error)**: Sai số phần trăm theo giá trị thực tế
   - Đánh giá khả năng dự đoán của mô hình
   - L2N19: 0.18% (xuất sắc)

---

## 7. CẤU TRÚC FILE TRONG PROJECT

### 7.1. Dữ liệu:
| File | Mô tả |
|------|-------|
| `data3.xlsx` | **Dữ liệu huấn luyện chính** (5.220 mẫu, 7 biến) |
| `data.xlsx` | Dữ liệu phụ/gốc |

### 7.2. Tài liệu hướng dẫn:
| File | Mô tả |
|------|-------|
| `2025 HD cac buoc xay dung mo hinh.docx` | **Hướng dẫn các bước xây dựng mô hình** |
| `Bảng hướng dẫn lấy số liệu.docx` | Phân tích chi tiết từng mô hình + bảng công suất theo tham số |
| `pj3.docx` | File kết quả mẫu (biểu đồ + bảng metrics) |

### 7.3. Kết quả huấn luyện:
| File | Mô tả |
|------|-------|
| `prediction_with_charts.xlsx` | **Kết quả dự đoán mẫu** (Actual vs Predicted + Error metrics) |
| `7 node.xlsx` | So sánh MAE qua các epoch (7 node vs 50 node) |
| `bảng so sánh.xlsx` | Bảng so sánh tổng hợp các mô hình |

### 7.4. Kết quả từng mô hình (12 models):
| File Excel | File Hình ảnh | Mô tả |
|------------|---------------|-------|
| `l2n19.xlsx` | `l2n19.JPG` | Kết quả mô hình L2N19 (tốt nhất) |
| `l2n31.xlsx` | `l2n31.JPG` | Kết quả mô hình L2N31 |
| `l2n47.xlsx` | `l2n47.JPG` | Kết quả mô hình L2N47 |
| `L3n27.xlsx` | `l3n27.JPG` | Kết quả mô hình L3N27 |
| `l3n35.xlsx` | `l3n35.JPG` | Kết quả mô hình L3N35 |
| `l3n39.xlsx` | `l3n39.JPG` | Kết quả mô hình L3N39 |
| `l4N43.xlsx` | `l4n43.JPG` | Kết quả mô hình L4N43 |
| `l4n55.xlsx` | `l4n55.JPG` | Kết quả mô hình L4N55 |
| `l4n67.xlsx` | `l4n67.JPG` | Kết quả mô hình L4N67 |
| `l5n79.xlsx` | `l5n79.JPG` | Kết quả mô hình L5N79 |
| `l5n131.xlsx` | `l5n131.JPG` | Kết quả mô hình L5N131 |
| `l5N135.xlsx` | `l5N135.JPG` | Kết quả mô hình L5N135 |

**Mỗi file Excel chứa 2 sheets:**
- `Dự đoán`: Giá trị thực tế, dự đoán, sai số tuyệt đối, sai số tương đối
- `Chỉ số sai số`: MAE, MSE, RMSE, nRMSE, MAPE, thời gian dự đoán

---

## 8. KẾT QUẢ BÀN GIAO (Không có source code)

### 8.1. Outputs cần thiết:
1. ✅ **Bảng hệ số tương quan Pearson** giữa các biến
2. ✅ **Biểu đồ quá trình huấn luyện** (MAE theo epoch cho các cấu hình)
3. ✅ **Bảng so sánh 12 mô hình** với đầy đủ metrics (MAE, MSE, RMSE, nRMSE, MAPE)
4. ✅ **Biểu đồ giá trị dự đoán vs thực tế** (Predicted vs Actual)
5. ✅ **Biểu đồ phân bổ sai số PE-APE** (Percentage Error - Absolute Percentage Error)
6. ✅ **12 hình ảnh JPG** minh họa kết quả training của từng mô hình

### 8.2. Chỉ số tổng hợp (từ `prediction_with_charts.xlsx`):
- **RMSE:** 85.56
- **MAPE:** 0.66%
- **nRMSE:** 0.025

### 8.3. Format bàn giao:
- File Word (.docx): Hướng dẫn và phân tích
- File Excel (.xlsx): Dữ liệu và kết quả số
- Hình ảnh (.JPG): Biểu đồ visualization

---

## 9. CÔNG NGHỆ SỬ DỤNG

- **Ngôn ngữ:** Python
- **Framework:** Mạng Neuron Nhân tạo (ANN)
- **Thư viện dự kiến:**
  - `scikit-learn` (MLPRegressor)
  - `pandas` (data processing)
  - `numpy` (numerical computing)
  - `matplotlib/seaborn` (visualization)
- **Metrics:** MAE, MSE, RMSE, MAPE, nRMSE

---

## 10. KHUYẾN NGHỊ

### Cho triển khai hiện tại:
✅ **Sử dụng mô hình L2N19** cho hệ thống production

### Cho nghiên cứu tiếp theo:
1. **Cải thiện L5N79**: Mô hình này có độ ổn định cao, nếu giảm được sai số có thể là lựa chọn tốt cho các điều kiện vận hành phức tạp
2. **Data augmentation**: Thu thập thêm dữ liệu để cải thiện khả năng tổng quát hóa
3. **Feature engineering**: Thử nghiệm thêm các biến tương tác hoặc biến phái sinh
4. **Ensemble methods**: Kết hợp nhiều mô hình (L2N19 + L3N27) để tăng độ tin cậy
5. **Hyperparameter tuning**: Tinh chỉnh learning rate, batch size, số epoch

---

## 11. KẾT LUẬN

Project đã thành công xây dựng mô hình ANN để dự báo công suất băng tải với độ chính xác cao:

- ✅ **Mô hình tối ưu L2N19** đạt MAPE 0.18% (xuất sắc)
- ✅ Đã thử nghiệm đầy đủ 12 cấu hình khác nhau
- ✅ Phân tích chi tiết ưu/nhược điểm từng mô hình
- ✅ Kết quả bàn giao đầy đủ (biểu đồ, bảng số liệu, phân tích)
- ✅ Không có source code (theo yêu cầu)

**Mô hình sẵn sàng triển khai vào môi trường thực tế.**

---

*Tài liệu tóm tắt được tạo tự động từ phân tích các file trong project.*  
*Ngày tạo: 2026-02-10*
