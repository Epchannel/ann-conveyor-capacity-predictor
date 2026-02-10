# BÁO CÁO KẾT QUẢ: SO SÁNH PHƯƠNG ÁN A VÀ B
## Dự báo công suất băng tải bằng mạng neuron nhân tạo (ANN)

**Ngày báo cáo:** 2026-02-11  
**Dữ liệu:** `data3.xlsx` — 5.220 mẫu, 7 biến  
**Công cụ:** Python, scikit-learn, XGBoost, LightGBM

---

## 1. TỔNG QUAN

### 1.1. Bối cảnh

Mô hình gốc **L2N19** (kiến trúc 8-4-1) đạt MAPE = 0.180% khi sử dụng cả 6 features. Tuy nhiên, phân tích dữ liệu phát hiện vấn đề nghiêm trọng:

```
sản lượng    = công suất tb × 8     (chính xác 100%)
tải tiêu thụ = công suất tb × 40    (gần chính xác 100%)
```

→ Ba biến `công suất tb`, `sản lượng`, `tải tiêu thụ` thực chất là **cùng một biến** chỉ khác hệ số tỷ lệ.

### 1.2. Ma trận tương quan Pearson với target (`tải tiêu thụ`)

| Biến | Tương quan | Phân loại |
|------|-----------|-----------|
| công suất tb | **1.0000** | Tuyến tính hoàn hảo (biến phái sinh) |
| sản lượng | **1.0000** | Tuyến tính hoàn hảo (biến phái sinh) |
| ca | **-0.9643** | Rất mạnh |
| góc nghiêng | 0.0042 | Rất yếu |
| độ ẩm | 0.0016 | Gần như không tương quan |
| nhiệt độ | 0.0012 | Gần như không tương quan |

### 1.3. Hai phương án thử nghiệm

| | Phương Án A | Phương Án B |
|---|-------------|-------------|
| **Mục tiêu** | Tối ưu mô hình hiện tại | Xây dựng lại bài toán có ý nghĩa |
| **Features** | 5 (loại `sản lượng`, giữ `công suất tb`) | 4 biến độc lập + Polynomial degree 2 |
| **Biến đầu vào** | độ ẩm, nhiệt độ, góc nghiêng, ca, **công suất tb** | độ ẩm, nhiệt độ, góc nghiêng, ca |
| **Số features thực tế** | 5 | 14 (sau polynomial transform) |
| **Tổng mô hình thử** | 23 | 27 |

---

## 2. KẾT QUẢ PHƯƠNG ÁN A (5 FEATURES)

### 2.1. Kết quả

| Rank | Mô hình | MAE | MAPE (%) | R² |
|------|---------|-----|----------|-----|
| #1 | GB_500 | **0.0000** | **0.0000** | **1.000000** |
| #2 | GB_300 | 0.0000 | 0.0000 | 1.000000 |
| #3 | RF_500 | 0.0000 | 0.0000 | 1.000000 |
| #4 | RF_300 | 0.0000 | 0.0000 | 1.000000 |
| #5 | Stacking(GB_500+GB_300+RF_500) | 0.0000 | 0.0000 | 1.000000 |

### 2.2. Phân tích

Tất cả top 5 mô hình đều đạt **MAE = 0, MAPE = 0%, R² = 1.0** — kết quả "hoàn hảo" về mặt số liệu.

**Nguyên nhân:** Biến `công suất tb` có quan hệ tuyến tính hoàn hảo với target (`tải tiêu thụ ≈ công suất tb × 40`). Các mô hình tree-based (Gradient Boosting, Random Forest) dễ dàng tìm ra quy luật nhân hệ số cố định này, dẫn đến sai số bằng 0.

**Bản chất:** Mô hình không thực sự "dự đoán" — nó chỉ thực hiện **phép nhân đã biết trước**. Bốn biến còn lại (độ ẩm, nhiệt độ, góc nghiêng, ca) hoàn toàn bị bỏ qua bởi model.

### 2.3. Ưu điểm

- ✅ Sai số cực thấp (MAE ≈ 0) nếu đã biết `công suất tb`
- ✅ Phù hợp cho bài toán **kiểm chứng ngược** (sanity check): xác nhận mối quan hệ tuyến tính giữa các biến
- ✅ Chứng minh rõ ràng vấn đề multicollinearity trong dữ liệu
- ✅ Thời gian train nhanh, model đơn giản

### 2.4. Nhược điểm

- ❌ **Không có giá trị dự đoán thực tế**: cần biết `công suất tb` (đã vận hành) mới tính được target
- ❌ **Vấn đề rò rỉ dữ liệu (data leakage)**: biến đầu vào là biến phái sinh trực tiếp từ target
- ❌ **Không có ý nghĩa khoa học**: model chỉ học phép nhân cố định, không phát hiện pattern nào mới
- ❌ **Không triển khai được**: không thể dự đoán công suất trước khi hệ thống hoạt động
- ❌ Kết quả R² = 1.0 là **cờ đỏ** (red flag) cho thấy data leakage, không phải dấu hiệu model tốt

---

## 3. KẾT QUẢ PHƯƠNG ÁN B (4 FEATURES + POLYNOMIAL)

### 3.1. Kết quả

| Rank | Mô hình | MAE | MAPE (%) | R² |
|------|---------|-----|----------|-----|
| #1 | **MLP_(32)** | **286.92** | **2.6981** | **0.927606** |
| #2 | Ridge | 287.58 | 2.7037 | 0.927209 |
| #3 | Stacking(MLP_32+Ridge+MLP_64_32) | 287.72 | 2.7047 | 0.927241 |
| #4 | XGB_Optimized | 287.84 | 2.7045 | 0.927066 |
| #5 | LGBM_Optimized | 287.87 | 2.7057 | 0.927013 |

### 3.2. Phân tích

**Mô hình chiến thắng: MLP_(32)** — mạng neuron 1 hidden layer với 32 nodes.

- **MAPE = 2.698%**: Sai số trung bình ~2.7% so với giá trị thực — xuất sắc cho bài toán dự đoán từ điều kiện môi trường
- **R² = 0.9276**: Mô hình giải thích được 92.76% phương sai của target
- **Top 5 rất đồng đều** (MAE chênh < 1 giữa #1 và #5): chứng tỏ mô hình ổn định, không overfitting

### 3.3. Đặc điểm đáng chú ý

1. **MLP_(32) đơn giản nhất lại thắng**: Kiến trúc 1 layer, 32 nodes — không cần deep network
2. **Ridge regression (#2) cạnh tranh sát**: Chứng tỏ dữ liệu có cấu trúc tuyến tính mạnh sau polynomial transform
3. **Stacking không cải thiện đáng kể**: Top 3 models quá đồng đều, ensemble không thêm giá trị
4. **XGBoost và LightGBM sau tuning cũng ngang bằng**: Bài toán đã đạt giới hạn với feature set hiện tại

### 3.4. Ưu điểm

- ✅ **Có ý nghĩa dự đoán thực tế**: dự báo công suất TRƯỚC KHI vận hành hệ thống
- ✅ **Chỉ cần 4 biến input đơn giản**: dễ đo lường (cảm biến nhiệt độ, độ ẩm, góc cố định, ca làm việc)
- ✅ **MAPE 2.698% là xuất sắc** cho bài toán regression từ điều kiện môi trường
- ✅ **R² = 0.9276**: giải thích tốt phương sai dữ liệu
- ✅ **Model ổn định**: top 5 đồng đều, không phụ thuộc vào 1 thuật toán cụ thể
- ✅ **Kiến trúc đơn giản**: MLP 1 layer — dễ triển khai, inference nhanh
- ✅ **Có giá trị khoa học**: phát hiện mối quan hệ thực giữa điều kiện vận hành và công suất

### 3.5. Nhược điểm

- ❌ **MAPE cao hơn nhiều so với A**: 2.698% vs 0.000% (nhưng đây là trade-off chấp nhận được)
- ❌ **MAE = 286.92 kW**: sai số trung bình ~287 kW trên dải 8.960-12.400 kW
- ❌ **Biến `ca` chi phối** (corr = -0.964): model phụ thuộc chủ yếu vào 1 biến, 3 biến còn lại đóng góp ít
- ❌ **Giới hạn của dữ liệu**: 4 biến độc lập không đủ thông tin để dự đoán chính xác hơn
- ❌ **Chưa tối ưu sâu**: chưa dùng Bayesian optimization (Optuna), chưa thử multi-seed ensemble

---

## 4. SO SÁNH TỔNG HỢP

### 4.1. Bảng so sánh chỉ số

| Chỉ số | L2N19 gốc | Phương Án A | Phương Án B |
|--------|-----------|-------------|-------------|
| **Features** | 6 (tất cả) | 5 (loại sản lượng) | 4 (chỉ biến độc lập) + poly |
| **MAE** | 19.57 | 0.0000 | 286.92 |
| **MAPE** | 0.180% | 0.000% | 2.698% |
| **R²** | ~0.999 | 1.000 | 0.928 |
| **Dùng biến phái sinh?** | Có | Có | **Không** |
| **Giá trị dự đoán** | Thấp | Không | **Cao** |
| **Triển khai thực tế** | Hạn chế | Không thể | **Khả thi** |
| **Ý nghĩa khoa học** | Thấp | Không | **Cao** |

### 4.2. Đánh giá tổng quan

```
Phương Án A:  Sai số = 0  nhưng  Giá trị = 0   → ❌ Không khuyến nghị
Phương Án B:  Sai số = 2.7%  nhưng  Giá trị = Cao  → ✅ Khuyến nghị sử dụng
```

### 4.3. Ví dụ minh họa

| Tình huống | Phương Án A | Phương Án B |
|-----------|-------------|-------------|
| **Đầu vào** | Cần biết: độ ẩm=95%, nhiệt độ=29°C, góc=22°, ca=1, **công suất tb=280kW** | Chỉ cần: độ ẩm=95%, nhiệt độ=29°C, góc=22°, ca=1 |
| **Câu hỏi** | "Đã biết công suất trung bình, tải tiêu thụ là bao nhiêu?" | "Với điều kiện môi trường này, dự đoán tải tiêu thụ?" |
| **Tính chất** | Đoán số đã biết trước | Dự đoán thực sự |
| **Ứng dụng** | Kiểm chứng (verification) | Lập kế hoạch (planning) |

---

## 5. ĐỀ XUẤT CẢI TIẾN: PHƯƠNG ÁN B1

### 5.1. Mục tiêu

Cải thiện kết quả Phương Án B (MAPE = 2.698%) bằng các kỹ thuật nâng cao, nhắm đến:

| Chỉ số | B hiện tại | B1 mục tiêu | Cải thiện |
|--------|-----------|-------------|-----------|
| MAE | 286.92 | < 275 | > 4% |
| MAPE | 2.698% | < 2.50% | > 7% |
| R² | 0.9276 | > 0.935 | tăng |

### 5.2. Kỹ thuật cải tiến

#### A. Feature Engineering nâng cao

| Kỹ thuật | Mô tả | Kỳ vọng |
|----------|-------|---------|
| **Polynomial degree 3** | Thêm cubic terms (x³, x²y, xyz...) → 34 features | Nắm bắt phi tuyến bậc cao |
| **Manual interaction features** | Tương tác 3 chiều: `độ_ẩm × nhiệt_độ × ca`, etc. → 28 features | Domain-specific knowledge |
| **Cyclical encoding** | Sin/Cos transform cho nhiệt độ, độ ẩm | Nắm bắt tính chu kỳ |
| **Ratio & log features** | `độ_ẩm / nhiệt_độ`, `log(nhiệt_độ)` | Quan hệ phi tuyến |
| **Tự động chọn feature set** | Test 5 bộ features × 3 scalers → chọn tốt nhất | Tối ưu hóa input |

#### B. Bayesian Optimization (Optuna)

| Model | Số trials | Tham số tìm kiếm |
|-------|----------|-------------------|
| **MLP** | 200 | Layers (1-4), nodes (8-128), activation, solver, alpha, LR, batch_size |
| **XGBoost** | 200 | n_estimators (100-2000), max_depth (3-15), LR, regularization, gamma |
| **LightGBM** | 200 | n_estimators (100-2000), num_leaves (15-200), LR, regularization |

So với B (dùng RandomizedSearchCV 100 iter), B1 dùng **Optuna TPE sampler** — thông minh hơn, hiệu quả hơn, tìm kiếm không gian rộng hơn.

#### C. Multi-seed Ensemble

```
Mỗi model tốt nhất (MLP, XGB, LGBM) được train 10 lần với seed khác nhau
→ Average prediction → Giảm variance → Kết quả robust hơn
→ Mega Ensemble: average tất cả 30 models (3 loại × 10 seeds)
```

#### D. Advanced Stacking

| Variant | Base Models | Meta-learner |
|---------|-------------|-------------|
| V1 | MLP + XGB + LGBM (Optuna best) | Ridge |
| V2 | MLP + XGB + LGBM + RF + GB | Ridge |
| V3 | MLP + XGB + LGBM | XGBoost (meta) |
| Voting | MLP + XGB + LGBM | Average |

### 5.3. Pipeline thực hiện

```
Phase 1: Feature Selection
    → Test 5 feature sets × 3 scalers × MLP_32 baseline
    → Chọn combo tốt nhất

Phase 2: Optuna MLP (200 trials)
    → Tìm kiến trúc & hyperparameters tối ưu

Phase 3: Optuna XGBoost (200 trials)
    → Tối ưu tree-based model mạnh nhất

Phase 4: Optuna LightGBM (200 trials)
    → Tối ưu model nhanh nhất

Phase 5: Multi-seed Ensemble
    → 10 seeds × 3 models + Mega Ensemble (30 models)

Phase 6: Advanced Stacking
    → 3 Stacking variants + Voting
```

### 5.4. Thời gian dự kiến

| Phase | Thời gian ước tính |
|-------|-------------------|
| Phase 1 (Feature Selection) | 5-10 phút |
| Phase 2 (Optuna MLP) | 15-30 phút |
| Phase 3 (Optuna XGBoost) | 10-20 phút |
| Phase 4 (Optuna LightGBM) | 5-15 phút |
| Phase 5 (Multi-seed) | 5-10 phút |
| Phase 6 (Stacking) | 5-15 phút |
| **Tổng cộng** | **45-100 phút** |

### 5.5. Lý do kỳ vọng cải thiện

1. **Feature set tối ưu chưa được tìm**: B dùng cố định Poly2 + Standard — có thể Poly3 hoặc Manual features tốt hơn
2. **MLP chưa được tune sâu**: B chỉ thử vài cấu hình cố định, B1 dùng Optuna tìm trong không gian rộng hơn nhiều
3. **Multi-seed giảm variance**: Một lần train có thể rơi vào local minimum, average 10 lần cho kết quả ổn định hơn
4. **Ensemble 30 models mạnh hơn 3 models**: Mega Ensemble kết hợp đa dạng thuật toán + đa dạng random initialization

---

## 6. KẾT LUẬN

### 6.1. Phát hiện chính

1. **Phương Án A (R² = 1.0) xác nhận vấn đề data leakage**: `công suất tb` là biến phái sinh trực tiếp từ target, khiến mọi model đạt sai số = 0 mà không có giá trị dự đoán
2. **Phương Án B (MAPE = 2.698%) cho kết quả thực tế có ý nghĩa**: Chỉ từ 4 biến điều kiện môi trường, model dự đoán được công suất với sai số ~2.7%
3. **MLP đơn giản (1 layer, 32 nodes) đã đủ tốt**: Không cần deep network cho bài toán này

### 6.2. Khuyến nghị

| Ưu tiên | Hành động |
|---------|----------|
| **#1** | **Sử dụng Phương Án B** cho mục đích dự báo và lập kế hoạch sản xuất |
| **#2** | **Chạy Phương Án B1** để cải thiện thêm hiệu suất (mục tiêu MAPE < 2.5%) |
| **#3** | Báo cáo Phương Án A như **bằng chứng multicollinearity**, không dùng làm model production |
| **#4** | Thu thập thêm biến đầu vào (tải trọng hàng hóa, tuổi thọ dây curoa, tốc độ gió...) để cải thiện sâu hơn |

### 6.3. Tóm tắt một dòng

> **Phương Án B (MAPE = 2.698%) là lựa chọn tối ưu** — kết quả có ý nghĩa khoa học, ứng dụng thực tế, và có tiềm năng cải thiện thêm qua Phương Án B1.

---

*Báo cáo được tạo dựa trên kết quả thực nghiệm từ Phương Án A và B.*  
*Ngày tạo: 2026-02-11*
