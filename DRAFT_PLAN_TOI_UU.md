# DRAFT PLAN: TỐI ƯU MÔ HÌNH DỰ BÁO CÔNG SUẤT BĂNG TẢI

## 0. HIỆN TRẠNG

### Mô hình hiện tại tốt nhất: L2N19 (8-4-1)

| Chỉ số | Giá trị hiện tại | Mục tiêu cải thiện |
|--------|------------------|---------------------|
| MAE | 19.57 | < 10 |
| RMSE | 20.63 | < 15 |
| MAPE | 0.180% | < 0.10% |
| nRMSE | 0.006 | < 0.003 |

---

## ⚠️ 1. VẤN ĐỀ NGHIÊM TRỌNG PHÁT HIỆN TRONG DATA

### 1.1. Multicollinearity hoàn hảo (correlation = 1.000)

Phân tích dữ liệu `data3.xlsx` cho thấy:

```
sản lượng     = công suất tb × 8    (chính xác 100%)
tải tiêu thụ  = sản lượng × 5       (gần chính xác 100%)
tải tiêu thụ  = công suất tb × 40   (gần chính xác 100%)
```

**→ 3 biến `công suất tb`, `sản lượng`, `tải tiêu thụ` thực chất là CÙNG MỘT BIẾN chỉ khác hệ số tỷ lệ.**

### 1.2. Ma trận tương quan Pearson thực tế:

| Feature | Tương quan với Target |
|---------|----------------------|
| công suất tb | **1.0000** (tuyến tính hoàn hảo) |
| sản lượng | **1.0000** (tuyến tính hoàn hảo) |
| ca | **-0.9643** (rất mạnh) |
| góc nghiêng | 0.0042 (rất yếu) |
| độ ẩm | 0.0016 (gần như không tương quan) |
| nhiệt độ | 0.0012 (gần như không tương quan) |

### 1.3. Hệ quả:

- Mô hình hiện tại đang **"gian lận"** vì dùng `công suất tb` và `sản lượng` (là biến phái sinh trực tiếp từ target) làm input
- MAPE thấp (0.18%) chủ yếu vì mô hình **chỉ cần học phép nhân × 40**
- Các biến thực sự độc lập (độ ẩm, nhiệt độ, góc nghiêng, ca) gần như KHÔNG đóng góp vào dự đoán

---

## 2. PHƯƠNG ÁN TỐI ƯU

### PHƯƠNG ÁN A: Giữ nguyên cách tiếp cận hiện tại (cải thiện nhỏ)
> Giữ tất cả 6 features, tối ưu mô hình để đạt sai số thấp hơn

### PHƯƠNG ÁN B: Xây dựng lại bài toán có ý nghĩa (khuyến nghị mạnh)
> Chỉ dùng 4 biến độc lập (độ ẩm, nhiệt độ, góc nghiêng, ca) để dự đoán target

### PHƯƠNG ÁN C: Kết hợp cả hai
> Chạy song song 2 phương án, so sánh kết quả

---

## 3. CHI TIẾT PHƯƠNG ÁN A - TỐI ƯU MÔ HÌNH HIỆN TẠI

### Phase A1: Cải thiện tiền xử lý dữ liệu

#### A1.1. Feature Scaling (ưu tiên cao)
```
Hiện tại: Không rõ có normalize hay chưa
Đề xuất: Thử nghiệm cả 2 phương pháp
```

| Phương pháp | Công thức | Ưu điểm |
|-------------|-----------|---------|
| MinMaxScaler | (x - min) / (max - min) | Giữ range [0, 1], phù hợp activation sigmoid |
| StandardScaler | (x - mean) / std | Phù hợp cho activation ReLU, data phân phối chuẩn |
| RobustScaler | (x - median) / IQR | Bền vững với outlier |

#### A1.2. Loại bỏ Multicollinearity
```
Hành động: BẮT BUỘC loại bỏ ít nhất 1 trong 2 biến:
  - Loại "sản lượng" (vì sản lượng = công suất tb × 8, hoàn toàn dư thừa)
  - Hoặc loại "công suất tb" (giữ sản lượng)
Lý do: 2 biến correlation = 1.0 gây nhiễu gradient descent
```

#### A1.3. Feature Engineering
```python
# Các feature mới đề xuất:
df['do_am_x_nhiet_do'] = df['độ ẩm'] * df['nhiệt độ']        # Tương tác nhiệt-ẩm
df['do_am_squared'] = df['độ ẩm'] ** 2                         # Phi tuyến
df['nhiet_do_squared'] = df['nhiệt độ'] ** 2                   # Phi tuyến
df['goc_x_ca'] = df['góc nghiêng'] * df['ca']                  # Tương tác góc-ca
df['heat_index'] = df['độ ẩm'] * df['nhiệt độ'] / 100          # Chỉ số nhiệt
```

#### A1.4. Train/Test Split tối ưu
```
Hiện tại: Không rõ tỷ lệ split
Đề xuất: 
  - 80/20 split (stratified theo ca)
  - Hoặc dùng K-Fold Cross Validation (k=5 hoặc k=10)
```

### Phase A2: Tối ưu kiến trúc mạng

#### A2.1. Systematic Hyperparameter Search

```python
param_grid = {
    'hidden_layer_sizes': [
        (4,),               # 1 layer, 4 nodes (baseline đơn giản)
        (8,),               # 1 layer, 8 nodes
        (16,),              # 1 layer, 16 nodes
        (8, 4),             # 2 layers (tương tự L2N19 nhưng nhiều node hơn)
        (16, 8),            # 2 layers lớn hơn
        (8, 8),             # 2 layers đều
        (16, 8, 4),         # 3 layers giảm dần
        (8, 4, 4),          # 3 layers nhỏ
    ],
    'activation': ['relu', 'tanh', 'logistic', 'identity'],
    'solver': ['adam', 'lbfgs', 'sgd'],
    'alpha': [0.0001, 0.001, 0.01, 0.1],           # L2 regularization
    'learning_rate_init': [0.001, 0.005, 0.01],
    'max_iter': [500, 1000, 2000],
    'early_stopping': [True],
    'validation_fraction': [0.1, 0.15],
    'batch_size': [32, 64, 128, 256],
}
```

#### A2.2. Chiến lược tìm kiếm

| Phương pháp | Số thử nghiệm | Thời gian ước tính | Ưu tiên |
|-------------|----------------|---------------------|---------|
| GridSearchCV | Toàn bộ grid | ~2-4 giờ | Nếu có thời gian |
| RandomizedSearchCV | 200-500 combos | ~30-60 phút | **Khuyến nghị** |
| Bayesian Optimization (Optuna) | 100-200 trials | ~20-40 phút | Tốt nhất |

#### A2.3. K-Fold Cross Validation

```python
from sklearn.model_selection import KFold, cross_val_score

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
# Đánh giá mô hình trên 5 folds
# → Kết quả đáng tin cậy hơn single train/test split
```

### Phase A3: Kỹ thuật huấn luyện nâng cao

#### A3.1. Early Stopping
```
Dừng training khi validation loss không giảm sau N epochs
→ Tránh overfitting, tiết kiệm thời gian
→ patience = 10-20 epochs
```

#### A3.2. Learning Rate Scheduling
```
adaptive: Giảm learning rate khi loss không giảm
→ Giúp hội tụ tốt hơn ở giai đoạn cuối training
```

#### A3.3. Multiple Random Seeds
```
Chạy mỗi cấu hình 5-10 lần với seed khác nhau
→ Lấy trung bình kết quả
→ Đánh giá tính ổn định của mô hình
```

---

## 4. CHI TIẾT PHƯƠNG ÁN B - XÂY DỰNG LẠI BÀI TOÁN (KHUYẾN NGHỊ)

### Mục tiêu:
Dự đoán `tải tiêu thụ` CHỈ từ 4 biến thực sự độc lập:
- **Độ ẩm** (8 giá trị: 90-97%)
- **Nhiệt độ** (4 giá trị: 27-30°C)
- **Góc nghiêng** (2 giá trị: 20°, 22°)
- **Ca** (2 giá trị: ca 1, ca 3)

### Thách thức:
- Các biến này có tương quan RẤT THẤP với target (< 0.005), ngoại trừ `ca` (-0.964)
- Trong cùng nhóm điều kiện (cùng độ ẩm, nhiệt độ, góc, ca), std trung bình = 320.82 kW
- → Sai số nội tại trong dữ liệu khá lớn

### Phase B1: Feature Engineering chuyên sâu

```python
# One-hot encoding cho biến phân loại
df_encoded = pd.get_dummies(df, columns=['ca', 'góc nghiêng'])

# Polynomial features
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, interaction_only=False)
X_poly = poly.fit_transform(X)
# → Tạo features: x1², x2², x1*x2, x1*x3, ...

# Cyclical encoding cho nhiệt độ (nếu cần)
df['nhiet_do_sin'] = np.sin(2 * np.pi * df['nhiệt độ'] / 30)
df['nhiet_do_cos'] = np.cos(2 * np.pi * df['nhiệt độ'] / 30)
```

### Phase B2: Thử nghiệm nhiều thuật toán

| Thuật toán | Ưu điểm | Nhược điểm | Ưu tiên |
|------------|---------|------------|---------|
| **ANN (MLP)** | Linh hoạt, phi tuyến | Cần nhiều data, dễ overfit | Hiện tại |
| **Random Forest** | Robust, ít overfit | Không ngoại suy tốt | ⭐ Nên thử |
| **Gradient Boosting (XGBoost/LightGBM)** | Chính xác cao, xử lý tốt feature ít | Cần tuning kỹ | ⭐⭐ Khuyến nghị |
| **SVR (Support Vector Regression)** | Tốt với data nhỏ | Chậm với data lớn | Nên thử |
| **Linear Regression + Polynomial** | Đơn giản, dễ hiểu | Hạn chế phi tuyến | Baseline |
| **ElasticNet** | Regularization tốt | Linear | Baseline |

### Phase B3: Ensemble Methods

```python
# Stacking: Kết hợp nhiều mô hình
from sklearn.ensemble import StackingRegressor

estimators = [
    ('mlp', MLPRegressor(hidden_layer_sizes=(16, 8))),
    ('rf', RandomForestRegressor(n_estimators=200)),
    ('xgb', XGBRegressor(n_estimators=200)),
    ('svr', SVR(kernel='rbf'))
]

stacking = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge(alpha=1.0),
    cv=5
)

# Voting: Trung bình kết quả
from sklearn.ensemble import VotingRegressor
voting = VotingRegressor(estimators=estimators)
```

### Phase B4: Neural Network nâng cao

#### B4.1. Kiến trúc sâu hơn với regularization (PyTorch/Keras)

```python
import torch.nn as nn

class ImprovedANN(nn.Module):
    def __init__(self, input_size=4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),          # Batch Normalization
            nn.ReLU(),
            nn.Dropout(0.2),             # Dropout regularization

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)
```

#### B4.2. Training với kỹ thuật nâng cao

```python
# Learning rate scheduling
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10
)

# Early stopping tùy chỉnh
# Cosine annealing
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=50, T_mult=2
)
```

---

## 5. KẾ HOẠCH THỰC HIỆN (TIMELINE)

### Giai đoạn 1: Chuẩn bị & phân tích (1-2 ngày)

| Bước | Công việc | Ưu tiên |
|------|-----------|---------|
| 1.1 | Phân tích EDA (Exploratory Data Analysis) đầy đủ | ⭐⭐⭐ |
| 1.2 | Vẽ biểu đồ phân phối, correlation heatmap | ⭐⭐⭐ |
| 1.3 | Xác nhận lại mối quan hệ tuyến tính giữa 3 biến cuối | ⭐⭐⭐ |
| 1.4 | Quyết định phương án (A, B, hoặc C) | ⭐⭐⭐ |
| 1.5 | Chuẩn bị pipeline tiền xử lý dữ liệu | ⭐⭐ |

### Giai đoạn 2: Thực nghiệm nhanh (2-3 ngày)

| Bước | Công việc | Ưu tiên |
|------|-----------|---------|
| 2.1 | Loại bỏ biến `sản lượng` (dư thừa hoàn toàn) | ⭐⭐⭐ |
| 2.2 | Áp dụng StandardScaler/MinMaxScaler | ⭐⭐⭐ |
| 2.3 | Chạy K-Fold Cross Validation trên L2N19 cải tiến | ⭐⭐⭐ |
| 2.4 | Thử RandomizedSearchCV cho hyperparameters | ⭐⭐ |
| 2.5 | Test với activation functions khác nhau | ⭐⭐ |

### Giai đoạn 3: Thử nghiệm sâu (3-5 ngày)

| Bước | Công việc | Ưu tiên |
|------|-----------|---------|
| 3.1 | Thử XGBoost, Random Forest, SVR | ⭐⭐⭐ |
| 3.2 | Feature engineering (polynomial, interaction) | ⭐⭐ |
| 3.3 | Ensemble stacking/voting | ⭐⭐ |
| 3.4 | Bayesian Optimization (Optuna) cho best model | ⭐⭐ |
| 3.5 | Chạy phương án B (4 biến độc lập) | ⭐⭐ |

### Giai đoạn 4: Đánh giá & báo cáo (1-2 ngày)

| Bước | Công việc | Ưu tiên |
|------|-----------|---------|
| 4.1 | So sánh toàn bộ mô hình (bảng + biểu đồ) | ⭐⭐⭐ |
| 4.2 | Phân tích residuals (sai số dư) | ⭐⭐ |
| 4.3 | Tạo báo cáo kết quả cuối cùng | ⭐⭐⭐ |
| 4.4 | Xuất kết quả dạng Excel + biểu đồ | ⭐⭐⭐ |

---

## 6. DANH SÁCH THỬ NGHIỆM CỤ THỂ

### 6.1. Nhóm thử nghiệm tiền xử lý (Preprocessing)

| ID | Thử nghiệm | Features | Scaler | Kỳ vọng |
|----|-------------|----------|--------|---------|
| P1 | Baseline (hiện tại) | 6 features gốc | Không | MAPE = 0.18% |
| P2 | Loại `sản lượng` | 5 features | StandardScaler | MAPE < 0.15% |
| P3 | Loại `sản lượng` + `công suất tb` | 4 features | StandardScaler | MAPE ~ 2-5% (thực tế hơn) |
| P4 | 4 features + polynomial degree 2 | ~14 features | StandardScaler | MAPE ~ 1-3% |
| P5 | 4 features + interaction terms | ~10 features | MinMaxScaler | MAPE ~ 1-4% |

### 6.2. Nhóm thử nghiệm mô hình (Model)

| ID | Mô hình | Cấu hình | Features | Kỳ vọng |
|----|---------|----------|----------|---------|
| M1 | MLP (sklearn) | (8, 4), adam, relu | P2 | MAPE < 0.15% |
| M2 | MLP (sklearn) | RandomSearch best | P2 | MAPE < 0.12% |
| M3 | MLP (sklearn) | Optuna best | P2 | MAPE < 0.10% |
| M4 | XGBoost | Default | P2 | MAPE < 0.12% |
| M5 | XGBoost | Tuned | P2 | MAPE < 0.08% |
| M6 | LightGBM | Tuned | P2 | MAPE < 0.08% |
| M7 | Random Forest | 500 trees | P2 | MAPE < 0.15% |
| M8 | SVR | RBF kernel | P2 | MAPE < 0.12% |
| M9 | Stacking (M3+M5+M7) | 5-fold | P2 | MAPE < 0.07% |
| M10 | MLP (PyTorch) | BatchNorm + Dropout | P2 | MAPE < 0.10% |

### 6.3. Nhóm thử nghiệm thực tế (chỉ 4 biến độc lập)

| ID | Mô hình | Features | Kỳ vọng |
|----|---------|----------|---------|
| R1 | MLP | P3 (4 features) | MAPE ~ 2-5% |
| R2 | XGBoost | P3 (4 features) | MAPE ~ 2-4% |
| R3 | XGBoost | P4 (polynomial) | MAPE ~ 1-3% |
| R4 | Stacking | P4 (polynomial) | MAPE ~ 1-2% |
| R5 | PyTorch ANN | P3 + BatchNorm | MAPE ~ 1-3% |

---

## 7. CÁC THƯ VIỆN PYTHON CẦN CÀI ĐẶT

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
pip install xgboost lightgbm optuna
pip install torch  # nếu dùng PyTorch
pip install shap   # giải thích mô hình
```

---

## 8. CẤU TRÚC CODE ĐỀ XUẤT

```
project/
├── data/
│   └── data3.xlsx                  # Dữ liệu gốc
├── notebooks/
│   ├── 01_EDA.ipynb                # Phân tích khám phá dữ liệu
│   ├── 02_preprocessing.ipynb      # Tiền xử lý
│   ├── 03_baseline_mlp.ipynb       # MLP baseline
│   ├── 04_hyperparameter_tuning.ipynb  # Tối ưu tham số
│   ├── 05_other_models.ipynb       # XGBoost, RF, SVR
│   ├── 06_ensemble.ipynb           # Ensemble methods
│   └── 07_final_comparison.ipynb   # So sánh cuối cùng
├── src/
│   ├── data_loader.py              # Đọc và xử lý data
│   ├── feature_engineering.py      # Tạo features
│   ├── models.py                   # Định nghĩa models
│   ├── training.py                 # Training pipeline
│   ├── evaluation.py               # Metrics & evaluation
│   └── visualization.py            # Biểu đồ
├── results/
│   ├── comparison_table.xlsx       # Bảng so sánh
│   ├── charts/                     # Biểu đồ kết quả
│   └── best_model.pkl              # Mô hình tốt nhất
├── requirements.txt
└── README.md
```

---

## 9. METRICS ĐÁNH GIÁ

### 9.1. Metrics chính (primary)

| Metric | Công thức | Ý nghĩa | Mục tiêu |
|--------|-----------|---------|----------|
| **MAE** | mean(\|y - ŷ\|) | Sai số trung bình tuyệt đối | Thấp nhất có thể |
| **MAPE** | mean(\|y - ŷ\| / \|y\|) × 100% | Sai số phần trăm | < 0.10% (PA A), < 2% (PA B) |
| **RMSE** | sqrt(mean((y - ŷ)²)) | Sai số bình phương trung bình | Thấp nhất có thể |

### 9.2. Metrics phụ (secondary)

| Metric | Ý nghĩa | Mục đích |
|--------|---------|---------|
| **nRMSE** | RMSE chuẩn hóa | So sánh giữa datasets khác nhau |
| **R²** | Hệ số xác định | Tỷ lệ phương sai được giải thích (> 0.99) |
| **MAE_CV** | MAE qua Cross Validation | Đánh giá tính tổng quát |
| **Training Time** | Thời gian huấn luyện | Đánh giá hiệu quả |

### 9.3. Phương pháp đánh giá

- **5-Fold Cross Validation**: Kết quả trung bình ± std trên 5 folds
- **Stratified Split**: Đảm bảo phân bổ đều theo `ca` và `góc nghiêng`
- **Multiple Seeds**: Chạy 5-10 lần với seed khác nhau, báo cáo mean ± std
- **Residual Analysis**: Phân tích phân phối sai số dư

---

## 10. KẾT QUẢ BÀN GIAO DỰ KIẾN

### 10.1. Bảng so sánh mở rộng

| Mô hình | MAE | RMSE | MAPE | nRMSE | R² | Training Time |
|---------|-----|------|------|-------|----|----|
| L2N19 (gốc) | 19.57 | 20.63 | 0.180% | 0.006 | ? | ? |
| L2N19 (tối ưu) | ? | ? | ? | ? | ? | ? |
| XGBoost (tuned) | ? | ? | ? | ? | ? | ? |
| Stacking Ensemble | ? | ? | ? | ? | ? | ? |
| ... | ... | ... | ... | ... | ... | ... |

### 10.2. Biểu đồ cần xuất

1. Correlation Heatmap (cập nhật)
2. Feature Importance (từ XGBoost/RF)
3. Learning Curves (train vs validation)
4. Predicted vs Actual (scatter + line plot)
5. Residual Distribution (histogram + Q-Q plot)
6. Model Comparison (bar chart)
7. SHAP Values (giải thích mô hình)

### 10.3. File Excel kết quả

- Prediction results cho mỗi mô hình (Actual, Predicted, Error)
- Sheet tổng hợp metrics
- Sheet hyperparameters tối ưu

---

## 11. RỦI RO & LƯU Ý

| Rủi ro | Mức độ | Giải pháp |
|--------|--------|-----------|
| Overfitting khi thêm features | Cao | Dùng Cross Validation, Regularization |
| MAPE tăng khi loại bỏ biến phái sinh | Rất cao | Chấp nhận - đây là kết quả thực tế hơn |
| Thời gian tuning quá lâu | Trung bình | Dùng Optuna thay vì GridSearch |
| Data không đủ đa dạng | Trung bình | Data augmentation hoặc thu thập thêm |
| Kết quả khác nhau giữa các lần chạy | Thấp | Cố định random seed, báo cáo mean ± std |

---

## 12. TÓM TẮT KHUYẾN NGHỊ

### Hành động ưu tiên cao nhất (làm ngay):

1. **Loại bỏ biến `sản lượng`** (correlation = 1.0 với `công suất tb`, hoàn toàn dư thừa)
2. **Áp dụng StandardScaler** cho tất cả features
3. **Chuyển sang 5-Fold Cross Validation** thay vì single split
4. **Thêm Early Stopping** (patience=15)
5. **Chạy RandomizedSearchCV** với 200+ tổ hợp tham số

### Hành động ưu tiên trung bình (tuần sau):

6. **Thử XGBoost và LightGBM** (thường vượt trội ANN trên tabular data)
7. **Feature Engineering** (polynomial degree 2)
8. **Ensemble Stacking** (kết hợp 3 mô hình tốt nhất)

### Hành động dài hạn (nếu có thời gian):

9. **Xây dựng phương án B** (4 biến độc lập) cho ý nghĩa thực tiễn
10. **SHAP analysis** để giải thích mô hình
11. **PyTorch ANN** với BatchNorm + Dropout + LR Scheduling

---

*Draft Plan được tạo dựa trên phân tích dữ liệu thực tế từ project.*
*Ngày tạo: 2026-02-10*
