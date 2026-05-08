# IEEE-CIS Fraud Detection
---

# კონკურსის მოკლე მიმოხილვა
IEEE-CIS Fraud Detection კონკურსის მიზანია საბანკო ტრანზაქციების მონაცემებზე დაყრდნობით თაღლითური ტრანზაქციების გამოვლენა.

---

# ჩემი მიდგომა
პრობლემის გადასაჭრელად გამოვიყენე სხვადასხვა ML არქიტექტურა. თითოეული მოდელისთვის შევქმენი sklearn Pipeline, რომელიც მოიცავს preprocessing-ს, feature engineering-ს, feature selection-ს და თავად მოდელს. ყველა ექსპერიმენტი დავარეგისტრირე MLflow-ში DagsHub-ზე.

---

# რეპოზიტორიის სტრუქტურა

```
IEEE-CIS-Fraud-Detection/
│
├── model_experiment_LogisticRegression.ipynb  ← Logistic Regression ექსპერიმენტები
├── model_experiment_DecisionTree.ipynb        ← Decision Tree ექსპერიმენტები
├── model_experiment_RandomForest.ipynb        ← Random Forest ექსპერიმენტები
├── model_experiment_AdaBoost.ipynb            ← AdaBoost ექსპერიმენტები
├── model_experiment_XGBoost.ipynb             ← XGBoost ექსპერიმენტები
├── model_inference.ipynb                      ← საუკეთესო მოდელით პროგნოზი
└── README.md                                  ← პროექტის აღწერა
```

---

# ფაილების განმარტება

| Notebook | აღწერა |
|---|---|
| `model_experiment_LogisticRegression.ipynb` | Logistic Regression-ის ექსპერიმენტები სხვადასხვა solver-ით, C პარამეტრით და null threshold-ით |
| `model_experiment_DecisionTree.ipynb` | Decision Tree-ის ექსპერიმენტები max_depth და min_samples_split-ით, overfitting-ის დემონსტრაცია |
| `model_experiment_RandomForest.ipynb` | Random Forest-ის ექსპერიმენტები n_estimators და max_depth-ით |
| `model_experiment_AdaBoost.ipynb` | AdaBoost-ის ექსპერიმენტები n_estimators, learning_rate და base estimator depth-ით |
| `model_experiment_XGBoost.ipynb` | XGBoost-ის ექსპერიმენტები WOE/IV feature selection-ით, threshold tuning-ით |
| `model_inference.ipynb` | MLflow Model Registry-იდან საუკეთესო pipeline-ის ჩატვირთვა და Kaggle submission გენერაცია |

---

# Feature Engineering

ყველა notebook-ში Feature Engineering ხდება `FraudPreprocessor` კლასის შიგნით `_feature_engineering()` მეთოდით:

| Feature | ახალი სვეტის ალგორითმი | აღწერა |
|---|---|---|
| `log_TransactionAmt` | `log1p(TransactionAmt)` | თანხის ლოგარითმი, skewness-ის შემცირება |
| `Transaction_hour` | `(TransactionDT // 3600) % 24` | ტრანზაქციის საათი |
| `Transaction_day` | `(TransactionDT // 86400) % 7` | ტრანზაქციის დღე |
| `TransactionAmt_freq` | frequency encoding | თანხის სიხშირე მონაცემებში |
| `card1_freq` | frequency encoding | ბარათის სიხშირე მონაცემებში |

XGBoost notebook-ში დამატებით გამოვიყენეთ `TransactionAmt_freq` და `card1_freq` — frequency encoding, რომელიც კარგად ასახავს ბარათის გამოყენების სიხშირეს.

---

## კატეგორიული ცვლადების რიცხვითში გადაყვანა

ყველა notebook-ში გამოვიყენეთ **One Hot Encoding** კატეგორიული სვეტებისთვის:
- `handle_unknown='ignore'` — უცნობი კატეგორიების დასამუშავებლად test set-ზე
- `sparse_output=False` — numpy array-ის დასაბრუნებლად

---

## NaN მნიშვნელობების დამუშავება

| სვეტის ტიპი | სტრატეგია |
|---|---|
| Numerical | `SimpleImputer(strategy='median')` |
| Categorical | `SimpleImputer(strategy='most_frequent')` |

---

## Cleaning მიდგომები

1. **Null Threshold** — სვეტები >50% ან >70% null მნიშვნელობით წაიშალა
   - `null_thresh=0.5` — 50%-ზე მეტი null-ის მქონე სვეტები იშლება
   - `null_thresh=0.7` — 70%-ზე მეტი null-ის მქონე სვეტები იშლება (XGBoost-ში საუკეთესო შედეგი)
2. **train/test merge** — `train_transaction` და `train_identity` გაერთიანდა `TransactionID`-ზე LEFT JOIN-ით
3. **stratify=y** — train/test split-ზე გამოვიყენეთ stratification კლასების პროპორციის შესანარჩუნებლად

---

# Feature Selection

გამოვიყენეთ ორი მიდგომა, ორივე `FraudPreprocessor` კლასის შიგნით:

## I. Correlation Filter

სვეტები კორელაციის threshold-ზე მეტი კორელაციით ერთმანეთთან წაიშლება (ნაკლებად მნიშვნელოვანი):
- `corr_thresh=0.9` — 0.9-ზე მაღალი კორელაციის მქონე სვეტები იშლება
- გამოვიყენეთ ყველა notebook-ში

## II. WOE/IV Filter (XGBoost notebook-ში)

Weight of Evidence / Information Value — თითოეული feature-ის სიძლიერე target-თან მიმართებაში:

| IV მნიშვნელობა | ინტერპრეტაცია |
|---|---|
| < 0.02 | უსარგებლო — იშლება |
| 0.02 - 0.1 | სუსტი |
| 0.1 - 0.3 | საშუალო |
| > 0.3 | ძლიერი |

`iv_thresh=0.02` — IV < 0.02 მქონე სვეტები წაიშალა როგორც უსარგებლო.

---

# Training

## ტესტირებული მოდელები

### Logistic Regression
- `class_weight='balanced'` — imbalanced data-ს გასამკლავებლად
- სხვადასხვა `solver`: lbfgs, saga
- სხვადასხვა `C` (regularization): 0.1, 1.0

| Run | Train AUC | Test AUC | CV AUC |
|---|---|---|---|
| lbfgs C=1.0 null=0.5 | 0.8394 | 0.8363 | 0.8372 ± 0.0033 |
| lbfgs C=1.0 null=0.7 | 0.8418 | 0.8396 | 0.8395 ± 0.0032 |
| saga C=1.0 null=0.5 | ~0.84 | ~0.84 | - |
| lbfgs C=0.1 null=0.5 | ~0.83 | ~0.83 | - |

---

### Decision Tree
- `class_weight='balanced'`
- სხვადასხვა `max_depth`: None, 5, 10
- სხვადასხვა `min_samples_split`: 2, 50

| Run | Train AUC | Test AUC | CV AUC |
|---|---|---|---|
| depth=None split=2 | 1.0000 | 0.7769 | 0.7652 ± 0.0047 |
| depth=5 split=2 | 0.8353 | 0.8315 | 0.8334 ± 0.0037 |
| depth=10 split=2 | 0.9014 | 0.8672 | 0.8671 ± 0.0032 |
| depth=10 split=50 | 0.9014 | 0.8681 | 0.8665 ± 0.0031 |

**Overfitting-ის დემონსტრაცია:** `depth=None` — Train AUC=1.0 მაგრამ Test AUC=0.77, კლასიკური overfitting. `max_depth=5`-ით პრობლემა გამოსწორდა.

---

### Random Forest
- `class_weight='balanced'`, `n_jobs=-1`
- სხვადასხვა `n_estimators`: 100, 200
- სხვადასხვა `max_depth`: None, 10, 15

| Run | Train AUC | Test AUC | CV AUC |
|---|---|---|---|
| n=100 depth=None | 1.0000 | 0.9411 | 0.9368 ± 0.0031 |
| n=100 depth=10 | 0.9015 | 0.8855 | 0.8877 ± 0.0028 |
| n=200 depth=10 | 0.9058 | 0.8892 | 0.8897 ± 0.0034 |
| n=200 depth=15 | 0.9616 | 0.9193 | 0.9156 ± 0.0033 |

Random Forest-მა Decision Tree-ს მნიშვნელოვნად აჯობა — bagging-მა variance შეამცირა.

---

### AdaBoost
- Base estimator: `DecisionTreeClassifier`
- სხვადასხვა `n_estimators`: 50, 100, 200
- სხვადასხვა `learning_rate`: 0.1, 0.5, 1.0
- სხვადასხვა base `max_depth`: 1, 2, 3

| Run | Train AUC | Test AUC | CV AUC |
|---|---|---|---|
| n=50 lr=1.0 depth=1 | 0.8463 | 0.8452 | 0.8447 ± 0.0020 |
| n=100 lr=1.0 depth=1 | 0.8549 | 0.8525 | 0.8535 ± 0.0031 |
| n=100 lr=0.5 depth=2 | ~0.86 | ~0.86 | - |
| n=200 lr=0.5 depth=2 | ~0.87 | ~0.87 | - |

AdaBoost-ი ძალიან სტაბილური იყო — Train და Test AUC თითქმის იდენტური, overfitting არ შეინიშნება.

---

### XGBoost
- `scale_pos_weight=27.58` — imbalanced data-ს გასამკლავებლად
- `tree_method='hist'` — სწრაფი ტრენინგი
- Threshold tuning — optimal threshold recall ≥ 75%-სთვის
- WOE/IV feature selection (`iv_thresh=0.02`)

| Run | Train AUC | Test AUC | CV AUC |
|---|---|---|---|
| n=100 depth=5 lr=0.1 | 0.9234 | 0.9109 | 0.9122 ± 0.0024 |
| n=200 depth=5 lr=0.05 | 0.9250 | 0.9119 | 0.9131 ± 0.0030 |
| n=200 depth=6 lr=0.1 | 0.9632 | 0.9381 | 0.9372 ± 0.0025 |
| n=300 depth=6 lr=0.05 | 0.9532 | 0.9321 | 0.9311 ± 0.0023 |
| **n=300 depth=8 lr=0.05** | **0.9800** | **0.9499** | **0.9454 ± 0.0020** |

---

## Hyperparameter ოპტიმიზაციის მიდგომა

თითოეული მოდელისთვის გავტესტეთ სხვადასხვა კომბინაციები:
- **null_thresh**: 0.5 vs 0.7 — რამდენი სვეტი წაიშალოს null-ების მიხედვით
- **corr_thresh**: 0.9 — კორელაციის ზღვარი
- **მოდელის სპეციფიკური პარამეტრები** — max_depth, n_estimators, learning_rate, subsample

---

## საბოლოო მოდელის შერჩევის დასაბუთება

საუკეთესო მოდელად შეირჩა **XGBoost (n=300, depth=8, lr=0.05, null=0.7)**:
- **Test AUC: 0.9499** — ყველაზე მაღალი ყველა ექსპერიმენტს შორის
- **CV AUC: 0.9454 ± 0.0020** — სტაბილური, დაბალი variance
- **Kaggle Public Score: 0.8807**
- **Kaggle Private Score: 0.9171**
<img width="1151" height="188" alt="image" src="https://github.com/user-attachments/assets/022f90df-9b99-437e-87fb-5db30107c7a8" />

---

# MLflow Tracking

## MLflow ექსპერიმენტების ბმული
[DagsHub MLflow](https://dagshub.com/Nestor-Dzadzamia/IEEE-CIS-Fraud-Detection.mlflow)

## ექსპერიმენტების სტრუქტურა

| Experiment | Run-ების რაოდენობა |
|---|---|
| `LogisticRegression_Training` | 4 |
| `DecisionTree_Training` | 4 |
| `RandomForest_Training` | 4 |
| `AdaBoost_Training` | 6 |
| `XGBoost_Training` | 7 |

## ჩაწერილი მეტრიკები

თითოეულ run-ში დაილოგა:
- `train_roc_auc`, `test_roc_auc` — overfitting-ის დეტექტისთვის
- `cv_mean_roc_auc`, `cv_std_roc_auc` — მოდელის სტაბილურობა
- `tn`, `fp`, `fn`, `tp` — confusion matrix
- Classification report მეტრიკები (precision, recall, f1) — ყველა კლასისთვის
- XGBoost-ში დამატებით: `optimal_threshold`, `adj_f1`, `adj_precision`, `adj_recall`

## საუკეთესო მოდელის შედეგები

| მეტრიკა | მნიშვნელობა |
|---|---|
| Train ROC-AUC | 0.9800 |
| Test ROC-AUC | 0.9499 |
| CV ROC-AUC | 0.9454 ± 0.0020 |
| Kaggle Public Score | 0.8807 |
| Kaggle Private Score | 0.9171 |
