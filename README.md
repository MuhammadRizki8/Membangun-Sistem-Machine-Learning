# Eksperimen*SML* Muhammad Rizki

Template Eksperimen Machine Learning untuk Klasifikasi Penyakit Jantung menggunakan Heart Failure Prediction Dataset dari Kaggle.

## 📊 Dataset Information

- **Dataset**: Heart Failure Prediction Dataset
- **Source**: [Kaggle - Heart Failure Prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- **Size**: 918 samples, 12 features
- **Task**: Binary Classification (Heart Disease: 0/1)
- **Features**: Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope, HeartDisease

## 🏗️ Repository Structure

```
Eksperimen_SML_[Nama-siswa]/
├── .github/
│   └── workflows/
│       └── preprocessing-workflow.yml    # GitHub Actions workflow
├── heart-failure-prediction/                           # Raw dataset directory
│   └── heart.csv
├── preprocessing/
│   ├── Eksperimen_[Nama-siswa].ipynb   # Experimentation notebook
│   ├── automate_[Nama-siswa].py        # Automated preprocessing script
│   └── heart_preprocessing/            # Processed dataset directory
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       ├── y_test.csv
│       └── heart_processed.csv
├── requirements.txt                     # Python dependencies
├── README.md                           # This file
└── preprocessing_report.md             # Automated preprocessing report
```

## 🎯 Kriteria Submission

### ✅ Basic (2 pts)

- [x] Melakukan tahapan experimentation secara manual
- [x] Data loading pada notebook
- [x] EDA (Exploratory Data Analysis) pada notebook
- [x] Preprocessing pada notebook

### ✅ Skilled (3 pts)

- [x] Semua tahap Basic terpenuhi
- [x] File `automate_[Nama-siswa].py` dengan fungsi preprocessing otomatis
- [x] Konversi dari proses eksperimen dengan struktur berbeda
- [x] Mengembalikan data siap latih

### ✅ Advance (4 pts)

- [x] Semua tahap Skilled terpenuhi
- [x] GitHub Actions workflow untuk preprocessing otomatis
- [x] Repository dengan struktur folder sesuai kriteria
- [x] Actions mengembalikan dataset terproses

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/username/Eksperimen_SML_[Nama-siswa].git
cd Eksperimen_SML_[Nama-siswa]

# Install dependencies
pip install -r requirements.txt
```

### 2. Manual Experimentation

Buka dan jalankan notebook `preprocessing/Eksperimen_[Nama-siswa].ipynb` di Jupyter atau Google Colab:

```bash
jupyter notebook preprocessing/Eksperimen_[Nama-siswa].ipynb
```

### 3. Automated Preprocessing

```python
from preprocessing.automate_[Nama-siswa] import preprocess_heart_disease_data

# Run automated preprocessing
X_train, X_test, y_train, y_test, preprocessor = preprocess_heart_disease_data('heart.csv')

# Use preprocessed data for model training
print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")
```

### 4. GitHub Actions (Automatic)

Workflow akan otomatis berjalan ketika:

- Push ke branch `main` atau `master`
- Ada perubahan pada folder `preprocessing/` atau file `automate_*.py`
- Manual trigger melalui GitHub Actions tab

## 📝 Preprocessing Steps

### 1. Data Loading

- Load dataset dari CSV file
- Validasi struktur dan tipe data

### 2. Missing Value Handling

- Identifikasi missing values
- Handle nilai 0 yang tidak valid pada Cholesterol dan RestingBP
- Imputasi dengan median untuk nilai numerik

### 3. Duplicate Removal

- Deteksi dan hapus data duplikat
- Preserve data integrity

### 4. Categorical Encoding

- **One-Hot Encoding**: ChestPainType, RestingECG, ST_Slope
- **Label Encoding**: Sex, ExerciseAngina (binary variables)

### 5. Feature Scaling

- **StandardScaler** untuk fitur numerik: Age, RestingBP, Cholesterol, MaxHR, Oldpeak
- Preserve distribusi data

### 6. Train-Test Split

- Ratio: 80% training, 20% testing
- Stratified split untuk menjaga distribusi target
- Random state: 42 (reproducible)

## 📊 EDA Highlights

### Dataset Overview

- **Total Samples**: 918
- **Features**: 11 input features + 1 target
- **Target Distribution**: ~55% Heart Disease, ~45% No Heart Disease
- **Missing Values**: None (explicit)
- **Data Quality Issues**: Zero values in Cholesterol (172 samples)

### Key Insights

1. **Age Distribution**: Most patients between 50-60 years
2. **Gender**: More male patients (79%) than female (21%)
3. **Chest Pain**: ASY (Asymptomatic) most common type
4. **Heart Rate**: Lower max HR associated with heart disease
5. **Exercise Angina**: Strong predictor of heart disease

## 🤖 GitHub Actions Workflow

### Triggers

- Push to main/master branch
- Changes in preprocessing files
- Manual workflow dispatch

### Steps

1. **Environment Setup**: Python 3.9, dependencies installation
2. **Data Download**: Kaggle API integration (optional)
3. **Preprocessing**: Run automated pipeline
4. **Validation**: Quality checks on processed data
5. **Artifacts**: Upload processed files
6. **Reporting**: Generate preprocessing report

### Secrets Configuration

For Kaggle dataset download, add these secrets to your repository:

- `KAGGLE_USERNAME`: Your Kaggle username
- `KAGGLE_KEY`: Your Kaggle API key

## 📋 Usage Examples

### Using Preprocessor Class

```python
from preprocessing.automate_[Nama-siswa] import HeartDiseasePreprocessor

# Initialize preprocessor
preprocessor = HeartDiseasePreprocessor()

# Load and preprocess data
df = preprocessor.load_data('heart.csv')
X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)

# Transform new data
new_data_processed = preprocessor.transform(new_data)

# Save processed data
preprocessor.save_processed_data(X_train, X_test, y_train, y_test, './output/')
```

### Direct Function Call

```python
from preprocessing.automate_[Nama-siswa] import preprocess_heart_disease_data

# One-line preprocessing
result = preprocess_heart_disease_data('heart.csv', './output/')
X_train, X_test, y_train, y_test, preprocessor = result
```

## 🔧 Customization

### Modifying Preprocessing Steps

Edit `automate_[Nama-siswa].py` to customize:

- Feature selection
- Encoding strategies
- Scaling methods
- Train-test split ratio

### Adding New Features

1. Update `HeartDiseasePreprocessor` class
2. Modify column definitions
3. Test with sample data
4. Update documentation

## 📊 Model Training Example

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load preprocessed data
X_train, X_test, y_train, y_test, _ = preprocess_heart_disease_data('heart.csv')

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

## 🐛 Troubleshooting

### Common Issues

1. **Kaggle API Error**

   ```bash
   # Check API credentials
   cat ~/.kaggle/kaggle.json
   # Re-download credentials from Kaggle
   ```

2. **Missing Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **File Path Issues**
   ```python
   # Use absolute paths
   import os
   file_path = os.path.abspath('heart.csv')
   ```

### GitHub Actions Issues

1. **Workflow Not Triggering**

   - Check file paths in trigger conditions
   - Verify branch names (main vs master)

2. **Kaggle Download Fails**
   - Add KAGGLE_USERNAME and KAGGLE_KEY secrets
   - Check dataset URL

## 📚 References

- [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Template MSML Guidelines](link-to-template)

## 👥 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

- **Author**: [Nama-siswa]
- **Email**: [email@domain.com]
- **GitHub**: [@username](https://github.com/username)

---

⭐ **Star this repository if it helped you!**
