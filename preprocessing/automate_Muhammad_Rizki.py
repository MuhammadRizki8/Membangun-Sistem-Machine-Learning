"""
Automated Preprocessing Pipeline for Heart Disease Classification
Template MSML - Otomatisasi Data Preprocessing
Author: [Nama-siswa]
Dataset: Heart Failure Prediction Dataset

This module contains functions to automatically preprocess the heart disease dataset
and return training-ready data.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')
from dotenv import load_dotenv
load_dotenv()
import os

class HeartDiseasePreprocessor:
    """
    Automated preprocessing pipeline for Heart Disease dataset
    """
    
    def __init__(self):
        """Initialize preprocessor with default parameters"""
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.is_fitted = False
        
        # Define column types
        self.numerical_columns = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
        self.binary_categorical = ['Sex', 'ExerciseAngina']
        self.multi_categorical = ['ChestPainType', 'RestingECG', 'ST_Slope']
        self.target_column = 'HeartDisease'
    
    def load_data(self, file_path):
        """
        Load dataset from CSV file
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        try:
            df = pd.read_csv(file_path)
            print(f"✅ Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except Exception as e:
            print(f"❌ Error loading dataset: {str(e)}")
            return None
    
    def handle_missing_values(self, df):
        """
        Handle missing values and invalid zero values
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with handled missing values
        """
        df_clean = df.copy()
        
        # Handle Cholesterol = 0 (replace with median)
        cholesterol_zero_count = (df_clean['Cholesterol'] == 0).sum()
        if cholesterol_zero_count > 0:
            cholesterol_median = df_clean[df_clean['Cholesterol'] > 0]['Cholesterol'].median()
            df_clean.loc[df_clean['Cholesterol'] == 0, 'Cholesterol'] = cholesterol_median
            print(f"✅ Replaced {cholesterol_zero_count} zero values in Cholesterol with median: {cholesterol_median}")
        
        # Handle RestingBP = 0 (replace with median) 
        restingbp_zero_count = (df_clean['RestingBP'] == 0).sum()
        if restingbp_zero_count > 0:
            restingbp_median = df_clean[df_clean['RestingBP'] > 0]['RestingBP'].median()
            df_clean.loc[df_clean['RestingBP'] == 0, 'RestingBP'] = restingbp_median
            print(f"✅ Replaced {restingbp_zero_count} zero values in RestingBP with median: {restingbp_median}")
        
        # Check for other missing values
        missing_count = df_clean.isnull().sum().sum()
        if missing_count > 0:
            print(f"⚠️ Found {missing_count} missing values")
            # Handle other missing values if needed
            df_clean = df_clean.fillna(df_clean.median(numeric_only=True))
            print("✅ Filled remaining missing values with median")
        else:
            print("✅ No missing values found")
            
        return df_clean
    
    def remove_duplicates(self, df):
        """
        Remove duplicate rows
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe without duplicates
        """
        initial_count = len(df)
        df_unique = df.drop_duplicates()
        final_count = len(df_unique)
        removed_count = initial_count - final_count
        
        if removed_count > 0:
            print(f"✅ Removed {removed_count} duplicate rows")
        else:
            print("✅ No duplicate rows found")
            
        return df_unique
    
    def encode_categorical_variables(self, df, fit=True):
        """
        Encode categorical variables
        
        Args:
            df (pd.DataFrame): Input dataframe
            fit (bool): Whether to fit encoders or use existing ones
            
        Returns:
            pd.DataFrame: Dataframe with encoded categorical variables
        """
        df_encoded = df.copy()
        
        # One-hot encoding for multi-categorical variables
        df_encoded = pd.get_dummies(df_encoded, 
                                   columns=self.multi_categorical,
                                   prefix=['ChestPain', 'RestingECG', 'ST_Slope'])
        
        # Label encoding for binary categorical variables
        for col in self.binary_categorical:
            if col in df_encoded.columns:
                if fit:
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col])
                    self.label_encoders[col] = le
                    print(f"✅ Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
                else:
                    if col in self.label_encoders:
                        df_encoded[col] = self.label_encoders[col].transform(df_encoded[col])
                    else:
                        print(f"⚠️ No encoder found for {col}")
        
        return df_encoded
    
    def scale_features(self, X, fit=True):
        """
        Scale numerical features
        
        Args:
            X (pd.DataFrame): Feature dataframe
            fit (bool): Whether to fit scaler or use existing one
            
        Returns:
            pd.DataFrame: Dataframe with scaled features
        """
        X_scaled = X.copy()
        
        # Get numerical columns that exist in the dataframe
        existing_numerical = [col for col in self.numerical_columns if col in X_scaled.columns]
        
        if existing_numerical:
            if fit:
                X_scaled[existing_numerical] = self.scaler.fit_transform(X_scaled[existing_numerical])
                print(f"✅ Fitted and scaled numerical features: {existing_numerical}")
            else:
                X_scaled[existing_numerical] = self.scaler.transform(X_scaled[existing_numerical])
                print(f"✅ Scaled numerical features: {existing_numerical}")
        
        return X_scaled
    
    def split_features_target(self, df):
        """
        Split features and target variable
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            tuple: (X, y) features and target
        """
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataset")
        
        X = df.drop(self.target_column, axis=1)
        y = df[self.target_column]
        
        print(f"✅ Split features and target: {X.shape[1]} features, {len(y)} samples")
        return X, y
    
    def train_test_split(self, X, y, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            test_size (float): Proportion of test set
            random_state (int): Random seed
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"✅ Train-test split completed:")
        print(f"   Training set: {len(X_train)} samples")
        print(f"   Testing set: {len(X_test)} samples")
        print(f"   Target distribution in training set: {y_train.value_counts(normalize=True).to_dict()}")
        
        return X_train, X_test, y_train, y_test
    
    def fit_transform(self, df):
        """
        Complete preprocessing pipeline (fit and transform)
        
        Args:
            df (pd.DataFrame): Raw dataset
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test) preprocessed and split data
        """
        print("\n" + "="*50)
        print("AUTOMATED PREPROCESSING PIPELINE")
        print("="*50)
        
        # Step 1: Handle missing values
        print("\n🔧 Step 1: Handling missing values...")
        df_clean = self.handle_missing_values(df)
        
        # Step 2: Remove duplicates
        print("\n🔄 Step 2: Removing duplicates...")
        df_unique = self.remove_duplicates(df_clean)
        
        # Step 3: Encode categorical variables
        print("\n🏷️ Step 3: Encoding categorical variables...")
        df_encoded = self.encode_categorical_variables(df_unique, fit=True)
        
        # Step 4: Split features and target
        print("\n🎯 Step 4: Splitting features and target...")
        X, y = self.split_features_target(df_encoded)
        
        # Step 5: Scale features
        print("\n⚖️ Step 5: Scaling features...")
        X_scaled = self.scale_features(X, fit=True)
        
        # Step 6: Train-test split
        print("\n🔀 Step 6: Train-test split...")
        X_train, X_test, y_train, y_test = self.train_test_split(X_scaled, y)
        
        # Mark as fitted
        self.is_fitted = True
        
        print("\n✅ PREPROCESSING COMPLETED SUCCESSFULLY!")
        print(f"📊 Final dataset shape: {X_scaled.shape}")
        print(f"🎯 Ready for model training!")
        
        return X_train, X_test, y_train, y_test
    
    def transform(self, df):
        """
        Transform new data using fitted preprocessors
        
        Args:
            df (pd.DataFrame): New dataset to transform
            
        Returns:
            pd.DataFrame: Transformed features
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform. Use fit_transform() first.")
        
        print("\n🔄 Transforming new data...")
        
        # Apply same preprocessing steps
        df_clean = self.handle_missing_values(df)
        df_encoded = self.encode_categorical_variables(df_clean, fit=False)
        
        # Handle target column if present
        if self.target_column in df_encoded.columns:
            X = df_encoded.drop(self.target_column, axis=1)
        else:
            X = df_encoded
        
        X_scaled = self.scale_features(X, fit=False)
        
        print("✅ Data transformed successfully!")
        return X_scaled
    
    def save_processed_data(self, X_train, X_test, y_train, y_test, output_dir="./"):
        """
        Save processed data to CSV files
        
        Args:
            X_train, X_test, y_train, y_test: Processed data splits
            output_dir (str): Output directory path
        """
        try:
            X_train.to_csv(f"{output_dir}X_train_processed.csv", index=False)
            X_test.to_csv(f"{output_dir}X_test_processed.csv", index=False)
            y_train.to_csv(f"{output_dir}y_train_processed.csv", index=False)
            y_test.to_csv(f"{output_dir}y_test_processed.csv", index=False)
            
            print(f"\n💾 Processed data saved to {output_dir}")
            print("   Files created:")
            print("   - X_train_processed.csv")
            print("   - X_test_processed.csv") 
            print("   - y_train_processed.csv")
            print("   - y_test_processed.csv")
            
        except Exception as e:
            print(f"❌ Error saving processed data: {str(e)}")

# =====================================
# MAIN FUNCTION FOR DIRECT EXECUTION
# =====================================

def preprocess_heart_disease_data(input_file_path, output_dir="./"):
    """
    Main function to preprocess heart disease data
    
    Args:
        input_file_path (str): Path to raw dataset
        output_dir (str): Directory to save processed files
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test) processed data
    """
    # Initialize preprocessor
    preprocessor = HeartDiseasePreprocessor()
    
    # Load raw data
    print("📁 Loading raw dataset...")
    df = preprocessor.load_data(input_file_path)
    
    if df is None:
        return None
    
    # Run complete preprocessing pipeline
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)
    
    # Save processed data
    preprocessor.save_processed_data(X_train, X_test, y_train, y_test, output_dir)
    
    return X_train, X_test, y_train, y_test, preprocessor

# =====================================
# EXAMPLE USAGE AND TESTING
# =====================================

if __name__ == "__main__":
    """
    Example usage of the automated preprocessing pipeline
    """
    print("🚀 Heart Disease Data Preprocessing Pipeline")
    print("="*60)
    
    # Example file paths (adjust as needed)
    # input_file = "../heart-failure-prediction/heart.csv"  # Raw dataset
    # output_directory = "./processed"   # Output directory
    input_file = os.getenv("INPUT_FILE", "heart-failure-prediction/heart.csv")
    output_directory = os.getenv("OUTPUT_DIRECTORY", "preprocessing/")
    
    try:
        # Run preprocessing
        result = preprocess_heart_disease_data(input_file, output_directory)
        
        if result is not None:
            X_train, X_test, y_train, y_test, preprocessor = result
            
            # Display summary
            print("\n📋 PREPROCESSING SUMMARY:")
            print("="*40)
            print(f"Training Features Shape: {X_train.shape}")
            print(f"Testing Features Shape: {X_test.shape}")
            print(f"Training Target Shape: {y_train.shape}")
            print(f"Testing Target Shape: {y_test.shape}")
            print(f"Feature Names: {list(X_train.columns)}")
            
            # Example: Transform new data (optional)
            print("\n🔄 Example: Transform new data...")
            try:
                # Load same data as example of new data transformation
                new_data = preprocessor.load_data(input_file)
                if new_data is not None:
                    # Remove a few rows to simulate new data
                    new_data_sample = new_data.head(10)
                    transformed_new_data = preprocessor.transform(new_data_sample)
                    print(f"✅ New data transformed: {transformed_new_data.shape}")
            except Exception as e:
                print(f"⚠️ New data transformation example failed: {str(e)}")
            
            print("\n🎉 PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY!")
            
        else:
            print("❌ Preprocessing failed!")
            
    except Exception as e:
        print(f"❌ Error in preprocessing pipeline: {str(e)}")
        
    print("\n" + "="*60)
    print("📚 USAGE INSTRUCTIONS:")
    print("="*60)
    print("1. Import this module: from automate_preprocessing import preprocess_heart_disease_data")
    print("2. Call function: X_train, X_test, y_train, y_test, preprocessor = preprocess_heart_disease_data('heart.csv')")
    print("3. Use returned data for model training")
    print("4. Use preprocessor.transform() for new data")
    print("\n📁 Generated Files:")
    print("- X_train_processed.csv")
    print("- X_test_processed.csv") 
    print("- y_train_processed.csv")
    print("- y_test_processed.csv")