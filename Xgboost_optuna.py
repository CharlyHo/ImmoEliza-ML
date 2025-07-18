import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
warnings.filterwarnings('ignore')

class HousePricePredictor:
    def __init__(self, csv_file_path, save_path="best_model.pkl"):
        """
        Initialize the predictor with data from CSV file
        """
        self.csv_file_path = csv_file_path
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_model = None
        self.scaler = StandardScaler()
        self.save_path = save_path
    
        
    def load_and_prepare_data(self):
        """
        Load data from CSV and prepare features and target
        """
        print("Loading data...")
        self.data = pd.read_csv(self.csv_file_path)
        print(f"Data shape: {self.data.shape}")
        
         
        # Display basic info about the dataset
        print("\nDataset Info:")
        print(self.data.info())
        print("\nMissing values:")
        print(self.data.isnull().sum())
        
        # Handle missing values if any
        if self.data.isnull().sum().sum() > 0:
            print("Handling missing values...")
            # Fill numerical columns with median
            numerical_cols = self.data.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if col != 'price':
                    self.data[col].fillna(self.data[col].median(), inplace=True)
    
        # Remove postCode column and separate features and target
        columns_to_drop = ['price']
        if 'postCode' in self.data.columns:
            columns_to_drop.append('postCode')  
            
        self.X = self.data.drop(columns_to_drop, axis=1)
        self.y = self.data['price']
        
        print(f"Features shape: {self.X.shape}")
        print(f"Target shape: {self.y.shape}")
        print(f"Feature columns: {list(self.X.columns)}")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")

    
 
    def basic_xgboost_model(self):

        # using model as .joblib if existing
        if os.path.exists(self.save_path):
            print(f"[SKIP] Model file already exists at '{self.save_path}'. Skipping training.")
            self.model = joblib.load(self.save_path)
            return 
        
        """
        Train a basic XGBoost model with chosen parameters
        """
        print("\n" + "="*50)
        print("TRAINING BASIC XGBOOST MODEL")
        print("="*50)
        
        # Create and train basic model
        basic_model = xgb.XGBRegressor(
            random_state=42,
            n_estimators=1000,
            learning_rate=0.1,
            max_depth=6
        )
        
        basic_model.fit(self.X_train, self.y_train)
        
        # Make predictions
        y_pred_train = basic_model.predict(self.X_train)
        y_pred_test = basic_model.predict(self.X_test)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
        train_mae = mean_absolute_error(self.y_train, y_pred_train)
        test_mae = mean_absolute_error(self.y_test, y_pred_test)
        train_r2 = r2_score(self.y_train, y_pred_train)
        test_r2 = r2_score(self.y_test, y_pred_test)
        
        print(f"Basic XGBoost Results:")
        print(f"Train RMSE: {train_rmse:,.2f}")
        print(f"Test RMSE: {test_rmse:,.2f}")
        print(f"Train MAE: {train_mae:,.2f}")
        print(f"Test MAE: {test_mae:,.2f}")
        print(f"Train R²: {train_r2:.4f}")
        print(f"Test R²: {test_r2:.4f}")
        
        return basic_model, test_rmse
    
    def objective(self, trial):
        """
        Objective function for Optuna opti'
        """
        # Suggest hyperparameters
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 6),
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
            'random_state': 42
        }
        
        # Create model with suggested parameters
        model = xgb.XGBRegressor(**params)
        
        # Perform cross-validation
        cv_scores = cross_val_score(
            model, self.X_train, self.y_train, 
            cv=5, scoring='neg_root_mean_squared_error'
        )
        
        # Return the mean CV score (Optuna minimizes, so we return negative RMSE)
        return -cv_scores.mean()
    
    def optimize_with_optuna(self, n_trials=1500):
        """
        Optimize XGBoost hyperparameters using Optuna
        """
        print("\n" + "="*50)
        print("OPTIMIZING XGBOOST WITH OPTUNA")
        print("="*50)
        print(f"Running {n_trials} trials...")
        
        # Create study
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)
        
        print(f"\nOptimization completed!")
        print(f"Best RMSE: {study.best_value:,.2f}")
        print(f"Best parameters: {study.best_params}")
        
        # Train final model with best parameters
        self.best_model = xgb.XGBRegressor(**study.best_params)
        self.best_model.fit(self.X_train, self.y_train)
        
        # Make predictions
        y_pred_train = self.best_model.predict(self.X_train)
        y_pred_test = self.best_model.predict(self.X_test)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
        train_mae = mean_absolute_error(self.y_train, y_pred_train)
        test_mae = mean_absolute_error(self.y_test, y_pred_test)
        train_r2 = r2_score(self.y_train, y_pred_train)
        test_r2 = r2_score(self.y_test, y_pred_test)
        
        print(f"\nOptimized XGBoost Results:")
        print(f"Train RMSE: {train_rmse:,.2f}")
        print(f"Test RMSE: {test_rmse:,.2f}")
        print(f"Train MAE: {train_mae:,.2f}")
        print(f"Test MAE: {test_mae:,.2f}")
        print(f"Train R²: {train_r2:.4f}")
        print(f"Test R²: {test_r2:.4f}")
        
        return study, test_rmse
    
    #saving .joblib
    def save(self):
        if self.best_model is None:
            raise ValueError("Model should be trained before saving.")
        
        # ensure directory exists
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        
        joblib.dump(self.best_model, self.save_path)
        print(f"[SAVE] Model saved in file '{self.save_path}'.")

    

def main():
    """
    Main function to run the house price prediction pipeline
    """
    # Initialize predictor 
    predictor = HousePricePredictor("ml_ready_real_estate_data_soft_filled.csv")

    
    # Load and prepare data
    predictor.load_and_prepare_data()
    
    # Train basic XGBoost model
    basic_model, basic_rmse = predictor.basic_xgboost_model()
    
    # Optimize with Optuna (+ chosing number of try)
    study, optimized_rmse = predictor.optimize_with_optuna(n_trials=150)
    
    # Calculate improvement
    improvement = ((basic_rmse - optimized_rmse) / basic_rmse) * 100
    print(f"\nImprovement: {improvement:.2f}% reduction in RMSE")

    # Saving model
    predictor.save()
    

if __name__ == "__main__":
   main()