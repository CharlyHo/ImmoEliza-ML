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
warnings.filterwarnings('ignore')

class HousePricePredictor:
    def __init__(self, csv_file_path):
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
    
    def add_lat_lon(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add latitude and longitude coordinates based on postal codes
        """
        print("Adding geographic coordinates...")
        df["postCode"] = df["postCode"].astype(str)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(base_dir, "ImmoEliza-ML", "georef-belgium-postal-codes.csv")
        
        try:
            geo_df = pd.read_csv(data_path, delimiter=";")
            geo_df[["lat", "lon"]] = geo_df["Geo Point"].str.split(",", expand=True)
            geo_df["lat"] = geo_df["lat"].astype(float)
            geo_df["lon"] = geo_df["lon"].astype(float)
            geo_df["postCode"] = geo_df["Post code"].astype(str)
            
            # Merge geographic data
            df = df.merge(geo_df[["postCode", "lat", "lon"]], on="postCode", how="left")
            
            # Check how many postal codes were matched
            matched_count = df[['lat', 'lon']].dropna().shape[0]
            total_count = df.shape[0]
            print(f"Geographic coordinates added for {matched_count}/{total_count} records ({matched_count/total_count*100:.1f}%)")
            
        
            
        except FileNotFoundError:
            print(f"Warning: Geographic data file not found at {data_path}")
        
        return df
        
    def load_and_prepare_data(self):
        """
        Load data from CSV and prepare features and target
        """
        print("Loading data...")
        self.data = pd.read_csv(self.csv_file_path)
        print(f"Data shape: {self.data.shape}")
        
        # Add geographic coordinates first
        if 'postCode' in self.data.columns:
            self.data = self.add_lat_lon(self.data)
        else:
            print("Warning: 'postCode' column not found. Skipping geographic feature addition.")
        
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
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
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
    
    def plot_feature_importance(self):
        """
        Plot feature importance from the best model
        """
        if self.best_model is None:
            print("No optimized model found. Please run optimization first.")
            return
        
        # Get feature importance
        importance = self.best_model.feature_importances_
        feature_names = self.X.columns
        
        # Create DataFrame for easier plotting
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=True)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.xlabel('Feature Importance')
        plt.title('XGBoost Feature Importance (Including Geographic Features)')
        plt.tight_layout()
        plt.show()
        
        # Print top 10 features
        print("\nTop 10 Most Important Features:")
        top_features = importance_df.tail(10)
        for idx, row in top_features.iterrows():
            print(f"{row['feature']}: {row['importance']:.4f}")
    
    def plot_predictions(self):
        """
        Plot actual vs predicted values
        """
        if self.best_model is None:
            print("No optimized model found. Please run optimization first.")
            return
        
        y_pred_test = self.best_model.predict(self.X_test)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(self.y_test, y_pred_test, alpha=0.6)
        plt.plot([self.y_test.min(), self.y_test.max()], 
                 [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.title('Actual vs Predicted House Prices (With Geographic Features)')
        plt.tight_layout()
        plt.show()
    
    def plot_geographic_analysis(self):
        """
        Plot geographic distribution of houses and prices
        """
        if 'lat' not in self.data.columns or 'lon' not in self.data.columns:
            print("No geographic coordinates available for plotting.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Geographic distribution of houses
        axes[0].scatter(self.data['lon'], self.data['lat'], alpha=0.6, s=10)
        axes[0].set_xlabel('Longitude')
        axes[0].set_ylabel('Latitude')
        axes[0].set_title('Geographic Distribution of Houses')
        
        # Plot 2: Price heatmap by location
        scatter = axes[1].scatter(self.data['lon'], self.data['lat'], 
                                c=self.data['price'], alpha=0.6, s=10, cmap='viridis')
        axes[1].set_xlabel('Longitude')
        axes[1].set_ylabel('Latitude')
        axes[1].set_title('House Prices by Geographic Location')
        plt.colorbar(scatter, ax=axes[1], label='Price (€)')
        
        plt.tight_layout()
        plt.show()
    
    def predict_new_house(self, house_features):
        """
        Predict price for a new house
        """
        if self.best_model is None:
            print("No optimized model found. Please run optimization first.")
            return None
        
        # Convert to DataFrame if it's a dict
        if isinstance(house_features, dict):
            house_features = pd.DataFrame([house_features])
        
        # Make prediction
        prediction = self.best_model.predict(house_features)
        return prediction[0]

def main():
    """
    Main function to run the house price prediction pipeline
    """
    # Initialize predictor 
    predictor = HousePricePredictor('ml_ready_real_estate_data_soft_filled.csv')
    
    # Load and prepare data
    predictor.load_and_prepare_data()
    
    # Train basic XGBoost model
    basic_model, basic_rmse = predictor.basic_xgboost_model()
    
    # Optimize with Optuna (+ chosing number of try)
    study, optimized_rmse = predictor.optimize_with_optuna(n_trials=150)
    
    # Calculate improvement
    improvement = ((basic_rmse - optimized_rmse) / basic_rmse) * 100
    print(f"\nImprovement: {improvement:.2f}% reduction in RMSE")
    
    # Plot results
    predictor.plot_feature_importance()
    predictor.plot_predictions()
    predictor.plot_geographic_analysis() 
    
    # Example predictions for completely new houses 
    print("\n" + "="*50)
    print("EXAMPLE PREDICTIONS ON NEW HOUSES (WITH GEOGRAPHIC DATA)")
    print("="*50)
    
    # Example 1: Small apartment in Brussels area
    small_apartment = {
        'bedroomCount': 1.0,
        'bathroomCount': 1.0,
        'habitableSurface': 65.0,
        'toiletCount': 1.0,
        'terraceSurface': 0.0,
        'gardenSurface': 0.0,
        'province_encoded': 1.0,
        'type_encoded': 1,
        'subtype_encoded': 1,
        'epcScore_encoded': 3.0,
        'hasAttic_encoded': 0,
        'hasGarden_encoded': 0,
        'hasAirConditioning_encoded': 0,
        'hasArmoredDoor_encoded': 1,
        'hasVisiophone_encoded': 1,
        'hasTerrace_encoded': 0,
        'hasOffice_encoded': 0,
        'hasSwimmingPool_encoded': 0,
        'hasFireplace_encoded': 0,
        'hasBasement_encoded': 0,
        'hasDressingRoom_encoded': 0,
        'hasDiningRoom_encoded': 0,
        'hasLift_encoded': 1,
        'hasHeatPump_encoded': 0,
        'hasPhotovoltaicPanels_encoded': 0,
        'hasLivingRoom_encoded': 1,
        'lat': 50.8503,  # Brussels latitude
        'lon': 4.3517    # Brussels longitude
    }
    
    # Example 2: Large family house in Antwerp area
    large_house = {
        'bedroomCount': 4.0,
        'bathroomCount': 3.0,
        'habitableSurface': 250.0,
        'toiletCount': 3.0,
        'terraceSurface': 40.0,
        'gardenSurface': 500.0,
        'province_encoded': 1.0,
        'type_encoded': 1,
        'subtype_encoded': 2,
        'epcScore_encoded': 6.0,
        'hasAttic_encoded': 1,
        'hasGarden_encoded': 1,
        'hasAirConditioning_encoded': 1,
        'hasArmoredDoor_encoded': 1,
        'hasVisiophone_encoded': 1,
        'hasTerrace_encoded': 1,
        'hasOffice_encoded': 1,
        'hasSwimmingPool_encoded': 1,
        'hasFireplace_encoded': 1,
        'hasBasement_encoded': 1,
        'hasDressingRoom_encoded': 1,
        'hasDiningRoom_encoded': 1,
        'hasLift_encoded': 0,
        'hasHeatPump_encoded': 1,
        'hasPhotovoltaicPanels_encoded': 1,
        'hasLivingRoom_encoded': 1,
        'lat': 51.2194,  # Antwerp latitude
        'lon': 4.4025    # Antwerp longitude
    }
    
    # Example 3: Modern eco-friendly house in Ghent area
    eco_house = {
        'bedroomCount': 3.0,
        'bathroomCount': 2.0,
        'habitableSurface': 150.0,
        'toiletCount': 2.0,
        'terraceSurface': 20.0,
        'gardenSurface': 200.0,
        'province_encoded': 1.0,
        'type_encoded': 1,
        'subtype_encoded': 1,
        'epcScore_encoded': 7.0,  
        'hasAttic_encoded': 1,
        'hasGarden_encoded': 1,
        'hasAirConditioning_encoded': 0,
        'hasArmoredDoor_encoded': 1,
        'hasVisiophone_encoded': 1,
        'hasTerrace_encoded': 1,
        'hasOffice_encoded': 1,
        'hasSwimmingPool_encoded': 0,
        'hasFireplace_encoded': 1,
        'hasBasement_encoded': 0,
        'hasDressingRoom_encoded': 1,
        'hasDiningRoom_encoded': 1,
        'hasLift_encoded': 0,
        'hasHeatPump_encoded': 1,  
        'hasPhotovoltaicPanels_encoded': 1,  
        'hasLivingRoom_encoded': 1,
        'lat': 51.0543,  # Ghent latitude
        'lon': 3.7174    # Ghent longitude
    }
    
    # Make predictions
    examples = [
        ("Small Apartment in Brussels (65m²)", small_apartment),
        ("Large Family House in Antwerp (250m²)", large_house),
        ("Modern Eco-Friendly House in Ghent (150m²)", eco_house)
    ]
    
    for name, house_data in examples:
        predicted_price = predictor.predict_new_house(house_data)
        print(f"{name}: €{predicted_price:,.2f}")
    

if __name__ == "__main__":
   main()