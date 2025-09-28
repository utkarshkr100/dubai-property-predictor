import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
from datetime import datetime

class HistoricalEnhancedPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=25, min_samples_split=10)
        self.label_encoders = {}
        self.feature_columns = []
        self.is_trained = False
        self.project_price_tiers = {}
        self.area_price_tiers = {}
        self.project_maturity = {}
        self.area_diversity = {}
        self.price_stats = {}

    def preprocess_historical_data(self, csv_file_path, output_path=None):
        """Preprocess historical data to create enhanced features"""
        print("Preprocessing historical data...")

        chunk_size = 100000
        processed_chunks = []

        try:
            for chunk_num, chunk in enumerate(pd.read_csv(csv_file_path, chunksize=chunk_size)):
                print(f"Processing chunk {chunk_num + 1}...")

                # Filter for sales only
                sales_chunk = chunk[chunk['procedure_name_en'] == 'Sell'].copy()

                if len(sales_chunk) == 0:
                    continue

                # Map historical columns to current format
                sales_chunk = self.map_historical_columns(sales_chunk)

                # Clean data
                sales_chunk = sales_chunk.dropna(subset=['ACTUAL_AREA', 'TRANS_VALUE'])
                sales_chunk = sales_chunk[sales_chunk['ACTUAL_AREA'] > 0]
                sales_chunk = sales_chunk[sales_chunk['TRANS_VALUE'] > 0]

                # Remove extreme outliers
                q99 = sales_chunk['TRANS_VALUE'].quantile(0.99)
                q01 = sales_chunk['TRANS_VALUE'].quantile(0.01)
                sales_chunk = sales_chunk[
                    (sales_chunk['TRANS_VALUE'] >= q01) &
                    (sales_chunk['TRANS_VALUE'] <= q99)
                ]

                if len(sales_chunk) > 0:
                    processed_chunks.append(sales_chunk)

                # Process first 15 chunks for manageable training
                if chunk_num >= 14:
                    break

            # Combine all processed chunks
            if processed_chunks:
                historical_df = pd.concat(processed_chunks, ignore_index=True)
                print(f"Combined historical data: {len(historical_df)} transactions")

                # Create enhanced features
                historical_df = self.create_historical_features(historical_df)

                if output_path:
                    historical_df.to_csv(output_path, index=False)
                    print(f"Preprocessed data saved to: {output_path}")

                return historical_df

        except Exception as e:
            print(f"Error preprocessing historical data: {e}")
            return None

    def map_historical_columns(self, df):
        """Map historical column names to current format"""
        column_mapping = {
            'area_name_en': 'AREA_EN',
            'property_type_en': 'PROP_TYPE_EN',
            'property_sub_type_en': 'PROP_SB_TYPE_EN',
            'property_usage_en': 'USAGE_EN',
            'project_name_en': 'PROJECT_EN',
            'rooms_en': 'ROOMS_EN',
            'procedure_area': 'ACTUAL_AREA',
            'actual_worth': 'TRANS_VALUE',
            'instance_date': 'INSTANCE_DATE',
            'reg_type_en': 'IS_OFFPLAN_EN',
            'nearest_metro_en': 'NEAREST_METRO_EN',
            'nearest_mall_en': 'NEAREST_MALL_EN',
            'has_parking': 'PARKING'
        }

        # Rename columns
        mapped_df = df.rename(columns=column_mapping)

        # Add missing columns with defaults
        if 'PROP_SB_TYPE_EN' not in mapped_df.columns:
            mapped_df['PROP_SB_TYPE_EN'] = 'Unit'

        if 'IS_OFFPLAN_EN' not in mapped_df.columns:
            mapped_df['IS_OFFPLAN_EN'] = 'Ready'

        # Clean project names
        mapped_df['PROJECT_EN'] = mapped_df['PROJECT_EN'].fillna('')

        return mapped_df

    def create_historical_features(self, df):
        """Create enhanced features using historical data insights"""
        print("Creating enhanced features from historical data...")

        # Calculate price per sqft
        df['PRICE_PER_SQFT'] = df['TRANS_VALUE'] / df['ACTUAL_AREA']

        # 1. PROJECT PRICE TIERS (based on historical analysis)
        project_price_medians = df.groupby('PROJECT_EN')['PRICE_PER_SQFT'].median()

        # Use historical thresholds
        def assign_project_tier(price_per_sqft):
            if pd.isna(price_per_sqft):
                return 'Mid'
            elif price_per_sqft < 8165:
                return 'Budget'
            elif price_per_sqft < 9764:
                return 'Economy'
            elif price_per_sqft < 11904:
                return 'Mid'
            elif price_per_sqft < 16993:
                return 'Premium'
            else:
                return 'Luxury'

        df['PROJECT_PRICE_TIER'] = df['PROJECT_EN'].map(project_price_medians).apply(assign_project_tier)
        self.project_price_tiers = dict(zip(project_price_medians.index, project_price_medians.apply(assign_project_tier)))

        # 2. AREA PRICE TIERS (based on historical analysis)
        area_price_medians = df.groupby('AREA_EN')['PRICE_PER_SQFT'].median()

        def assign_area_tier(price_per_sqft):
            if pd.isna(price_per_sqft):
                return 'Mid'
            elif price_per_sqft < 4660:
                return 'Budget'
            elif price_per_sqft < 7373:
                return 'Economy'
            elif price_per_sqft < 9783:
                return 'Mid'
            elif price_per_sqft < 15862:
                return 'Premium'
            else:
                return 'Luxury'

        df['AREA_PRICE_TIER'] = df['AREA_EN'].map(area_price_medians).apply(assign_area_tier)
        self.area_price_tiers = dict(zip(area_price_medians.index, area_price_medians.apply(assign_area_tier)))

        # 3. PROJECT MATURITY (years active)
        df['INSTANCE_DATE'] = pd.to_datetime(df['INSTANCE_DATE'], dayfirst=True, errors='coerce')
        project_years = df.groupby('PROJECT_EN')['INSTANCE_DATE'].agg(['min', 'max'])
        project_years['years_active'] = (project_years['max'] - project_years['min']).dt.days / 365.25

        def assign_maturity(years):
            if pd.isna(years) or years < 1:
                return 'New'
            elif years < 3:
                return 'Young'
            elif years < 7:
                return 'Mature'
            else:
                return 'Established'

        project_maturity_map = project_years['years_active'].apply(assign_maturity).to_dict()
        df['PROJECT_MATURITY'] = df['PROJECT_EN'].map(project_maturity_map).fillna('New')
        self.project_maturity = project_maturity_map

        # 4. AREA DIVERSITY (number of unique projects)
        area_project_counts = df.groupby('AREA_EN')['PROJECT_EN'].nunique()

        def assign_diversity(count):
            if count < 5:
                return 'Low'
            elif count < 15:
                return 'Medium'
            elif count < 30:
                return 'High'
            else:
                return 'Very High'

        area_diversity_map = area_project_counts.apply(assign_diversity).to_dict()
        df['AREA_DIVERSITY'] = df['AREA_EN'].map(area_diversity_map).fillna('Low')
        self.area_diversity = area_diversity_map

        # 5. TEMPORAL FEATURES
        df['SALE_YEAR'] = df['INSTANCE_DATE'].dt.year
        df['SALE_QUARTER'] = df['INSTANCE_DATE'].dt.quarter

        # Market cycle (simplified)
        def assign_market_cycle(year):
            if pd.isna(year):
                return 'Current'
            elif year <= 2008:
                return 'Pre-Crisis'
            elif year <= 2012:
                return 'Crisis'
            elif year <= 2018:
                return 'Recovery'
            else:
                return 'Current'

        df['MARKET_CYCLE'] = df['SALE_YEAR'].apply(assign_market_cycle)

        # 6. BASIC FEATURES
        df['AREA_SIZE_CATEGORY'] = pd.cut(df['ACTUAL_AREA'],
                                         bins=[0, 50, 100, 200, 500, float('inf')],
                                         labels=['Tiny', 'Small', 'Medium', 'Large', 'XL'])

        df['SIZE_EFFICIENCY'] = np.log1p(df['ACTUAL_AREA'])

        # 7. EXPECTED PRICE based on historical patterns
        area_room_medians = df.groupby(['AREA_EN', 'ROOMS_EN'])['PRICE_PER_SQFT'].median()
        df['EXPECTED_PRICE_PER_SQFT'] = df.apply(
            lambda row: area_room_medians.get((row['AREA_EN'], row['ROOMS_EN']),
                                            df['PRICE_PER_SQFT'].median()), axis=1)

        # Store price statistics
        self.price_stats = {
            'median': df['TRANS_VALUE'].median(),
            'mean': df['TRANS_VALUE'].mean(),
            'price_per_sqft_median': df['PRICE_PER_SQFT'].median()
        }

        print(f"Enhanced features created. Dataset shape: {df.shape}")
        return df

    def encode_features(self, df, fit=True):
        """Encode categorical features"""
        categorical_features = ['AREA_EN', 'PROP_TYPE_EN', 'PROP_SB_TYPE_EN',
                               'ROOMS_EN', 'USAGE_EN', 'IS_OFFPLAN_EN',
                               'AREA_SIZE_CATEGORY', 'PROJECT_PRICE_TIER',
                               'AREA_PRICE_TIER', 'PROJECT_MATURITY',
                               'AREA_DIVERSITY', 'MARKET_CYCLE']

        encoded_df = df.copy()

        for feature in categorical_features:
            if feature in df.columns:
                if fit:
                    if feature not in self.label_encoders:
                        self.label_encoders[feature] = LabelEncoder()
                    encoded_df[feature] = self.label_encoders[feature].fit_transform(
                        df[feature].astype(str)
                    )
                else:
                    if feature in self.label_encoders:
                        # Handle unseen categories
                        unique_values = set(self.label_encoders[feature].classes_)
                        encoded_values = []
                        for val in df[feature].astype(str):
                            if val in unique_values:
                                encoded_values.append(
                                    self.label_encoders[feature].transform([val])[0]
                                )
                            else:
                                encoded_values.append(0)  # Default for unseen
                        encoded_df[feature] = encoded_values

        return encoded_df

    def train_on_historical_data(self, historical_csv_path):
        """Train model using historical data"""
        print("=== TRAINING ON HISTORICAL DATA ===")

        # Preprocess historical data
        historical_df = self.preprocess_historical_data(
            historical_csv_path,
            output_path="data/preprocessed_historical.csv"
        )

        if historical_df is None:
            print("Failed to preprocess historical data")
            return self

        # Encode features
        encoded_df = self.encode_features(historical_df, fit=True)

        # Select enhanced feature set
        self.feature_columns = ['AREA_EN', 'PROP_TYPE_EN', 'PROP_SB_TYPE_EN',
                               'ROOMS_EN', 'USAGE_EN', 'IS_OFFPLAN_EN', 'ACTUAL_AREA',
                               'AREA_SIZE_CATEGORY', 'PROJECT_PRICE_TIER', 'AREA_PRICE_TIER',
                               'PROJECT_MATURITY', 'AREA_DIVERSITY', 'SIZE_EFFICIENCY',
                               'EXPECTED_PRICE_PER_SQFT', 'SALE_QUARTER', 'MARKET_CYCLE']

        # Filter features that exist
        available_features = [f for f in self.feature_columns if f in encoded_df.columns]
        print(f"Using {len(available_features)} features: {available_features}")

        X = encoded_df[available_features]
        y = encoded_df['TRANS_VALUE']

        # Temporal split (use 80% of historical data for training)
        split_date = historical_df['INSTANCE_DATE'].quantile(0.8)
        train_mask = historical_df['INSTANCE_DATE'] <= split_date

        X_train = X[train_mask]
        X_test = X[~train_mask]
        y_train = y[train_mask]
        y_test = y[~train_mask]

        print(f"Training set: {len(X_train)} transactions")
        print(f"Validation set: {len(X_test)} transactions")

        # Train model
        print("Training historical enhanced model...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        self.feature_columns = available_features

        # Calculate metrics
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)

        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        print(f"Historical model training completed!")
        print(f"Training MAE: AED {train_mae:,.0f}")
        print(f"Validation MAE: AED {test_mae:,.0f}")
        print(f"Training R²: {train_r2:.3f}")
        print(f"Validation R²: {test_r2:.3f}")

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nFeature Importance:")
        for _, row in feature_importance.head(10).iterrows():
            print(f"{row['feature']}: {row['importance']:.3f}")

        return self

    def predict_price_range(self, area, property_type, property_subtype,
                           rooms, usage, is_offplan, actual_area, project=None):
        """Predict using historical insights"""
        if not self.is_trained:
            raise ValueError("Model must be trained first!")

        # Create input data
        input_data = pd.DataFrame({
            'AREA_EN': [area],
            'PROP_TYPE_EN': [property_type],
            'PROP_SB_TYPE_EN': [property_subtype],
            'ROOMS_EN': [rooms],
            'USAGE_EN': [usage],
            'IS_OFFPLAN_EN': [is_offplan],
            'ACTUAL_AREA': [actual_area]
        })

        # Add historical-based features
        # Project price tier
        project_tier = self.project_price_tiers.get(project, 'Mid')
        input_data['PROJECT_PRICE_TIER'] = [project_tier]

        # Area price tier
        area_tier = self.area_price_tiers.get(area, 'Mid')
        input_data['AREA_PRICE_TIER'] = [area_tier]

        # Project maturity
        project_maturity = self.project_maturity.get(project, 'New')
        input_data['PROJECT_MATURITY'] = [project_maturity]

        # Area diversity
        area_diversity = self.area_diversity.get(area, 'Medium')
        input_data['AREA_DIVERSITY'] = [area_diversity]

        # Market cycle (current)
        input_data['MARKET_CYCLE'] = ['Current']

        # Basic features
        if actual_area <= 50:
            area_size_cat = 'Tiny'
        elif actual_area <= 100:
            area_size_cat = 'Small'
        elif actual_area <= 200:
            area_size_cat = 'Medium'
        elif actual_area <= 500:
            area_size_cat = 'Large'
        else:
            area_size_cat = 'XL'

        input_data['AREA_SIZE_CATEGORY'] = [area_size_cat]
        input_data['SIZE_EFFICIENCY'] = [np.log1p(actual_area)]
        input_data['SALE_QUARTER'] = [datetime.now().quarter]
        input_data['EXPECTED_PRICE_PER_SQFT'] = [self.price_stats.get('price_per_sqft_median', 1500)]

        # Encode features
        encoded_input = self.encode_features(input_data, fit=False)

        # Select available features
        available_features = [f for f in self.feature_columns if f in encoded_input.columns]
        X_input = encoded_input[available_features]

        # Predict
        predicted_price = self.model.predict(X_input)[0]
        predicted_price = max(predicted_price, 100000)

        # Confidence interval
        margin = 0.15  # Historical model should be more confident
        price_low = predicted_price * (1 - margin)
        price_high = predicted_price * (1 + margin)

        return {
            'predicted_price': predicted_price,
            'price_range_low': price_low,
            'price_range_high': price_high,
            'price_per_sqft': predicted_price / actual_area if actual_area > 0 else 0,
            'project_tier': project_tier,
            'area_tier': area_tier,
            'project_maturity': project_maturity,
            'area_diversity': area_diversity
        }

    def save_model(self, filename):
        """Save the historical enhanced model"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'label_encoders': self.label_encoders,
                'feature_columns': self.feature_columns,
                'is_trained': self.is_trained,
                'project_price_tiers': self.project_price_tiers,
                'area_price_tiers': self.area_price_tiers,
                'project_maturity': self.project_maturity,
                'area_diversity': self.area_diversity,
                'price_stats': self.price_stats
            }, f)

    def load_model(self, filename):
        """Load the historical enhanced model"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.label_encoders = data['label_encoders']
            self.feature_columns = data['feature_columns']
            self.is_trained = data['is_trained']
            self.project_price_tiers = data.get('project_price_tiers', {})
            self.area_price_tiers = data.get('area_price_tiers', {})
            self.project_maturity = data.get('project_maturity', {})
            self.area_diversity = data.get('area_diversity', {})
            self.price_stats = data.get('price_stats', {})

if __name__ == "__main__":
    # Train on historical data
    predictor = HistoricalEnhancedPredictor()
    predictor.train_on_historical_data('old-data/Transactions.csv')
    predictor.save_model('models/historical_enhanced_model.pkl')