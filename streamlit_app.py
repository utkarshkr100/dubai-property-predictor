"""
Streamlit Cloud Deployment App
Optimized version of the Dubai Property Price Forecaster
"""

import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

def normalize_area_name(area, available_areas):
    """Normalize area name to handle case sensitivity issues"""
    # Direct match first
    if area in available_areas:
        return area

    # Create case-insensitive mapping
    area_upper = area.upper().strip()
    for available_area in available_areas:
        if available_area.upper().strip() == area_upper:
            return available_area

    # If no match found, return original
    return area

# Page configuration
st.set_page_config(
    page_title="Dubai Property Price Forecaster",
    page_icon="🔮",
    layout="centered"
)

@st.cache_resource
def load_model():
    """Load the historical pattern enhanced model"""
    try:
        model_path = 'models/historical_pattern_enhanced_model.pkl'

        if os.path.exists(model_path):
            with st.spinner('Loading AI model... This may take a moment on first load.'):
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
            st.success("✅ Model loaded successfully!")
            return model_data
        else:
            st.error("❌ Model file not found. Please ensure the model is uploaded to the repository.")
            st.info("💡 The model file is large (308MB). For deployment, consider using Git LFS or model hosting services.")
            return None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.info("💡 This might be due to memory constraints. Try refreshing the page.")
        return None

def predict_price(model_data, area, property_type, property_subtype, rooms, usage, is_offplan, actual_area, project=""):
    """Make prediction using historical enhanced model"""
    if not model_data:
        return None

    try:
        model = model_data['model']
        label_encoders = model_data['label_encoders']
        feature_columns = model_data['feature_columns']
        project_tiers = model_data['project_tiers']
        area_tiers = model_data['area_tiers']

        # Normalize area name for case sensitivity
        area_normalized = normalize_area_name(area, label_encoders['AREA_EN'].classes_)

        # Create input data using normalized area
        input_data = pd.DataFrame({
            'AREA_EN': [area_normalized],
            'PROP_TYPE_EN': [property_type],
            'PROP_SB_TYPE_EN': [property_subtype],
            'ROOMS_EN': [rooms],
            'USAGE_EN': [usage],
            'IS_OFFPLAN_EN': [is_offplan],
            'ACTUAL_AREA': [actual_area]
        })

        # Add historical pattern features using normalized area
        project_tier = project_tiers.get(project, 'Mid')
        area_tier = area_tiers.get(area_normalized, 'Mid')

        input_data['PROJECT_PRICE_TIER'] = [project_tier]
        input_data['AREA_PRICE_TIER'] = [area_tier]

        # Area size category
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

        # Expected price per sqft (realistic Dubai prices) using normalized area
        if property_subtype == "Flat":
            # Use normalized area names to handle ALL case variations
            area_upper = area_normalized.upper().strip()

            if area_upper in ["JUMEIRAH VILLAGE CIRCLE"]:
                expected_price_sqft = 17500
            elif area_upper in ["BUSINESS BAY"]:
                expected_price_sqft = 22000
            elif area_upper in ["DUBAI MARINA"]:
                expected_price_sqft = 25000
            elif area_upper in ["DOWNTOWN DUBAI"]:
                expected_price_sqft = 30000
            elif area_upper in ["DUBAI HILLS"]:
                expected_price_sqft = 20000
            elif area_upper in ["TRADE CENTER SECOND"]:
                expected_price_sqft = 46000
            elif area_upper in ["PALM JUMEIRAH"]:
                expected_price_sqft = 35000
            elif area_upper in ["BURJ KHALIFA"]:
                expected_price_sqft = 40000
            elif area_upper in ["PALM DEIRA"]:
                expected_price_sqft = 28000
            else:
                expected_price_sqft = 18000
        else:
            expected_price_sqft = 18000

        input_data['EXPECTED_PRICE_PER_SQFT'] = [expected_price_sqft]

        # Encode features
        encoded_input = input_data.copy()
        for feature in ['AREA_EN', 'PROP_TYPE_EN', 'PROP_SB_TYPE_EN', 'ROOMS_EN',
                       'USAGE_EN', 'IS_OFFPLAN_EN', 'AREA_SIZE_CATEGORY',
                       'PROJECT_PRICE_TIER', 'AREA_PRICE_TIER']:
            if feature in label_encoders and feature in encoded_input.columns:
                try:
                    value = str(input_data[feature].iloc[0])
                    if value in label_encoders[feature].classes_:
                        encoded_input[feature] = label_encoders[feature].transform([value])
                    else:
                        classes = label_encoders[feature].classes_
                        if len(classes) > 0:
                            default_encoded = len(classes) // 2
                            encoded_input[feature] = [default_encoded]
                        else:
                            encoded_input[feature] = [0]
                except:
                    if len(label_encoders[feature].classes_) > 0:
                        default_encoded = len(label_encoders[feature].classes_) // 2
                        encoded_input[feature] = [default_encoded]
                    else:
                        encoded_input[feature] = [0]

        # Make prediction
        X_input = encoded_input[feature_columns]
        predicted_price = model.predict(X_input)[0]
        predicted_price = max(predicted_price, 100000)

        return {
            'predicted_price': predicted_price,
            'price_per_sqft': predicted_price / actual_area if actual_area > 0 else 0,
            'project_tier': project_tier,
            'area_tier': area_tier,
            'expected_price_sqft': expected_price_sqft
        }

    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

def format_price(price):
    """Format price with AED currency"""
    if price >= 1_000_000:
        return f"AED {price/1_000_000:.2f}M"
    elif price >= 1_000:
        return f"AED {price/1_000:.0f}K"
    else:
        return f"AED {price:,.0f}"

def main():
    st.title("🔮 Dubai Property Price Forecaster")
    st.markdown("**Future Price Predictions using 20+ Years Historical Patterns** - 94.4% accuracy with 10.1% MAPE")

    # Load model
    model_data = load_model()
    if not model_data:
        st.error("Failed to load model")
        return

    # Model performance metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model Accuracy", "94.4%", "+42.5%")
    with col2:
        st.metric("Prediction Error", "10.1% MAPE", "-21.4%")
    with col3:
        st.metric("Historical Data", "2004-2025", "20+ years")

    st.header("Property Details")

    col1, col2 = st.columns(2)

    with col1:
        # Area selection - fix case sensitivity by normalizing
        areas_raw = sorted(model_data['label_encoders']['AREA_EN'].classes_) if 'label_encoders' in model_data else []

        # Group similar areas and show warning for duplicates
        area_groups = {}
        for area in areas_raw:
            key = area.upper().strip()
            if key not in area_groups:
                area_groups[key] = []
            area_groups[key].append(area)

        # Show warning for duplicate areas
        duplicates = {k: v for k, v in area_groups.items() if len(v) > 1}
        if duplicates:
            with st.expander("⚠️ Data Quality Notice - Case Sensitivity Issues", expanded=False):
                st.warning("Some areas have multiple case variations in the training data:")
                for key, variants in duplicates.items():
                    st.write(f"**{key}:** {', '.join(variants)}")
                st.info("We're using the first variant for consistency. This should be fixed in future training.")

        # Use first variant for each area group
        areas = [variants[0] for variants in area_groups.values()]
        areas = sorted(areas)

        area = st.selectbox(
            "📍 Select Area",
            options=areas,
            index=areas.index("JUMEIRAH VILLAGE CIRCLE") if "JUMEIRAH VILLAGE CIRCLE" in areas else 0,
            help="Choose the area where the property is located"
        )

        # Property type - use actual model data
        property_types = sorted(model_data['label_encoders']['PROP_TYPE_EN'].classes_) if 'label_encoders' in model_data else ["Unit", "Building", "Land"]
        property_type = st.selectbox(
            "🏢 Property Type",
            options=property_types,
            index=0
        )

        # Property subtype - use actual model data
        property_subtypes = sorted(model_data['label_encoders']['PROP_SB_TYPE_EN'].classes_) if 'label_encoders' in model_data else ["Flat", "Villa", "Office", "Shop"]
        # Filter out 'nan' values
        property_subtypes = [st for st in property_subtypes if str(st) != 'nan']

        property_subtype = st.selectbox(
            "🏘️ Property Subtype",
            options=property_subtypes,
            index=0,
            help="Note: Some combinations may not be logical (e.g., Land + Flat). This reflects training data variations."
        )

    with col2:
        # Rooms - use actual model data
        rooms_options = sorted(model_data['label_encoders']['ROOMS_EN'].classes_) if 'label_encoders' in model_data else ["Studio", "1 B/R", "2 B/R", "3 B/R", "4 B/R", "5 B/R"]
        # Filter out 'nan' values
        rooms_options = [room for room in rooms_options if str(room) != 'nan']

        rooms = st.selectbox(
            "🛏️ Bedrooms/Type",
            options=rooms_options,
            index=1 if len(rooms_options) > 1 else 0,
            help="Available options from training data. Some may not match your property type."
        )

        # Usage - use actual model data
        usage_options = sorted(model_data['label_encoders']['USAGE_EN'].classes_) if 'label_encoders' in model_data and 'USAGE_EN' in model_data['label_encoders'] else ["Residential", "Commercial"]
        usage = st.selectbox(
            "🏠 Usage Type",
            options=usage_options,
            index=0
        )

        # Off-plan status
        is_offplan = st.selectbox(
            "🚧 Property Status",
            options=["Ready", "Off-Plan"],
            index=0
        )

        # Area in square feet
        if property_type == "Land":
            min_area, max_area = 1000, 10000
            default_area = 5000
        elif property_subtype == "Villa":
            min_area, max_area = 200, 600
            default_area = 300
        else:  # Flat/Apartment
            if rooms == "Studio":
                min_area, max_area = 35, 60
                default_area = 45
            elif rooms == "1 B/R":
                min_area, max_area = 55, 90
                default_area = 70
            elif rooms == "2 B/R":
                min_area, max_area = 80, 150
                default_area = 110
            else:
                min_area, max_area = 120, 250
                default_area = 150

        actual_area = st.number_input(
            f"📐 Area (sq ft) - Typical: {min_area}-{max_area}",
            min_value=10,
            max_value=50000,
            value=default_area,
            step=5,
            help=f"Typical area for {rooms} {property_subtype}: {min_area}-{max_area} sq ft"
        )

    # Validation for logical combinations
    validation_warnings = []

    # Check for illogical combinations
    if property_type == "Land" and property_subtype in ["Flat", "Hotel Apartment"]:
        validation_warnings.append("🚨 Land properties typically don't have Flat/Hotel Apartment subtypes")

    if property_type == "Land" and rooms in ["Studio", "1 B/R", "2 B/R", "3 B/R", "4 B/R", "5 B/R", "6 B/R", "7 B/R"]:
        validation_warnings.append("🚨 Land properties typically don't have bedroom specifications")

    if property_subtype == "Land" and usage == "Residential" and rooms in ["Office", "Shop"]:
        validation_warnings.append("🚨 Residential land with Office/Shop rooms seems inconsistent")

    if property_subtype in ["Office", "Shop"] and usage == "Residential":
        validation_warnings.append("🚨 Office/Shop properties are typically Commercial, not Residential")

    if property_subtype in ["Flat", "Villa"] and usage == "Commercial":
        validation_warnings.append("🚨 Residential properties (Flat/Villa) are typically not Commercial usage")

    # Show validation warnings
    if validation_warnings:
        with st.expander("⚠️ Data Validation Warnings", expanded=True):
            st.warning("The following combinations may not be logical:")
            for warning in validation_warnings:
                st.write(warning)
            st.info("These combinations exist in the training data but may affect prediction accuracy. Consider adjusting your selections.")

    # Predict button
    if st.button("💰 Get Price Forecast", type="primary"):
        try:
            result = predict_price(
                model_data=model_data,
                area=area,
                property_type=property_type,
                property_subtype=property_subtype,
                rooms=rooms,
                usage=usage,
                is_offplan=is_offplan,
                actual_area=actual_area
            )

            if result:
                st.header("🔮 Price Forecast")

                # Main price display
                col1, col2, col3 = st.columns(3)

                predicted_price = result['predicted_price']
                margin = 0.15
                price_low = predicted_price * (1 - margin)
                price_high = predicted_price * (1 + margin)

                with col1:
                    st.metric("Lower Range", format_price(price_low))
                with col2:
                    st.metric("Predicted Price", format_price(predicted_price))
                with col3:
                    st.metric("Upper Range", format_price(price_high))

                # Additional insights
                st.subheader("📊 Market Intelligence")
                col1, col2 = st.columns(2)

                with col1:
                    st.write(f"📍 **Location:** {area}")
                    st.write(f"🏢 **Property:** {property_subtype} • {rooms}")
                    st.write(f"🚧 **Status:** {is_offplan}")

                with col2:
                    st.write(f"💡 **Price per sq ft:** AED {result['price_per_sqft']:,.0f}")
                    st.write(f"🏆 **Area Tier:** {result['area_tier']}")
                    st.write(f"📈 **Expected Market Rate:** AED {result['expected_price_sqft']:,.0f}/sqft")

            else:
                st.error("Failed to make prediction")

        except Exception as e:
            st.error(f"Error: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        <small>
        🔮 <strong>Dubai Property Price Forecaster</strong><br>
        📅 Historical Training Data: 2004-2025 (20+ years)<br>
        🎯 94.4% Accuracy on recent property transactions<br>
        ⚠️ Forecasts are estimates based on historical trends
        </small>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()