import streamlit as st
import pandas as pd
import pickle

def main():
    # Title of the web app
    st.title('Human Development Prediction in Tanzania')

    # Define the input fields
    year = st.number_input('Year', min_value=2020, max_value=2060)
    high_tech_exports = st.number_input('High-technology exports (manufactured exports)', min_value=0.0)
    export_value_index = st.number_input('Export value index', min_value=0.0)
    merch_exports_usd = st.number_input('Merchandise exports (current US$)', min_value=0.0)
    insurance_financial_services = st.number_input('Insurance and financial services (commercial service exports)', min_value=0.0)
    agricultural_raw_materials = st.number_input('Agricultural raw materials exports (merchandise exports)', min_value=0.0)
    computer_communications_services = st.number_input('Computer, communications and other services (commercial service imports)', min_value=0.0)
    merch_trade_gdp = st.number_input('Merchandise trade (GDP)', min_value=0.0)
    tourism_receipts = st.number_input('International tourism, receipts for travel items (current US$)', min_value=0.0)

    # Button to make predictions
    if st.button('Predict Human Development'):
        # Create a DataFrame for the input data
        input_data = pd.DataFrame({
            'Year': [year],
            'High-technology exports (manufactured exports)': [high_tech_exports],
            'Export value index ': [export_value_index],
            'Merchandise exports (current US$)': [merch_exports_usd],
            'Insurance and financial services (commercial service exports)': [insurance_financial_services],
            'Agricultural raw materials exports (merchandise exports)': [agricultural_raw_materials],
            'Computer, communications and other services (commercial service imports)': [computer_communications_services],
            'Merchandise trade (GDP)': [merch_trade_gdp],
            'International tourism, receipts for travel items (current US$)': [tourism_receipts]
        })

        # Debugging: Display the input data
        st.write("Input Data:")
        st.write(input_data)

        # Load the pre-trained linear regression model
        model_filename = 'human_model.pkl'
        try:
            with open(model_filename, 'rb') as file:
                model = pickle.load(file)
            st.write(f"Model loaded successfully: {model}")
        except Exception as e:
            st.write(f"Error loading model: {e}")
            return

        # Ensure the model has a predict method
        if not hasattr(model, 'predict'):
            st.write("The loaded model does not have a 'predict' method.")
            return

        # Ensure the model is an instance of the expected class (e.g., LinearRegression)
        from sklearn.linear_model import LinearRegression
        if not isinstance(model, LinearRegression):
            st.write("The loaded model is not an instance of 'LinearRegression'.")
            return

        # Make predictions using the loaded model
        try:
            prediction = model.predict(input_data)
            # Display the prediction
            st.write(f'Predicted Human Development: {prediction[0]}')
        except Exception as e:
            st.write(f"Error making prediction: {e}")

if __name__ == '__main__':
    main()

# To run this Streamlit app, use the following command in the terminal:
# streamlit run app.py
