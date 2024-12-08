import streamlit as st
import tensorflow as tf
import numpy as np
from custom_loss import WeightedMSELoss
from helper_functions import *
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Load scaler pickle files with joblib
scaler = joblib.load('./ML_II_capstone/scalers/scaler.pkl')
target_scaler = joblib.load('./ML_II_capstone/scalers/target_scaler.pkl')

# create instance of custom loss function. needed to load the model
custom_loss_fn = WeightedMSELoss(cd_targets=cd_targets, higher_weight=50.0, base_weight=1.0)

# Load pre-trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('./models/trained_model.h5')
    return model

def main():
    st.title("Machine Learning Prediction App")
    st.write("This app uses a pre-trained machine learning model to make predictions on randomly generated data.")

    # Load the model
    model = load_model()

    # User input for data generation
    n_samples = st.slider("Number of data points to generate:", 10, 500, 100)
    noise = st.slider("Noise level in data:", 0.0, 1.0, 0.1)

    # Button to generate data
    if st.button("Generate Random Data"):
        data = generate_synthetic_data(sensor_limits, n_samples)

        # add out-of-spec samples
        num_out_of_spec_samples = int(n_samples * 0.1)
        data = generate_out_of_spec_samples(data, sensor_limits, num_out_of_spec_samples)

        # add CD values
        data = add_cd_values(data, sensor_limits, cd_targets)

        # add CD InSpec labels
        data = add_cd_spec_labels(data, cd_targets)

        if noise > 0:
            data = add_noise_to_samples(df=data, noise_level=noise, sensor_limits=sensor_limits, n_samples=n_samples)

        # Store generated data in session state
        st.session_state['data'] = data
        st.session_state['predictions'] = None  # Clear predictions when new data is generated
        st.dataframe(data, use_container_width=True)

    # Section to confirm making predictions
    if 'data' in st.session_state:
        data = st.session_state['data']

        # drop the VPP columns
        cols = data.columns.tolist()
        cols = [col for col in cols if 'VPP' not in col]
        data = data[cols]

        # Preprocess data
        sensor_columns = [col for col in data.columns if 'CD' not in col]
        cd_columns = ['CD1', 'CD4', 'CD5', 'CD151']

        # split data and targets
        X = data[sensor_columns]
        y = data[cd_columns]

        # Scale data
        X_scaled = scaler.fit_transform(X)

        # Scale target variables
        y_scaled = target_scaler.fit_transform(y)

        pred_button = st.button("Make Predictions on Generated Data")

        if pred_button or st.session_state.get('predictions') is not None:
            # Predict on test data if predictions are not already stored
            if st.session_state.get('predictions') is None:
                y_pred_scaled = model.predict(X_scaled)

                # Inverse transform predictions and actual values
                y_pred = target_scaler.inverse_transform(y_pred_scaled)
                y_test_actual = target_scaler.inverse_transform(y_scaled)

                # Store predictions in session state
                st.session_state['predictions'] = {
                    'y_pred': y_pred,
                    'y_test_actual': y_test_actual
                }
            else:
                y_pred = st.session_state['predictions']['y_pred']
                y_test_actual = st.session_state['predictions']['y_test_actual']

            st.write("Predictions made successfully!")

            # dict to store metrics
            mse = {}
            rmse = {}
            mape = {}

            for i, cd in enumerate(cd_columns):
                mse[cd] = mean_squared_error(y_test_actual[:, i], y_pred[:, i])
                rmse[cd] = np.sqrt(mse[cd])
                mape[cd] = mean_absolute_percentage_error(y_test_actual[:, i], y_pred[:, i])
                st.write(f'{cd} - MSE: {mse[cd]:.6f}, RMSE: {rmse[cd]:.6f}, MAPE: {mape[cd]:.4f}')

            from helper_functions import st_plot_helper
            
            fig_list = st_plot_helper(y_true=y_test_actual, y_pred=y_pred, cd_columns=cd_columns, cd_targets=cd_targets)
            for fig in fig_list:
                st.pyplot(fig)

            # Plotting predictions using Plotly
            # fig = go.Figure()
            # for i, cd in enumerate(cd_columns):
            #     fig.add_trace(go.Scatter(
            #         x=np.arange(len(y_pred[:, i])),
            #         y=y_pred[:, i],
            #         mode='markers',
            #         name=f'Predicted {cd}',
            #         marker=dict(
            #             symbol='circle',
            #             color='green',
            #             size=8
            #         )
            #     ))

            #     # Check if the user wants to see the true values
            #     if st.checkbox(f"Show True Values for {cd}", key=f'show_true_{cd}'):
            #         fig.add_trace(go.Scatter(
            #             x=np.arange(len(y_test_actual[:, i])),
            #             y=y_test_actual[:, i],
            #             mode='markers',
            #             name=f'True {cd}',
            #             marker=dict(
            #                 symbol='cross',
            #                 color='red',
            #                 size=8
            #             )
            #         ))

            # fig.update_layout(title="Predicted vs True Values", xaxis_title="Data Point Index", yaxis_title="Value")
            # st.plotly_chart(fig)

if __name__ == "__main__":
    main()
