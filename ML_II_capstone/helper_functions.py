import random
from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# values for sensors are upper limit, lower limit and mean
# only exception is RF. this is just going to be a random value between the listed limits
sensor_limits = {
    'STEP1_press_avg': [20.0, 19.9, 20.0],
    'STEP2_press_avg': [15.0, 14.9, 15.0],
    'STEP3_press_avg': [50.0, 49.6, 49.8],
    'STEP4_press_avg': [30.0, 30.0, 30.0],
    'STEP5_press_avg': [100.0, 100.0, 100.0],
    'STEP6_press_avg': [59.8, 59.4, 59.6],
    'STEP7_press_avg': [70.0, 69.9, 70.0],
    'STEP8_press_avg': [40.0, 39.9, 40.0],
    'STEP1_Press_std': [0.65, 0.05, 0.13],
    'STEP2_Press_std': [0.59, 0.0, 0.07],
    'STEP3_Press_std': [1.58, 0.0, 0.69],
    'STEP4_Press_std': [0.34, 0.0, 0.1],
    'STEP5_Press_std': [0.53, 0.09, 0.21],
    'STEP6_Press_std': [1.85, 0.39, 0.98],
    'STEP7_Press_std': [0.33, 0.0, 0.14],
    'STEP8_Press_std': [0.46, 0.0, 0.13],
    'STEP1_Current_avg': [2.0, 0.0, 0.1],
    'STEP2_Current_avg': [2.0, 0.0, 0.0],
    'STEP3_Current_avg': [2.0, 0.0, 0.0],
    'STEP4_Current_avg': [2.0, 0.0, 0.2],
    'STEP5_Current_avg': [2.0, 0.0, 0.1],
    'STEP6_Current_avg': [2.0, 0.0, 0.0],
    'STEP7_Current_avg': [2.0, 0.0, 0.0],
    'STEP8_Current_avg': [2.0, 0.0, 0.0],
    'STEP1_Etch_Time': [40.0, 40.0, 40.0],
    'STEP2_Etch_Time': [18.5, 18.5, 18.5],
    'STEP3_Etch_Time': [7.0, 7.0, 7.0],
    'STEP4_Etch_Time': [60.0, 60.0, 60.0],
    'STEP5_Etch_Time': [60.0, 60.0, 60.0],
    'STEP6_Etch_Time': [7.0, 7.0, 7.0],
    'STEP7_Etch_Time': [12.9, 11.1, 11.9],
    'STEP8_Etch_Time': [12.9, 11.1, 11.9],
    'STEP1_Lower_Edge_Temp_Avg': [18.1, 18.0, 18.0],
    'STEP2_Lower_Edge_Temp_Avg': [18.0, 17.8, 17.9],
    'STEP3_Lower_Edge_Temp_Avg': [27.5, 25.5, 26.2],
    'STEP4_Lower_Edge_Temp_Avg': [27.1, 27.0, 27.0],
    'STEP5_Lower_Edge_Temp_Avg': [60.0, 58.7, 59.5],
    'STEP6_Lower_Edge_Temp_Avg': [61.3, 60.6, 60.9],
    'STEP7_Lower_Edge_Temp_Avg': [61.3, 60.8, 61.0],
    'STEP8_Lower_Edge_Temp_Avg': [61.3, 60.8, 61.0],
    'STEP1_Lower_Temp_Avg_E': [18.1, 18.0, 18.0],
    'STEP2_Lower_Temp_Avg_E': [18.0, 17.8, 17.9],
    'STEP3_Lower_Temp_Avg_E': [24.0, 23.4, 23.7],
    'STEP4_Lower_Temp_Avg_E': [23.0, 23.0, 23.0],
    'STEP5_Lower_Temp_Avg_E': [55.9, 54.8, 55.4],
    'STEP6_Lower_Temp_Avg_E': [57.2, 56.6, 56.9],
    'STEP7_Lower_Temp_Avg_E': [57.2, 56.9, 57.0],
    'STEP8_Lower_Temp_Avg_E': [57.2, 56.9, 57.0],
    'STEP1_Lower_VPP_Hardlimit_E': [0.0, 0.0, 0.0],
    'STEP2_Lower_VPP_Hardlimit_E': [0.0, 0.0, 0.0],
    'STEP3_Lower_VPP_Hardlimit_E': [549.6, 519.4, 535.0],
    'STEP4_Lower_VPP_Hardlimit_E': [0.0, 0.0, 0.0],
    'STEP5_Lower_VPP_Hardlimit_E': [1661.0, 1559.5, 1614.8],
    'STEP6_Lower_VPP_Hardlimit_E': [420.0, 395.2, 408.8],
    'STEP7_Lower_VPP_Hardlimit_E': [306.7, 282.2, 294.2],
    'STEP8_Lower_VPP_Hardlimit_E': [239.4, 215.2, 228.6],
    'STEP1_Lower_VPP_Std_E': [0.0, 0.0, 0.0],
    'STEP2_Lower_VPP_Std_E': [0.0, 0.0, 0.0],
    'STEP3_Lower_VPP_Std_E': [15.39, 3.42, 8.75],
    'STEP4_Lower_VPP_Std_E': [0.0, 0.0, 0.0],
    'STEP5_Lower_VPP_Std_E': [123.55, 103.68, 113.78],
    'STEP6_Lower_VPP_Std_E': [8.32, 1.64, 4.55],
    'STEP7_Lower_VPP_Std_E': [8.88, 2.37, 4.35],
    'STEP8_Lower_VPP_Std_E': [3.0, 0.0, 1.55],
    'STEP1_Upper_VPP_Hardlimit_E': [1895.8, 1749.8, 1817.7],
    'STEP2_Upper_VPP_Hardlimit_E': [1262.2, 1162.9, 1212.4],
    'STEP3_Upper_VPP_Hardlimit_E': [1118.6, 1020.5, 1067.6],
    'STEP4_Upper_VPP_Hardlimit_E': [1091.7, 1003.1, 1044.1],
    'STEP5_Upper_VPP_Hardlimit_E': [0.0, 0.0, 0.0],
    'STEP6_Upper_VPP_Hardlimit_E': [1062.4, 973.0, 1014.2],
    'STEP7_Upper_VPP_Hardlimit_E': [1061.2, 976.9, 1014.5],
    'STEP8_Upper_VPP_Hardlimit_E': [1606.8, 1457.3, 1531.0],
    'RF': [987.85716, 6.71667, 322.29572],
}

# upper limit, lower limit, target
cd_targets = {
    'CD1': [0.04232, 0.03585, 0.0385],
    'CD4': [0.03022, 0.02685, 0.0285],
    'CD5': [0.03065, 0.02712, 0.0285],
    'CD151': [0.04463, 0.03648, 0.04],
}

def generate_synthetic_data(sensor_limits: dict, num_wafers: int) -> pd.DataFrame:
    """
    Generate synthetic sensor data for a given number of wafers.

    Parameters:
        sensor_limits (dict): Dictionary where keys are sensor names and values are lists of [upper, lower, target].
        num_wafers (int): Number of wafers to generate data for.

    Returns:
        pd.DataFrame: DataFrame containing synthetic sensor data for each wafer.
    """
    data = []
    
    # get the list of sensor names
    sensor_names = list(sensor_limits.keys())
    sensor_names.sort()  # Sort for consistency
    sensors_except_rf = [s for s in sensor_names if s != 'RF']

    # extract RF limits
    upper_rf, lower_rf, target_rf = sensor_limits['RF']
    rf_total_range = upper_rf - lower_rf
    
    # calculate the lowest and highest 10% of the range. values in these ranges will trigger sensor adjustments
    rf_low_threshold = lower_rf + 0.1 * rf_total_range
    rf_high_threshold = upper_rf - 0.1 * rf_total_range

    for wafer in range(num_wafers):
        wafer_data = {}
        
        # generate 'RF' value
        rf_value = np.random.uniform(low=lower_rf, high=upper_rf)
        wafer_data['RF'] = rf_value
        
        # check if RF is within the lowest or highest 10%
        adjust_sensors = False
        if rf_value <= rf_low_threshold or rf_value >= rf_high_threshold:
            adjust_sensors = True
        
        # generate other sensor values
        for sensor in sensors_except_rf:
            upper, lower, target = sensor_limits[sensor]
            
            # handle fixed value sensors
            if upper == lower:
                # 95% of the time, the value will be the same as the limit, otherwise add random noise
                if random.random() > .05:
                    value = upper
                else:
                    # Add slight noise to fixed values
                    value = target + np.random.normal(loc=0, scale=target * 0.001)
            else:
                # Use normal distribution centered at target
                std_dev = (upper - lower) / 6  # 99.7% data within limits
                value = np.random.normal(loc=target, scale=std_dev)
                value = max(min(value, upper), lower)
            
            wafer_data[sensor] = value
        
        # adjust two random sensors if RF is in extreme ranges
        if adjust_sensors:
            # exclude sensors with fixed values
            adjustable_sensors = [s for s in sensors_except_rf if sensor_limits[s][0] != sensor_limits[s][1]]
            
            # ensure there are enough sensors to adjust
            if len(adjustable_sensors) >= 2:
                sensors_to_adjust = np.random.choice(adjustable_sensors, size=2, replace=False)
            else:
                sensors_to_adjust = adjustable_sensors  # adjust what is available

            for sensor in sensors_to_adjust:
                upper, lower, _ = sensor_limits[sensor]
                # 33% chance of adjusting values in lowest and upper most RF hours to have random sensors out of spec
                if np.random.rand() < 0.66:
                    # below lower limit by 5% of the range
                    deviation = (upper - lower) * 0.05
                    value = lower - deviation
                else:
                    # above upper limit by 5% of the range
                    deviation = (upper - lower) * 0.05
                    value = upper + deviation
                wafer_data[sensor] = value

        data.append(wafer_data)
    
    df = pd.DataFrame(data)
    return df


def add_noise_to_samples(
    df: pd.DataFrame,
    n_samples: int,
    noise_level: float = 0.05,
    sensor_limits: dict = None,
    sensors_to_modify: list = None
):
    """
    Randomly samples n_samples rows from df and adds noise to their values.

    Parameters:
        df (pd.DataFrame): The DataFrame containing synthetic data.
        n_samples (int): The number of rows to randomly sample and modify.
        noise_level (float): The percentage of noise to add (default is 5%).
        sensor_limits (dict): Dictionary of sensor limits to ensure values stay within bounds.
        sensors_to_modify (List[str]): List of sensor names to add noise to. If None, all sensors are modified.

    Returns:
        pd.DataFrame: The modified DataFrame with noise added to selected rows.
    """
    # Make a copy of the DataFrame to avoid modifying the original data
    df_noisy = df.copy()
    
    # Randomly select n_samples indices
    selected_indices = df_noisy.sample(n=n_samples).index
    
    # Iterate over selected indices and add noise
    for idx in selected_indices:
        # Original row values
        original_values = df_noisy.loc[idx]

        # If sensors_to_modify is provided, modify only those sensors
        if sensors_to_modify is None:
            sensors = df_noisy.columns
        else:
            sensors = sensors_to_modify
        
        noisy_values = original_values.copy()
        
        # add noise to the selected sensors
        for sensor in sensors:
            upper, lower, _ = sensor_limits.get(sensor, (None, None, None)) # get limits if they were provided
            value = original_values[sensor]

            if upper is not None and upper == lower:
                continue # skip sensors with fixed values

            noise = np.random.normal(loc=0, scale=noise_level * abs(value))
            noisy_value = value + noise
            
            # if limits are passed, clip values to stay within limits
            if sensor_limits and sensor in sensor_limits:
                upper, lower, _ = sensor_limits[sensor]
                max_limit = max(upper, lower)
                min_limit = min(upper, lower)
                noisy_value = np.clip(noisy_value, min_limit, max_limit)
            
            noisy_values[sensor] = noisy_value
        
        # Update the DataFrame
        df_noisy.loc[idx] = noisy_values
    
    return df_noisy


def generate_out_of_spec_samples(df: pd.DataFrame, sensor_limits: dict, num_out_of_spec_samples: int) -> pd.DataFrame:
    """
    Generates new out-of-spec samples and adds them to the original DataFrame.

    Parameters:
        df (pd.DataFrame): Original DataFrame with synthetic data.
        sensor_limits (dict): Dictionary with sensor limits.
        num_out_of_spec_samples (int): Number of out-of-spec samples to generate.

    Returns:
        pd.DataFrame: DataFrame with new out-of-spec samples appended.
    """
    data = df.copy()
    sensor_names = list(sensor_limits.keys())
    sensors_except_rf = [s for s in sensor_names if s != 'RF']

    # extract RF limits
    upper_rf, lower_rf, _ = sensor_limits['RF']
    rf_total_range = upper_rf - lower_rf
    rf_low_threshold = lower_rf + 0.1 * rf_total_range
    rf_high_threshold = upper_rf - 0.1 * rf_total_range

    out_of_spec_samples = []

    for _ in range(num_out_of_spec_samples):
        wafer_data = {}
        
        # generate RF value
        # randomly decide if RF should be in the extreme ranges
        if np.random.rand() < 0.5:
            # set R' to be very low (lowest 10%)
            rf_value = np.random.uniform(low=lower_rf, high=rf_low_threshold)
        else:
            # set R' to be very high (highest 10%)
            rf_value = np.random.uniform(low=rf_high_threshold, high=upper_rf)
        wafer_data['RF'] = rf_value
        
        # generate other sensor values
        for sensor in sensors_except_rf:
            upper, lower, target = sensor_limits[sensor]
            
            # handle fixed value sensors
            if upper == lower:
                value = upper
            else:
                std_dev = (upper - lower) / 6  # 99.7% within limits
                value = np.random.normal(loc=target, scale=std_dev)
                value = max(min(value, upper), lower)
            
            wafer_data[sensor] = value
        
        # adjust one or more sensors to be out-of-spec
        # randomly choose how many sensors to adjust (at least one)
        num_sensors_to_adjust = np.random.randint(1, min(4, len(sensors_except_rf)) + 1)

        # select sensors to adjust
        adjustable_sensors = [s for s in sensors_except_rf if sensor_limits[s][0] != sensor_limits[s][1]]
        sensors_to_adjust = np.random.choice(adjustable_sensors, size=num_sensors_to_adjust, replace=False)
        
        for sensor in sensors_to_adjust:
            upper, lower, _ = sensor_limits[sensor]
            # decide randomly to exceed upper or lower limit
            if np.random.rand() < 0.5:
                # below lower limit by 5% of the range
                deviation = (upper - lower) * 0.25
                value = lower - deviation
            else:
                # above upper limit by 5% of the range
                deviation = (upper - lower) * 0.25
                value = upper + deviation
            wafer_data[sensor] = value

        out_of_spec_samples.append(wafer_data)
    
    # convert to df
    out_of_spec_df = pd.DataFrame(out_of_spec_samples)
    
    # append out-of-spec samples to the original df
    combined_df = pd.concat([data, out_of_spec_df], ignore_index=True)
    
    return combined_df

def add_cd_values(df: pd.DataFrame, sensor_limits: dict, cd_targets: dict) -> pd.DataFrame:
    """
    Calculates CD values based on sensor readings and adds them to the DataFrame.
    Allows CDs to go out of spec and adds extra weight if sensors are significantly beyond their limits.

    Parameters:
        df (pd.DataFrame): DataFrame containing synthetic sensor data.
        sensor_limits (dict): Dictionary of sensor limits {sensor: [upper, lower, target]}.
        cd_targets (dict): Dictionary of CD targets {CD: [upper, lower, target]}.

    Returns:
        pd.DataFrame: DataFrame with added CD columns.
    """
    df_with_cd = df.copy()
    sensors = [sensor for sensor in df.columns if sensor not in ['RF', 'InSpec']]

    deviations = pd.DataFrame(index=df_with_cd.index)
    for sensor in sensors:
        upper, lower, target = sensor_limits[sensor]
        sensor_range = upper - lower
        value = df_with_cd[sensor]
        deviation = value - target
        if sensor_range != 0:
            relative_deviation = deviation / sensor_range
        else:
            relative_deviation = 0.0  # for fixed sensors
        deviations[sensor] = relative_deviation

    significant_deviation = pd.DataFrame(index=df_with_cd.index)
    for sensor in sensors:
        upper, lower, _ = sensor_limits[sensor]
        sensor_range = upper - lower
        value = df_with_cd[sensor]
        significant_threshold = 0.05 * sensor_range  # 5% of the sensor range

        if sensor_range == 0:
            significant_deviation[sensor] = False
        else:
            is_beyond_upper = value > (upper + significant_threshold)
            is_beyond_lower = value < (lower - significant_threshold)
            significant_deviation[sensor] = is_beyond_upper | is_beyond_lower

    for cd in cd_targets.keys():
        upper_cd, lower_cd, target_cd = cd_targets[cd]
        df_with_cd[cd] = target_cd

        base_weight = 1.0
        base_weights = pd.Series(base_weight, index=sensors)
        total_base_weight = base_weights.sum()
        normalized_base_weights = base_weights / total_base_weight

        weights = pd.DataFrame(
            np.tile(normalized_base_weights.values, (df_with_cd.shape[0], 1)),
            index=df_with_cd.index,
            columns=sensors
        )

        extra_weight_factor = 10.0
        for sensor in sensors:
            weights.loc[significant_deviation[sensor], sensor] *= extra_weight_factor

        weighted_deviation = (weights * deviations).sum(axis=1)

        cd_range = upper_cd - lower_cd
        adjustment_factor = 2.0  # New adjustment factor
        df_with_cd[cd] += weighted_deviation * cd_range * adjustment_factor

    return df_with_cd

def add_cd_spec_labels(df: pd.DataFrame, cd_targets: dict) -> pd.DataFrame:
    for cd in cd_targets.keys():
        upper_cd, lower_cd, _ = cd_targets[cd]
        df[f'{cd}_InSpec'] = df[cd].between(lower_cd, upper_cd, inclusive='both')
    return df

def plot_helper(y_true, y_pred, cd_columns, cd_targets, plot_type='scatter'):
    """
    Plots true CD values and predicted CD values on the same plot for each CD.
    Different markers are used for true and predicted values.
    Also draws horizontal lines for upper limit, lower limit, and target for each CD.
    
    Parameters:
        y_true (numpy.ndarray or pandas.DataFrame): True CD values. Shape: (n_samples, n_cds)
        y_pred (numpy.ndarray or pandas.DataFrame): Predicted CD values. Shape: (n_samples, n_cds)
        cd_columns (list): List of CD column names.
        cd_targets (dict): Dictionary with CD targets {CD: [upper, lower, target]}.
        plot_type (str): Type of plot ('scatter', 'line', 'error'). Default is 'scatter'.
    
    Returns:
        None. Displays the plots.
    """
    # Ensure y_true and y_pred are numpy arrays
    if isinstance(y_true, pd.DataFrame) or isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.DataFrame) or isinstance(y_pred, pd.Series):
        y_pred = y_pred.values

    n_cds = len(cd_columns)
    n_samples = y_true.shape[0]
    sample_indices = np.arange(n_samples)

    for i, cd in enumerate(cd_columns):
        plt.figure(figsize=(10, 6))
        if plot_type == 'scatter':
            # Plot true values
            plt.scatter(sample_indices, y_true[:, i], color='blue', marker='o', label='True Values', alpha=0.6)
            # Plot predicted values
            plt.scatter(sample_indices, y_pred[:, i], color='red', marker='x', label='Predicted Values', alpha=0.6)
        elif plot_type == 'line':
            # Plot true values
            plt.plot(sample_indices, y_true[:, i], color='blue', label='True Values')
            # Plot predicted values
            plt.plot(sample_indices, y_pred[:, i], color='red', label='Predicted Values')
        elif plot_type == 'error':
            # Plot prediction errors
            errors = y_true[:, i] - y_pred[:, i]
            plt.scatter(sample_indices, errors, color='purple', marker='.', label='Prediction Error', alpha=0.6)
            plt.ylabel(f'Error in {cd} Value')
        else:
            raise ValueError("plot_type must be 'scatter', 'line', or 'error'.")

        plt.xlabel('Sample Index')
        if plot_type != 'error':
            plt.ylabel(f'{cd} Value')
            plt.title(f'True vs. Predicted Values for {cd}')
        else:
            plt.title(f'Prediction Error for {cd}')

        # Extract limits and target for the current CD
        upper_cd, lower_cd, target_cd = cd_targets[cd]

        # Draw horizontal lines for upper limit, lower limit, and target
        plt.axhline(y=upper_cd, color='green', linestyle='--', linewidth=1.5, label='Upper Limit')
        plt.axhline(y=lower_cd, color='orange', linestyle='--', linewidth=1.5, label='Lower Limit')
        plt.axhline(y=target_cd, color='black', linestyle='-', linewidth=1.5, label='Target')

        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def st_plot_helper(y_true, y_pred, cd_columns, cd_targets, plot_type='scatter'):
    """
    Plots true CD values and predicted CD values on the same plot for each CD.
    Different markers are used for true and predicted values.
    Also draws horizontal lines for upper limit, lower limit, and target for each CD.
    
    Parameters:
        y_true (numpy.ndarray or pandas.DataFrame): True CD values. Shape: (n_samples, n_cds)
        y_pred (numpy.ndarray or pandas.DataFrame): Predicted CD values. Shape: (n_samples, n_cds)
        cd_columns (list): List of CD column names.
        cd_targets (dict): Dictionary with CD targets {CD: [upper, lower, target]}.
        plot_type (str): Type of plot ('scatter', 'line', 'error'). Default is 'scatter'.
    
    Returns:
        list: A list of matplotlib.figure.Figure objects for each CD.
    """
    # Ensure y_true and y_pred are numpy arrays
    if isinstance(y_true, pd.DataFrame) or isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.DataFrame) or isinstance(y_pred, pd.Series):
        y_pred = y_pred.values

    n_cds = len(cd_columns)
    n_samples = y_true.shape[0]
    sample_indices = np.arange(n_samples)

    figures = []  # List to store each CD's figure

    for i, cd in enumerate(cd_columns):
        # Create a new figure for each CD
        fig, ax = plt.subplots(figsize=(10, 6))

        if plot_type == 'scatter':
            # Plot true values
            ax.scatter(sample_indices, y_true[:, i], color='blue', marker='o', label='True Values', alpha=0.6)
            # Plot predicted values
            ax.scatter(sample_indices, y_pred[:, i], color='red', marker='x', label='Predicted Values', alpha=0.6)
        elif plot_type == 'line':
            # Plot true values
            ax.plot(sample_indices, y_true[:, i], color='blue', label='True Values')
            # Plot predicted values
            ax.plot(sample_indices, y_pred[:, i], color='red', label='Predicted Values')
        elif plot_type == 'error':
            # Plot prediction errors
            errors = y_true[:, i] - y_pred[:, i]
            ax.scatter(sample_indices, errors, color='purple', marker='.', label='Prediction Error', alpha=0.6)
            ax.set_ylabel(f'Error in {cd} Value')
        else:
            raise ValueError("plot_type must be 'scatter', 'line', or 'error'.")

        ax.set_xlabel('Sample Index')
        if plot_type != 'error':
            ax.set_ylabel(f'{cd} Value')
            ax.set_title(f'True vs. Predicted Values for {cd}')
        else:
            ax.set_title(f'Prediction Error for {cd}')

        # Extract limits and target for the current CD
        upper_cd, lower_cd, target_cd = cd_targets[cd]

        # Draw horizontal lines for upper limit, lower limit, and target
        ax.axhline(y=upper_cd, color='green', linestyle='--', linewidth=1.5, label='Upper Limit')
        ax.axhline(y=lower_cd, color='orange', linestyle='--', linewidth=1.5, label='Lower Limit')
        ax.axhline(y=target_cd, color='black', linestyle='-', linewidth=1.5, label='Target')

        ax.legend()
        ax.grid(True)
        fig.tight_layout()

        # Append the figure to the list
        figures.append(fig)

    return figures
