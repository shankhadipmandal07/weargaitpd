import os
import glob
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from collections import defaultdict

# --- Configuration ---
DATA_DIR = "./download_data"
CTRL_DEMO_FILE = "CONTROLS - Demographic+Clinical - datasetV1.csv"
PD_DEMO_FILE = "PD - Demographic+Clinical - datasetV1.csv"

def load_demographics():
    """
    Loads and merges the demographic and clinical data for both Controls and PD patients.
    Returns a dictionary keyed by Subject ID.
    """
    try:
        # The actual headers are on the second row (index 1) in these specific CSVs
        ctrl_df = pd.read_csv(CTRL_DEMO_FILE, header=1)
        pd_df = pd.read_csv(PD_DEMO_FILE, header=1)
        
        # Standardize the 'Age' column for PD patients
        if 'Age (years)' in pd_df.columns:
            pd_df.rename(columns={'Age (years)': 'Age'}, inplace=True)
            
        cols_to_keep = ['Subject ID', 'Age', 'Gender', 'Sex', 'Race', 'Height (in)', 'Weight (kg)', 'Modified Hoehn & Yahr Score']
        
        # Keep only existing columns to avoid errors
        ctrl_cols = [c for c in cols_to_keep if c in ctrl_df.columns]
        pd_cols = [c for c in cols_to_keep if c in pd_df.columns]
        
        ctrl_df = ctrl_df[ctrl_df['Subject ID'].notna()][ctrl_cols]
        pd_df = pd_df[pd_df['Subject ID'].notna()][pd_cols]
        
        combined_df = pd.concat([ctrl_df, pd_df], ignore_index=True)
        # Rename the H&Y score to precisely match your requested name
        combined_df.rename(columns={'Modified Hoehn & Yahr Score': 'Hoehn and Yahr scale PD severity'}, inplace=True)
        
        combined_df.set_index('Subject ID', inplace=True)
        return combined_df.to_dict(orient='index')
    except Exception as e:
        print(f"Warning: Could not load demographic files properly. Error: {e}")
        return {}

def calculate_global_cop(df):
    """
    Calculates the force-weighted Global Center of Pressure (CoP).
    """
    f_l = pd.to_numeric(df['LTotalForce'], errors='coerce').fillna(0).values
    f_r = pd.to_numeric(df['RTotalForce'], errors='coerce').fillna(0).values
    
    cop_x_l = pd.to_numeric(df['LCoP_X'], errors='coerce').fillna(0).values
    cop_x_r = pd.to_numeric(df['RCoP_X'], errors='coerce').fillna(0).values
    
    cop_y_l = pd.to_numeric(df['LCoP_Y'], errors='coerce').fillna(0).values
    cop_y_r = pd.to_numeric(df['RCoP_Y'], errors='coerce').fillna(0).values
    
    time_raw = df['Time'].astype(str).str.replace(' sec', '', regex=False)
    time_arr = pd.to_numeric(time_raw, errors='coerce').values
    
    f_total = f_l + f_r
    valid_mask = (f_total > 0) & (~np.isnan(time_arr))
    
    f_total, f_l, f_r = f_total[valid_mask], f_l[valid_mask], f_r[valid_mask]
    cop_x_l, cop_x_r = cop_x_l[valid_mask], cop_x_r[valid_mask]
    cop_y_l, cop_y_r = cop_y_l[valid_mask], cop_y_r[valid_mask]
    valid_times = time_arr[valid_mask]
    
    cop_x_global = ((cop_x_l * f_l) + (cop_x_r * f_r)) / f_total
    cop_y_global = ((cop_y_l * f_l) + (cop_y_r * f_r)) / f_total
            
    return cop_x_global, cop_y_global, valid_times

def get_velocity_peaks(vel):
    """
    Replicates the 'velocity_peaks' function from the Quijoux dynamic.py script.
    Extracts zero crossings and positive/negative peak magnitudes.
    """
    current_peak = 0
    past_value = 0
    zero_crossings = 0
    pos_peaks, neg_peaks = [], []
    
    valid_vel = vel[vel != 0]
    if len(valid_vel) == 0:
        return 0, 0, 0, 0
        
    current_side = np.sign(valid_vel[0])

    for i, value in enumerate(vel):
        is_crossing = (value * past_value <= 0) and (i != 0) and (value != 0) and (np.sign(value) != current_side)

        if is_crossing:
            if zero_crossings > 0:
                if value < 0:
                    pos_peaks.append(abs(current_peak))
                elif value > 0:
                    neg_peaks.append(abs(current_peak))
            
            zero_crossings += 1
            current_side = np.sign(value)
            current_peak = 0

        if abs(value) > abs(current_peak):
            current_peak = value
            
        if value != 0:
            past_value = value
            
    mean_pos = np.mean(pos_peaks) if len(pos_peaks) > 0 else 0
    mean_neg = np.mean(neg_peaks) if len(neg_peaks) > 0 else 0
    all_peaks = pos_peaks + neg_peaks
    mean_all = np.mean(all_peaks) if len(all_peaks) > 0 else 0
    
    return zero_crossings, mean_pos, mean_neg, mean_all

def extract_features(cop_x, cop_y, time_arr):
    """
    Extracts all 21 dynamic Center of Pressure features based on the 
    Quijoux et al. implementation.
    """
    feature_names = [
        'mean_velocity_ML', 'mean_velocity_AP', 'mean_velocity_ML_AND_AP',
        'sway_area_per_second_ML_AND_AP', 'phase_plane_parameter_ML', 'phase_plane_parameter_AP',
        'LFS_ML_AND_AP', 'fractal_dimension_ML_AND_AP',
        'zero_crossing_SPD_ML', 'peak_velocity_pos_SPD_ML', 'peak_velocity_neg_SPD_ML', 'peak_velocity_all_SPD_ML',
        'zero_crossing_SPD_AP', 'peak_velocity_pos_SPD_AP', 'peak_velocity_neg_SPD_AP', 'peak_velocity_all_SPD_AP',
        'mean_peak_Sway_Density', 'mean_distance_peak_Sway_Density',
        'mean_frequency_ML', 'mean_frequency_AP', 'mean_frequency_ML_AND_AP'
    ]
    
    if len(cop_x) < 50:
        return {k: np.nan for k in feature_names}

    N = len(cop_x)
    duration = time_arr[-1] - time_arr[0]
    freq = (N - 1) / duration if duration > 0 else 100

    # 1. Align axes using PCA to detrend the walking path
    coords = np.column_stack((cop_x, cop_y))
    pca = PCA(n_components=2)
    aligned_coords = pca.fit_transform(coords)
    
    cop_ap = aligned_coords[:, 0]
    cop_ml = aligned_coords[:, 1]
    
    # Mean-centered signals
    ml_c = cop_ml - np.mean(cop_ml)
    ap_c = cop_ap - np.mean(cop_ap)
    
    # Velocities
    vel_ml = np.diff(cop_ml) * freq
    vel_ap = np.diff(cop_ap) * freq

    # --- 1. Sway Lengths & Mean Velocities ---
    sway_length_ml = np.sum(np.abs(np.diff(cop_ml)))
    sway_length_ap = np.sum(np.abs(np.diff(cop_ap)))
    sway_length_mlap = np.sum(np.sqrt(np.diff(cop_ml)**2 + np.diff(cop_ap)**2))

    mean_vel_ml = sway_length_ml * (freq / N)
    mean_vel_ap = sway_length_ap * (freq / N)
    mean_vel_mlap = sway_length_mlap * (freq / N)

    # --- 2. Sway Area per Second ---
    triangles = np.abs(ml_c[1:] * ap_c[:-1] - ml_c[:-1] * ap_c[1:])
    sway_area_per_sec = np.sum(triangles) / (2 * duration)

    # --- 3. Phase Plane Parameters ---
    ppp_ml = np.sqrt(np.var(cop_ml) + np.var(vel_ml))
    ppp_ap = np.sqrt(np.var(cop_ap) + np.var(vel_ap))

    # --- 4. Geometry-Based Metrics (LFS & Fractal Dimension) ---
    cov = np.cov(cop_ml, cop_ap)
    area_ce = 2 * np.pi * np.sqrt(max(0, np.linalg.det(cov))) # 95% Confidence Ellipse
    if area_ce == 0: area_ce = 0.0001

    lfs = sway_length_mlap / area_ce
    d = np.sqrt(area_ce * 4 / np.pi)
    fd = np.log(N) / (np.log(N) + np.log(d) - np.log(sway_length_mlap)) if sway_length_mlap > 0 else np.nan

    # --- 5. Velocity Peak Distribution ---
    zc_ml, pos_ml, neg_ml, all_ml = get_velocity_peaks(vel_ml)
    zc_ap, pos_ap, neg_ap, all_ap = get_velocity_peaks(vel_ap)

    # --- 6. Sway Density (Optimized) ---
    radius = 0.05 
    radius_sq = radius ** 2
    sway_density = np.zeros(N)
    
    for i in range(N):
        count = 0
        ref_ap, ref_ml = cop_ap[i], cop_ml[i]
        for j in range(i, N):
            if ((cop_ap[j] - ref_ap)**2 + (cop_ml[j] - ref_ml)**2) <= radius_sq:
                count += 1
            else:
                break 
        sway_density[i] = count
        
    pos_peaks_idx = np.where((sway_density[1:-1] > sway_density[:-2]) & (sway_density[1:-1] > sway_density[2:]))[0] + 1
    if len(pos_peaks_idx) > 0:
        mean_peak_sd = np.mean(sway_density[pos_peaks_idx])
        peak_coords = np.column_stack((cop_ml[pos_peaks_idx], cop_ap[pos_peaks_idx]))
        mean_dist_peak_sd = np.mean(np.linalg.norm(np.diff(peak_coords, axis=0), axis=1)) if len(peak_coords) > 1 else 0.0
    else:
        mean_peak_sd = np.max(sway_density)
        mean_dist_peak_sd = 0.0

    # --- 7. Mean Frequencies ---
    dist_ml = np.mean(np.abs(ml_c))
    spd_ml = np.abs(vel_ml)
    mean_freq_ml = (1 / (4 * np.sqrt(2))) * (np.mean(spd_ml) / dist_ml) if dist_ml != 0 else 0

    dist_ap = np.mean(np.abs(ap_c))
    spd_ap = np.abs(vel_ap)
    mean_freq_ap = (1 / (4 * np.sqrt(2))) * (np.mean(spd_ap) / dist_ap) if dist_ap != 0 else 0

    dist_mlap = np.mean(np.sqrt(ml_c**2 + ap_c**2))
    spd_mlap = np.sqrt(vel_ml**2 + vel_ap**2)
    mean_freq_mlap = (1 / (2 * np.pi)) * (np.mean(spd_mlap) / dist_mlap) if dist_mlap != 0 else 0

    return {
        'mean_velocity_ML': mean_vel_ml,
        'mean_velocity_AP': mean_vel_ap,
        'mean_velocity_ML_AND_AP': mean_vel_mlap,
        'sway_area_per_second_ML_AND_AP': sway_area_per_sec,
        'phase_plane_parameter_ML': ppp_ml,
        'phase_plane_parameter_AP': ppp_ap,
        'LFS_ML_AND_AP': lfs,
        'fractal_dimension_ML_AND_AP': fd,
        'zero_crossing_SPD_ML': zc_ml,
        'peak_velocity_pos_SPD_ML': pos_ml,
        'peak_velocity_neg_SPD_ML': neg_ml,
        'peak_velocity_all_SPD_ML': all_ml,
        'zero_crossing_SPD_AP': zc_ap,
        'peak_velocity_pos_SPD_AP': pos_ap,
        'peak_velocity_neg_SPD_AP': neg_ap,
        'peak_velocity_all_SPD_AP': all_ap,
        'mean_peak_Sway_Density': mean_peak_sd,
        'mean_distance_peak_Sway_Density': mean_dist_peak_sd,
        'mean_frequency_ML': mean_freq_ml,
        'mean_frequency_AP': mean_freq_ap,
        'mean_frequency_ML_AND_AP': mean_freq_mlap
    }

def main():
    results_by_task = defaultdict(list)
    
    # 1. Load the demographic dictionaries for mapping later
    print("Loading demographic and clinical data...")
    demo_dict = load_demographics()
    
    # Ensure standard ordering of demographic columns in the output files
    demo_columns = ['Age', 'Gender', 'Sex', 'Race', 'Height (in)', 'Weight (kg)', 'Hoehn and Yahr scale PD severity']
    
    for group in ['Control', 'PD']:
        folder_path = os.path.join(DATA_DIR, group)
        if not os.path.exists(folder_path):
            print(f"Directory not found: {folder_path}. Skipping.")
            continue
            
        # 2. Process all CSV files in the directory
        csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
        print(f"\nProcessing {len(csv_files)} files for {group} group...")
        
        for file_path in csv_files:
            file_name = os.path.basename(file_path)
            
            # Skip _mat files automatically
            if '_mat' in file_name:
                continue
                
            # Extract Subject ID and Task from the filename (e.g., HC100_HurriedPace.csv)
            parts = file_name.replace('.csv', '').split('_', 1)
            if len(parts) < 2:
                print(f"Skipping {file_name}: format does not match Subject_Task.csv")
                continue
                
            subject_id = parts[0]
            task_name = parts[1]
            
            try:
                # 3. Read raw data and extract metrics
                df = pd.read_csv(file_path, low_memory=False)
                cop_x, cop_y, time_arr = calculate_global_cop(df)
                features = extract_features(cop_x, cop_y, time_arr)

                # Add this check to skip the patient if they have invalid CoP data
                if pd.isna(features['mean_velocity_ML']):
                    print(f"Skipping {file_name}: Insufficient or invalid CoP data.")
                    continue
                
                # 4. Construct the patient's row
                row_data = {
                    'Subject': subject_id, 
                    'Group': group, 
                    'Task': task_name,
                    'File': file_name
                }
                
                # 5. Append demographics using the subject ID (if missing, it inserts NaN)
                patient_demo = demo_dict.get(subject_id, {})
                for col in demo_columns:
                    row_data[col] = patient_demo.get(col, np.nan)
                
                # 6. Append dynamic features
                row_data.update(features)
                
                # Add to the dictionary categorized by task name
                results_by_task[task_name].append(row_data)
                
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
                
    # 7. Create separate CSVs for every task dynamically
    if results_by_task:
        for task, results in results_by_task.items():
            final_df = pd.DataFrame(results)
            output_file = f"CoP_dynamic_features_{task}_All_21.csv"
            final_df.to_csv(output_file, index=False)
            print(f"✅ Created {output_file} containing {len(results)} total patients.")
    else:
        print("\n⚠️ No data was processed.")

if __name__ == "__main__":
    main()