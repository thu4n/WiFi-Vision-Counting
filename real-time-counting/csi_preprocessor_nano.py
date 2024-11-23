import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from hampel import hampel
import joblib

class ESP32:
    """Parse ESP32 Wi-Fi Channel State Information (CSI) obtained using ESP32 CSI Toolkit by Hernandez and Bulut.
    ESP32 CSI Toolkit: https://stevenmhernandez.github.io/ESP32-CSI-Tool/
    """

    NULL_SUBCARRIERS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 64, 65, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 382, 383]

    def __init__(self, csi_file):
        self.csi_file = csi_file
        self.__read_file()

    def __read_file(self):
        """Read RAW CSI file (.csv) using Pandas and return a Pandas dataframe
        """
        self.csi_df = pd.read_csv(self.csi_file, error_bad_lines=False)

    def seek_file(self):
        """Seek RAW CSI file
        """
        return self.csi_df

    def filter_by_sig_mode(self, sig_mode):
        """Filter CSI data by signal mode
        Args:
            sig_mode (int):
            0 : Non - High Throughput Signals (non-HT)
            1 : HIgh Throughput Signals (HT)
        """
        self.csi_df = self.csi_df.loc[self.csi_df['sig_mode'] == sig_mode]
        return self

    def get_csi(self):
        """Read CSI string as Numpy array

        The CSI data collected by ESP32 contains channel frequency responses (CFR) represented by two signed bytes (imaginary, real) for each sub-carriers index
        The length (bytes) of the CSI sequency depends on the CFR type
        CFR consist of legacy long training field (LLTF), high-throughput LTF (HT-LTF), and space- time block code HT-LTF (STBC-HT-LTF)
        Ref: https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-guides/wifi.html#wi-fi-channel-state-information

        NOTE: Not all 3 field may not be present (as represented in table and configuration)
        """
        raw_csi_data = self.csi_df['CSI_DATA'].copy()

        # csi_data = []
        # for csi_datum in raw_csi_data:
        #     csi_datum = csi_datum.strip('[ ]')
        #     data = [int]
        csi_data = np.array([np.fromstring(csi_datum.strip('[ ]'), dtype=int, sep = ' ') for csi_datum in raw_csi_data])
        csi_data = [csi_datum for csi_datum in csi_data if len(csi_datum) == 128]
        self.csi_data = np.array(csi_data)

        return self

    # NOTE: Currently does not provide support for all signal subcarrier types
    def remove_null_subcarriers(self):
        """Remove NULL subccodearriers from CSI
        """
        # Non-HT Signals (20 Mhz) - non STBC
        if self.csi_data.shape[1] == 128:
            remove_null_subcarriers = self.NULL_SUBCARRIERS[:24]
        # HT Signals (40 Mhz) - non STBC
        elif self.csi_data.shape[1] == 384:
            remove_null_subcarriers = self.NULL_SUBCARRIERS
        else:
            return self
        print("Removing null subcarries")
        csi_data_T = self.csi_data.T
        csi_data_T_clean = np.delete(csi_data_T, remove_null_subcarriers, 0)
        csi_data_clean = csi_data_T_clean.T
        self.csi_data = csi_data_clean

        return self

    def get_amplitude_from_csi(self):
        """Calculate the Amplitude (or Magnitude) from CSI
        Ref: https://farside.ph.utexas.edu/teaching/315/Waveshtml/node88.html
        """
        try:
            self.amplitude = np.array([np.sqrt(data[::2]**2 + data[1::2]**2) for data in self.csi_data])
            # amplitude = np.array([np.sqrt(data[::2]**2 + data[1::2]**2) for data in self.csi_data])
            print("Amplitude extracted")
            return self
        except IndexError as e:
            print("IndexError:", e)
            print("Check the structure of self.csi_data")
            return None

def extract_amplitude(raw_data):
    csi_array = (
        ESP32(raw_data)
            .get_csi()
            .remove_null_subcarriers()
            .get_amplitude_from_csi()
    )
    amp_df = pd.DataFrame(csi_array.amplitude, index=None)
    return amp_df

def denoise_data(amp_df):
    filtered_df = pd.DataFrame()
    for col in amp_df.columns:
        col_series = amp_df[col]
        # Hampel filter
        hampel_filtered = hampel(col_series, window_size=10, imputation=True)
        # Savitzky-Golay filter
        sg_filtered = savgol_filter(hampel_filtered, window_length=11, polyorder=3)
        filtered_df[col] = sg_filtered
    print("Denoised amplitude")
    return filtered_df

def extract_features(filtered_df_with_rssi):
    """
    Args:
        filtered_df_with_rssi: numpy array containing CSI amplitude data (shape: [packets, subcarriers])

    Returns:
        A dictionary containing the calculated statistics.
    """
    features = {}
    for i in range(filtered_df_with_rssi.shape[1] - 1): # Loop through each subcarrier
      features[f'std_subcarrier_{i}'] = np.std(filtered_df_with_rssi[:, i]) # Standard deviation
      features[f'mean_subcarrier_{i}'] = np.mean(filtered_df_with_rssi[:, i]) # The average amplitude value
      features[f'max_subcarrier_{i}'] = np.max(filtered_df_with_rssi[:, i])
      features[f'min_subcarrier_{i}'] = np.min(filtered_df_with_rssi[:, i])
      features[f'qtu_subcarrier_{i}'] = np.percentile(filtered_df_with_rssi[:, i], 75) # Upper quartile
      features[f'qtl_subcarrier_{i}'] = np.percentile(filtered_df_with_rssi[:, i], 25) # Lower quartile
      features[f'iqr_subcarrier_{i}'] = features[f'qtu_subcarrier_{i}'] - features[f'qtl_subcarrier_{i}']

    for i in range(2, filtered_df_with_rssi.shape[1] - 3):  # Skip the first and last 2 subcarriers
        num_cols = filtered_df_with_rssi.shape[1]
        start_idx = max(0, i - 2)
        end_idx = min(num_cols, i + 2 + 1)
        adjacent_data = np.delete(filtered_df_with_rssi[:, start_idx:end_idx], i - start_idx, axis=1)
        # Calculate the amplitude difference for the current subcarrier
        amplitude_difference = np.sum(np.abs(adjacent_data - filtered_df_with_rssi[:, [i]]), axis=1)
        features[f'adj_subcarrier_{i}'] = np.mean(amplitude_difference)

    euclidean_distances = []
    for i in range(1, filtered_df_with_rssi.shape[0]):  # Loop through packets starting from the second
      distances = np.linalg.norm(filtered_df_with_rssi[i, :] - filtered_df_with_rssi[i-1, :])
      euclidean_distances.append(distances)

    features['euc'] = np.median(euclidean_distances)
    features = pd.DataFrame([features])
    print("Features extracted")
    return features

def process_csi_from_csv(csi_path):
    raw_amp = extract_amplitude(csi_path)
    filtered_amp = denoise_data(raw_amp)

    temp_df = filtered_amp.copy()
    temp_np_array = np.array(temp_df) # Turn into numpy array for easier matrix calculation

    temp_df = extract_features(temp_np_array)
    scaler = joblib.load('modelzoo/2111_scaler_fold_2.pkl')
    final_csi_data = scaler.transform(temp_df)
    print("Data scaled")
    return final_csi_data
