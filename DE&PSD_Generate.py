import numpy as np
import scipy.io as sio
import os
from scipy.signal import butter, lfilter
from pathlib import Path

from preprocessors import DEAPDataset, Sequence
from preprocessors import BinaryLabel
from preprocessors import Raw2TNCF, RemoveBaseline, TNCF2NCF, ChannelToLocation

def bandpass_filter(data, low, high, fs, order=5):
    b, a = butter(order, [low / (fs / 2), high / (fs / 2)], btype='band')
    return lfilter(b, a, data, axis=-1)

def compute_de(signal):
    var = np.var(signal, ddof=1)
    return 0.5 * np.log(2 * np.pi * np.e * var) if var > 0 else 0.0

def compute_PSD(signal):
    return np.sum(signal**2)

def extract_de_features(subject_data, sample_rate=128):
    raw_data = subject_data['feature']  # shape: (2400, 128, 9, 9)
    raw_label = subject_data['label']   # shape: (2400,)

    raw_data = raw_data.reshape(40, 60 * sample_rate, 9, 9)  # (40, 7680, 9, 9)
    raw_label = raw_label.reshape(40, 60)  # (40, 60)

    window_size = 128
    step_size = 128
    freq_bands = [(4, 8), (8, 14), (14, 31), (31, 45)]

    num_trials = raw_data.shape[0]
    num_windows = 7680 // step_size   # 60
    total_windows = num_windows

    DE_features = np.zeros((num_trials, total_windows, len(freq_bands)*2, 9, 9))

    for trial_idx in range(num_trials):
        trial_data = raw_data[trial_idx]  # (7680, 9, 9)

        for win_idx in range(num_windows):
            start = win_idx * step_size
            end = start + window_size
            window = trial_data[start:end]  # (128, 9, 9)

            for band_idx, (low_f, high_f) in enumerate(freq_bands):
                filtered = bandpass_filter(window.transpose(1, 2, 0), low_f, high_f, sample_rate)
                for i in range(9):
                    for j in range(9):
                        signal = filtered[i, j]
                        DE_features[trial_idx, win_idx, band_idx, i, j] = compute_de(signal)
                        DE_features[trial_idx, win_idx, band_idx + 4, i, j] = compute_PSD(signal)


    # reshape 为 (2400, 4, 9, 9)
    data = DE_features.reshape(-1, 8, 9, 9)

    # 将标签flatten 为 (2400,)
    valence_labels = np.repeat(raw_label, 1, axis=1).reshape(1, -1)  # (1, 2400)

    return data, valence_labels

def standardize_along_samples_per_channel(data):
    # 创建空数组用于存放结果
    data_standardized = np.zeros_like(data)

    # 遍历每个通道（第1轴）
    for ch in range(data.shape[1]):  # 8个通道
        # 对第 ch 个通道的所有样本进行标准化
        channel_data = data[:, ch, :, :]  # shape = (1200, 8, 9)
        mean = np.mean(channel_data, axis=0, keepdims=True)  # shape = (1, 8, 9)
        std = np.std(channel_data, axis=0, keepdims=True) + 1e-6
        # 标准化：每个样本的位置减去该位置的通道均值并除以标准差
        data_standardized[:, ch, :, :] = (channel_data - mean) / std

    return data_standardized
# 主处理流程
def process_all_subjects(preprocessors_results, save_dir='./DE&PSD_feature'):
    os.makedirs(save_dir, exist_ok=True)

    for idx in range(1, 33):
        subject_id = f's{idx:02d}'
        subject_data = preprocessors_results[subject_id]

        print(f"Processing {subject_id}...")

        data, valence_labels = extract_de_features(subject_data)
        data = standardize_along_samples_per_channel(data)
        print(data.shape)
        print(valence_labels.shape)

        save_path = os.path.join(save_dir, f'{subject_id}.mat')
        sio.savemat(save_path, {
            'data': data,  # (4800, 8, 9, 9)
            'valence_labels': valence_labels  # (1, 4800)
        })

    print("All subjects processed and saved.")

if __name__ == '__main__':
    DATASET_BASE_DIR = Path('./eeg_dataset')
    DATASET_FOLD_DIR = DATASET_BASE_DIR / 'DEAP'
    PREPROCESSED_EEG_DIR = DATASET_FOLD_DIR / 'data_preprocessed_python'

    label_preprocessors = {'label': Sequence([BinaryLabel()])}
    feature_preprocessors = {
        'feature':
        Sequence([Raw2TNCF(),
                  RemoveBaseline(),
                  TNCF2NCF(),
                  ChannelToLocation()])
    }

    preprocessors_results = DEAPDataset(
        PREPROCESSED_EEG_DIR, label_preprocessors,
        feature_preprocessors)('./dataset/deap_binary_valence_dataset.pkl')

    process_all_subjects(preprocessors_results)