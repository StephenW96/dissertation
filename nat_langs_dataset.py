import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import os

## Creates a native language dataset of mel spectrograms and labels for training classifier

class NatLangsDataset(Dataset):

    def __init__(self, dataframe, audio_dir, transformation, target_sample_rate, num_samples, device):
        # file with path, name, labels

        # self.annotations = dataframe.iloc[200:204]
        self.annotations = dataframe

        # path from cslu to each lang and speaker
        self.audio_dir = audio_dir
        # cpu or cuda
        self.device = device
        # mel spectrogram/ mfcc
        self.transformation = transformation.to(self.device)
        # 8000
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    def __len__(self):
        # how much data
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        # signal -> (num_channels, samples) -> (2, 16000) --> (1, 16000) ---> look for my data
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)
        return signal, label

    def _cut_if_necessary(self, signal):
        # signal -> Tensor -> (1, num_samples) -> (1, 50000) will only go up to num of samples
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            # [1, 1, 1] -> [1, 1, 1, 0, 0]
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal


    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _get_audio_sample_path(self, index):
        #specific to annotation file -> coords change depending on this
        folders = self.annotations.iloc[index, 0]
        path = os.path.join(self.audio_dir, folders, self.annotations.iloc[index, 1])
        return path

    def _get_audio_sample_label(self, index) :
        #for my data -> first 2 characs of each file is its class, e.g. FR, HU, etc.
        return self.annotations.iloc[index, 2]

if __name__ == "__main__":
    ANNOTATIONS_FILE = '/afs/inf.ed.ac.uk/user/s21/s2118613/dissertation/foreign_data.csv'
    AUDIO_DIR = '/group/corporapublic/cslu_22_lang/speech/'
    SAMPLE_RATE = 8000
    NUM_SAMPLES = 8000

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft = 1024,
        hop_length=512,
        n_mels=64
    )

    nld = NatLangsDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, device)

    print(f'There are {len(nld)} samples in the dataset.')
    print(nld[0])

    signal, label = nld[0]



