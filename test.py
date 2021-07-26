import torch
from torch import nn
from torch.utils.data import DataLoader
import torchaudio
import pandas as pd

from nat_langs_dataset import NatLangsDataset
from CNN_model import CNNNetwork

# PATH = '/afs/inf.ed.ac.uk/user/s21/s2118613/PycharmProjects/Simple_CNN/l1_classifier_melspec.pth'
PATH = '/Users/stephenwalters/Documents/msc_speech_and_language_processing/dissertation/dissertation/l1_classifier_melspec.pth'

# Test file with labels
#TE_ANNOTATIONS_FILE = '/afs/inf.ed.ac.uk/user/s21/s2118613/dissertation/cslu_fae_labels.csv'
# TE_ANNOTATIONS_FILE = '/content/gdrive/MyDrive/dissertation_data/annotations/cslu_fae_labels.csv'
TE_ANNOTATIONS_FILE = '/Users/stephenwalters/Documents/msc_speech_and_language_processing/dissertation/dissertation_data/annotations/cslu_fae_labels.csv'

# Test audio directory
#TE_AUDIO_DIR = '/afs/inf.ed.ac.uk/user/s21/s2118613/cslu_fae/speech/'
# TE_AUDIO_DIR = '/content/gdrive/MyDrive/dissertation_data/cslu_fae/speech/'
TE_AUDIO_DIR = '/Users/stephenwalters/Documents/msc_speech_and_language_processing/dissertation/dissertation_data/cslu_fae/speech'


# Hyperparameters
BATCH_SIZE = 1

# Sample rate hyperparameter
SAMPLE_RATE = 8000
# num samples = sample rate -> means 1 seconds worth of audio
NUM_SAMPLES = 8000



if __name__ == "__main__":

    # if the GPU is available - use it
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    hop_length = 512
    n_fft = 1024
    # defining mel-spectrogram data input
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=64
    )

    # MFCC alternative to mel-spectrogram - calculates the MFCC on the DB-scaled Mel spectrogram
    mfcc = torchaudio.transforms.MFCC(
        sample_rate=SAMPLE_RATE,
        # 12-13 is sufficient for English - 20 for tonal langs, maybe accent info
        n_mfcc=20,
        melkwargs={'hop_length': hop_length,
                   'n_fft': n_fft}
    )

    # Test data
    test_dataframe = pd.read_csv(TE_ANNOTATIONS_FILE)
    test_data = NatLangsDataset(test_dataframe, TE_AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, device)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)


    # dataiter = iter(test_dataloader)
    # feature, label = dataiter.next()
    # print(feature.shape)
    # print(label)

    # Load in pretrained model
    net = CNNNetwork().to(device)
    net.load_state_dict(torch.load(PATH))

    loss_fn = nn.CrossEntropyLoss()


    class_mapping = {
        'BP': 0,
        'CA': 1,
        'GE': 2,
        'MA': 3,
        'RU': 4,
        'SP': 5
    }


    with torch.no_grad():
        acc = 0
        for data in test_dataloader:
            input, target = data
            # for batches > 1, move torch
            target = torch.tensor([class_mapping[x] for x in target])

            # calculate outputs by running images through network
            prediction = net(input)

            _,predicted = torch.max(prediction.data, 1)
            if predicted == target:
                acc += 1



        print('Model Accuracy:', acc/len(test_dataloader)*100,'%')
