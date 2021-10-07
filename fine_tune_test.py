import torch
from torch import nn
from torch.utils.data import DataLoader
import torchaudio
from sklearn import metrics
import pandas as pd
from nat_langs_dataset import NatLangsDataset
import random
import numpy as np


# Change to fit CNN dims
from CNN_model_ft import CNNetwork

# Set Random seeds
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

ROOT = '/disk/scratch1/s2118613/dissertation/'
FILE = 'ft_linear_75training.pth' 
PATH = ROOT+FILE

# All model names
# Xft_conv1_100training.pth  Xft_conv2_100training.pth  Xft_linear_100training.pth
# Xft_conv1_25training.pth   Xft_conv2_25training.pth   Xft_linear_25training.pth
# Xft_conv1_50training.pth   Xft_conv2_50training.pth   Xft_linear_50training.pth
# Xft_conv1_75training.pth   Xft_conv2_75training.pth   Xft_linear_75training.pth
# fae_100training.pth fae_75training.pth fae_50training.pth fae_25training.pth

# Test file with labels
TE_ANNOTATIONS_FILE = '/disk/scratch1/s2118613/dissertation/annotations/cslu_fae_aug.csv'

# Test audio directory
TE_AUDIO_DIR = '/disk/scratch1/s2118613/dissertation/cslu_fae_merged/speech'

# Hyperparameters
BATCH_SIZE = 256

# Sample rate hyperparameter
SAMPLE_RATE = 8000
# num samples = sample rate -> means 1 seconds worth of audio
NUM_SAMPLES = 8000

def my_collate(batch):
    class_mapping = {
        'BP': 0,
        'CA': 1,
        'GE': 2,
        'MA': 3,
        'RU': 4,
        'SP': 5
    }

    x, y = [], []
    for utterance, target in batch:
        for second in utterance:
            x.append(second)
            y.append(class_mapping[target])
    y = torch.LongTensor(y)
    # print([i.shape for i in x])
    x = torch.stack(x)

    return x, y


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
        n_mfcc=40, #12,13,40,60
        melkwargs={'hop_length': hop_length,
                   'n_fft': n_fft}
    )

    # Train-dev/Test split
    df = pd.read_csv(TE_ANNOTATIONS_FILE)
    train_sub, dev_test_sub = [], []
    cut = 0.80
    for el in df.groupby('path'):
        el = el[1]
        thres = int(len(el) * cut)
        train_sub.append(el[:thres])
        dev_test_sub.append(el[thres:])
    train_set = pd.concat(train_sub)
    dev_test_set = pd.concat(dev_test_sub)

    #Split into dev and test
    dev_sub, test_sub = [], []
    for el in dev_test_set.groupby('path'):
        el = el[1]
        thres = int(len(el) * 0.5)
        dev_sub.append(el[:thres])
        test_sub.append(el[thres:])
    ft_dev_set = pd.concat(dev_sub)
    ft_test_set = pd.concat(test_sub)

    hop_length_cut = 2000

    # instantiating our dataset object and create data loader
    # Training data + dataloader
    # train_data = NatLangsDataset(ft_train_set, FT_AUDIO_DIR, mfcc, SAMPLE_RATE, NUM_SAMPLES, hop_length_cut, device)
    # train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, collate_fn=my_collate, shuffle=True)

    # Dev data + dataloader
    # dev_data = NatLangsDataset(ft_dev_set, FT_AUDIO_DIR, mfcc, SAMPLE_RATE, NUM_SAMPLES, hop_length_cut, device)
    # dev_dataloader = DataLoader(dev_data, batch_size=BATCH_SIZE, collate_fn=my_collate, shuffle=True)

    # Test data + dataloader
    test_data = NatLangsDataset(ft_test_set, TE_AUDIO_DIR, mfcc, SAMPLE_RATE, NUM_SAMPLES, hop_length_cut, device)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, collate_fn=my_collate, shuffle=True)

    # Load in pretrained model
    net = CNNetwork().to(device)
    net.load_state_dict(torch.load(PATH))

    loss_fn = nn.CrossEntropyLoss()

    loss_sum = 0
    targets_total = []
    predictions_total = []
    i = 0
    with torch.no_grad():
        for input, target in test_dataloader:

            # put data to device
            input = input.to(device)
            target = target.to(device)

            # calculate loss
            prediction = net(input.to(device))
            loss = loss_fn(prediction, target).to(device)

            # accumulate loss for average
            loss_sum += loss

            # argmax of classes = prediction
            prediction_acc = torch.argmax(prediction, dim=1).to(device)

            # Convert prediction tensor to np array and concatenate into list of predictions
            prediction_acc = prediction_acc.cpu()
            prediction_acc = prediction_acc.numpy()
            predictions_total += list(prediction_acc)

            # Convert target tensor to np array and concatenate  into list of targets
            target = target.cpu()
            target_acc = target.numpy()
            targets_total += list(target_acc)

            # print batch number
            i += 1
            print("Test", i)

        print(FILE)
        loss = loss_sum / len(test_dataloader)
        print(f"Test loss: {(loss)}")

        print(metrics.classification_report(targets_total, predictions_total))

