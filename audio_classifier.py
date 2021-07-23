import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
import torchaudio
from sklearn.model_selection import train_test_split
import pandas as pd

from nat_langs_dataset import NatLangsDataset
from CNN_model import CNNNetwork


#### Training Model

# Hyperparameters
BATCH_SIZE = 1
EPOCHS = 10
LEARNING_RATE = 0.001

# Train file with labels
#TR_ANNOTATIONS_FILE = '/afs/inf.ed.ac.uk/user/s21/s2118613/dissertation/cslu_22_labels.csv'
TR_ANNOTATIONS_FILE = '/content/gdrive/MyDrive/dissertation_data/annotations/cslu_22_labels.csv'

# AUDIO_DIR = '/group/corporapublic/cslu_22_lang/speech/'

# Train audio directory
#TR_AUDIO_DIR = '/afs/inf.ed.ac.uk/user/s21/s2118613/cslu_22_lang/speech/'
TR_AUDIO_DIR = '/content/gdrive/MyDrive/dissertation_data/cslu_22_lang/speech/'

# Sample rate hyperparameter
SAMPLE_RATE = 8000
# num samples = sample rate -> means 1 seconds worth of audio
NUM_SAMPLES = 8000


def train_val_dataset(dataset, val_split=0.05):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split, stratify=True)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets


def train_single_epoch(model, train_dataloader, val_dataloader, loss_fn, optimiser, device):
    # Dict for mapping classes to numbers
    class_mapping = {
        'BP': 0,
        'CA': 1,
        'GE': 2,
        'MA': 3,
        'RU': 4,
        'SP': 5
    }

    loss_sum = 0
    acc = 0
    i = 0
    for input, target in train_dataloader:
        # Map classes to number, convert batch to tensor
        target_tensor = torch.tensor([class_mapping[x] for x in target]).to(device)
        # print(target_tensor)

        # calculate loss
        prediction = model(input.to(device))
        loss = loss_fn(prediction, target_tensor)
        loss_sum += loss

        if prediction == target:
            acc += 1
        i+=1
        #print(i)
        # if i == 200:
        #   break
        

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"Training loss: {(loss_sum/len(train_dataloader))}")
    print(f"Training Accuracy: {acc/len(train_dataloader)}")
    
    loss_val_sum = 0
    acc = 0
    with torch.no_grad():
        for input_val, target_val in val_dataloader:
            target_val_tensor = torch.tensor([class_mapping[x] for x in target_val]).to(device)
            prediction_val = model(input_val.to(device))
            loss_val = loss_fn(prediction_val, target_val_tensor)

            loss_val_sum += loss_val

            if prediction_val == target:
              acc += 1

    print(f"Dev loss: {(loss_val_sum/len(val_dataloader))}")
    print(f"Dev Accuracy: {(acc/len(val_dataloader))}")



def train(model, train_dataloader, val_dataloader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i + 1}")
        train_single_epoch(model, train_dataloader, val_dataloader, loss_fn, optimiser, device)
        print("---------------------------")
    print("Finished training")


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

    # Train-Validation split
    df = pd.read_csv(TR_ANNOTATIONS_FILE)
    train_sub, val_sub = [], []
    cut = 0.95
    for el in df.groupby('label'):
        el = el[1]
        thres = int(len(el) * cut)
        train_sub.append(el[:thres])
        val_sub.append(el[thres:])
    train_sub = pd.concat(train_sub)
    val_sub = pd.concat(val_sub)


    # instantiating our dataset object and create data loader
    # Training data
    train_data = NatLangsDataset(train_sub, TR_AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES,
                                 device)
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE)

    # Validation data
    val_data = NatLangsDataset(val_sub, TR_AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES,
                               device)
    val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE)

    # dataiter = iter(train_dataloader)
    # feature, label = dataiter.next()
    # print(feature.shape)
    # print(label)

    # construct model and assign it to device
    cnn_net = CNNNetwork().to(device)
    # print(cnn_net)

    # initialise loss function + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn_net.parameters(),
                                 lr=LEARNING_RATE)

    # train model
    train(cnn_net, train_dataloader, val_dataloader, loss_fn, optimiser, device, EPOCHS)

    # save model
    torch.save(cnn_net.state_dict(), "l1_classifier_melspec.pth")
    print("Trained feed forward net saved at l1_classifier.pth")
