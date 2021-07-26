import torch
from torch import nn
from torch.utils.data import DataLoader
import torchaudio
from sklearn import metrics
import pandas as pd
import time
from nat_langs_dataset import NatLangsDataset
from CNN_model import CNNNetwork


#### Training Model

# Hyperparameters
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001

# Train file with labels
# TR_ANNOTATIONS_FILE = '/afs/inf.ed.ac.uk/user/s21/s2118613/dissertation/cslu_22_labels.csv'
# TR_ANNOTATIONS_FILE = '/content/gdrive/MyDrive/dissertation_data/annotations/cslu_22_labels.csv'
TR_ANNOTATIONS_FILE = '/Users/stephenwalters/Documents/msc_speech_and_language_processing/dissertation/dissertation_data/annotations/cslu_22_labels.csv'

# AUDIO_DIR = '/group/corporapublic/cslu_22_lang/speech/'

# Train audio directory
# TR_AUDIO_DIR = '/afs/inf.ed.ac.uk/user/s21/s2118613/cslu_22_lang/speech/'
# TR_AUDIO_DIR = '/content/gdrive/MyDrive/dissertation_data/cslu_22_lang/speech/'
TR_AUDIO_DIR = '/Users/stephenwalters/Documents/msc_speech_and_language_processing/dissertation/dissertation_data/cslu_22_lang/speech'

# Sample rate hyperparameter
SAMPLE_RATE = 8000
# num samples = sample rate -> means 1 seconds worth of audio
NUM_SAMPLES = 8000


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

    i = 0
    targets_total = []
    predictions_total = []
    for input, target in train_dataloader:

        # Map classes to number, convert batch to tensor
        target_tensor = torch.tensor([class_mapping[x] for x in target]).to(device)
        # print(target_tensor)

        # calculate loss
        prediction = model(input.to(device))
        loss = loss_fn(prediction, target_tensor)

        # accumulate loss for average
        loss_sum += loss

        # argmax of classes = prediction
        prediction_acc = torch.argmax(prediction, dim=1)


        # Convert prediction tensor to np array and concatenate  into list of predictions
        prediction_acc = prediction_acc.numpy()
        predictions_total += list(prediction_acc)

        # Convert target tensor to np array and concatenate  into list of targets
        target_acc = target_tensor.numpy()
        targets_total += list(target_acc)

        i+=1
        print(i)


        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"Training loss: {(loss_sum/len(train_dataloader))}")

    prfs = metrics.precision_recall_fscore_support(targets_total, predictions_total)
    print(f"p = {prfs[0]}, r = {prfs[1]}, f = {prfs[2]}, s = {prfs[3]}")
    print(metrics.classification_report(targets_total, predictions_total))
    
    loss_val_sum = 0
    targets_val_total = []
    predictions_val_total = []
    i=0
    with torch.no_grad():
        for input_val, target_val in val_dataloader:
            target_val_tensor = torch.tensor([class_mapping[x] for x in target_val]).to(device)
            prediction_val = model(input_val.to(device))
            loss_val = loss_fn(prediction_val, target_val_tensor)

            loss_val_sum += loss_val

            # argmax of classes = prediction
            prediction_val_acc = torch.argmax(prediction_val, dim=1)

            # Convert prediction tensor to np array and concatenate  into list of predictions
            prediction_val_acc = prediction_val_acc.numpy()
            predictions_val_total += list(prediction_val_acc)

            # Convert target tensor to np array and concatenate  into list of targets
            target_val_acc = target_val_tensor.numpy()
            targets_val_total += list(target_val_acc)

            i+=1
            print("Dev",i)

    print(f"Dev loss: {(loss_val_sum/len(val_dataloader))}")

    prfs = metrics.precision_recall_fscore_support(targets_val_total, predictions_val_total)
    print(f"Dev p = {prfs[0]}, r = {prfs[1]}, f = {prfs[2]}, s = {prfs[3]}")
    print(metrics.classification_report(targets_val_total, predictions_val_total))


def train(model, train_dataloader, val_dataloader, loss_fn, optimiser, device, epochs):
    b = time.time()
    for i in range(epochs):
        print(f"Epoch {i + 1}")
        a = time.time()
        train_single_epoch(model, train_dataloader, val_dataloader, loss_fn, optimiser, device)
        print(f'Epoch {i} time: {a-b}')
        b = a
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
        # for alternate MFCC experiment
        # n_mfcc=12,
        melkwargs={'hop_length': hop_length,
                   'n_fft': n_fft}
    )

    # Train-Validation split
    df = pd.read_csv(TR_ANNOTATIONS_FILE)
    train_sub, val_sub = [], []

    # % of data used in training
    cut = 0.95

    for el in df.groupby('path'):
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
