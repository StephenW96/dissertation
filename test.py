import torch
from torch import nn
from torch.utils.data import DataLoader
import torchaudio
import pandas as pd
from nat_langs_dataset import NatLangsDataset
from sklearn import metrics

# Change to fit CNN dims
from CNN_40mfcc_model import CNNetwork

# PATH = '/afs/inf.ed.ac.uk/user/s21/s2118613/PycharmProjects/Simple_CNN/l1_classifier_melspec.pth'
# PATH = '/Users/stephenwalters/Documents/msc_speech_and_language_processing/dissertation/dissertation/l1_classifier_melspec.pth'
ROOT = '/disk/scratch1/s2118613/dissertation/models/'
PATH = ROOT+'l1_classifier_40mfcc.pth'

# Test file with labels
#TE_ANNOTATIONS_FILE = '/afs/inf.ed.ac.uk/user/s21/s2118613/dissertation/cslu_fae_labels.csv'
# TE_ANNOTATIONS_FILE = '/content/gdrive/MyDrive/dissertation_data/annotations/cslu_fae_labels.csv'
#TE_ANNOTATIONS_FILE = '/Users/stephenwalters/Documents/msc_speech_and_language_processing/dissertation/dissertation_data/annotations/cslu_fae_labels.csv'
TE_ANNOTATIONS_FILE = '/disk/scratch1/s2118613/dissertation/annotations/cslu_fae_aug.csv'

# Test audio directory
#TE_AUDIO_DIR = '/afs/inf.ed.ac.uk/user/s21/s2118613/cslu_fae/speech/'
# TE_AUDIO_DIR = '/content/gdrive/MyDrive/dissertation_data/cslu_fae/speech/'
# TE_AUDIO_DIR = '/Users/stephenwalters/Documents/msc_speech_and_language_processing/dissertation/dissertation_data/cslu_fae/speech'
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

    # Test data
    test_dataframe = pd.read_csv(TE_ANNOTATIONS_FILE)

    hop_length_cut=2000

    test_data = NatLangsDataset(test_dataframe, TE_AUDIO_DIR, mfcc, SAMPLE_RATE, NUM_SAMPLES, hop_length_cut, device)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, collate_fn=my_collate, shuffle=True)



    # dataiter = iter(test_dataloader)
    # feature, label = dataiter.next()
    # print(feature.shape)
    # print(label)

    # Load in pretrained model
    net = CNNetwork().to(device)
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
            # i += 1
            # print("Test Batch", i)

            # # for batches > 1, move torch
            # target = torch.tensor([class_mapping[x] for x in target])
            #
            # # calculate outputs by running images through network
            # prediction = net(input)
            #
            # _,predicted = torch.max(prediction.data, 1)
            # if predicted == target:
            #     acc += 1

    test_loss = loss_sum / len(test_dataloader)
    print(f"Test loss: {(test_loss)}")

    print(metrics.classification_report(targets_total, predictions_total))
    quit()

        # print('Model Accuracy:', acc/len(test_dataloader)*100,'%')

