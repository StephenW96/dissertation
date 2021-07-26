import torch
from torch import nn
from torch.utils.data import DataLoader
import torchaudio
from sklearn import metrics
import pandas as pd
import time
from nat_langs_dataset import NatLangsDataset
from CNN_model import CNNNetwork

# Test file with labels
# FT_ANNOTATIONS_FILE = '/afs/inf.ed.ac.uk/user/s21/s2118613/dissertation/cslu_fae_labels.csv'
# FT_ANNOTATIONS_FILE = '/content/gdrive/MyDrive/dissertation_data/annotations/cslu_fae_labels.csv'
FT_ANNOTATIONS_FILE = '/Users/stephenwalters/Documents/msc_speech_and_language_processing/dissertation/dissertation_data/annotations/cslu_fae_labels.csv'


# Test audio directory
# FT_AUDIO_DIR = '/afs/inf.ed.ac.uk/user/s21/s2118613/cslu_fae/speech/'
# FT_AUDIO_DIR = '/content/gdrive/MyDrive/dissertation_data/cslu_fae/speech/'
FT_AUDIO_DIR = '/Users/stephenwalters/Documents/msc_speech_and_language_processing/dissertation/dissertation_data/cslu_fae/speech'

# Hyperparameters
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001

# Sample rate hyperparameter
SAMPLE_RATE = 8000
# num samples = sample rate -> means 1 seconds worth of audio
NUM_SAMPLES = 8000

# Size of fine-tuning training data
TRAIN_SIZE = .5

def freeze_network(model):
    for name, p in model.named_parameters():
        # Freeze all but linear layers
        if "linear" not in name:
            p.requires_grad = False
        else:
            p.requires_grad = True


def train(model, train_dataloader, loss_fn, optimiser, device, epochs):
    b = time.time()
    for i in range(epochs):
        print(f"Epoch {i + 1}")
        a = time.time()
        train_single_epoch(model, train_dataloader, loss_fn, optimiser, device)
        print(f'Epoch {i} time: {a - b}')
        b = a
        print("---------------------------")
    print("Finished training")


def train_single_epoch(model, train_dataloader, loss_fn, optimiser, device):
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

        i += 1
        print(i)

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"Fine-Tuning Training loss: {(loss_sum / len(train_dataloader))}")

    prfs = metrics.precision_recall_fscore_support(targets_total, predictions_total)
    print(f"p = {prfs[0]}, r = {prfs[1]}, f = {prfs[2]}, s = {prfs[3]}")
    print(metrics.classification_report(targets_total, predictions_total))




def test(test_dataloader, net):
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


    # Train-Test split
    df = pd.read_csv(FT_ANNOTATIONS_FILE)
    print(df)
    ft_train_sub, test_sub = [], []
    cut = 0.95
    for el in df.groupby('path'):
        el = el[1]
        thres = int(len(el) * cut)
        ft_train_sub.append(el[:thres])
        test_sub.append(el[thres:])
    df = pd.concat(ft_train_sub)
    test_sub = pd.concat(test_sub)
    print(df)

    # Adjust size of training data for experiments (ignore leftover data)
    ft_train_sub, _ = [], []
    cut_train = TRAIN_SIZE
    for el in df.groupby('path'):
        el = el[1]
        thres = int(len(el) * cut_train)
        ft_train_sub.append(el[:thres])
    ft_train_sub = pd.concat(ft_train_sub)
    print(ft_train_sub)


    # instantiating our dataset object and create data loader
    # Training data
    train_data = NatLangsDataset(ft_train_sub, FT_AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES,
                                 device)
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE)

    # Validation data - Test data
    test_data = NatLangsDataset(test_sub, FT_AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES,
                               device)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)


    # Load in pretrained cnn
    PRETRAINED = '/Users/stephenwalters/Documents/msc_speech_and_language_processing/dissertation/dissertation/l1_classifier_melspec.pth'
    ft_net = CNNNetwork().to(device)
    ft_net.load_state_dict(torch.load(PRETRAINED))

    # Freeze Convolutional Layers
    freeze_network(ft_net)

    # initialise loss function + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(ft_net.parameters(),
                                 lr=LEARNING_RATE)

    # train model
    # No dev set because a) not enough data, b) out of domain fine-tuning means less likely overfitting
    train(ft_net, train_dataloader, loss_fn, optimiser, device, EPOCHS)



    # save model
    torch.save(ft_net.state_dict(), "fine_tuned_melspec.pth")
    print("Trained feed forward net saved at fine_tuned_melspec.pth")


    # Test model
    TEST_PATH = 'path to fine_tuned_melspec.pth'

    test_net = CNNNetwork().to(device)
    test_net.load_state_dict(torch.load(TEST_PATH))
    test(test_dataloader, test_net)