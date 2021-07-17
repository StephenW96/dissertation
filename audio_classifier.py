import torch
from torch import nn
from torch.utils.data import DataLoader
import torchaudio

from nat_langs_dataset import NatLangsDataset
from CNN_model import CNNNetwork

# need to change hyperparameters
BATCH_SIZE = 1
EPOCHS = 10
LEARNING_RATE = 0.001

ANNOTATIONS_FILE = '/afs/inf.ed.ac.uk/user/s21/s2118613/dissertation/limited_data.csv'
#AUDIO_DIR = '/group/corporapublic/cslu_22_lang/speech/'
AUDIO_DIR = '/afs/inf.ed.ac.uk/user/s21/s2118613/cslu_22_lang/speech/'
SAMPLE_RATE = 8000
# num samples = sample rate -> means 1 seconds worth of audio
NUM_SAMPLES = 8000


def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader


def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)

        # calculate loss
        prediction = model(input)
        loss = loss_fn(prediction, target)

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"loss: {loss.item()}")


def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_single_epoch(model, data_loader, loss_fn, optimiser, device)
        print("---------------------------")
    print("Finished training")


if __name__ == "__main__":
    # if the GPU is available - use it
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    # defining mel-spectrogram data input
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    # MFCC alternative to mel-spectrogram - calculates the MFCC on the DB-scaled Mel spectrogram
    mfcc = torchaudio.transforms.MFCC(
        sample_rate=SAMPLE_RATE,
        # 12-13 is sufficient for English - 20 for tonal langs, maybe accent info
        n_mfcc=20
    )


    # instantiating our dataset object and create data loader
    nld = NatLangsDataset(ANNOTATIONS_FILE, AUDIO_DIR, mfcc, SAMPLE_RATE, NUM_SAMPLES, device)

    train_dataloader = create_data_loader(nld, BATCH_SIZE)
    dataiter = iter(train_dataloader)
    feature, label = dataiter.next()
    print(feature.shape)
    feature, label = dataiter.next()
    print(feature.shape)

    quit()

    # construct model and assign it to device
    cnn_net = CNNNetwork().to(device)
    #print(cnn_net)

    # initialise loss function + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn_net.parameters(),
                                 lr=LEARNING_RATE)

    # train model
    train(cnn_net, train_dataloader, loss_fn, optimiser, device, EPOCHS)

    # save model
    torch.save(cnn_net.state_dict(), "6_langs_classifier_cnn.pth")
    print("Trained feed forward net saved at 6_langs_classifier_cnn.pth")