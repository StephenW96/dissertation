import torch
from torch import nn
from torch.utils.data import DataLoader
import torchaudio
from sklearn import metrics
import pandas as pd
import time
from nat_langs_dataset import NatLangsDataset
from CNN_model_ft import CNNetwork
import pickle as pk
import random
import numpy as np

# Set Random seeds
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

#### Fine-tuning model

# Hyperparameters
BATCH_SIZE = 256
EPOCHS = 200
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 0.0001

# Path to CSV with file locations & labels
FT_ANNOTATIONS_FILE = '/disk/scratch1/s2118613/dissertation/annotations/cslu_fae_aug.csv'

# Path to Audio dir with dirs for langs & speakers
FT_AUDIO_DIR = '/disk/scratch1/s2118613/dissertation/cslu_fae_merged/speech'

# Sample rate hyperparameter
SAMPLE_RATE = 8000
# num samples = sample rate -> means 1 seconds worth of audio
NUM_SAMPLES = 8000

# Size of fine-tuning training data .25/.5/.75/1
TRAIN_SIZE = 1
MODEL_NAME = 'ft_linear_100training'


def train_single_epoch(model, train_dataloader, val_dataloader, loss_fn, optimiser, device):

    loss_sum = 0
    i = 0
    targets_total = []
    predictions_total = []

    for input, target in train_dataloader:
        # put data to device
        input = input.to(device)
        target = target.to(device)

        # calculate loss
        prediction = model(input.to(device))
        loss = loss_fn(prediction, target).to(device)

        # accumulate loss for average
        loss_sum += loss

        # argmax of classes = prediction
        prediction_acc = torch.argmax(prediction, dim=1).to(device)

        # Convert prediction tensor to np array and concatenate into list of predictions
        prediction_acc = prediction_acc.cpu()
        prediction_acc = prediction_acc.numpy()
        predictions_total += list(prediction_acc)

        # Convert target tensor to np array and concatenate into list of targets
        target = target.cpu()
        target_acc = target.numpy()
        targets_total += list(target_acc)

        # print batch number
        i += 1
        print(i)

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    train_loss = loss_sum / len(train_dataloader)
    print(f"Fine-Tuning Training loss: {train_loss}")

    print(metrics.classification_report(targets_total, predictions_total))

    # Accuracy score for the epoch (train)
    train_acc = metrics.accuracy_score(targets_total, predictions_total)

    loss_val_sum = 0
    targets_val_total = []
    predictions_val_total = []
    i = 0
    with torch.no_grad():
        for input_val, target_val in val_dataloader:
            # put data to device
            input_val = input_val.to(device)
            target_val = target_val.to(device)

            # calculate loss
            prediction_val = model(input_val.to(device))
            dev_loss = loss_fn(prediction_val, target_val).to(device)

            # accumulate loss for average
            loss_val_sum += dev_loss

            # argmax of classes = prediction
            prediction_val_acc = torch.argmax(prediction_val, dim=1).to(device)

            # Convert prediction tensor to np array and concatenate into list of predictions
            prediction_val_acc = prediction_val_acc.cpu()
            prediction_val_acc = prediction_val_acc.numpy()
            predictions_val_total += list(prediction_val_acc)

            # Convert target tensor to np array and concatenate  into list of targets
            target_val = target_val.cpu()
            target_val_acc = target_val.numpy()
            targets_val_total += list(target_val_acc)

            # print batch number
            i += 1
            print("Dev", i)

            # No backpropagation for validation set
    dev_loss = loss_val_sum / len(val_dataloader)
    print(f"Fine-Tuning Dev loss: {(dev_loss)}")

    print(metrics.classification_report(targets_val_total, predictions_val_total))

    # Accuracy score for the epoch (dev)
    dev_acc = metrics.accuracy_score(targets_val_total, predictions_val_total)

    return train_loss, dev_loss, train_acc, dev_acc


def train(model, train_dataloader, val_dataloader, loss_fn, optimiser, device, epochs):

    min = 100000
    counter = 0
    train_loss_array = []
    dev_loss_array = []
    train_acc_array = []
    dev_acc_array = []

    for i in range(epochs):
        # Time epoch
        a = time.time()
        print(f"Epoch {i + 1}")
        # Train epoch
        train_loss, dev_loss, train_acc, dev_acc = train_single_epoch(model, train_dataloader, val_dataloader, loss_fn, optimiser, device)
        b = time.time()
        print(f'Epoch {i + 1} time: {b - a}')
        print("---------------------------")

        # Convert loss tensor to scalar
        train_loss = train_loss.cpu()
        train_loss = train_loss.detach().numpy()
        dev_loss = dev_loss.cpu()
        dev_loss = dev_loss.numpy()

        # Append train & dev losses to array
        train_loss_array.append(train_loss)
        dev_loss_array.append(dev_loss)

        # Append train & dev accuracies to array
        train_acc_array.append(train_acc)
        dev_acc_array.append(dev_acc)

        # if current epochs val loss value < best loss so far --> set new best loss
        if dev_loss < min:
            min = dev_loss
            counter = 0

            # Save current best model
            torch.save(ft_net.state_dict(), f"./{MODEL_NAME}.pth")

        # if current loss is worse than best loss --> counter
        else:
            counter +=1

        # if loss doesnt improve in 5 successive epochs end training
        # if counter == 100:
        #     print("Finished training")
        #     quit()

    print("Finished training")

    # ft_conv1_100training

    with open(f'./{MODEL_NAME}_losses.pk', 'wb') as f:
        pk.dump((train_loss_array, dev_loss_array), f, protocol=pk.HIGHEST_PROTOCOL)

    with open(f'./{MODEL_NAME}_accuracies.pk', 'wb') as f:
        pk.dump((train_acc_array, dev_acc_array), f, protocol=pk.HIGHEST_PROTOCOL)


def test(test_dataloader, net):

    with torch.no_grad():
        for input, target in test_dataloader:

            # put data to device
            input = input.to(device)
            target = target.to(device)

            # calculate loss
            prediction = model(input.to(device))
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

    # MFCC alternative to mel-spectrogram - calculates the MFCC on the DB-scaled Mel spectrogram
    mfcc = torchaudio.transforms.MFCC(
        sample_rate=SAMPLE_RATE,
        n_mfcc=40, # For experiments: 12, 13, 40, 60
        # for alternate MFCC experiment
        melkwargs={'hop_length': hop_length,
                   'n_fft': n_fft}
    )

    # Train-dev/Test split
    df = pd.read_csv(FT_ANNOTATIONS_FILE)
    train_sub, dev_test_sub = [], []
    cut = 0.80
    for el in df.groupby('path'):
        el = el[1]
        thres = int(len(el) * cut)
        train_sub.append(el[:thres])
        dev_test_sub.append(el[thres:])
    train_set = pd.concat(train_sub)
    dev_test_set = pd.concat(dev_test_sub)
    #print(df)

    #Split into dev and test
    dev_sub, test_sub = [], []
    for el in dev_test_set.groupby('path'):
        el = el[1]
        thres = int(len(el) * 0.5)
        dev_sub.append(el[:thres])
        test_sub.append(el[thres:])
    ft_dev_set = pd.concat(dev_sub)
    ft_test_set = pd.concat(test_sub)

    # Adjust size of training data for experiments (ignore leftover data)
    final_train_sub, _ = [], []
    cut_train = TRAIN_SIZE
    for el in df.groupby('path'):
        el = el[1]
        thres = int(len(el) * cut_train)
        final_train_sub.append(el[:thres])
    ft_train_set = pd.concat(final_train_sub)
    # print(ft_train_sub)

    hop_length_cut = 2000

    # instantiating our dataset object and create data loader
    # Training data + dataloader
    train_data = NatLangsDataset(ft_train_set, FT_AUDIO_DIR, mfcc, SAMPLE_RATE, NUM_SAMPLES, hop_length_cut, device)
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, collate_fn=my_collate, shuffle=True)

    # Dev data + dataloader
    dev_data = NatLangsDataset(ft_dev_set, FT_AUDIO_DIR, mfcc, SAMPLE_RATE, NUM_SAMPLES, hop_length_cut, device)
    dev_dataloader = DataLoader(dev_data, batch_size=BATCH_SIZE, collate_fn=my_collate, shuffle=True)

    # Test data + dataloader
    test_data = NatLangsDataset(ft_test_set, FT_AUDIO_DIR, mfcc, SAMPLE_RATE, NUM_SAMPLES, hop_length_cut, device)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)


    # Path to Pre-trained CNN
    PRETRAINED = '/disk/scratch1/s2118613/dissertation/models/l1_classifier_40mfcc.pth'

    # Load in Pre-trained CNN and freeze layers
    # 'conv0' = no frozen layers
    # 'conv1' = 1 frozen layer
    # 'conv2' = 2 frozen layers
    # 'linear' = all frozen CNN layers
    
    ft_net = CNNetwork(fine_tune='linear').to(device)
    ft_net.load_state_dict(torch.load(PRETRAINED))

    # initialise loss function + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(ft_net.parameters(),
                                 lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # train model
    train(ft_net, train_dataloader, dev_dataloader, loss_fn, optimiser, device, EPOCHS)

    # save model
    SAVE_PATH = '/disk/scratch1/s2118613/dissertation/ft_models/'
    torch.save(ft_net.state_dict(), SAVE_PATH+MODEL_NAME)
    print(f"Trained feed forward net saved at {SAVE_PATH}{MODEL_NAME}")


    # Test model
    TEST_PATH = f'{SAVE_PATH}{MODEL_NAME}'

    test_net = CNNetwork().to(device)
    test_net.load_state_dict(torch.load(TEST_PATH))
    test(test_dataloader, test_net)
