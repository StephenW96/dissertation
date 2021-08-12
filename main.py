# import numpy as np
# import matplotlib.pyplot as plt
# import pprint
# import librosa
# import librosa.display
# import glob
#
# print(len(glob.glob('/group/corporapublic/timit/original/train/dr1/fcjf0/*.wav')))
#
# sound_file = '/group/corporapublic/timit/original/test/dr1/faks0/sa1.wav'
# y, sr = librosa.load(sound_file)
#
# D = librosa.stft(y)  # STFT of y
# S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
#
# plt.figure()
# librosa.display.specshow(S_db, cmap='gray')
# plt.show()

import pickle as pk
import matplotlib.pyplot as plt
with open('./l1_classifier_mfcc_losses.pk', 'rb') as f:
  train_loss, dev_loss = pk.load(f)

plt.plot(train_loss, label='train')
plt.plot(dev_loss, label='dev')
plt.legend()
plt.show()

with open('./l1_classifier_mfcc_accuracies.pk', 'rb') as g:
  train_acc, dev_acc = pk.load(g)

plt.plot(train_acc, label='train')
plt.plot(dev_acc, label='dev')
plt.legend()
plt.show()