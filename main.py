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
with open('./l1_classifier_melspec_losses.pk', 'rb') as f:
  train_loss, dev_loss = pk.load(f)


data1 = train_loss
data2 = dev_loss
plt.plot(data1, label = 'train')
plt.plot(data2, label = 'dev')
plt.legend()
plt.show()