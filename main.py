# Plot loss
import pickle as pk
import matplotlib.pyplot as plt
with open('./l1_classifier_mfcc_losses.pk', 'rb') as f:
  train_loss, dev_loss = pk.load(f)

plt.plot(train_loss, label='train')
plt.plot(dev_loss, label='dev')
plt.legend()
plt.show()

# Plot accuracy
with open('./l1_classifier_mfcc_accuracies.pk', 'rb') as g:
  train_acc, dev_acc = pk.load(g)

plt.plot(train_acc, label='train')
plt.plot(dev_acc, label='dev')
plt.legend()
plt.show()