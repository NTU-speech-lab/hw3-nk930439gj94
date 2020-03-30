import matplotlib.pyplot as plt
import numpy as np

log_acc = np.load('log/log_acc.npy')
log_loss = np.load('log/log_loss.npy')

plt.subplot(2,1,1)
plt.plot(log_acc, 'b', label='accuracy')
plt.legend()

plt.subplot(2,1,2)
plt.plot(log_loss, 'r', label='loss')
plt.xlabel('epoch')
plt.legend()

plt.tight_layout()
plt.show()