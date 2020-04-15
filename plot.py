import matplotlib.pyplot as plt
import numpy as np
import os
import sys

log_train_acc = np.load(os.path.join(sys.argv[1], 'log_train_acc.npy'))
log_val_acc = np.load(os.path.join(sys.argv[1], 'log_val_acc.npy'))

plt.plot(log_train_acc, 'r', label='train_acc')
plt.legend()
plt.plot(log_val_acc, 'b', label='val_acc')
plt.legend()

plt.tight_layout()
plt.show()