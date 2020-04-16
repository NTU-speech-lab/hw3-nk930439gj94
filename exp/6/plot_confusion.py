import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

confusion = np.load('confusion.npy')

df_cm = pd.DataFrame(confusion, range(11), range(11))
sn.heatmap(df_cm, annot=True, cmap="YlGnBu", vmax=100) # font size

plt.xlabel('prediction')
plt.ylabel('answer')

plt.show()