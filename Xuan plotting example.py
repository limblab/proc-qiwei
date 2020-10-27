import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import numpy as np

# Generating data samples
x = np.arange(1000)
y = np.random.randn(1000)
fs = 100 # 100 Hz
N = 500 # How long you want to plot

rcParams['font.family'] = 'Arial'
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

# pre-set
fig = plt.figure('Random', figsize = (12, 4))
ax = plt.subplot(111)

# do real plotting
plt.plot(x[:N]/fs, y[:N], linewidth = 1.5, color = 'green')

# set labels
plt.xlabel('Time (s)', fontsize = 18)
plt.ylabel('Y value', fontsize = 18)

# set ticks
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)

plt.title('Random', fontsize = 20)

sns.despine()
plt.tight_layout()