#!/usr/bin/env python
# coding: utf-8

# In[1]:



import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams


# In[2]:



test_logs_rmc = pd.read_csv('../../results/gan/logs/test_logs.csv')
test_result_logs_rmc = pd.read_csv('../../results/gan/logs/test_result_logs.csv')

# test_logs_lstm = pd.read_csv('../../results/lstm/logs/test_logs.csv')
# test_result_logs_lstm = pd.read_csv('../../results/lstm/logs/test_result_logs.csv')


# In[3]:



res = test_logs_rmc.groupby('epoch').mean()


# In[4]:



X = 39
YMIN = 0.15
YMAX = 0.85
COLOR = 'k'
LINESTYLE = '--'


# In[9]:



#rcParams['figure.dpi'] = 400
#
#plt.subplot(2, 2, 1)
#plt.plot(res['selfBLEU_2'])
#plt.axvline(x=X, ymin=YMIN, ymax=YMAX, color=COLOR, linestyle=LINESTYLE)
#plt.xlabel('epochs')
#plt.ylabel('Self-BLEU-2')
#
#plt.subplot(2, 2, 2)
#plt.plot(res['selfBLEU_3'])
#plt.axvline(x=X, ymin=YMIN, ymax=YMAX, color=COLOR, linestyle=LINESTYLE)
#plt.xlabel('epochs')
#plt.ylabel('Self-BLEU-3')
#
#plt.subplot(2, 2, 3)
#plt.plot(res['selfBLEU_4'])
#plt.axvline(x=X, ymin=YMIN, ymax=YMAX, color=COLOR, linestyle=LINESTYLE)
#plt.xlabel('epochs')
#plt.ylabel('Self-BLEU-4')
#
#plt.subplot(2, 2, 4)
#plt.plot(res['selfBLEU_5'])
#plt.axvline(x=X, ymin=YMIN, ymax=YMAX, color=COLOR, linestyle=LINESTYLE)
#plt.xlabel('epochs')
#plt.ylabel('Self-BLEU-5')
#
#plt.tight_layout()
#plt.savefig('./figures/rmc_self_bleu_trend.png')


# In[10]:



rcParams['figure.dpi'] = 400

plt.subplot(2, 2, 1)
plt.plot(res['pMMD'])
plt.axvline(x=X, ymin=YMIN, color=COLOR, linestyle=LINESTYLE)
plt.xlabel('epochs')
plt.ylabel('Pitch MMD')

plt.subplot(2, 2, 2)
plt.plot(res['dMMD'])
plt.axvline(x=X, ymin=YMIN, color=COLOR, linestyle=LINESTYLE)
plt.xlabel('epochs')
plt.ylabel('Duration MMD')

plt.subplot(2, 2, 3)
plt.plot(res['rMMD'])
plt.axvline(x=X, ymin=YMIN, ymax=YMAX, color=COLOR, linestyle=LINESTYLE)
plt.xlabel('epochs')
plt.ylabel('Rest MMD')

plt.subplot(2, 2, 4)
plt.plot(res['oMMD'])
plt.axvline(x=X, ymin=YMIN, ymax=YMAX, color=COLOR, linestyle=LINESTYLE)
plt.xlabel('epochs')
plt.ylabel('Overall MMD')

plt.tight_layout()
plt.savefig('./figures/rmc_mmd_trend.png')


# In[7]:



# test_result_logs = pd.concat([test_result_logs_lstm, test_result_logs_rmc], ignore_index=True, sort=False)
#
#
# # In[13]:
#
#
#
# stats = test_result_logs.groupby('method').agg(['mean', 'std'])
# stats.loc['C-Hybrid-GAN']


# In[ ]:




