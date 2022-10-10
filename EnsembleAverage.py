
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#data
my_data = pd.read_csv('epData.csv', header=None)
arr = np.reshape(my_data.values, (-1, 512)) # 512  data point per epoch
arr1 = arr.T
time = np.linspace(0,160,512)
points = np.linspace(0,512,1)
y = np.mean(arr1, axis=1)

#plot
f, ax = plt.subplots(nrows=6,ncols=1,figsize=(25,50))

ax[0].plot(time,arr1, linewidth = 2); ax[0].set_ylabel('EPSPs(uV)', fontsize=18);ax[0].set_xlabel('time(ms)', fontsize=18); ax[0].set_title("Fig1: 1000 epochs pattern with mean in bold blue line", fontsize=18)
ax[0].plot(time,y, linewidth = 4,  label="mean" )
ax[0].legend(loc="upper right")
ax[1].imshow(arr1, aspect ='auto'); ax[1].set_title('Fig2: Imageshow with auto aspect', fontsize=18)
ax[2].plot(time, arr1[:,:100], linewidth = 5, alpha = .6); ax[2].set_title('Fig3: Corrletion of mean with first 100 epochs', fontsize=18); ax[2].set_ylabel('EPSPs(uV)',fontsize=18);ax[2].set_xlabel('time(ms)',fontsize=18);
ax[2].plot(time,np.mean(arr1[:,:100],axis=1), linewidth = 6,  label="mean" )
ax[2].legend(loc="upper right")
ax[3].plot(time, y, linewidth = 4, alpha =0.6, label="mean of  first 1000 epochs");ax[3].set_ylabel('Average EPSPs(uV)',fontsize=18);ax[3].set_xlabel('time(ms)',fontsize=18); ax[3].set_title("Fig4: Mean EPSPs of 1000/500/100 epochs",fontsize=18)
ax[3].plot(time, np.mean(arr1[:,:500], axis=1),linewidth =4, alpha =0.4, label="mean of  first 500 epochs");
ax[3].plot(time, np.mean(arr1[:,:100], axis=1), linewidth =4, alpha =0.6, label="mean of  first 100 epochs");
ax[3].legend(loc="upper right")
ax[4].plot(arr1, norm.pdf(arr1,np.mean(arr1),np.std(arr1)),".r", lw=1, alpha=0.9, label='norm pdf'); ax[4].set_title('Fig5: Probability Density of all 1000 epochs', fontsize=18);ax[4].set_ylabel('Probability',fontsize=18);ax[4].set_xlabel('EPSPs value',fontsize=18);
ax[5].plot(y, norm.pdf(y, np.mean(y),np.std(y)),".r", lw=1, alpha=0.9, label='norm pdf'); ax[5].set_title('Fig6: Probability Density of all Mean value of all epochs', fontsize=18);ax[5].set_ylabel('Probability',fontsize=18);ax[5].set_xlabel('Mean EPSPs value',fontsize=18);

plt.show()

# Iterate through the 1000 epcohs
for i in range(1000):
    # Subset of one epoch
    subset = sorted(arr1[:,i])
    # Draw the density plot
    sns.distplot(subset, hist = False, kde = True,
                 kde_kws = {'linewidth': 2})
    
# Plot formatting
plt.title('Fig7: Proability density of individual of 1000 epcohs')
plt.xlabel('EPSPs')
plt.ylabel('Density')
plt.show()