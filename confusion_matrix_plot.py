from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np

PATH="/home/oscar/Desktop/keras_proyect/"

data = pd.read_csv(PATH+'confusion_matrix_primary_site.csv')
#data.index = data.iloc[:,0]
del data['Unnamed: 0']
data.index = data.columns
data.to_csv(PATH+'confusion_matrix_primary_site2.csv')

df_cm = data
index = data.index
df_cm = np.array(df_cm)
cm_sum = np.sum(df_cm, axis=1, keepdims=True)
cm_perc = df_cm / cm_sum.astype(float) * 100
annot = np.empty_like(df_cm).astype(str)
nrows, ncols = df_cm.shape
for i in range(nrows):
    for j in range(ncols):
        c = df_cm[i, j]
        p = cm_perc[i, j]
        if i == j:
            s = cm_sum[i]
            annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
        elif c == 0:
            annot[i, j] = '0'
        else:
            annot[i, j] = '%.1f%%\n%d' % (p, c)

df_cm = pd.DataFrame(df_cm, index=list(data.index), columns=list(data.index))
a = pd.DataFrame(cm_perc,index=list(data.index), columns=list(data.index))
a.columns.name = "Predicted tissue"
a.index.name = "Tissue of origin"
fig, ax = plt.subplots(figsize=(30,20))
sn.heatmap(a,  annot=annot, fmt='', ax=ax, annot_kws={"size": 12},linewidths=.4)
#"Blues"cmap="Greens",
sn.set(font_scale=2)
#sn.heatmap(df_cm, annot=annot, fmt='', ax=ax, annot_kws={"size": 8},linewidths=.5)
plt.savefig("confusion_matrix_primary_site2.png",bbox_inches='tight')

#plt.show()
plt.close()
