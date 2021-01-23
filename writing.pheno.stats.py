
dfpehno = pandas.read_feather("/home/oscar/Desktop/keras.proyect/data/phenodata.dataframe.all.15000.feather")
del dfpehno["index"]
dfpehno2 = dfpehno.set_index('file_name')
dfpehno2.columns
dfpehno2 = dfpehno2[dfpehno2.index.isin(dfsamples.columns)]


dfpehno2.fillna(value="unknown", inplace=True)
# Diseases counts
dftmp = dfpehno2.loc[:, "disease"]
# Primary Sites
dftmp = dfpehno2.loc[:, "sample_type"]
# Sampel Tissue
dftmp = dfpehno2.loc[:, "primary_site"]
############################################3
#### BARPLOT with diseases
from collections import Counter
import matplotlib.pyplot as plt
import csv
tmp = Counter(dftmp)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
plt.bar(tmp.keys(),tmp.values())
plt.show()
############################################

with open('samples.tissue.csv','w') as csvfile:
    fieldnames=['tissue','samples']
    writer=csv.writer(csvfile)
    writer.writerow(fieldnames)
    for key, value in tmp.items():
        writer.writerow([key] + [value]) 

