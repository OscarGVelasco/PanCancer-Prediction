from keras.models import load_model
import json
import os
import convolutionalkerasfunctionalapicorefunctions as convfunc
import pandas
from keras.utils import np_utils
import numpy
import csv
from sklearn.metrics import classification_report, confusion_matrix

# Import app configuration variables
from config import config

# load model
print("Loading DNN model: ",config.FILEMODEL)
model = load_model(config.DATAOUTPUT+config.FILEMODEL)
# summarize model.
model.summary()

#Now read the file back into a Python list object
#print("Loading Validation samples...")
#with open(DATAOUTPUT+TESTSAMPLEFILE, 'r') as f:
#    validation_samples = json.loads(f.read())

# Pancancer genes features
print("Loading Pancancer genes csv.")
with open(config.DATAOUTPUT+config.PANCANCER_INDEXES, 'r') as f2:
    pancancer_genes = json.loads(f2.read())

# Transcription factors
print("Loading Transcription Factor genes.")
with open(config.DATAOUTPUT+config.TF_AND_PC_INDEXES, 'r') as f3:
    tf_and_pancancer_genes_index = json.loads(f3.read())

print("Length of the TF and Pancancer vector:",len(tf_and_pancancer_genes_index))

# Phenotypic data - clinical data
print("Loading phenodata of samples: ",config.PHENODATA_FILE)
dfpehno = pandas.read_feather(config.PATHDATA+config.PHENODATA_FILE)
#dfpehno = pandas.read_feather(PATHDATA+PHENODATA_FILE).set_index('index')
#print(dfpehno.columns)
#print(dfpehno.index)
print("Done.")

####### Loading the Data
# Raw Reads RNA-Seq
print("Loading RNA-Seq data file: ",config.RNASEQDATA_FILE)
dfsamples = pandas.read_feather(config.PATHDATA+config.RNASEQDATA_FILE).set_index('index')
if config.RNASEQDATA_IS_NORMALIZED==False:
    print("Nomalizing data to CPM.")
    dfsamples = convfunc.convert_rawreads_to_cpm(dfsamples)
else:
    print("Data is already normalized.")
dfsamples = dfsamples.loc[tf_and_pancancer_genes_index,:]

"""
print("Loading GDC data...")
dfsamples = pandas.read_feather(PATHDATA+RNASEQDATA_FILE).set_index('gene')
dfsamples = convfunc.convert_rawreads_to_cpm(dfsamples)
dfsamples.index = [item.split(".")[0] for item in dfsamples.index]
dfsamples = dfsamples.loc[tf_and_pancancer_genes_index,:]

print("Loading GTex data...")
gtex = pandas.pandas.read_feather(PATHDATA+GTEX_RNASEQDATA_FILE)
gtex = gtex.set_index("index")
gtex = convfunc.convert_rawreads_to_cpm(gtex)
gtex = gtex.loc[tf_and_pancancer_genes_index,:]

print("Merging GDC and GTex data...")
dfsamples = pandas.merge(gtex,dfsamples,left_index=True, right_index=True)
dfsamples = dfsamples.loc[:,validation_samples]
nfeatures = len(dfsamples.index)
print("The data has %i samples"%len(dfsamples.columns))
"""

encoder = convfunc.open_label_encoder(path=config.PATHDATA+config.LABELENCODER)

print("Creating bioimage... this may takes a while.")
X = convfunc.create_bioimage(dfsamples,pancancer_genes)
print("Predicting outcomes from data.")
Y_precicted = model.predict(numpy.array(X))
Y_precicted = numpy.argmax(Y_precicted, axis=1)
Y_precicted_label = encoder.inverse_transform(Y_precicted)
print(encoder.inverse_transform(Y_precicted))
"""
for i in range(0,len(Y_precicted_label)) :
    print("Sample:",dfpehno.iloc[i,:])
    print("Prediction:",Y_precicted_label[i])
    print("\n")
exit
"""
exit()
Y_observed = encoder.transform(dfpehno.loc[:,config.VARIABLE])
cnfMX = confusion_matrix(dfpehno.loc[:,config.VARIABLE],encoder.inverse_transform(Y_precicted))
cnfMX = pandas.DataFrame(cnfMX)
cnfMX.columns = encoder.classes_
cnfMX.indexes = encoder.classes_
print("Writing confusion matrix to file: ",config.CONFUSIONMATRIXFILE)

cnfMX.to_csv(config.DATAOUTPUT+config.CONFUSIONMATRIXFILE)
