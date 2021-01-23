from keras.models import load_model
import json
import os
import convolutionalkerasfunctionalapicorefunctions as convfunc
import pandas
from keras.utils import np_utils
import numpy
import csv
from sklearn.metrics import classification_report, confusion_matrix

#PATH="/home/oscar/Escritorio/keras_proyect/"
#PATH="/home/oscar/Desktop/keras_proyect/"
PATH="/lhome/ext/ibmc029/ibmc0292/keras_proyect/"
PATHDATA=PATH+"data/" # folder in which the data is stored
PATHBESTMODELS=PATH+"models/"
PATHTOINDEXES=PATH+"model_indexes/"
VARIABLE = "primary_site"
#VARIABLE = "meta_groups"
CONFUSIONMATRIXFILE = "confusion_matrix_primary_site_gtex_gdc.csv"
##################################
# SAVED FILES
DATAOUTPUT=PATH+"gdc_gtex_model/"

PHENODATA_FILE = "GDC.GTex.phenodatas.merged.feather"
#RNASEQDATA_FILE = "breast.metastasic.samples.RNAseq.feather"
RNASEQDATA_FILE = "dfsamples.GDC.15211.samples.clean1.57920.genes.deleted.less.5.reads.feather"
GTEX_RNASEQDATA_FILE = "gtex.raw.reads.formated.feather"
FILEMODEL = "DNN.best.model.by.acurracy.convolutional.GDC.GTEX.merged.hdf5"
FILEHISTORY = "model_history_log_GDC.GTEX.merged.csv"
LABELENCODER = "primary.site.gdc.gtex.label.encoder"
TESTSAMPLEFILE = 'test_sample_indexes_primary_site_merged.txt'
PANCANCER_INDEXES = 'gtex.gdc.pancancer.genes.index.sorted.txt'
TF_AND_PC_INDEXES = 'gtex.gdc.transcription.factor.TF.pancancer.genes.index.sorted.txt'

# load model
print("Loading DNN model...")
model = load_model(DATAOUTPUT+FILEMODEL)
# summarize model.
model.summary()

#Now read the file back into a Python list object
print("Loading Validation samples...")
with open(DATAOUTPUT+TESTSAMPLEFILE, 'r') as f:
    validation_samples = json.loads(f.read())

# Pancancer genes features
print("Loading Pancancer genes csv...")
with open(DATAOUTPUT+PANCANCER_INDEXES, 'r') as f2:
    pancancer_genes = json.loads(f2.read())

# Transcription factors
print("Loading Transcription Factor genes...")
with open(DATAOUTPUT+TF_AND_PC_INDEXES, 'r') as f3:
    tf_and_pancancer_genes_index = json.loads(f3.read())

print("Length of the TF and Pancancer vector:",len(tf_and_pancancer_genes_index))
# Phenotypic data - clinical data

print("Loading GDC and GTex phenodata...")
dfpehno = pandas.read_feather(PATHDATA+PHENODATA_FILE).set_index('file_name')
dfpehno = dfpehno.loc[validation_samples,:]
print("Done.")

####### Loading the Data
# Raw Reads RNA-Seq
print("Loading RNA-Seq data...")

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

original_data = dfsamples

for j in tf_and_pancancer_genes_index:
    tested_gene = j
    print("Testing gene: ",tested_gene)
    dfsamples = original_data
    dfsamples.loc[tested_gene,:] = 0
    GENECONFUSIONMATRIXFILE = tested_gene + '_' + CONFUSIONMATRIXFILE

    encoder = convfunc.open_label_encoder(path=PATHDATA+LABELENCODER)

    print("Creating bioimage... this may takes a while.")
    X = convfunc.create_bioimage(dfsamples,pancancer_genes)
    print("Predicting outcomes from data.")
    Y_precicted = model.predict(numpy.array(X))
    Y_precicted = numpy.argmax(Y_precicted, axis=1)
    Y_observed = encoder.transform(dfpehno.loc[:,VARIABLE])

    cnfMX = confusion_matrix(dfpehno.loc[:,VARIABLE],encoder.inverse_transform(Y_precicted))
    cnfMX = pandas.DataFrame(cnfMX)
    cnfMX.columns = encoder.classes_
    cnfMX.indexes = encoder.classes_
    print("Writing confusion matrix to file: ",GENECONFUSIONMATRIXFILE)
    cnfMX.to_csv(DATAOUTPUT+"feature_selection_genes/"+GENECONFUSIONMATRIXFILE)

print("Feature selection - Finished successfully.")