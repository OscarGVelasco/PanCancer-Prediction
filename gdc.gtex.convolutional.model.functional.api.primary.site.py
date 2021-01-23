import pandas
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from sklearn.pipeline import Pipeline
from matplotlib import pyplot
import numpy
from numpy import asarray
import os
import convolutionalkerasfunctionalapicorefunctions as convfunc
from DLNNdatagenerator import DataGenerator
import tensorflow.keras.backend as K
import tensorflow as tf
import time
import keras
import random
import json

# Import app configuration variables
from config import config


"""
# Model hyperparameters
N_CPUS=64
N_EPOCH = 300 # number of epochs
BATCH_SIZE = 30 # size of the batch
BATCH_SIZE_VALIDATION = 30
VERBOSE = 1 # show training phase messages and progress
OPTIMIZER = "adam" # model optimizer in use
LOSS_FUNCTION = 'categorical_crossentropy' # Loss function for model error measurement
TEST_SPLIT=0.7 # percentage of the data used for training
VALIDATION_SPLIT=0.3 # percentage of the training data used for validation during training
DROPOUT_RATE=0.4 # dropout rate - percentage of neurons to randomly drop and set to 0
#PATH="/home/oscar/Escritorio/keras_proyect/"
#PATH="/home/oscar/Desktop/keras_proyect/"
PATH="/lhome/ext/ibmc029/ibmc0292/keras_proyect/"
PATHDATA=PATH+"data/" # folder in which the data is stored
PATHBESTMODELS=PATH+"models/"
PATHTOINDEXES=PATH+"model_indexes/"
PHENODATA_FILE = "GDC.GTex.phenodatas.merged.feather"
RNASEQDATA_FILE = "dfsamples.GDC.15211.samples.clean1.57920.genes.deleted.less.5.reads.feather"
GTEX_RNASEQDATA_FILE = "gtex.raw.reads.formated.feather"
# SAVE FILES
DATAOUTPUT=PATH+"gdc_gtex_model_dimvalidation/"
FILEMODEL = "DNN.best.model.by.acurracy.convolutional.GDC.GTEX.merged.hdf5"
FILEHISTORY = "model_history_log_GDC.GTEX.merged.csv"
FILEENCODER = "primary.site.gdc.gtex.label.encoder"
##################################
"""
ncpus = convfunc.basic_config()

# Base Path
os.chdir(config.PATH)

# Phenotypic data - clinical data
print("Loading GDC phenodata...")
dfpehno = pandas.read_feather(config.PATHDATA+config.PHENODATA_FILE).set_index('file_name')
print("Done.")
a = dfpehno.index

####### Loading the Data
# Raw Reads RNA-Seq
print("Loading GDC data...")
dfsamples = pandas.read_feather(config.PATHDATA+config.RNASEQDATA_FILE).set_index('gene')
dfsamples = convfunc.convert_rawreads_to_cpm(dfsamples)
# ENSEMBL genes index are in forma: ENS_NNNN.1 : split by . and erase the .X part:
dfsamples.index = [item.split(".")[0] for item in dfsamples.index]
print("Done.")
# GTEX RNA-Seq data
print("Loading GTex data...")
gtex = pandas.pandas.read_feather(config.PATHDATA+config.GTEX_RNASEQDATA_FILE)
gtex = gtex.set_index("index")
#gtex.index = [item.split(".")[0] for item in gtex.index]
#del gtex["Description"]
gtex = convfunc.convert_rawreads_to_cpm(gtex)

print("Merging GDC and GTex data...")
dfsamples = pandas.merge(gtex,dfsamples,left_index=True, right_index=True)
b = dfsamples.columns
sample_indexes = list(set(set(a) & set(b)))
dfsamples = dfsamples.loc[:,sample_indexes]
dfpehno = dfpehno.loc[sample_indexes,:]
dfpehno = dfpehno.reindex(index = sample_indexes)
print("RNA-Seq Dataframe shape:",dfsamples.shape)
print("Phenodata Dataframe shape:",dfpehno.shape)
# Dataframes with the same features and same ordering:
nsamples = len(sample_indexes)
print("Done.")
print("Total number of samples available:",nsamples)

# Training and test labels
# Splitting dataset into: training (% training_split) and test (%test split)
# **We take into consideration the split for EACH class:
training_sample_indexes = list()
test_sample_indexes = list()
keys = list(set(dfpehno.primary_site))
for labels in keys:
    sample_indexes = dfpehno[dfpehno.primary_site == labels].index
    trainingSamples = int(round(len(sample_indexes)*config.TEST_SPLIT,0))
    j = list(random.sample(set(sample_indexes), trainingSamples))
    training_sample_indexes = training_sample_indexes + j
    test_sample_indexes = list(test_sample_indexes) + list(sample_indexes[~sample_indexes.isin(j)])

training_sample_indexes_boolean = dfpehno.index.isin(training_sample_indexes)
test_sample_indexes_boolean = ~dfpehno.index.isin(training_sample_indexes)


# Saving the test samples ids to a file:
with open(config.DATAOUTPUT+'test_sample_indexes_primary_site_merged.txt', 'w') as f:
    f.write(json.dumps(test_sample_indexes))

# Pancancer genes features
print("Loading Pancancer genes csv...")
with open(config.PATHTOINDEXES+'pancancer.genes.index.sorted.txt', 'r') as f2:
    pancancer_genes = json.loads(f2.read())

pancancer_genes = list(set(dfsamples.index) & set(pancancer_genes))
pancancer_genes = sorted(pancancer_genes)
# Saving the TF+pancancer genes ids INDEX to a file:
with open(config.DATAOUTPUT+config.PANCANCER_INDEXES, 'w') as f:
    f.write(json.dumps(pancancer_genes))

# Transcription factors
print("Loading Transcription Factor genes...")
with open(config.PATHTOINDEXES+config.TF_AND_PC_INDEXES, 'r') as f3:
    tf_and_pancancer_genes_index = json.loads(f3.read())

tf_and_pancancer_genes_index_common = list(set(dfsamples.index) & set(tf_and_pancancer_genes_index))
tf_and_pancancer_genes_index_common = set(tf_and_pancancer_genes_index_common)
tf_and_pancancer_genes_index_common = sorted(tf_and_pancancer_genes_index_common)
# Saving the TF+pancancer genes ids INDEX to a file:
with open(config.DATAOUTPUT+config.TF_AND_PC_INDEXES, 'w') as f:
    f.write(json.dumps(tf_and_pancancer_genes_index_common))

dfsamples = dfsamples[~dfsamples.index.duplicated(keep='first')]
dfsamples = dfsamples.loc[tf_and_pancancer_genes_index_common,:]
#dfsamples = dfsamples.reindex(index = tf_and_pancancer_genes_index_common)
nfeatures = len(dfsamples.index)
print("Using as rows and signal a number of -cancer driven- and -TFactors- genes of:",nfeatures)
print("Variable: tf_and_pancancer_genes_index_common has elements:",len(tf_and_pancancer_genes_index_common))
print("Done.")
#exit()

# Equal sortening with the outputs Y'
dfsamples_training = dfsamples.iloc[:,training_sample_indexes_boolean]
#dfsamples_training = dfsamples.reindex(columns = training_sample_indexes_boolean)
dfsamples_test = dfsamples.iloc[:,test_sample_indexes_boolean]
#dfsamples_test = dfsamples.reindex(columns = test_sample_indexes_boolean)
####### Model Variables Definition
# split into input (X) and output (Y) variables
# Input Variable - convert samples to bioimages using cancer-genes
# Create the output variables using the target labels
sample_indexes = dfpehno.index

Y_primary_disease = convfunc.convert_to_dummy(dfpehno.loc[sample_indexes,config.VARIABLE],save=1,path=config.PATHDATA+config.FILEENCODER)
Y_primary_disease_test = Y_primary_disease[test_sample_indexes_boolean]
Y_primary_disease = Y_primary_disease[training_sample_indexes_boolean]

# Layer model definition
noutputs_Y_primary_disease = Y_primary_disease.shape[1]

input_pixels = [nfeatures,len(pancancer_genes)+1,1]
print("Image input dimension in pixels:")
print(input_pixels)

# Calculating the equivalent megapixels and resolution of the bioimage
bio_megapixels = (input_pixels[0] * input_pixels[1])/1e6
print("The images to be processed have a bio-resolution of: %f Megapixels" %(bio_megapixels))

##########################
# Data Generators
print("Creating the data generator...")
# Primary Site generators
training_generator = DataGenerator(dfsamples_training, training_sample_indexes, Y_primary_disease,None, pancancer_genes, batch_size=config.BATCH_SIZE, shuffle=True)
test_generator = DataGenerator(dfsamples_test, test_sample_indexes, Y_primary_disease_test,None, pancancer_genes, batch_size=config.BATCH_SIZE_VALIDATION, shuffle=False,testdata=1)

""" 
# Pick a raw sample to convert to gray image:
x = dfsamples.iloc[:,1]
#convfunc.convert_to_bioimage(one_sample,pancancer_genes)
img = pandas.DataFrame((x - x.mean())/x.std(ddof=0))
for gene in pancancer_genes:
	img = pandas.merge(img,pandas.DataFrame((x - x.loc[gene,])/x.std(ddof=0)),left_index=True, right_index=True)
"""


print("Creating the DLNN model with Keras and TensorFlow...")
model = convfunc.primary_site_baseline_model_thiner(input_pixels=input_pixels,noutputs=noutputs_Y_primary_disease,loss_function = config.LOSS_FUNCTION,dropout_rate = config.DROPOUT_RATE)
#model = convfunc.tumor_type_baseline_model(input_pixels=input_pixels,noutputs=noutputs_Y_tissue_of_origin,loss_function = LOSS_FUNCTION,dropout_rate = DROPOUT_RATE)
#model = convfunc.multilabel_baseline_model(hyparameters_primary_disease=primary_disease,hyparameters_tissue_of_origin=tissue_of_origin, input_pixels=input_pixels,loss_function = LOSS_FUNCTION,dropout_rate = DROPOUT_RATE)
# summarize layers
print(model.summary())

# CALLBACKS to save automatically the best model we find during training:
print("Defining callbacks...")
print(" - Best model will be saved on file:",config.FILEMODEL)
checkpoint = ModelCheckpoint(config.DATAOUTPUT+config.FILEMODEL, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
print(" - History stats of training will be save to:",config.FILEHISTORY)
csv_logger = CSVLogger(config.DATAOUTPUT+config.FILEHISTORY, append=True)
callbacks_list = [checkpoint,csv_logger]


start = time.time()
# Training the Network with custom data generator- WARNING it could take a long time
model.fit(x=training_generator,
            #validation_data=(numpy.array(mytestdata),[Y_primary_disease_test,Y_tissue_of_origin_test]),
            #validation_data=(numpy.array(mytestdata),Y_primary_disease_test),
            validation_data=test_generator,
            use_multiprocessing=True,
            workers=config.N_CPUS,
            epochs=config.N_EPOCH,
            #validation_steps = len(test_sample_indexes),
            #validation_steps = 1,
            # // BATCH_SIZE_VALIDATION,
            callbacks=callbacks_list,
            max_queue_size=100
        )
end = time.time()
print('Total training time: %f seconds' % (end-start))
exit()

