import pandas
import numpy
import pickle
from keras.layers import Input
from keras.models import Sequential
from keras.layers import Dense
from keras.models import Model
from keras.layers import Dropout
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import ZeroPadding2D
from keras.layers import Flatten
from keras.utils import plot_model
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers import PReLU
from keras.initializers import Constant
from keras import backend as K
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from time import sleep
import multiprocessing
from keras.optimizers import Adam
from keras.optimizers import SGD

# Basic Keras and Tensorflow initial config
def basic_config(floatpointer='float32',numgpus=0):
	print("**************************************************")
	print("  - D 	L 	n		N		*`--#")
	if(K.floatx() != floatpointer):
		K.set_floatx(floatpointer)
		print("Float pointer changed to: %s" %(floatpointer))
	else:
		print("Using numerical %s for the model." % (K.floatx()))
	try:
		print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
	except Exception as e:
		print(e)
	try:
		ncpus = len(tf.config.experimental.list_physical_devices('CPU'))
		print("Num CPUs Available: ", len(tf.config.experimental.list_physical_devices('CPU')))
		print("Num CPUs Available: with multiprocessing.cpu_count()")
		print(multiprocessing.cpu_count())
	except Exception as e:
		print(e)
	print("Initializing")
	print("**********************************************")
	sleep(3)
	return(ncpus)

# define base model
def baseline_model(layers,kernel_sizes,input_pixels,strides,loss_function,dropout_rate):
	# create model
	# Input Layer
	model_input = Input(shape=(input_pixels[0],input_pixels[1],1))
	
	# Hidden layers as specify by :layers:
	n_conv_hiddenlayers = len(kernel_sizes)-1
	
	modelHidden = model_input
	for i in range(0,n_conv_hiddenlayers):
		modelHidden = layer_model(modelHidden,layers[i],kernel_sizes[i],strides[i],dropout_rate)

	# fully-connected layer
	modelHidden = Flatten()(modelHidden)
	modelHidden = Dense(512, kernel_initializer='normal')(modelHidden)
	modelHidden = PReLU(alpha_initializer=Constant(value=0.25))(modelHidden)
	modelHidden = BatchNormalization()(modelHidden)
	modelHidden = Dropout(dropout_rate)(modelHidden)
	
	# Classification model:
	# Output layer: Softmax
	modelHidden = Dense(layers[len(layers)-1], activation='softmax')(modelHidden)
	# Compile model:
	model = Model(inputs = model_input,outputs = modelHidden)
	model.compile(loss = loss_function, optimizer='adam', metrics=['accuracy'])
	return model


def convert_to_bioimage(x,cancergenes):
	from sklearn.preprocessing import MinMaxScaler
	import matplotlib.pyplot as plt
	print("Converting transcriptomic data to a geyscale image using the pancancer genes.")
	print("Shape of the sample:",x.shape)
	#x = x.loc[cancergenes,].astype('float64')
	img = pandas.DataFrame((x - x.mean())/x.std(ddof=0))
	for gene in cancergenes:
		img = pandas.merge(img,pandas.DataFrame((x - x.loc[gene,])/x.std(ddof=0)),left_index=True, right_index=True)
	
	# Normalize between [0,255] as float64 to create a greyscale image
	scaler = MinMaxScaler(feature_range=(0,255))
	# Pick max and min of the entire dataframe and fit the scaler:
	x_sample = [img.values.flatten().min(), img.values.flatten().max()]
	scaler.fit(numpy.array(x_sample)[:, numpy.newaxis]) # reshape data to satisfy fit() method requirements
	# Scale dataframe using all the parameters:
	img = scaler.transform(img)
	try:
		plt.imshow(img, cmap='gray', vmin=0, vmax=255)
		plt.savefig("bioimage_greyscaled.png")
	except Exception as e:
		print(e)
	return img

def create_bioimage_deprecated(dfsamples,pancancer_genes):
	bioimages = []
	for col in dfsamples.columns:
		colimage = pandas.DataFrame((dfsamples[col] - dfsamples[col].mean())/dfsamples[col].std(ddof=0))
		for gene in pancancer_genes:
			colimage = pandas.merge(colimage,pandas.DataFrame((dfsamples[col] - dfsamples.loc[gene,col])/dfsamples[col].std(ddof=0)),left_index=True, right_index=True)
		colimage = numpy.reshape(numpy.array(colimage), colimage.shape + (1,))
		bioimages.append(colimage)
	return(bioimages)
## Vectorized version that is at least x20 faster
def create_bioimage(dfsamples,pancancer_genes):
	def f(x, y):
		return numpy.array( (x - y)/x.std(),dtype='float32')
	fv = numpy.vectorize(f,signature='(n),()->(n)')
	bioimages = list()
	for i,col in dfsamples.iteritems():
		colimage = numpy.transpose(fv(numpy.array(col),[col.mean()]+list(col[pancancer_genes])))
		colimage = numpy.reshape(colimage, colimage.shape + (1,))
		bioimages.append(colimage)
	return(bioimages)
# Converts samples raw reads to normalized Counts Per Million CPM
def convert_rawreads_to_cpm(sample):
	return((sample.apply(lambda x: pandas.to_numeric((x * 1e6) / sum(x), downcast='float'))).set_index(sample.index))
# Convert a discrete variable with classes to a dummy 0-1 encoded var
def convert_to_dummy(feature,save=0,path="label.encoder"):
	encoder = LabelEncoder()
	encoder.fit(feature)
	if save == 1:
		with open(path, 'wb') as f:
			pickle.dump(encoder, f)
	# Encoder transforms categorical labels to numeric features:
	encoded_Y = encoder.transform(feature)
	# Convert integers to dummy variables (i.e. one hot encoded as 0's and 1's) and return:
	return(np_utils.to_categorical(encoded_Y))

def open_label_encoder(path='label.encoder'):
	with open(path, 'rb') as f:
		encoder = pickle.load(f)
	return(encoder)

def primary_site_baseline_model(input_pixels,noutputs,loss_function,dropout_rate):
	
	#
	#	ACHIEVED A 95% ACCURACY (0.95000) with TF + CANCER GENES (cancer genes targeted z-score)
	#	2 days of training on 15600 sample: training: 0.75		test: 0.25
	#	Trained on 2 GPUs and 64 CPUs
	# Input Layer
	print("Creating the input layer with input shape:",input_pixels)
	model_input = Input(shape=(input_pixels[0],input_pixels[1],1))

	# Convolutional stuck of 32 kernels each:
	model_hidden_primary_disease = Conv2D(32, kernel_size = (3,3), padding="same", data_format = "channels_last", activation='relu')(model_input)
	model_hidden_primary_disease = Conv2D(32, kernel_size = (3,3), padding="same", data_format = "channels_last", activation='relu')(model_hidden_primary_disease)
	model_hidden_primary_disease = Conv2D(32, kernel_size = (3,3), padding="same", data_format = "channels_last", activation='relu')(model_hidden_primary_disease)
	model_hidden_primary_disease = MaxPool2D((2,2), strides=(2,2),data_format = "channels_last")(model_hidden_primary_disease)
	
	# Convolutional stuck of 64 kernels each:
	model_hidden_primary_disease = Conv2D(filters = 64, kernel_size = (3,3), padding="same", data_format = "channels_last", activation='relu')(model_hidden_primary_disease)
	model_hidden_primary_disease = Conv2D(filters = 64, kernel_size = (3,3), padding="same", data_format = "channels_last", activation='relu')(model_hidden_primary_disease)
	model_hidden_primary_disease = Conv2D(filters = 64, kernel_size = (3,3), padding="same", data_format = "channels_last", activation='relu')(model_hidden_primary_disease)
	model_hidden_primary_disease = MaxPool2D((2,2), strides=(2,2),data_format = "channels_last")(model_hidden_primary_disease)
	
	# Convolutional stuck of 128 kernels each:
	model_hidden_primary_disease = Conv2D(filters = 128, kernel_size = (3,3), padding="same", activation='relu')(model_hidden_primary_disease)
	model_hidden_primary_disease = Conv2D(filters = 128, kernel_size = (3,3), padding="same", activation='relu')(model_hidden_primary_disease)
	model_hidden_primary_disease = Conv2D(filters = 128, kernel_size = (3,3), padding="same", activation='relu')(model_hidden_primary_disease)
	model_hidden_primary_disease = MaxPool2D((2,2), strides=(2,2),data_format = "channels_last")(model_hidden_primary_disease)

	# Convolutional stuck of 256 kernels each:
	model_hidden_primary_disease = Conv2D(filters = 256, kernel_size = (3,3), padding="same", activation='relu')(model_hidden_primary_disease)
	model_hidden_primary_disease = Conv2D(filters = 256, kernel_size = (3,3), padding="same", activation='relu')(model_hidden_primary_disease)
	model_hidden_primary_disease = Conv2D(filters = 256, kernel_size = (3,3), padding="same", activation='relu')(model_hidden_primary_disease)
	model_hidden_primary_disease = MaxPool2D((2,2), strides=(2,2),data_format = "channels_last")(model_hidden_primary_disease)

	# Convolutional stuck of 512 kernels each:
	model_hidden_primary_disease = Conv2D(512, kernel_size = (3,3), padding="same", activation='relu')(model_hidden_primary_disease)
	model_hidden_primary_disease = Conv2D(512, kernel_size = (3,3), padding="same", activation='relu')(model_hidden_primary_disease)
	model_hidden_primary_disease = MaxPool2D((2,2), strides=(2,2),data_format = "channels_last")(model_hidden_primary_disease)

	# Fully-connected layers:
	model_hidden_primary_disease = Flatten()(model_hidden_primary_disease)
	model_hidden_primary_disease = Dense(2064, kernel_initializer='normal',activation="relu")(model_hidden_primary_disease)
	model_hidden_primary_disease = Dropout(dropout_rate)(model_hidden_primary_disease)
	model_hidden_primary_disease = Dense(664, kernel_initializer='normal',activation="relu")(model_hidden_primary_disease)
	model_hidden_primary_disease = Dropout(dropout_rate)(model_hidden_primary_disease)
	model_hidden_primary_disease = Dense(86, kernel_initializer='normal',activation="relu")(model_hidden_primary_disease)
	
	# Classification model:
	# Output layer: Softmax
	model_hidden_primary_disease = Dense(noutputs, activation='softmax')(model_hidden_primary_disease)

	print("Compiling the model")
	model = Model(inputs = model_input,outputs = model_hidden_primary_disease)
	optimiz = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss = loss_function, optimizer=optimiz, metrics=["accuracy"])
	return model


def primary_site_baseline_model_thiner(input_pixels,noutputs,loss_function,dropout_rate):
	
	# SIMPLER MODEL
	#
	# Input Layer
	print("Creating the input layer with input shape:",input_pixels)
	model_input = Input(shape=(input_pixels[0],input_pixels[1],1))

	# Convolutional stuck of 32 kernels each:
	model_hidden_primary_disease = Conv2D(32, kernel_size = (3,3), padding="same", data_format = "channels_last", activation='relu')(model_input)
	model_hidden_primary_disease = Conv2D(32, kernel_size = (3,3), padding="same", data_format = "channels_last", activation='relu')(model_hidden_primary_disease)
	#model_hidden_primary_disease = Conv2D(32, kernel_size = (3,3), padding="same", data_format = "channels_last", activation='relu')(model_hidden_primary_disease)
	model_hidden_primary_disease = MaxPool2D((2,2), strides=(2,2),data_format = "channels_last")(model_hidden_primary_disease)
	
	# Convolutional stuck of 64 kernels each:
	model_hidden_primary_disease = Conv2D(filters = 64, kernel_size = (3,3), padding="same", data_format = "channels_last", activation='relu')(model_hidden_primary_disease)
	model_hidden_primary_disease = Conv2D(filters = 64, kernel_size = (3,3), padding="same", data_format = "channels_last", activation='relu')(model_hidden_primary_disease)
	#model_hidden_primary_disease = Conv2D(filters = 64, kernel_size = (3,3), padding="same", data_format = "channels_last", activation='relu')(model_hidden_primary_disease)
	model_hidden_primary_disease = MaxPool2D((2,2), strides=(2,2),data_format = "channels_last")(model_hidden_primary_disease)
	
	# Convolutional stuck of 128 kernels each:
	model_hidden_primary_disease = Conv2D(filters = 128, kernel_size = (3,3), padding="same", activation='relu')(model_hidden_primary_disease)
	model_hidden_primary_disease = Conv2D(filters = 128, kernel_size = (3,3), padding="same", activation='relu')(model_hidden_primary_disease)
	#model_hidden_primary_disease = Conv2D(filters = 128, kernel_size = (3,3), padding="same", activation='relu')(model_hidden_primary_disease)
	model_hidden_primary_disease = MaxPool2D((2,2), strides=(2,2),data_format = "channels_last")(model_hidden_primary_disease)

	# Convolutional stuck of 256 kernels each:
	model_hidden_primary_disease = Conv2D(filters = 256, kernel_size = (3,3), padding="same", activation='relu')(model_hidden_primary_disease)
	model_hidden_primary_disease = Conv2D(filters = 256, kernel_size = (3,3), padding="same", activation='relu')(model_hidden_primary_disease)
	#model_hidden_primary_disease = Conv2D(filters = 256, kernel_size = (3,3), padding="same", activation='relu')(model_hidden_primary_disease)
	model_hidden_primary_disease = MaxPool2D((2,2), strides=(2,2),data_format = "channels_last")(model_hidden_primary_disease)

	# Convolutional stuck of 512 kernels each:
	model_hidden_primary_disease = Conv2D(512, kernel_size = (3,3), padding="same", activation='relu')(model_hidden_primary_disease)
	#model_hidden_primary_disease = Conv2D(512, kernel_size = (3,3), padding="same", activation='relu')(model_hidden_primary_disease)
	model_hidden_primary_disease = MaxPool2D((2,2), strides=(2,2),data_format = "channels_last")(model_hidden_primary_disease)

	# Fully-connected layers:
	model_hidden_primary_disease = Flatten()(model_hidden_primary_disease)
	model_hidden_primary_disease = Dense(2064, kernel_initializer='normal',activation="relu")(model_hidden_primary_disease)
	model_hidden_primary_disease = Dropout(dropout_rate)(model_hidden_primary_disease)
	model_hidden_primary_disease = Dense(664, kernel_initializer='normal',activation="relu")(model_hidden_primary_disease)
	model_hidden_primary_disease = Dropout(dropout_rate)(model_hidden_primary_disease)
	model_hidden_primary_disease = Dense(86, kernel_initializer='normal',activation="relu")(model_hidden_primary_disease)
	
	# Classification model:
	# Output layer: Softmax
	model_hidden_primary_disease = Dense(noutputs, activation='softmax')(model_hidden_primary_disease)

	print("Compiling the model")
	model = Model(inputs = model_input,outputs = model_hidden_primary_disease)
	optimiz = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss = loss_function, optimizer=optimiz, metrics=["accuracy"])
	return model


def metagroups_baseline_model(input_pixels,noutputs,loss_function,dropout_rate):
	
	# Input Layer
	print("Creating the input layer with input shape:",input_pixels)
	model_input = Input(shape=(input_pixels[0],input_pixels[1],1))

	# Convolutional stuck of 32 kernels each:
	model_hidden_primary_disease = Conv2D(32, kernel_size = (3,3), padding="same", data_format = "channels_last", activation='relu')(model_input)
	model_hidden_primary_disease = Conv2D(32, kernel_size = (3,3), padding="same", data_format = "channels_last", activation='relu')(model_hidden_primary_disease)
	model_hidden_primary_disease = MaxPool2D((2,2), strides=(2,2),data_format = "channels_last")(model_hidden_primary_disease)
	
	# Convolutional stuck of 64 kernels each:
	model_hidden_primary_disease = Conv2D(filters = 64, kernel_size = (3,3), padding="same", data_format = "channels_last", activation='relu')(model_hidden_primary_disease)
	model_hidden_primary_disease = Conv2D(filters = 64, kernel_size = (3,3), padding="same", data_format = "channels_last", activation='relu')(model_hidden_primary_disease)
	model_hidden_primary_disease = MaxPool2D((2,2), strides=(2,2),data_format = "channels_last")(model_hidden_primary_disease)
	
	# Convolutional stuck of 128 kernels each:
	model_hidden_primary_disease = Conv2D(filters = 128, kernel_size = (3,3), padding="same", activation='relu')(model_hidden_primary_disease)
	model_hidden_primary_disease = Conv2D(filters = 128, kernel_size = (3,3), padding="same", activation='relu')(model_hidden_primary_disease)
	model_hidden_primary_disease = MaxPool2D((2,2), strides=(2,2),data_format = "channels_last")(model_hidden_primary_disease)

	# Convolutional stuck of 256 kernels each:
	model_hidden_primary_disease = Conv2D(filters = 256, kernel_size = (3,3), padding="same", activation='relu')(model_hidden_primary_disease)
	model_hidden_primary_disease = MaxPool2D((2,2), strides=(2,2),data_format = "channels_last")(model_hidden_primary_disease)

	# Convolutional stuck of 512 kernels each:
	model_hidden_primary_disease = Conv2D(512, kernel_size = (3,3), padding="same", activation='relu')(model_hidden_primary_disease)
	model_hidden_primary_disease = MaxPool2D((2,2), strides=(2,2),data_format = "channels_last")(model_hidden_primary_disease)

	# Fully-connected layers:
	model_hidden_primary_disease = Flatten()(model_hidden_primary_disease)
	model_hidden_primary_disease = Dense(2064, kernel_initializer='normal',activation="relu")(model_hidden_primary_disease)
	model_hidden_primary_disease = Dropout(dropout_rate)(model_hidden_primary_disease)
	model_hidden_primary_disease = Dense(664, kernel_initializer='normal',activation="relu")(model_hidden_primary_disease)
	model_hidden_primary_disease = Dropout(dropout_rate)(model_hidden_primary_disease)
	model_hidden_primary_disease = Dense(46, kernel_initializer='normal',activation="relu")(model_hidden_primary_disease)
	
	# Classification model:
	# Output layer: Softmax
	model_hidden_primary_disease = Dense(noutputs, activation='softmax')(model_hidden_primary_disease)

	print("Compiling the model")
	model = Model(inputs = model_input,outputs = model_hidden_primary_disease)
	optimiz = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss = loss_function, optimizer=optimiz, metrics=["accuracy"])
	return model


def tumor_type_baseline_model(input_pixels,noutputs,loss_function,dropout_rate):
	
	# Tumor type: solid, metastasic, bone marrow, blood, normal tissue...
	#
	# Input Layer
	print("Creating the input layer with input shape:",input_pixels)
	model_input = Input(shape=(input_pixels[0],input_pixels[1],1))

	# Convolutional stuck of 32 kernels each:
	model_hidden = Conv2D(32, kernel_size = (3,3), padding="same", data_format = "channels_last", activation='relu')(model_input)
	model_hidden = Conv2D(32, kernel_size = (3,3), padding="same", data_format = "channels_last", activation='relu')(model_hidden)
	model_hidden = MaxPool2D((2,2), strides=(2,2),data_format = "channels_last")(model_hidden)
	
	# Convolutional stuck of 64 kernels each:
	model_hidden = Conv2D(filters = 64, kernel_size = (3,3), padding="same", data_format = "channels_last", activation='relu')(model_hidden)
	#model_hidden = Conv2D(filters = 64, kernel_size = (3,3), padding="same", data_format = "channels_last", activation='relu')(model_hidden)
	model_hidden = MaxPool2D((2,2), strides=(2,2),data_format = "channels_last")(model_hidden)
	
	# Convolutional stuck of 128 kernels each:
	model_hidden = Conv2D(filters = 128, kernel_size = (3,3), padding="same", activation='relu')(model_hidden)
	#model_hidden = Conv2D(filters = 128, kernel_size = (3,3), padding="same", activation='relu')(model_hidden)
	model_hidden = MaxPool2D((2,2), strides=(2,2),data_format = "channels_last")(model_hidden)

	# Convolutional stuck of 256 kernels each:
	model_hidden = Conv2D(filters = 256, kernel_size = (3,3), padding="same", activation='relu')(model_hidden)
	#model_hidden = Conv2D(filters = 256, kernel_size = (3,3), padding="same", activation='relu')(model_hidden)
	model_hidden = MaxPool2D((2,2), strides=(2,2),data_format = "channels_last")(model_hidden)

	# Convolutional stuck of 512 kernels each:
	#model_hidden = Conv2D(512, kernel_size = (3,3), padding="same", activation='relu')(model_hidden)
	#model_hidden_primary_disease = Conv2D(512, kernel_size = (3,3), padding="same", activation='relu')(model_hidden_primary_disease)
	#model_hidden = MaxPool2D((2,2), strides=(2,2),data_format = "channels_last")(model_hidden)

	# Fully-connected layers:
	model_hidden = Flatten()(model_hidden)
	model_hidden = Dense(1064, kernel_initializer='normal',activation="relu")(model_hidden)
	model_hidden = Dropout(dropout_rate)(model_hidden)
	model_hidden = Dense(664, kernel_initializer='normal',activation="relu")(model_hidden)
	model_hidden = Dropout(dropout_rate)(model_hidden)
	model_hidden = Dense(36, kernel_initializer='normal',activation="relu")(model_hidden)
	
	# Classification model:
	# Output layer: Softmax
	model_hidden = Dense(noutputs, activation='softmax')(model_hidden)

	print("Compiling the model")
	model = Model(inputs = model_input,outputs = model_hidden)
	optimiz = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss = loss_function, optimizer=optimiz, metrics=["accuracy"])
	return model



# define base model for multi label classification
def multilabel_baseline_model(hyparameters_primary_disease,hyparameters_tissue_of_origin,input_pixels,loss_function,dropout_rate):
	# create model
	layers_primary_disease = hyparameters_primary_disease[0]
	kernel_primary_disease = hyparameters_primary_disease[1]
	strides_primary_disease = hyparameters_primary_disease[2]
	
	layers_tissue_of_origin = hyparameters_tissue_of_origin[0]
	kernel_tissue_of_origin = hyparameters_tissue_of_origin[1]
	strides_tissue_of_origin = hyparameters_tissue_of_origin[2]

	# Input Layer
	print("Creating the input shared layer...")
	model_input = Input(shape=(input_pixels[0],input_pixels[1],1))
	# shared_model = Conv2D(64, (7, 7),strides=3, padding="same",kernel_initializer="he_normal", kernel_regularizer=l2(0.0005))(model_input)

	# Primary Disease Model #
	print("Building Primary Disease model")
	# Hidden layers as specify by :layers:
	model_hidden_primary_disease = model_input

	model_hidden_primary_disease = Conv2D(32, kernel_size = (3,3), padding="same", data_format = "channels_last", activation='relu')(model_hidden_primary_disease)
	model_hidden_primary_disease = Conv2D(32, kernel_size = (3,3), padding="same", data_format = "channels_last", activation='relu')(model_hidden_primary_disease)
	model_hidden_primary_disease = Conv2D(32, kernel_size = (3,3), padding="same", data_format = "channels_last", activation='relu')(model_hidden_primary_disease)
	model_hidden_primary_disease = MaxPool2D((2,2), strides=(2,2),data_format = "channels_last")(model_hidden_primary_disease)
	#model_hidden_primary_disease = ZeroPadding2D((1,1), data_format = "channels_last")(model_input)
	
	model_hidden_primary_disease = Conv2D(filters = 64, kernel_size = (3,3), padding="same", data_format = "channels_last", activation='relu')(model_hidden_primary_disease)
	model_hidden_primary_disease = Conv2D(filters = 64, kernel_size = (3,3), padding="same", data_format = "channels_last", activation='relu')(model_hidden_primary_disease)
	#model_hidden_primary_disease = ZeroPadding2D((1,1),data_format = "channels_last")(model_hidden_primary_disease)
	model_hidden_primary_disease = Conv2D(64, kernel_size = (3,3), padding="same", data_format = "channels_last", activation='relu')(model_hidden_primary_disease)
	model_hidden_primary_disease = MaxPool2D((2,2), strides=(2,2),data_format = "channels_last")(model_hidden_primary_disease)
	
	#model_hidden_primary_disease = ZeroPadding2D((1,1),data_format = "channels_last")(model_hidden_primary_disease)
	model_hidden_primary_disease = Conv2D(128, kernel_size = (3,3), padding="same", activation='relu')(model_hidden_primary_disease)
	model_hidden_primary_disease = Conv2D(128, kernel_size = (3,3), padding="same", activation='relu')(model_hidden_primary_disease)
	#model_hidden_primary_disease = ZeroPadding2D((1,1),data_format = "channels_last")(model_hidden_primary_disease)
	model_hidden_primary_disease = Conv2D(128, kernel_size = (3,3), padding="same", activation='relu')(model_hidden_primary_disease)
	model_hidden_primary_disease = MaxPool2D((2,2), strides=(2,2),data_format = "channels_last")(model_hidden_primary_disease)

	#model_hidden_primary_disease = ZeroPadding2D((1,1),data_format = "channels_last")(model_hidden_primary_disease)
	model_hidden_primary_disease = Conv2D(256, kernel_size = (3,3), padding="same", activation='relu')(model_hidden_primary_disease)
	model_hidden_primary_disease = Conv2D(256, kernel_size = (3,3), padding="same", activation='relu')(model_hidden_primary_disease)
	#model_hidden_primary_disease = ZeroPadding2D((1,1),data_format = "channels_last")(model_hidden_primary_disease)
	model_hidden_primary_disease = Conv2D(256, kernel_size = (3,3), padding="same", activation='relu')(model_hidden_primary_disease)
	model_hidden_primary_disease = MaxPool2D((2,2), strides=(2,2),data_format = "channels_last")(model_hidden_primary_disease)

	#model_hidden_primary_disease = ZeroPadding2D((1,1),data_format = "channels_last")(model_hidden_primary_disease)
	model_hidden_primary_disease = Conv2D(512, kernel_size = (3,3), padding="same", activation='relu')(model_hidden_primary_disease)
	model_hidden_primary_disease = Conv2D(512, kernel_size = (3,3), padding="same", activation='relu')(model_hidden_primary_disease)
	model_hidden_primary_disease = MaxPool2D((2,2), strides=(2,2),data_format = "channels_last")(model_hidden_primary_disease)

	#for i in range(0,len(kernel_primary_disease)-1):
	#model_hidden_primary_disease = Conv2D(32, (7, 7),strides=3, padding="same",kernel_initializer="he_normal", kernel_regularizer=l2(0.0005))(model_input)
	#model_hidden_primary_disease = layer_model(model_hidden_primary_disease,64,7,3,dropout_rate)
	#model_hidden_primary_disease = layer_model(model_hidden_primary_disease,64,5,1,dropout_rate)
	i=0
	#model_hidden_primary_disease = layer_model(model_hidden_primary_disease,layers_primary_disease[i],kernel_primary_disease[i],strides_primary_disease[i],dropout_rate)
	i=1
	#model_hidden_primary_disease = layer_clean_model(model_hidden_primary_disease,layers_primary_disease[i],kernel_primary_disease[i],strides_primary_disease[i],dropout_rate)	
	#i=2
	#model_hidden_primary_disease = layer_model(model_hidden_primary_disease,layers_primary_disease[i],kernel_primary_disease[i],strides_primary_disease[i],dropout_rate)

	# fully-connected layer
	model_hidden_primary_disease = Flatten()(model_hidden_primary_disease)
	model_hidden_primary_disease = Dense(2064, kernel_initializer='normal',activation="relu")(model_hidden_primary_disease)
	#model_hidden_primary_disease = PReLU(alpha_initializer=Constant(value=0.25))(model_hidden_primary_disease)
	model_hidden_primary_disease = Dropout(dropout_rate)(model_hidden_primary_disease)
	model_hidden_primary_disease = Dense(664, kernel_initializer='normal',activation="relu")(model_hidden_primary_disease)
	#model_hidden_primary_disease = PReLU(alpha_initializer=Constant(value=0.25))(model_hidden_primary_disease)
	model_hidden_primary_disease = Dropout(dropout_rate)(model_hidden_primary_disease)
	#model_hidden_primary_disease = BatchNormalization()(model_hidden_primary_disease)
	#model_hidden_primary_disease = Dropout(dropout_rate)(model_hidden_primary_disease)	
	model_hidden_primary_disease = Dense(86, kernel_initializer='normal',activation="relu")(model_hidden_primary_disease)
	#model_hidden_primary_disease = Dense(40, kernel_initializer='normal')(model_hidden_primary_disease)
	#model_hidden_primary_disease = PReLU(alpha_initializer=Constant(value=0.25))(model_hidden_primary_disease)
	#model_hidden_primary_disease = BatchNormalization()(model_hidden_primary_disease)
	#model_hidden_primary_disease = Dropout(dropout_rate)(model_hidden_primary_disease)	

	# Classification model:
	# Output layer: Softmax
	model_hidden_primary_disease = Dense(layers_primary_disease[len(layers_primary_disease)-1], activation='softmax')(model_hidden_primary_disease)
	# model_hidden_primary_disease.name = "primary_disease"

	# Primary Disease Model #
	print("Building Tissue type model")
	# Hidden layers as specify by :layers:
	model_hidden_tissue_of_origin = model_input
	#for i in range(0,len(kernel_tissue_of_origin)-1):
	model_hidden_tissue_of_origin = Conv2D(64, (7, 7),strides=3, padding="same",kernel_initializer="he_normal", kernel_regularizer=l2(0.0005))(model_hidden_tissue_of_origin)
	#model_hidden_tissue_of_origin = layer_model(model_hidden_tissue_of_origin,64,7,3,dropout_rate)
	i=0
	model_hidden_tissue_of_origin = layer_model(model_hidden_tissue_of_origin,layers_tissue_of_origin[i],kernel_tissue_of_origin[i],strides_tissue_of_origin[i],dropout_rate)
	#i=1
	#model_hidden_tissue_of_origin = layer_clean_model(model_hidden_tissue_of_origin,layers_tissue_of_origin[i],kernel_tissue_of_origin[i],strides_tissue_of_origin[i],dropout_rate)
	i=2
	model_hidden_tissue_of_origin = layer_model(model_hidden_tissue_of_origin,layers_tissue_of_origin[i],kernel_tissue_of_origin[i],strides_tissue_of_origin[i],dropout_rate)
	
	# fully-connected layer
	model_hidden_tissue_of_origin = Flatten()(model_hidden_tissue_of_origin)
	# model_hidden_tissue_of_origin = Dense(1024, kernel_initializer='normal')(model_hidden_tissue_of_origin)
	# model_hidden_tissue_of_origin = PReLU(alpha_initializer=Constant(value=0.25))(model_hidden_tissue_of_origin)
	# model_hidden_tissue_of_origin = BatchNormalization()(model_hidden_tissue_of_origin)
	model_hidden_tissue_of_origin = Dense(300, kernel_initializer='normal')(model_hidden_tissue_of_origin)
	model_hidden_tissue_of_origin = PReLU(alpha_initializer=Constant(value=0.25))(model_hidden_tissue_of_origin)
	model_hidden_tissue_of_origin = Dense(50, kernel_initializer='normal')(model_hidden_tissue_of_origin)
	model_hidden_tissue_of_origin = PReLU(alpha_initializer=Constant(value=0.25))(model_hidden_tissue_of_origin)
	model_hidden_tissue_of_origin = BatchNormalization()(model_hidden_tissue_of_origin)
	model_hidden_tissue_of_origin = Dropout(dropout_rate)(model_hidden_tissue_of_origin)	
	model_hidden_tissue_of_origin = Dense(30, kernel_initializer='normal')(model_hidden_tissue_of_origin)
	model_hidden_tissue_of_origin = PReLU(alpha_initializer=Constant(value=0.25))(model_hidden_tissue_of_origin)
	model_hidden_tissue_of_origin = BatchNormalization()(model_hidden_tissue_of_origin)
	model_hidden_tissue_of_origin = Dropout(dropout_rate)(model_hidden_tissue_of_origin)	
	# Classification model:
	# Output layer: Softmax
	model_hidden_tissue_of_origin = Dense(layers_tissue_of_origin[len(layers_tissue_of_origin)-1], activation='softmax')(model_hidden_tissue_of_origin)
	# model_hidden_tissue_of_origin.name = "tissue_of_origin"

	# Compile model:
	print("Merging and compiling the model")
	#model = Model(inputs = model_input,outputs = [model_hidden_primary_disease,model_hidden_tissue_of_origin])
	model = Model(inputs = model_input,outputs = model_hidden_primary_disease)
	metrics={'model_hidden_primary_disease': 'accuracy','model_hidden_tissue_of_origin': 'accuracy'}
	optimiz = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
	#optimiz = Adam(lr=0.0001)
	model.compile(loss = loss_function, optimizer=optimiz, metrics=["accuracy"])
	return model

def layer_model(model,n_neuron,kernel_shape,strides,dropout_rate):
	# Hidden Layer
	# stack CONV layers, keeping the size of each filter
	model = Conv2D(n_neuron, (kernel_shape, kernel_shape),strides=strides, padding="same",kernel_initializer="he_normal", kernel_regularizer=l2(0.0005))(model)
	model = BatchNormalization()(model)
	model = PReLU(alpha_initializer=Constant(value=0.25))(model)
	model = Dropout(dropout_rate)(model)
	return model

def layer_clean_model(model,n_neuron,kernel_shape,strides,dropout_rate):
	# Hidden Layer
	# stack CONV layers, keeping the size of each filter
	model = Conv2D(n_neuron, (kernel_shape, kernel_shape),strides=strides, padding="same",kernel_initializer="he_normal", kernel_regularizer=l2(0.0005))(model)
	# model = BatchNormalization()(model)
	# model = PReLU(alpha_initializer=Constant(value=0.25))(model)
	# model = Dropout(dropout_rate)(model)
	return model