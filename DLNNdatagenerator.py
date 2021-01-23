import numpy as np
import keras

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, signal_df, list_IDs, labels_y1, labels_y2=None, zscore_targets=None, batch_size=32,shuffle=True,testdata=0):
        'Initialization'
        self.batch_size = batch_size
        self.labels = labels_y1
        self.labels2 = labels_y2
        self.list_IDs = list_IDs
        self.signal_df = signal_df
        self.zscore_targets = zscore_targets
        self.shuffle = shuffle
        self.on_epoch_end()
        self.testdata = testdata
        self.multilab = 1
        if self.labels2 is None:
            self.multilab = 0

    def __len__(self):
        'Denotes the number of batches per epoch'
        if(self.testdata):
            #print("__len function executed...")
            return(int(np.floor(len(self.list_IDs) / self.batch_size)))
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
       
        X = list()
        # Find list of IDs
        #if self.testdata:
        #    list_IDs_temp = self.list_IDs
        #    indexes = np.arange(len(self.list_IDs))
        #else:
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y, y2 = self.__data_generation(list_IDs_temp,indexes)
        if self.multilab == 0:
            return np.array(X), y
        return np.array(X), [y, y2]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def create_bioimage(self,dfsamples,pancancer_genes):
        def f(x, y):
            return np.array( (x - y)/x.std(),dtype='float32')
        fv = np.vectorize(f,signature='(n),()->(n)')
        bioimages = list()
        for i,col in dfsamples.iteritems():
            colimage = np.transpose(fv(np.array(col),[col.mean()]+list(col[pancancer_genes])))
            colimage = np.reshape(colimage, colimage.shape + (1,))
            bioimages.append(colimage)
        return(bioimages)

    def __data_generation(self, list_IDs_temp, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        y = y2 = None
        # Generate data
        #for ID in list_IDs_temp:
        # Store sample
        X = self.create_bioimage(self.signal_df.iloc[:,indexes],self.zscore_targets)
        # Store class
        y = self.labels[indexes]
        if self.multilab != 0:
            y2 = self.labels2[indexes]
            return(X, y, y2)
        return(X, y, y2)

