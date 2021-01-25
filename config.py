class Config(object):
    def __init__(self):
        print('Loading config:')
        print('\t - Model hyperparameters')
        print('\t - Local paths')
        print('\t - File names')
        """ Model hyperparameters """
        self.N_CPUS=64
        self.N_EPOCH = 300 # number of epochs
        self.BATCH_SIZE = 30 # size of the batch
        self.BATCH_SIZE_VALIDATION = 30
        self.VERBOSE = 1 # show training phase messages and progress
        self.OPTIMIZER = "adam" # model optimizer in use
        self.LOSS_FUNCTION = 'categorical_crossentropy' # Loss function for model error measurement
        self.TEST_SPLIT=0.7 # percentage of the data used for training
        self.VALIDATION_SPLIT=0.3 # percentage of the training data used for validation during training
        self.DROPOUT_RATE=0.4 # dropout rate - percentage of neurons to randomly drop and set to 0
        
        """ Local Paths """
        self.PATH="/lhome/ext/ibmc029/ibmc0292/keras_proyect/"
        self.PATHDATA=self.PATH+"data/" # folder in which the data is stored
        self.PATHBESTMODELS=self.PATH+"gdc_gtex_model_dimvalidation/"
        self.PATHTOINDEXES=self.PATH+"model_indexes/"

        """ Data """
        # MAIN GDC and GTEx data
        self.PHENODATA_FILE = "GDC.GTex.phenodatas.merged.feather"
        self.GTEX_RNASEQDATA_FILE = "gtex.raw.reads.formated.feather"
        self.RNASEQDATA_FILE = "dfsamples.GDC.15211.samples.clean1.57920.genes.deleted.less.5.reads.feather"
        # - LUNG CANCER SAMPLES - metastasic and no metastasic
        #self.PHENODATA_FILE = "lung.cancer.GSE162945.pheno.feather"
        #self.RNASEQDATA_FILE = "lung.cancer.GSE162945.raw.feather"
        # - COLON CANCER SAMPLES - caucasian american - no metastasic
        #self.PHENODATA_FILE = "colon.caucasian.american.pheno.feather"
        #self.RNASEQDATA_FILE = "colon.caucasian.american.raw.feather"
        # - COLON CANCER SAMPLES - African american - no metastasic
        #self.PHENODATA_FILE = "colon.african.american.pheno.feather"
        #self.RNASEQDATA_FILE = "colon.african.american.raw.feather"
        # - SINGLE-CELL GASGTRIC CANCER SAMPLES - metastasic
        #self.PHENODATA_FILE = "singlecell.gastric.metastasis.pheno.feather"
        #self.RNASEQDATA_FILE = "singlecell.gastric.metastasis.raw.reads.feather"
        #self.RNASEQDATA_FILE = "singlecell.gastric.metastasis.GMPR.normalized.reads.feather"
        self.RNASEQDATA_IS_NORMALIZED = True

        """ File names """
        self.FILEMODEL = "DNN.best.model.by.acurracy.convolutional.GDC.GTEX.merged.hdf5"
        self.FILEHISTORY = "model_history_log_GDC.GTEX.merged.csv"
        self.FILEENCODER = self.LABELENCODER = "primary.site.gdc.gtex.label.encoder"
        self.TESTSAMPLEFILE = 'test_sample_indexes_primary_site_merged.txt'
        self.PANCANCER_INDEXES = 'gtex.gdc.pancancer.genes.index.sorted.txt'
        self.TF_AND_PC_INDEXES = 'gtex.gdc.transcription.factor.TF.pancancer.genes.index.sorted.txt'

        """ Model evaluation variables """
        self.VARIABLE = "primary_site"
        #self.VARIABLE = "meta_groups"
        self.CONFUSIONMATRIXFILE = "colon_caucasian_american_confusion_matrix_primary_site_gtex_gdc.csv"
        self.DATAOUTPUT=self.PATH+"gdc_gtex_model_dimvalidation/"

config = Config()