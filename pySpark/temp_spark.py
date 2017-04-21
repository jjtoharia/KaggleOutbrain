# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal
"""

# 
# Esto en [pyspark | GoogleCloud] NO hace falta (ya hay una seasión de spark lanzada y un sparkContext creado):
# C:\Archivos de programa\Google\Cloud SDK>gcloud  compute  instances  start  cluster-jjtzapata-m  cluster-jjtzapata-w-0  cluster-jjtzapata-w-1  --zone europe-west1-d
# 
from pyspark.sql import SparkSession
miSparkSession = SparkSession \
                .builder \
                .appName("Spark-Outbrain-JJTZ") \
                .config("spark.some.config.option", "some-value") \
                .getOrCreate()
sc = miSparkSession.sparkContext
#SparkSession.builder.master("local[*]").appName("Outbrain-JJTZ2").getOrCreate()
# miSparkSession.stop()
# sc.stop()
#from pyspark import SparkConf, SparkContext
#conf = SparkConf().setMaster("local").setAppName("Outbrain-JJTZ")
#miSparkContext = SparkContext(conf = conf)

#from pyspark.sql.types import StringType
#from pyspark import SQLContext
#sqlContext = SQLContext(miSparkContext)

# 
# CARGAMOS DATOS:
# 
s_input_path = "C:/Users/jtoharia/Downloads/Kaggle_Outbrain/"
s_output_path = "C:/Users/jtoharia/Dropbox/AFI_JOSE/Kaggle/Outbrain/"

#f = sc.textFile(s_input_path + "clicks_train_spark.csv")      # 87.141.732 resgitros
f = sc.textFile(s_input_path + "clicks_train_debug_spark.csv") #     54.348 registros
f = sc.textFile(s_output_path + "clicks_train_debug_spark.csv") #     54.348 registros
f = sc.textFile("gs://jjtzapata/clicks_train_debug_spark.csv") #     54.348 registros
f = sc.textFile("/home/jjtzapata/clicks_train_debug_spark.csv") #     54.348 registros
# # NOTA: Para copiar a la máquina de gcloud (cuidado que lo copia a otro usuarioque no es jjtoharia, seguramente /home/jjtzapata!):
# gcloud compute copy-files "C:\Personal\Dropbox\AFI_JOSE\Kaggle\Outbrain\prueba.libsvm" cluster-jjtzapata-m: --zone europe-west1-d
# # NOTA: Para copiar al Google Storage gs://jjtzapata
# gsutil cp "C:\Personal\Dropbox\AFI_JOSE\Kaggle\Outbrain\prueba.libsvm" gs://jjtzapata
# gsutil cp "C:\Personal\Dropbox\AFI_JOSE\Kaggle\Outbrain\clicks_train_debug_spark.csv" gs://jjtzapata
# # Instancias (máquinas, clusters) Google Cloud Dataproc:
# # Para ver la IP externa: gcloud compute instances list
# gcloud compute instances start cluster-jjtzapata-m --zone europe-west1-d

# f.cache()
#f.count() # Tarda mucho! (6 min) 87.141.732
#Remove the first line (contains headers)
cabecera = f.first()
f = f.filter(lambda x: x != cabecera).map(lambda lin: lin.replace("\"","").replace("'","").split(","))
#f.count() # Tarda mucho! (6 min) 87.141.731
#f.take(1)

campos_enteros = ['display_id', 'ad_document_id', 'document_id', 'ad_id', 'clicked', 'numAds', 'platform', 'hora', 'dia', 'ad_campaign_id', 'ad_advertiser_id', 'source_id', 'publisher_id', 'ad_source_id', 'ad_publisher_id', 'pais_US', 'pais_GB' ,'pais_CA' ,'pais_resto']
campos_string =  ['uuid']  # Eliminados: 'geo_location', 'geo_loc.country', 'pais', 'publish_time', 'ad_publish_time'
                           # NOTA: Eliminado 'uuid' también (de clicks_train_debug_spark.csv)

from pyspark.sql.types import StringType, IntegerType, FloatType, StructField, StructType
def mi_estructura(nombre_campo):
    if(nombre_campo in campos_enteros):
        return(StructField(nombre_campo, IntegerType(), True))
    elif(nombre_campo in campos_string):
        return(StructField(nombre_campo, StringType(), True))
    else:
        return(StructField(nombre_campo, FloatType(), True))

campos = [mi_estructura(fld_name) for fld_name in cabecera.split(",")]
estructura = StructType(campos)

# toDF() NO FUNCIONA PORQUE LOS TIPOS NO COINCIDEN (?) full_trainset = f.toDF(estructura)
# ASÍ QUE LEEMOS DE NUEVO EL CSV, PERO AHORA CON LA ESTRUCTURA (SCHEMA):
full_trainset = spark.read.csv("gs://jjtzapata/clicks_train_debug_spark.csv", schema = estructura, header = True, mode = "DROPMALFORMED")

#full_trainset.createOrReplaceTempView("full_trainset")
#full_trainset.take(2)
#full_trainset.describe().show()

# 
# FIND CORRELATION BETWEEN PREDICTORS AND TARGET:
# 
for i in full_trainset.columns:
    if not( isinstance(full_trainset.select(i).take(1)[0][0], str) | isinstance(full_trainset.select(i).take(1)[0][0], unicode) ) :
        p = full_trainset.stat.corr("clicked",i)
        if(p > 0.5):
            print( "Correlation to OUTCOME (clicked) for ", i, p)

# 
# SELECCIONAMOS VARIABLES:
# 
from pyspark.ml.linalg import Vectors
def transformToLabeledPoint(row) :
    lp = ( row["clicked"], \
            Vectors.dense([
                row["numAds"], \
                row["timestamp"], \
                row["topics_prob"], \
                row["ad_topics_prob"], \
                row["entities_prob"], \
                row["ad_entities_prob"], \
                row["categories_prob"], \
                row["ad_categories_prob"]
        ]))
    return lp

train_lp = full_trainset.rdd.map(transformToLabeledPoint)
#train_lp.collect()[:5]
train_df = spark.createDataFrame(train_lp, ["label", "features"])# miSparkSession.createDataFrame(train_lp, ["label", "features"])
#train_df.select("label","features").show(10)

# 
# PCA (PRINCIPAL COMPONENTS):
# 
from pyspark.ml.feature import PCA
numComps = 3
bankPCA = PCA(k=numComps, inputCol="features", outputCol="pcaFeatures") # Nos quedamos con las 3 primeras componentes principales
pcaModel = bankPCA.fit(train_df)
pcaResult = pcaModel.transform(train_df).select("label","pcaFeatures")
pcaResult.show(truncate=False)

#### Hasta aquí todo bien (en Google Cloud Dataproc)!
# Para conectarse al Linux de Google Cloud Dataproc:
# - Instalar Google Cloud SDK o mejor usar la web (google cloud console) o usar Kitty (coñazo crear ssh keys, etc.)
# - abrimos Spark-Python (pyspark)
# - Ya está ("miSparkSession" es "spark" y "sc" es "sc")

# Para usar XGBoost, hay que instalarlo:
# 1.- en la consola de Linux: [ssh cluster-jjtzapata-m.europe-west1-d.evident-galaxy-150614]
#   git clone --recursive https://github.com/dmlc/xgboost
#   cd xgboost/
#   make -j4
#   sudo apt-get install python-setuptools
#   [Instalar NumPy, SciPy, etc. (TARDA UN HUEVO):] sudo apt-get install python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose
#   cd python-package
#   sudo python setup.py install
#
#   hadoop fs -copyFromLocal /home/jjtzapata/trainset.libsvm
#
import xgboost as xgb
dtrain = xgb.DMatrix("/home/jjtzapata/trainset.libsvm#dtrain.cache")
# NOTA "#dtrain.cache" es para la versión con caché de disco, para ficheros "GRANDES"...
#dtrain = xgb.DMatrix("hdfs:///trainset.libsvm/#dtrain.cache") # ESTO NO FUCNIONA IS XGBOOST NO ESTÁ COMPILADO CON LA OPCIÓN "HDFS"...
# dtrain = xgb.DMatrix(train_df.select("features"), label = train_df.select("label"))
param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic', 'eval_metric':'map'}
num_round = 20
cv_xr = xgb.cv(param, dtrain, num_boost_round=num_round)
cv_xr
# help(xgb.cv) # para ver todas las opciones!

# make prediction
dtest = xgb.DMatrix("/home/jjtzapata/testset.libsvm")
preds = bst.predict(dtest)
preds[1:10]
dtrain[1:10,1:3]
xr = xgb.XGBClassifier
cv_xr = xgb.cv.fit(full_trainset, y = full_trainset['clicked'])
xr.predict(X_test)

#
# H2O:
#
# git clone http://github.com/h2oai/sparkling-water
# cd sparkling-water
# sudo su
# # export SPARK_HOME="/path/to/spark/installation"
# export SPARK_HOME=/usr/lib/spark
# export MASTER='local[*]'
# mkdir -p $(pwd)/private/
# curl -s http://h2o-release.s3.amazonaws.com/h2o/rel-turing/10/Python/h2o-3.10.0.10-py2.py3-none-any.whl > $(pwd)/private/h2o.whl
# export H2O_PYTHON_WHEEL=$(pwd)/private/h2o.whl
# ./gradlew build -x check
# export HADOOP_HOME=/usr/lib/spark
cd sparkling-water
bin/pysparkling

#from operator import add
#wc = f.flatMap(lambda x: x.split(" ")).map(lambda x: (x,1)).reduceByKey(add)
#print(wc.collect())

#f.saveAsTextFile("clicks_train_prueba.csv")


# ********************************************************************************************************************************************
# ********************************************************************************************************************************************
# virtualenv + pyspark + keras + tensorlfow: [http://henning.kropponline.de/2016/09/17/running-pyspark-with-virtualenv/]
#
NOTA: Lo que sigue hay que hacerlo en cada máquina del cluster SPARK (master y nodos):
NOTA: Estamos en: jjtoharia@cluster-jjtzapata-m:~$ [pwd = /home/jjtoharia]  (o en cluster-jjtzapata-w0 o en cluster-jjtzapata-w1...)
sudo apt-get install python-pip
sudo pip install virtualenv
virtualenv kaggle
virtualenv --relocatable kaggle
source kaggle/bin/activate
# No sé si hace falta numpy, pero lo hice antes de instalar keras:
pip install numpy
# Mostrar la versión de numpy:
python -c "import numpy as np; print('Python numpy v. ' + np.version.version)"
pip install keras
pip install tensorflow
# Para Windows (64 bits), si pip no funciona: [https://www.tensorflow.org/get_started/os_setup#pip_installation_on_windows]
# conda install python=3.5 [para hacer un "downgrade" de Anaconda a Python 3.5, mientras tensorflow para Windows llega a la 3.6 o superior]
# # pip install --upgrade https://storage.googleapis.com/tensorflow/windows/cpu/tensorflow-0.12.1-cp35-cp35m-win_amd64.whl
# Verificar keras en python:
python -c "import keras"
# cambiar "theano" por "tensorflow", si hace falta - [Ctrl-X] - [Y]:
# nano .keras/keras.cnf
pip install pandas
pip install sklearn
# Para leer/guardar pesos en formato .hdf5:
pip install h5py
# Para compartir ficheros binarios (dataframes) entre R-Python
# https://www.google.es/amp/s/blog.rstudio.org/2016/03/29/feather/amp/
sudo apt-get install python-dev
pip install cython
pip install feather-format
# # otra forma de instalarlo (Windows Anaconda3, p.ej.)]
# conda install feather-format -c conda-forge

# elephas (para usar keras en SPARK):
# sudo apt-get install python-dev
# tarda...
pip install elephas
# para elephas (?):
pip install flask
# Para que funcione con keras v1.xxx:
pip install --upgrade --no-deps git+git://github.com/maxpumperla/elephas
sudo nano /etc/spark/conf/spark-env.sh
# Añadir al final del fichero spark-env.sh:
if [ -z "${PYSPARK_PYTHON}" ]; then
  export PYSPARK_PYTHON=/home/jjtoharia/kaggle/bin/python2.7
fi
NOTA: De esta forma no hace falta arrancar el virtualenv (con source xxx/bin/activate). Se usará lo instalado en ese lugar de cada máquina (master y nodos).
# sudo reboot
# ********************************************************************************************************************************************
# ********************************************************************************************************************************************

# [en cmd]
source kaggle/bin/activate
python
# [en python/pyspark] [NOTA: cluster-1-m es el nombre del servidor master del cluster de Spark, que antes fue cluster-jjtzapata-m]
import time
def timefunc(f):
    def f_timer(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        print f.__name__, ': ', '{:,.4f}'.format(end - start), ' segs.'
        return result
    return f_timer

s_input_path = 'kaggle/Outbrain/In/python/'
@timefunc
def from_feather_to_csv(fich = 'clicks_X_valid_4-1.feather', s_input_path = 'kaggle/Outbrain/In/python/'):
  from feather import read_dataframe as fthr_read_dataframe
  from numpy import savetxt as np_savetxt
  X = fthr_read_dataframe(s_input_path + fich)
  fich = fich.replace('.feather', '_para_spark.csv')
  # # Quitamos NAs (ponemos ceros): NO debería haber... (¡¡¡PERO HAY!!!) (uuid_pgvw_hora_min, p.ej.)
  # X[isnan(X)] = 0
  np_savetxt(s_input_path + fich, X, delimiter=',')
  print(fich, X.values.shape, ' Ok.')
  return(fich)

def from_feather_to_csv_all():
  from os.path import isfile as os_path_isfile
  for seq_len in range(2,13):
    for nF in range(1, 9999): # 1,...,(n-1)
      fichtr = 'clicks_X_train_' + str(seq_len) + '-' + str(nF) + '.feather'
      if not os_path_isfile(s_input_path + fichtr):
        break # Ya no hay más
      fich = 'clicks_X_train_' + str(seq_len) + '-' + str(nF) + '.feather'; fich = from_feather_to_csv(fich)
      fich = 'clicks_X_valid_' + str(seq_len) + '-' + str(nF) + '.feather'; fich = from_feather_to_csv(fich)
      fich = 'clicks_X_test_' + str(seq_len) + '-' + str(nF) + '.feather'; fich = from_feather_to_csv(fich)
      fich = 'clicks_y_train_' + str(seq_len) + '-' + str(nF) + '.feather'; fich = from_feather_to_csv(fich)
      fich = 'clicks_y_valid_' + str(seq_len) + '-' + str(nF) + '.feather'; fich = from_feather_to_csv(fich)
      fich = 'clicks_y_test_' + str(seq_len) + '-' + str(nF) + '.feather'; fich = from_feather_to_csv(fich)

from_feather_to_csv_all()

[en cmd]
# hadoop fs -copyFromLocal kaggle/Outbrain/In/python/clicks_*_*_*-*.csv
hadoop fs -rm  clicks_X_train_4.csv
hadoop fs -appendToFile kaggle/Outbrain/In/python/clicks_X_train_4-*_para_spark.csv  clicks_X_train_4.csv
hadoop fs -appendToFile kaggle/Outbrain/In/python/clicks_y_train_4-*_para_spark.csv  clicks_y_train_4.csv
hadoop fs -appendToFile kaggle/Outbrain/In/python/clicks_X_valid_4-*_para_spark.csv  clicks_X_valid_4.csv
hadoop fs -appendToFile kaggle/Outbrain/In/python/clicks_y_valid_4-*_para_spark.csv  clicks_y_valid_4.csv
hadoop fs -appendToFile kaggle/Outbrain/In/python/clicks_X_test_4-*_para_spark.csv  clicks_X_test_4.csv
hadoop fs -appendToFile kaggle/Outbrain/In/python/clicks_y_test_4-*_para_spark.csv  clicks_y_test_4.csv
hadoop fs -ls
ls -l kaggle/Outbrain/In/python/clicks_X_train_4-*.csv
[en pyspark] [NOTA: cluster-1-m es el nombre del servidor master del cluster de Spark, que antes fue cluster-jjtzapata-m]
# s_spark_inputpath = 'hdfs://cluster-1-m:8020/user/jjtoharia/'
# from pyspark.sql.types import StructType, StructField
# from pyspark.sql.types import DoubleType, IntegerType, StringType
# schema = StructType([
#     StructField("A", IntegerType()),
#     StructField("B", DoubleType()),
#     StructField("C", StringType())
# ])
# schema = StructType([StructField("A", DoubleType())])
# X = spark.read.csv(s_spark_inputpath + 'clicks_X_valid_4-1_para_spark.csv', header=False, mode="DROPMALFORMED", schema=schema)
# y = spark.read.csv(s_spark_inputpath + 'clicks_y_valid_4-1_para_spark.csv', header=False, mode="DROPMALFORMED", schema=schema)
# X.collect()[5]

s_spark_inputpath = 'hdfs://cluster-1-m:8020/user/jjtoharia/'
# Incluimos utilidad pyspark_csv al contexto de Spark:
sc.addPyFile('kaggle/pyspark_csv.py')
# E importamos lo que queremos de la misma:
import pyspark_csv as pycsv
txt_rdd = sc.textFile(s_spark_inputpath + 'clicks_X_valid_4.csv')
txt_rdd.count()
first_rec = txt_rdd.top(1)
first_rec = first_rec[0].split(',')
num_cols = len(first_rec)

from pandas import read_csv
from numpy import float64 as np_float64
X = read_csv(s_input_path + 'clicks_X_valid_4-1_para_spark.csv', dtype=np_float64, header = None)
X2 = read_csv(s_input_path + 'clicks_X_valid_4-2_para_spark.csv', dtype=np_float64, header = None)
y = read_csv(s_input_path + 'clicks_y_valid_4-1_para_spark.csv', dtype=np_float64, header = None)
y2 = read_csv(s_input_path + 'clicks_y_valid_4-2_para_spark.csv', dtype=np_float64, header = None)
from numpy import concatenate as np_concat # Para concatenar varios ficheros en uno (leer_y_reshape)
X = np_concat((X, X2), axis=0)
y = np_concat((y, y2), axis=0)
X.shape, y.shape
num_cols = X.shape[1]

# NOTA: Cuidado que se ordena (por la primera columna...)
dfX = pycsv.csvToDataFrame(sqlCtx, txt_rdd, columns=['Col_' + str(i) for i in range(0,num_cols)])

txt_rdd = sc.textFile(s_spark_inputpath + 'clicks_y_valid_4.csv')
# NOTA: Cuidado que se ordena (por la primera columna...)
dfy = pycsv.csvToDataFrame(sqlCtx, txt_rdd, columns=['Clicked'])

dfX.select(['Col_' + str(i) for i in range(0,4)]).show(10)
dfy.select('Clicked').show(10)
# Ahora estos DataFrame tienen que convertirse en uno como hace [rdd = to_simple_rdd(sc, X_train, y_train)]
PENDIENTE*****
from elephas.utils.rdd_utils import to_simple_rdd
rdd = to_simple_rdd(sc, X_train, Y_train)

[?]
sc.statusTracker().getActiveJobsIds()
sc.statusTracker().getActiveStageIds()

miSparkSession.stop()
sc.stop()
# --------------------------
# # from: https://github.com/maxpumperla/elephas
# from keras.models import Sequential
# from keras.layers.recurrent import LSTM
# from keras.layers.core import Dense, Dropout, Activation
# from keras.optimizers import SGD
# seq_len = 4
# model = Sequential()
# #model.add(Dense(128, input_dim=503))
# #model.add(Activation('relu'))
# model.add(LSTM(input_length=seq_len, input_dim=num_cols,  output_dim=lstm_neuronas_ini, dropout_W=dropout_in, dropout_U=dropout_U, return_sequences=(seq_len != 1))) # , activation='relu'))
# #model.add(Dropout(0.2))
# model.add(Dense(128))
# model.add(Activation('relu'))
# model.add(Dropout(0.2))
# model.add(Dense(2)) # Es 2 por culpa de to_categorical()
# model.add(Activation('softmax'))
# model.compile(loss='categorical_crossentropy', optimizer=SGD())
# model.get_weights()
# from pandas import read_csv
# from numpy import float64 as np_float64
# s_input_path = 'kaggle/Outbrain/In/python/'
# X = read_csv(s_input_path + 'clicks_X_valid_4-1_para_spark.csv', dtype=np_float64, header = None)
# X = X.values
# y = read_csv(s_input_path + 'clicks_y_valid_4-1_para_spark.csv', dtype=int, header = None)
# y = y.values
# from keras.utils.np_utils import to_categorical
# X2, y2 = mi_reshape(X, to_categorical(y), seq_len) # Ponemos dos clases (columnas) a y
# X.shape, y.shape, X2.shape, y2.shape
# from elephas.utils.rdd_utils import to_simple_rdd
# rdd = to_simple_rdd(sc, X, y_bin) # y[:,0])
# from elephas.spark_model import SparkModel
# from elephas import optimizers as elephas_optimizers
# adagrad = elephas_optimizers.Adagrad()
# mi_spark_model = SparkModel(sc,model, optimizer=adagrad, frequency='epoch', mode='asynchronous', num_workers=4)
# mi_spark_model.train(rdd, nb_epoch=20, batch_size=batchsize, verbose=0, validation_split=0.1)
# #scores = model.evaluate(X, y_bin, verbose=0, batch_size=batchsize)
# #print('1 - Loss: %.4f%%' % (100-scores[0]*100))
# #probs = model.predict_proba(X_test, batch_size=batchsize)[:,1] # Nos quedamos con las probs del "1"
# probs = mi_spark_model.predict(X_test)[:,1] # Nos quedamos con las probs del "1"
# print('1 - Loss: %.4f%%' % (100*(1-log_loss(y_bin[:,1], probs))))
# 
# ------------------------------------------------------------
import time
def timefunc(f):
    def f_timer(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        print f.__name__, ': ', '{:,.4f}'.format(end - start), ' segs.'
        return result
    return f_timer

from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
# -> import crear_modelo
@timefunc
def crear_modelo(seq_len, num_capas, num_cols, lstm_neuronas_ini, lstm_neuronas_mid, lstm_neuronas_fin, dropout_in, dropout_U, mi_loss, mi_optimizador, mis_metrics, b_Spark = False):
  print('Create the model:')
  model = Sequential()
  #model.add(Embedding(input_dim=top_words, output_dim=embedding_vector_length, input_length=seq_len))
  if(num_capas == 1):
    model.add(LSTM(input_length=seq_len, input_dim=num_cols,  output_dim=lstm_neuronas_ini, dropout_W=dropout_in, dropout_U=dropout_U, return_sequences=(seq_len != 1))) # , activation='relu'))
  else:
    model.add(LSTM(input_length=seq_len, input_dim=num_cols,  output_dim=lstm_neuronas_ini, dropout_W=dropout_in, dropout_U=dropout_U, return_sequences=True)) # , activation='relu'))
    if(num_capas == 2):
      model.add(LSTM(output_dim=lstm_neuronas_fin, dropout_W=dropout_in, dropout_U=dropout_U, return_sequences=(seq_len != 1))) # , activation='relu'))
    else:
      model.add(LSTM(output_dim=lstm_neuronas_mid, dropout_W=dropout_in, dropout_U=dropout_U, return_sequences=True)) # , activation='relu'))
      model.add(LSTM(output_dim=lstm_neuronas_fin, dropout_W=dropout_in, dropout_U=dropout_U, return_sequences=(seq_len != 1))) # , activation='relu'))
  # Capa de salida:
  model.add(LSTM(output_dim=(2 if b_Spark else 1), dropout_W=dropout_in, dropout_U=dropout_U, activation='sigmoid', return_sequences=(seq_len != 1)))
  model.compile(loss=mi_loss, optimizer=mi_optimizador, metrics=mis_metrics)
  print(model.summary())
  return(model)

# ########################################################
# PREPARAMOS PRIMERO EL RDD Y LO GUARDAMOS EN HADOOP:
# ########################################################
from keras.utils.np_utils import to_categorical
from numpy import reshape as np_reshape
from numpy import concatenate as np_concat # Para concatenar varios ficheros en uno (leer_y_reshape)
from pandas import read_csv
from numpy import float64 as np_float64
from os.path import isfile as os_path_isfile
def mi_reshape(X, y, seq_len = 1):
  if len(X.shape) == 3:
    X = np_reshape(X, (int((X.shape[0] * X.shape[1])/seq_len), seq_len, X.shape[2]))
  else:
    X = np_reshape(X, (int(X.shape[0]/seq_len), seq_len, X.shape[1]))
  if not y is None:
    if len(y.shape) == 3:
      y = np_reshape(y, (int((y.shape[0] * y.shape[1])/seq_len), seq_len, y.shape[2]))
    else:
      if seq_len != 1:
        y = np_reshape(y, (int(y.shape[0]/seq_len), seq_len, y.shape[1]))
    print(X.shape, y.shape)
  else:
    print(X.shape)
  return [X, y]

def mi_reshape_probs(probs, seq_len = 1):
  if len(probs.shape) == 3:
    if seq_len != probs.shape[1]:
      print('NOTA: La dimensión Seq_Len de probs NO coincide con el param. seq_len!')
    probs = np_reshape(probs, (probs.shape[0] * probs.shape[1], probs.shape[2]))
  print(probs.shape)
  return(probs)

b_Spark = True
s_input_path = 'kaggle/Outbrain/In/python/'
s_output_path = 'kaggle/Outbrain/Out/python/'
s_spark_inputpath = 'hdfs://cluster-1-m:8020/user/jjtoharia/'
numSparkWorkers = 4
def preparar_RDD(seq_len = 0):
  from elephas.utils.rdd_utils import to_simple_rdd
  from os import rename as os_rename
  for nF in range(1, 99): # 1,...,(n-1)
    fichtr = 'clicks_X_train_' + str(seq_len) + '-' + str(nF) + '_para_spark.csv'
    if os_path_isfile(s_input_path + fichtr):
      print('Leyendo ficheros train+valid ' + str(nF) + ' - numAds ' + str(seq_len) + '...')
      X_train = read_csv(s_input_path + 'clicks_X_train_' + str(seq_len) + '-' + str(nF) + '_para_spark.csv', dtype=np_float64, header = None).values
      y_train = read_csv(s_input_path + 'clicks_y_train_' + str(seq_len) + '-' + str(nF) + '_para_spark.csv', dtype=int, header = None).values
      X_valid = read_csv(s_input_path + 'clicks_X_valid_' + str(seq_len) + '-' + str(nF) + '_para_spark.csv', dtype=np_float64, header = None).values
      y_valid = read_csv(s_input_path + 'clicks_y_valid_' + str(seq_len) + '-' + str(nF) + '_para_spark.csv', dtype=int, header = None).values
      print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape)
      X_train, y_train = mi_reshape(X_train, to_categorical(y_train), seq_len)
      X_valid, y_valid = mi_reshape(X_valid, to_categorical(y_valid), seq_len)
      X_train = np_concat((X_train, X_valid), axis=0) # Incluimos validset dentro del trainset en Spark
      y_train = np_concat((y_train, y_valid), axis=0) # Incluimos validset dentro del trainset en Spark
      print(X_train.shape, y_train.shape)
      print('Creando RDD (train+valid) ' + str(nF) + ' - numAds ' + str(seq_len) + '...')
      rdd_ini = to_simple_rdd(sc, X_train, y_train)
      # Convertimos ndarray [ i.e. array(...) ] en list [ i.e. [...] ]:
      rdd_lista = rdd_ini.map(lambda i: map(lambda s: s.tolist(), i))
      # Y ahora guardamos como txt:
      rdd_lista.coalesce(numSparkWorkers, True).saveAsTextFile(s_spark_inputpath + 'clicks_train_seq' + str(seq_len) + '-' + str(nF) + '_rdd') # Forzamos a guardarlo en 4 trozos (al menos)
      print('Ok. Guardado en HDFS el RDD (train+valid) ' + str(nF) + ' - numAds ' + str(seq_len) + '.')
  os_rename(s_input_path + fichtr, s_input_path + 'ok_en_hdfs/' + 'clicks_X_train_' + str(seq_len) + '-' + str(nF) + '_para_spark.csv')

for seq_len in range(2,13):
  preparar_RDD(seq_len)

seq_len = 4

dropout_in = 0.3
dropout_U = 0.3
batchsize = 1000
num_capas = 1 # 1, 2 ó 3
lstm_neuronas_ini = 48 # 192
lstm_neuronas_mid = 24 # 48
lstm_neuronas_fin = 12 # 12

mi_early_stop = 10 # si no hay mejora (en val_loss) en N rounds seguidos, se detiene el fit (training)
iteraciones = 2

mi_loss = 'binary_crossentropy' # mi_loss = 'mean_absolute_error'
mi_optimizador = 'adam'
mis_metrics = ['accuracy'] # mis_metrics = ['precision']

# ########################################################
# # LEEMOS DATOS (rdd) YA PREPARADOS DESDE HADOOP:
# ########################################################
rdd_train_txt = sc.textFile(s_spark_inputpath + 'clicks_train_seq' + str(seq_len) + '-1_rdd')
from numpy import array as np_array
rdd_train_ok = rdd_train_txt.map(lambda s: eval(s)).map(lambda j: map(lambda s: np_array(s), j))
print(rdd_train_ok.getNumPartitions()) # Debería devolver numSparkWorkers == 4 (o más)

# Obtenemos el número de columnas (num_cols) y el tamaño de la secuencia (seq_len) del RDD:
primer_reg = rdd_train_ok.take(1)
seq_len = len(primer_reg[0][0]) # 4
num_cols = len(primer_reg[0][0][0]) # = 503
num_reg_train = rdd_train_ok.count()
print('seq_len = ', seq_len, 'num_cols = ', num_cols)

# ########################################################
# LEEMOS DATOS DE TEST (ndarray), PARA EVALUAR:
# ########################################################
X_test = read_csv(s_input_path + 'clicks_X_test_' + str(seq_len) + '-1_para_spark.csv', dtype=np_float64, header = None).values
y_test = read_csv(s_input_path + 'clicks_y_test_' + str(seq_len) + '-1_para_spark.csv', dtype=int, header = None).values
print(X_test.shape, y_test.shape)
X3_test, y3_test = mi_reshape(X_test, to_categorical(y_test), seq_len)
print(X3_test.shape, y3_test.shape)

# ########################################################
model=crear_modelo(seq_len, num_capas, num_cols, lstm_neuronas_ini, lstm_neuronas_mid, lstm_neuronas_fin, dropout_in, dropout_U, mi_loss, mi_optimizador, mis_metrics, b_Spark)
# ########################################################

from elephas.spark_model import SparkModel
from elephas import optimizers as elephas_optimizers
adagrad = elephas_optimizers.Adagrad()
mi_spark_model = SparkModel(sc,model, optimizer=adagrad, frequency='epoch', mode='asynchronous', num_workers=numSparkWorkers)
# ########################################################
print(' =============== ENTRENANDO... ================= ')
# ########################################################
from sklearn.metrics import log_loss
@timefunc
def entrenar_spark(mi_spark_model, rdd_train_ok, iteraciones, batchsize, verbose=0, validation_split=0.1):
  mi_spark_model.train(rdd_train_ok, nb_epoch=iteraciones, batch_size=batchsize, verbose=verbose, validation_split=validation_split)

@timefunc
def evaluar_spark(mi_spark_model, X3_test, y_test):
  seq_len = X_test.shape[1]
  #scores = model.evaluate(X3_test, y3_test, verbose=0, batch_size=batchsize)
  #print('1 - Loss: %.2f%%' % (100-scores[0]*100))
  #probs = model.predict_proba(X3_test, batch_size=batchsize)[:,1] # Nos quedamos con las probs del "1"
  probs = mi_spark_model.predict(X3_test)
  print(probs.shape)
  # probs = mi_reshape_probs(probs, seq_len)[:,1:] # Nos quedamos con las probs del "1" (por alguna razón aparecen a cero... ???)
  print('1 - Loss: %.4f%%' % (100*(1-log_loss(y_test, mi_reshape_probs(probs, seq_len)[:,1:]))))
  return(probs)

# ########################################################
print(' =============== ENTRENANDO... ================= ')
# ########################################################
iteraciones = 5
entrenar_spark(mi_spark_model, rdd_train_ok, iteraciones, batchsize, verbose = 1, validation_split = 0.2)
# ########################################################
print(' =============== EVALUAMOS...  ================= ')
# ########################################################
probs = evaluar_spark(mi_spark_model, X3_test, y_test)
probs[0:2]
# ########################################################
print(' ======== GUARDAMOS PREDS Y MODELO...  =========')
# ########################################################
from numpy import savetxt as np_savetxt
def descr_modelo(model, num_reg_train, batchsize, tipo_descr = 1):
  model_conf_capa_1 = model.get_config()[0]['config']
  num_cols = model_conf_capa_1['input_dim']
  dropout_in = model_conf_capa_1['dropout_W']
  dropout_U = model_conf_capa_1['dropout_U']
  num_capas = len(model.get_config()) - 1
  seq_len = model_conf_capa_1['input_length']
  lstm_neuronas_ini = model_conf_capa_1['output_dim']
  lstm_neuronas_mid = 0
  lstm_neuronas_fin = 0
  if(num_capas > 1):
    lstm_neuronas_fin = model.get_config()[1]['config']['output_dim']
    if(num_capas > 2):
      lstm_neuronas_mid = lstm_neuronas_fin
      lstm_neuronas_fin = model.get_config()[2]['config']['output_dim']
  if tipo_descr == 1:
    descr = 'bch-' + str(batchsize) + '_dri-' + str(dropout_in) + '_dru-' + str(dropout_U) + '_reg-' + str(num_reg_train) + '_col-' + str(num_cols)
    descr = descr + '_ini-' + str(lstm_neuronas_ini) + '_mid-' + str(lstm_neuronas_mid) + '_fin-' + str(lstm_neuronas_fin)
    descr = descr + '_seq-' + str(seq_len)
  else: # if tipo_descr == 2:
    descr = '(BatchSize = ' + str(batchsize) + ')' + '. (Dropout_in = ' + str(dropout_in) + '. Dropout_U = ' + str(dropout_U) + ')'
    descr = descr + '. (SeqLen = ' + str(seq_len) + ')'
    descr = descr + ' - (Nodos = '   + str(lstm_neuronas_ini)
    descr = descr + (            ',' + str(lstm_neuronas_mid) if num_capas == 3 else '')
    descr = descr + (            ',' + str(lstm_neuronas_fin) if num_capas >= 2 else '') + ')'
    descr = descr + ' - ' + str(num_reg_train) + ' regs/' + str(num_cols) + ' cols'
  return(descr)

def guardar_modelo_json(model, pre_post, batchsize):
  fichname = descr_modelo(model, num_reg_train, batchsize)
  fichname = 'modelo' + pre_post + '_' + fichname + '.json'
  print('Guardando modelo (json) [' + 'python/' + fichname + ']...')
  with open(s_output_path + fichname, 'w') as json_file:
      json_file.write(model.to_json())

@timefunc
def guardar_preds(model, X_test, indice = 0, b_csv = True, numAds = 0, numAdsFich = 0, num_reg_train = 0, num_cols = 0, iteraciones = 0, batchsize = 0, dropout_in = 0, dropout_U = 0):
  mi_X_train_shape = [num_reg_train, num_cols]
  str_fich = '_debug' if numAds == 0 else '_{n}-{m}'.format(n=numAds,m=numAdsFich)
  str_fich = 'test_probs' + str_fich + ('' if indice == 0 else '_' + str(indice))
  print('Guardando resultados (probs) en ' + 'python/' + str_fich + ('.csv' if b_csv else '.feather') + '...')
  #probs = model.predict_proba(X_test, batch_size=batchsize)
  probs = mi_spark_model.predict(X3_test)
  probs = mi_reshape_probs(probs, seq_len)[:,1:] # Volvemos a dos dimensiones
  # print(probs[1:10])
  if b_csv:
    np_savetxt(s_input_path + str_fich + '.csv', probs, delimiter=',')
  else:
    fthr_write_dataframe(DataFrame(probs), s_input_path + str_fich + '.feather')
  print('\nOk. [' + 'In/python/' + str_fich + ('.csv' if b_csv else '.feather') + ']')
  np_savetxt(s_output_path + str_fich + '_' + str(iteraciones) + '_' + str(batchsize) + '-' + str(num_reg_train) + '.log', mi_X_train_shape, delimiter=',')
  print('Ok. [' + 'python/' + str_fich + '_' + str(iteraciones) + '_' + str(batchsize) + '-' + str(num_reg_train) + '.log]')
  guardar_modelo_json(model, 'post', batchsize) # Guardamos estructura también al final.
  print('\nOk. (Iter = ' + str(iteraciones) + '. BatchSize = ' + str(batchsize) + ')' + '. (Dropout_in = ' + str(dropout_in) + '. Dropout_U = ' + str(dropout_U) + ') - ' + str(num_reg_train) + ' regs/' + str(num_cols) + ' cols')

guardar_preds(model, X3_test, 0, True, seq_len, 0, num_reg_train, num_cols, iteraciones, batchsize, dropout_in, dropout_U)
