# -*- coding: utf-8 -*-

# spark-submit --driver-memory 3G kaggle/Outbrain/lstm_preparar_RDD.py

print('Creando SparkSession...')
from pyspark.sql import SparkSession
miSparkSession = SparkSession \
                .builder \
                .appName("Spark-Outbrain-JJTZ-Preparar") \
                .config("spark.some.config.option", "some-value") \
                .getOrCreate()
sc = miSparkSession.sparkContext

import time
def timefunc(f):
    def f_timer(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        print f.__name__, ': ', '{:,.4f}'.format(end - start), ' segs.'
        return result
    return f_timer

# ########################################################
# PREPARAMOS PRIMERO LOS CSV (a partir de los .feather):
# ########################################################
from os.path import isfile as os_path_isfile
s_input_path = 'kaggle/Outbrain/In/python/'
def from_feather_to_csv(fich = 'clicks_X_valid_4-1.feather', s_input_path = 'kaggle/Outbrain/In/python/'):
  fich_dest = fich.replace('.feather', '_para_spark.csv')
  if not os_path_isfile(s_input_path + fich_dest):
    from feather import read_dataframe as fthr_read_dataframe
    from numpy import savetxt as np_savetxt
    X = fthr_read_dataframe(s_input_path + fich)
    np_savetxt(s_input_path + fich_dest, X, delimiter=',')
    print(fich_dest, X.values.shape, ' Ok.')
  return(fich_dest)

@timefunc
def from_feather_to_csv_all():
  for seq_len in range(2,13):
    print('seq_len = ' + str(seq_len) + '...')
    for nF in range(1, 9999): # 1,...,(n-1)
      fichtr = 'clicks_X_train_' + str(seq_len) + '-' + str(nF) + '.feather'
      if not os_path_isfile(s_input_path + 'ok_en_hdfs/' + fichtr.replace('.feather', '_para_spark.csv')):
        if not os_path_isfile(s_input_path + fichtr.replace('.feather', '_para_spark.csv')):
          if os_path_isfile(s_input_path + fichtr):
            fich = 'clicks_X_train_' + str(seq_len) + '-' + str(nF) + '.feather'; fich = from_feather_to_csv(fich)
            fich = 'clicks_X_valid_' + str(seq_len) + '-' + str(nF) + '.feather'; fich = from_feather_to_csv(fich)
            fich = 'clicks_X_test_' + str(seq_len) + '-' + str(nF) + '.feather'; fich = from_feather_to_csv(fich)
            fich = 'clicks_y_train_' + str(seq_len) + '-' + str(nF) + '.feather'; fich = from_feather_to_csv(fich)
            fich = 'clicks_y_valid_' + str(seq_len) + '-' + str(nF) + '.feather'; fich = from_feather_to_csv(fich)
            fich = 'clicks_y_test_' + str(seq_len) + '-' + str(nF) + '.feather'; fich = from_feather_to_csv(fich)

from_feather_to_csv_all()

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
  if not X is None:
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
  if not X is None:
    if not y is None:
      print(X.shape, y.shape)
    else:
      print(X.shape)
  else:
    if not y is None:
      print(y.shape)
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
    if not os_path_isfile(s_input_path + 'ok_en_hdfs/' + fichtr):
      if os_path_isfile(s_input_path + fichtr):
        print('Leyendo ficheros train+valid ' + str(nF) + ' - numAds ' + str(seq_len) + ' [' + fichtr + ']...')
        # X_train = X_train + X_valid:
        X_train = read_csv(s_input_path + 'clicks_X_train_' + str(seq_len) + '-' + str(nF) + '_para_spark.csv', dtype=np_float64, header = None).values
        X_valid = read_csv(s_input_path + 'clicks_X_valid_' + str(seq_len) + '-' + str(nF) + '_para_spark.csv', dtype=np_float64, header = None).values
        print(X_train.shape, X_valid.shape)
        X_train = mi_reshape(X_train, None, seq_len)[0]
        X_valid = mi_reshape(X_valid, None, seq_len)[0]
        X_train = np_concat((X_train, X_valid), axis=0) # Incluimos validset dentro del trainset en Spark
        del X_valid
        
        # y_train = y_train + y_valid:
        y_train = read_csv(s_input_path + 'clicks_y_train_' + str(seq_len) + '-' + str(nF) + '_para_spark.csv', dtype=int, header = None).values
        y_valid = read_csv(s_input_path + 'clicks_y_valid_' + str(seq_len) + '-' + str(nF) + '_para_spark.csv', dtype=int, header = None).values
        print(y_train.shape, y_valid.shape)
        y_train = mi_reshape(None, to_categorical(y_train), seq_len)[1]
        y_valid = mi_reshape(None, to_categorical(y_valid), seq_len)[1]
        y_train = np_concat((y_train, y_valid), axis=0) # Incluimos validset dentro del trainset en Spark
        del y_valid
        
        print(X_train.shape, y_train.shape)
        print('Creando RDD (train+valid) ' + str(nF) + ' - numAds ' + str(seq_len) + '[' + fichtr + ']...')
        rdd_ini = to_simple_rdd(sc, X_train, y_train)
        # Convertimos ndarray [ i.e. array(...) ] en list [ i.e. [...] ]:
        rdd_lista = rdd_ini.map(lambda i: map(lambda s: s.tolist(), i))
        # Y ahora guardamos como txt:
        rdd_lista.coalesce(numSparkWorkers, True).saveAsTextFile(s_spark_inputpath + 'clicks_train_seq' + str(seq_len) + '-' + str(nF) + '_rdd') # Forzamos a guardarlo en 4 trozos (al menos)
        print('Ok. Guardado en HDFS el RDD (train+valid) ' + str(nF) + ' - numAds ' + str(seq_len) + ' [' + fichtr + '].')
        os_rename(s_input_path + fichtr, s_input_path + 'ok_en_hdfs/' + fichtr)
        del rdd_ini, rdd_lista, X_train, y_train

for seq_len in range(2,13):
  preparar_RDD(seq_len)

print('Ok. Fin.')
