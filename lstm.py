# -*- coding: utf-8 -*-

# NOTA: Arrancar desde /home/jjtoharia/ con:
# source kaggle/bin/activate
# [Python]  - python -i kaggle/Outbrain/lstm.py   [seq_len [iteraciones [maxNumFichs [b_guardarDatos]]]]
# [pyspark] - spark-submit --driver-memory 4G kaggle/Outbrain/lstm_entrenar_RDD.py   [seq_len [iteraciones [maxNumFichs [b_guardarDatos]]]]

# NOTA: UNICODE - UTF8 - También! Si no hay acentos, conversiones -> UNICODE/UTF8 a UTF8 (ed. Unicode)
# NOTA: Para borrar todas la variables: %reset

from sys import argv
print(argv)

seq_len =        int(argv[1])        if(len(argv) > 1)  else 0     # Seq_Len (numAds)
iteraciones =    int(argv[2])        if(len(argv) > 2)  else 2     # Iteraciones del entrenamiento
maxNumFichs =    int(argv[3])        if(len(argv) > 3)  else 1     # Acumular n RDDs (n ficheros por numAd)
b_guardarDatos = (int(argv[4]) != 0) if(len(argv) > 4)  else False # Guardar datos al entrenar [entrenar_modelo()]

b_Spark = False

print('Params: [Seq_Len = ' + str(seq_len) + '] [Iteraciones = ' + str(iteraciones) + '] [numRDDs = ' + str(maxNumFichs) + ']')

if b_Spark:
  print('Creando SparkSession...')
  from pyspark.sql import SparkSession
  miSparkSession = SparkSession \
                  .builder \
                  .appName("Spark-Outbrain-JJTZ-Entrenar") \
                  .config("spark.some.config.option", "some-value") \
                  .getOrCreate()
  sc = miSparkSession.sparkContext
  # from pyspark.sql import SQLContext
  # sqlCtx = SQLContext(sc)

# import numpy as np
from pandas import DataFrame # para guardar arrays de numpy con fthr_write_dataframe
from numpy import float64 as np_float64
from numpy import load as np_load
#from numpy import sqrt
from numpy.random import seed as np_random_seed
from numpy import reshape as np_reshape
from numpy import save as np_save
from numpy import savetxt as np_savetxt
from feather import write_dataframe as fthr_write_dataframe
from numpy import vstack as np_vstack
# from numpy import zeros as np_zeros
from numpy import isnan
from numpy import concatenate as np_concat # Para concatenar varios ficheros en uno (leer_y_reshape)
from numpy import c_ as cbind
from feather import read_dataframe as fthr_read_dataframe
from keras.models import Sequential
#from keras.layers.core import Activation
#from keras.layers.embeddings import Embedding
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
#from keras.layers import Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
#from sklearn.preprocessing import MinMaxScaler
#from sklearn.metrics import mean_squared_error
# import os # os.listdir(path); os.remove(pathname); os.rename(old,new); isfile(pathname)
from os.path import isfile as os_path_isfile
from os.path import isdir as os_path_isdir
from os import remove as os_remove
from os import rename as os_rename
#import h5py
from time import time

import time
def timefunc(f):
    def f_timer(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        print(f.__name__, ': ', '{:,.4f}'.format(end - start), ' segs.')
        return(result)
    return(f_timer)

#in_out_neurons = 2
#hidden_neurons = 300

## load the dataset:
#import pandas
#dataframe = pandas.read_csv('international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
#dataset = dataframe.values
#dataset = dataset.astype('float32')
## normalize the dataset:
#scaler = MinMaxScaler(feature_range=(0, 1))
#dataset = scaler.fit_transform(dataset)
#
## split into train and test sets
#train_size = int(len(dataset) * 0.67)
#test_size = len(dataset) - train_size
#train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
#print(len(train), len(test))

## load the dataset but only keep the top n words, zero the rest
#from keras.datasets import imdb
#top_words = 5000
#(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=top_words)
#print(X_train.shape, y_train.shape)

# 
print('CARGAMOS DATOS:')
# 
s_input_path = 'C:/Users/jtoharia/Downloads/Kaggle_Outbrain/python/'
if not os_path_isdir(s_input_path):
  s_input_path = 'kaggle/Outbrain/In/python/'

if not os_path_isdir(s_input_path):
  s_input_path = 'In/python/'

if not os_path_isdir(s_input_path):
  print('\nERROR: s_input_path [' + s_input_path + '] NO encontrado!\n')
  quit()

s_output_path = 'C:/Users/jtoharia/Dropbox/AFI_JOSE/Kaggle/Outbrain/python/'
if not os_path_isdir(s_output_path):
  s_output_path = 'C:/Personal/Dropbox/AFI_JOSE/Kaggle/Outbrain/python/'

if not os_path_isdir(s_output_path):
  s_output_path = 'kaggle/Outbrain/Out/python/'

if not os_path_isdir(s_output_path):
  s_output_path = 'Out/python/'

if not os_path_isdir(s_output_path):
  print('\nERROR: s_output_path [' + s_output_path + '] NO encontrado!\n')
  quit()

print('input_path = ' + s_input_path)
print('output_path = ' + s_output_path)
#import matplotlib.pyplot as plt





from numpy import mean as np_mean
def apk(actual, predicted, k=10):
    """
    Computes the average precision at k. This function computes the average prescision at k between two lists of items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]
    
    score = 0.0
    num_hits = 0.0
    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)
    
    if not actual:
        return 0.0
    
    return(score / min(len(actual), k))

def mapk(actual, predicted, k=12):
    """
    Computes the mean average precision at k. This function computes the mean average prescision at k between two lists of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return(np_mean([apk(a,p,k) for a,p in zip(actual.tolist(), predicted.tolist())]))

def mapk12(actual, predicted):
    return(np_mean([apk(a,p,12) for a,p in zip(actual, predicted)]))

# NO FUNCIONA!!! from tensorflow import to_double as tf_to_float64 # tf.to_double(x, name='ToDouble'): Casts a tensor to type float64.
# NO FUNCIONA!!! from tensorflow import transpose as tf_transpose
# NO FUNCIONA!!! from tensorflow.contrib.metrics import streaming_sparse_average_precision_at_k as tf_mapk
# NO FUNCIONA!!! # from tensorflow import unstack as tf_unstack
# NO FUNCIONA!!! def mapk12_tf(actual, predicted):
# NO FUNCIONA!!!     tensor_actual = tf_to_int64(actual) # labels tienen que ser enteros int64
# NO FUNCIONA!!!     tensor_predic = tf_to_float32(predicted) # preds tienen que ser floats float
# NO FUNCIONA!!!     tensor_actual = tf_transpose(tensor_actual, (0, 2, 1)) # La última dimensión del tensor (la tercera) tiene que ser la de sequences (que es la segunda en keras-RNN)
# NO FUNCIONA!!!     tensor_predic = tf_transpose(tensor_predic, (0, 2, 1)) # La última dimensión del tensor (la tercera) tiene que ser la de sequences (que es la segunda en keras-RNN)
# NO FUNCIONA!!!     tensor_actual = tf_transpose(tensor_actual) # Recuperamos filas y columnas en su sitio (creo...) (?)
# NO FUNCIONA!!!     tensor_predic = tf_transpose(tensor_predic) # Recuperamos filas y columnas en su sitio (creo...) (?)
# NO FUNCIONA!!!     return(tf_mapk(predictions = tensor_predic, labels = tensor_actual, k = 12, weights=None, metrics_collections=None, updates_collections=None, name=None)[0])

from scipy import argmax as sp_argmax
from scipy import argsort as sp_argsort
from numpy import int64 as np_int64
def mi_mapk12(actual, predicted):
  labels = [[sp_argmax(label)] for label in np_int64(actual[:,:,0].tolist())]
  preds = [list(reversed(sp_argsort(prob))) for prob in predicted[:,:,0].tolist()]
  return(mapk12(labels, preds))

from tensorflow import argmax as tf_argmax
from tensorflow.python.ops.nn import top_k as tf_top_k
from tensorflow import unstack as tf_unstack
from tensorflow.contrib.metrics import streaming_sparse_average_precision_at_k as tf_mapk
from tensorflow import to_int64 as tf_to_int64
from tensorflow import to_float as tf_to_float32
def mi_mapk12_tf(actual, predicted):
  tensor_actual = [tf_argmax(actual, axis=0)]  
  vals, tensor_predic = list(tf_top_k( tf_unstack(predicted, num=None, axis=2) , k=2)) # top_k returns (values, indices)
  tensor_actual = tf_to_int64(actual) # labels tienen que ser enteros int64
  tensor_predic = tf_to_float32(predicted) # preds tienen que ser floats float
  return(tf_mapk(predictions = tensor_predic, labels = tensor_actual, k = 2)[0])




#scaler = MinMaxScaler(feature_range=(0, 1))

def leer_datos(tipo, numAds = 0, numAdsFich = 0):  # tipo es "train" / "valid" / "test"
  # str_fich = '_debug' if numAds == 0 else '_{n}-{m:06d}'.format(n=numAds,m=numAdsFich)
  str_fich = '_debug' if numAds == 0 else '_{n}-{m}'.format(n=numAds,m=numAdsFich)
  if os_path_isfile(s_input_path + 'clicks_X_' + tipo + str_fich + '.csv'):
    from pandas import read_csv
    print('Cargando ' + tipo + 'set' + str_fich + ' (csv)...')
    X = read_csv(s_input_path + 'clicks_X_' + tipo + str_fich + '.csv', dtype=np_float64)
    print('Guardando ' + tipo + 'set (feather)...')
    fthr_write_dataframe(X, s_input_path + 'clicks_X_' + tipo + str_fich + '.feather')
    #print('No normalizando ' + tipo + '...')
    X = X.values # scaler.fit_transform(X)
    #print('Guardando ' + tipo + 'set (npy)...')
    #np_save(s_input_path + 'clicks_X_' + tipo + str_fich + '.npy', X)
    y = read_csv(s_input_path + 'clicks_y_' + tipo + str_fich + '.csv')
    fthr_write_dataframe(y, s_input_path + 'clicks_y_' + tipo + str_fich + '.feather')
    y = y.values
    #np_save(s_input_path + 'clicks_y_' + tipo + str_fich + '.npy', y)
    if os_path_isfile(s_input_path + 'clicks_X_' + tipo + str_fich + '.bak.csv'):
      os_remove(s_input_path + 'clicks_X_' + tipo + str_fich + '.bak.csv')
      os_remove(s_input_path + 'clicks_y_' + tipo + str_fich + '.bak.csv')
    os_rename(s_input_path + 'clicks_X_' + tipo + str_fich + '.csv', s_input_path + 'clicks_X_' + tipo + str_fich + '.bak.csv')
    os_rename(s_input_path + 'clicks_y_' + tipo + str_fich + '.csv', s_input_path + 'clicks_y_' + tipo + str_fich + '.bak.csv')
  
  elif os_path_isfile(s_input_path + 'clicks_X_' + tipo + str_fich + '.feather'):
    print('Cargando ' + tipo + 'set' + str_fich + ' (feather)...')
    X = fthr_read_dataframe(s_input_path + 'clicks_X_' + tipo + str_fich + '.feather')
    if X.columns[-1] == 'numAds':
      print('NOTA: Copiamos columna numAds.')
      b_conNumAds = True
      X_numAds = X.values[:,-1:] * 10 + 2 # Desnormalizamos
      X = X.values
      # NO la quitamos, que ya viene normalizada: X = X.values[:,0:-1] # Quitamos last column (numAds)
    else:
      X = X.values
    y = fthr_read_dataframe(s_input_path + 'clicks_y_' + tipo + str_fich + '.feather')
    y = y.values
  
  elif os_path_isfile(s_input_path + 'clicks_X_' + tipo + str_fich + '.npy'):
    print('Cargando ' + tipo + 'set' + str_fich + ' (npy)...')
    X = np_load(s_input_path + 'clicks_X_' + tipo + str_fich + '.npy')
    y = np_load(s_input_path + 'clicks_y_' + tipo + str_fich + '.npy')
  
  else:
    print('Fichero ' + 'clicks_X_' + tipo + str_fich + ' NO encontrado.')
    X, y = (None, None)
  
  if not X is None:
    print(X.shape, y.shape)
  
  return [X, y]

#plt.plot(train_y)
#plt.show()

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

def leer_y_reshape(tipo, seq_len = 1, numAds = 0, numAdsFich = 0, X_ant = None, y_ant = None):
  # Leemos csv (y creamos npy para acelerar lecturas posteriores, y renombramos el csv):
  # NOTA: Si hay 'feather' lo leemos también
  X, y = leer_datos(tipo, numAds, numAdsFich)
  if not X is None:
    # CORREGIDO # Quitamos NAs (ponemos ceros): NO debería haber... (¡¡¡PERO HAY!!!) (uuid_pgvw_hora_min, p.ej.)
    # CORREGIDO X[isnan(X)] = 0
    # Padding:
    # ## # truncate and pad input sequences
    # ## from keras.preprocessing import sequence
    # ## X_train = sequence.pad_sequences(train_X.values, dtype=np_float64, maxlen=seq_len)
    # ## X_test = sequence.pad_sequences(valid_X.values, dtype=np_float64, maxlen=seq_len)
    # print('Padding (' + tipo + ') para tener secuencias completas...')
    # seq_len = 1 # 12
    # l_old = train_X.shape[0]
    # l_new = seq_len * (1 + int(train_X.shape[0]/seq_len))
    # for i in range(l_old, l_new):
    #   train_X = np_vstack((train_X, np_zeros(train_X.shape[1])))
    #   train_y = np_vstack((train_y, 0))
    # l_old = valid_X.shape[0]
    # l_new = seq_len * (1 + int(valid_X.shape[0]/seq_len))
    # for i in range(l_old, l_new):
    #   valid_X = np_vstack((valid_X, np_zeros(valid_X.shape[1])))
    #   valid_y = np_vstack((valid_y, 0))
    # Reshape:
    print('Reshape input (' + tipo + '_' + str(numAds) + '-' + str(numAdsFich) + ' to be [samples, time steps(seq.len.), features]...')
    X, y = mi_reshape(X, y, seq_len)
    if not X_ant is None:
      X = np_concat((X_ant, X), axis=0)
      y = np_concat((y_ant, y), axis=0)
      print(X.shape, y.shape)
    
  return [X, y]

def crear_modelo(seq_len, num_capas, num_cols, lstm_neuronas_ini, lstm_neuronas_mid, lstm_neuronas_fin, dropout_in, dropout_U, mi_loss, mi_optimizador, mis_metrics):
  print('Create the model:')
  model = Sequential()
  #model.add(Embedding(input_dim=top_words, output_dim=embedding_vector_length, input_length=seq_len))
  if(num_capas == 1):
    model.add(LSTM(input_length=seq_len, input_dim=num_cols,  output_dim=lstm_neuronas_ini, dropout_W=dropout_in, dropout_U=dropout_U, return_sequences=True)) # , activation='relu'))
  else:
    model.add(LSTM(input_length=seq_len, input_dim=num_cols,  output_dim=lstm_neuronas_ini, dropout_W=dropout_in, dropout_U=dropout_U, return_sequences=True)) # , activation='relu'))
    if(num_capas == 2):
      model.add(LSTM(output_dim=lstm_neuronas_fin, dropout_W=dropout_in, dropout_U=dropout_U, return_sequences=True)) # , activation='relu'))
    else:
      model.add(LSTM(output_dim=lstm_neuronas_mid, dropout_W=dropout_in, dropout_U=dropout_U, return_sequences=True)) # , activation='relu'))
      model.add(LSTM(output_dim=lstm_neuronas_fin, dropout_W=dropout_in, dropout_U=dropout_U, return_sequences=True)) # , activation='relu'))
  # Capa de salida:
  model.add(LSTM(output_dim=1, dropout_W=dropout_in, dropout_U=dropout_U, activation='sigmoid', return_sequences=True))
  model.compile(loss=mi_loss, optimizer=mi_optimizador, metrics=mis_metrics)
  print(model.summary())
  return(model)

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
def entrenar_modelo(model, X_train, y_train, X_valid, y_valid, batchsize, iteraciones, mi_early_stop = 5, mi_shuffle = True, b_guardar = True):
  if b_guardar:
    guardar_modelo_json(model, 'prev', batchsize) # Guardamos estructura ANTES de empezar...
  num_reg_train = X_train.shape[0]
  print('Entrenando el modelo (Iter = ' + str(iteraciones) + '). ' + descr_modelo(model, num_reg_train, batchsize, tipo_descr = 2))
  np_random_seed(1234)
  if b_guardar:
    fichname = descr_modelo(model, num_reg_train, batchsize)
    fichname = 'weights_' + fichname + '__{epoch:02d}-{val_loss:.4f}.hdf5'
    callbacks = [
      EarlyStopping(monitor='val_loss', patience=mi_early_stop, verbose=1),
      ModelCheckpoint(s_output_path + fichname, monitor='val_loss', save_best_only=True, verbose=0)
    ]
  else:
    callbacks = [
      EarlyStopping(monitor='val_loss', patience=mi_early_stop, verbose=1)
    ]  
  #model.fit(X_train, y_train, validation_data=(X_valid, y_valid), nb_epoch=iteraciones, batch_size=batchsize)
  model.fit(X_train, y_train, validation_data=(X_valid, y_valid), nb_epoch=iteraciones, batch_size=batchsize, shuffle=mi_shuffle, callbacks=callbacks)

def leer_pesos_modelo(model, mi_loss, mi_optimizador, mis_metrics, fichero_pesos = 'weights-21-0.42_bch-1024_dri-0.1_dru-0.1_reg-50000_col-506.hdf5'):
  # load weights
  if not os_path_isfile(s_output_path + fichero_pesos):
    print('ERROR: Fichero [' + fichero_pesos + '] NO encontrado.')
  else:
    print('Cargando pesos y compilando modelo...')
    model.load_weights(s_output_path + fichero_pesos)
    model.compile(loss=mi_loss, optimizer=mi_optimizador, metrics=mis_metrics)
  return(model)

#preds = model.predict_classes(X_test, batch_size=batchsize)
# #print(preds[1:10] - y_test[1:10])
# print(y_test[1:10])

def evaluar_modelo(model, X, y, txt, batchsize, eval_ini = None, verbose = 1):
  if verbose > 0:
    print('Evaluamos el modelo (' + txt + '):')
  scores = model.evaluate(X, y, verbose=0, batch_size=batchsize)
  probs = model.predict_proba(X, batch_size=batchsize)
  if verbose > 0:
    print(np_vstack((probs[0:10].T, y[0:10].T)).T)
  # preds = model.predict_classes(X, batch_size=batchsize)
  # print(np_vstack((preds[0:10].T, y[0:10].T)).T)
  mapk12_fin = mi_mapk12(y, probs)
  if verbose > 0:
    if not eval_ini is None:
      print('1-Loss (' + txt + ') BEFORE = %.2f%%' % (100-eval_ini[0]*100))
      print('1-Loss (' + txt + ') AFTER  = %.2f%%' % (100-scores[0]*100))
      print('mapk12 (' + txt + ') BEFORE = ', eval_ini[1])
      print('mapk12 (' + txt + ') AFTER  = ', mapk12_fin)
    else:
      print('1-Loss (' + txt + ') = %.2f%%' % (100-scores[0]*100))
      print('mapk12 (' + txt + ') = ', mapk12_fin)
  return([scores[0], mapk12_fin]);

def guardar_preds(model, X_test, indice = 0, b_csv = False, numAds = 0, numAdsFich = 0, num_reg_train = 0, num_cols = 0, iteraciones = 0, batchsize = 0, dropout_in = 0, dropout_U = 0):
  str_fich = '_debug' if numAds == 0 else '_{n}-{m}'.format(n=numAds,m=numAdsFich)
  str_fich = 'test_probs' + str_fich + ('' if indice == 0 else '_' + str(indice))
  print('Guardando resultados (probs) en ' + 'python/' + str_fich + ('.csv' if b_csv else '.feather') + '...')
  probs = model.predict_proba(X_test, batch_size=batchsize)
  probs = mi_reshape_probs(probs, seq_len = numAds) # Volvemos a dos dimensiones
  # print(probs[1:10])
  if b_csv:
    np_savetxt(s_input_path + str_fich + '.csv', probs, delimiter=',')
  else:
    fthr_write_dataframe(DataFrame(probs), s_input_path + str_fich + '.feather')
  print('\nOk. [' + 'In/python/' + str_fich + ('.csv' if b_csv else '.feather') + ']')
  np_savetxt(s_output_path + str_fich + '_' + str(iteraciones) + '_' + str(batchsize) + '-' + str(num_reg_train) + '.log', [num_reg_train, num_cols], delimiter=',')
  print('Ok. [' + 'python/' + str_fich + '_' + str(iteraciones) + '_' + str(batchsize) + '-' + str(num_reg_train) + '.log]')
  guardar_modelo_json(model, 'post', batchsize) # Guardamos estructura también al final.
  print('\nOk. (Iter = ' + str(iteraciones) + '. BatchSize = ' + str(batchsize) + ')' + '. (Dropout_in = ' + str(dropout_in) + '. Dropout_U = ' + str(dropout_U) + ') - ' + str(num_reg_train) + ' regs/' + str(num_cols) + ' cols')

def guardar_pesos(model, num_reg_train, batchsize, iteraciones, val_loss):
  model.save_weights(s_output_path + 'weights_' + descr_modelo(model, num_reg_train, batchsize) + '__{epoch:02d}-{val_loss:.4f}.hdf5'.format(epoch=iteraciones, val_loss=val_loss))

def generar_test_submit_probs(model, minBatch=1, maxBatch=32):
  maxBatch += 1
  submit_subpath = 'submit/'
  # scaler = MinMaxScaler(feature_range=(0, 1))
  if not os_path_isfile(s_input_path + submit_subpath + 'clicks_X_test_submit_{n:03d}_{m:03d}.feather'.format(n=maxBatch-1,m=32)):
    if not os_path_isfile(s_input_path + submit_subpath + 'testset_norm_{n:03d}_{m:03d}.npy'.format(n=maxBatch-1,m=32)):
      from pandas import read_csv
      for numBatch in range(minBatch, maxBatch):
        for numSubBatch in range(1, 33):
          fichorig = 'clicks_X_test_submit_{n:03d}_{m:03d}.csv'.format(n=numBatch,m=numSubBatch)
          fichdest = 'testset_norm_{n:03d}_{m:03d}.npy'.format(n=numBatch,m=numSubBatch)
          if not os_path_isfile(s_input_path + submit_subpath + fichdest + '.OK'):
            if not os_path_isfile(s_input_path + submit_subpath + fichdest):
              if os_path_isfile(s_input_path + submit_subpath + fichorig):
                print('Leyendo [' + fichorig + ']...')
                X = read_csv(s_input_path + submit_subpath + fichorig, dtype=np_float64)
                print(X.shape)
                print('No normalizando...')
                X = X.values # scaler.fit_transform(X)
                print('Guardando [' + submit_subpath + fichdest + ']...')
                np_save(s_input_path + submit_subpath + fichdest, X)
            else:
              print('Ok [' + submit_subpath + fichdest + ']')
          else:
            print('Ok [' + submit_subpath + fichdest + ']')
  
  n_submitsOk, n_submitsPend, modelos = [0, 0, None]
  inicio = time.time()
  for numBatch in range(minBatch, maxBatch):
    for numSubBatch in range(1, 33):
      fichorig = 'clicks_X_test_submit_{n:03d}_{m:03d}.feather'.format(n=numBatch,m=numSubBatch)
      fichdest = 'test_submit_probs_{n:03d}_{m:03d}.csv'.format(n=numBatch,m=numSubBatch)
      if not os_path_isfile(s_input_path + submit_subpath + fichdest):
        b_conNumAds = False
        X_numAds = None
        if not os_path_isfile(s_input_path + submit_subpath + fichorig):
          fichorig = 'testset_norm_{n:03d}_{m:03d}.npy'.format(n=numBatch,m=numSubBatch)
          if os_path_isfile(s_input_path + submit_subpath + fichorig):
            print('Leyendo [' + fichorig + ']...')
            X = np_load(s_input_path + submit_subpath + fichorig)
          else:
            n_submitsPend += 1
            if n_submitsPend == 1:
              print('NOTA: Falta el fichero [' + fichorig + '] (y puede que los siguientes...)')
        else:
          print('Leyendo [' + fichorig + ']...')
          X = fthr_read_dataframe(s_input_path + submit_subpath + fichorig)
          if X.columns[-1] == 'numAds':
            print('NOTA: Copiamos columna numAds.')
            b_conNumAds = True
            X_numAds = X.values[:,-1:] * 10 + 2 # Desnormalizamos
            X_numAds = X_numAds.astype(int) # Por si acaso
            X = X.values
            # NO la quitamos, que ya viene normalizada: X = X.values[:,0:-1] # Quitamos last column (numAds)
            if modelos is None:
              modelos, batchsizes = leer_modelos_numAds()
          else:
            X = X.values
            X_numAds = None
            modelos, batchsizes = [list(model), list(batchsize)] # Listas de ...
        # CORREGIDO # Quitamos NAs (ponemos ceros): NO debería haber... (¡¡¡PERO HAY!!!) (uuid_pgvw_hora_min, p.ej.)
        # CORREGIDO X[isnan(X)] = 0
        print('Predicting [' + fichorig + ']...')
        probs = predecir_testprobs(modelos, X, batchsizes, X_numAds)
        ## if not b_conNumAds:
        ##   X = np_reshape(X, (X.shape[0], 1, X.shape[1]))
        ##   # X = np_reshape(X, (int(X.shape[0]/seq_len), seq_len, X.shape[1]))
        ##   probs = model.predict_proba(X, batch_size=batchsize)
        ## else:
        ##   X = np_reshape(X, (X.shape[0], 1, X.shape[1]))
        ##   # X = np_reshape(X, (int(X.shape[0]/seq_len), seq_len, X.shape[1]))
        ##   probs = model.predict_proba(X, batch_size=batchsize) # PENDIENTE: Usar varios modelos en función del numAds ('''pero sin desordenar X!!!)
        print('Guardando resultados (probs) [' + fichdest + ']...\n')
        np_savetxt(s_input_path + submit_subpath + fichdest, probs, delimiter=',')
        n_submitsOk += 1
        if os_path_isfile(s_input_path + submit_subpath + fichorig):
          os_rename(s_input_path + submit_subpath + fichorig, s_input_path + submit_subpath + fichorig + '.OK')
      else:
        n_submitsOk += 1
        print('Ok [' + submit_subpath + fichdest + ']')
        if os_path_isfile(s_input_path + submit_subpath + fichorig):
          os_rename(s_input_path + submit_subpath + fichorig, s_input_path + submit_subpath + fichorig + '.OK')
  
  if n_submitsPend != 0:
    print('NOTA: Faltan ' + str(n_submitsPend) + '/' + str(n_submitsPend + n_submitsOk) + ' ficheros (.npy). NO se han creado todos los test_submit_probs!')
  
  final = time.time()
  print('Ok. (s_input_path = ' + s_input_path + ') - ' + '{x:,.1f}'.format(x = (final - inicio) / (60 if final-inicio>60 else 1)) + (' mins.' if final-inicio>60 else ' segs.'))

def leer_modelos_numAds():
  batchsizes = [1024,1024,1024,1024,1024,1024,1024,1024,512,1024,512] # numAds=2,3,....,12
  modelos = list()
  prvloss, prvoptimizador, prvmetrics = ['binary_crossentropy', 'adam', ['accuracy']]
  model = crear_modelo(seq_len=2, num_capas=3, num_cols=503, lstm_neuronas_ini=475, lstm_neuronas_mid=475, lstm_neuronas_fin=475, dropout_in=0.2, dropout_U=0.2, mi_loss=prvloss, mi_optimizador=prvoptimizador, mis_metrics=prvmetrics)
  leer_pesos_modelo(model, prvloss, prvoptimizador, prvmetrics, 'weights_bch-1024_dri-0.2_dru-0.2_reg-401325_col-503_ini-475_mid-475_fin-475_seq-2__25-0.5551.hdf5')
  modelos.append(model)
  model = crear_modelo(seq_len=3, num_capas=3, num_cols=503, lstm_neuronas_ini=475, lstm_neuronas_mid=475, lstm_neuronas_fin=475, dropout_in=0.2, dropout_U=0.2, mi_loss=prvloss, mi_optimizador=prvoptimizador, mis_metrics=prvmetrics)
  leer_pesos_modelo(model, prvloss, prvoptimizador, prvmetrics, 'weights_bch-1024_dri-0.2_dru-0.2_reg-281407_col-503_ini-475_mid-475_fin-475_seq-3__28-0.5215.hdf5')
  modelos.append(model)
  model = crear_modelo(seq_len=4, num_capas=3, num_cols=503, lstm_neuronas_ini=475, lstm_neuronas_mid=475, lstm_neuronas_fin=475, dropout_in=0.2, dropout_U=0.2, mi_loss=prvloss, mi_optimizador=prvoptimizador, mis_metrics=prvmetrics)
  leer_pesos_modelo(model, prvloss, prvoptimizador, prvmetrics, 'weights_bch-1024_dri-0.2_dru-0.2_reg-215619_col-503_ini-475_mid-475_fin-475_seq-4__27-0.4841.hdf5')
  modelos.append(model)
  model = crear_modelo(seq_len=5, num_capas=3, num_cols=503, lstm_neuronas_ini=475, lstm_neuronas_mid=475, lstm_neuronas_fin=475, dropout_in=0.2, dropout_U=0.2, mi_loss=prvloss, mi_optimizador=prvoptimizador, mis_metrics=prvmetrics)
  leer_pesos_modelo(model, prvloss, prvoptimizador, prvmetrics, 'weights_bch-1024_dri-0.2_dru-0.2_reg-160051_col-503_ini-475_mid-475_fin-475_seq-5__43-0.4383.hdf5')
  modelos.append(model)
  model = crear_modelo(seq_len=6, num_capas=3, num_cols=503, lstm_neuronas_ini=475, lstm_neuronas_mid=475, lstm_neuronas_fin=475, dropout_in=0.2, dropout_U=0.2, mi_loss=prvloss, mi_optimizador=prvoptimizador, mis_metrics=prvmetrics)
  leer_pesos_modelo(model, prvloss, prvoptimizador, prvmetrics, 'weights_bch-1024_dri-0.2_dru-0.2_reg-137828_col-503_ini-475_mid-475_fin-475_seq-6__42-0.3983.hdf5')
  modelos.append(model)
  model = crear_modelo(seq_len=7, num_capas=3, num_cols=503, lstm_neuronas_ini=475, lstm_neuronas_mid=475, lstm_neuronas_fin=475, dropout_in=0.2, dropout_U=0.2, mi_loss=prvloss, mi_optimizador=prvoptimizador, mis_metrics=prvmetrics)
  leer_pesos_modelo(model, prvloss, prvoptimizador, prvmetrics, 'weights_bch-1024_dri-0.2_dru-0.2_reg-72986_col-503_ini-475_mid-475_fin-475_seq-7__27-0.3617.hdf5')
  modelos.append(model)
  model = crear_modelo(seq_len=8, num_capas=3, num_cols=503, lstm_neuronas_ini=475, lstm_neuronas_mid=475, lstm_neuronas_fin=475, dropout_in=0.2, dropout_U=0.2, mi_loss=prvloss, mi_optimizador=prvoptimizador, mis_metrics=prvmetrics)
  leer_pesos_modelo(model, prvloss, prvoptimizador, prvmetrics, 'weights_bch-1024_dri-0.2_dru-0.2_reg-104146_col-503_ini-475_mid-475_fin-475_seq-8__61-0.3337.hdf5')
  modelos.append(model)
  model = crear_modelo(seq_len=9, num_capas=3, num_cols=503, lstm_neuronas_ini=475, lstm_neuronas_mid=475, lstm_neuronas_fin=475, dropout_in=0.2, dropout_U=0.2, mi_loss=prvloss, mi_optimizador=prvoptimizador, mis_metrics=prvmetrics)
  leer_pesos_modelo(model, prvloss, prvoptimizador, prvmetrics, 'weights_bch-1024_dri-0.2_dru-0.2_reg-95921_col-503_ini-475_mid-475_fin-475_seq-9__24-0.3077.hdf5')
  modelos.append(model)
  model = crear_modelo(seq_len=10, num_capas=3, num_cols=503, lstm_neuronas_ini=475, lstm_neuronas_mid=475, lstm_neuronas_fin=475, dropout_in=0.2, dropout_U=0.2, mi_loss=prvloss, mi_optimizador=prvoptimizador, mis_metrics=prvmetrics)
  leer_pesos_modelo(model, prvloss, prvoptimizador, prvmetrics, 'weights_bch-512_dri-0.2_dru-0.2_reg-80536_col-503_ini-475_mid-475_fin-475_seq-10__108-0.2841.hdf5')
  modelos.append(model)
  model = crear_modelo(seq_len=11, num_capas=3, num_cols=503, lstm_neuronas_ini=475, lstm_neuronas_mid=475, lstm_neuronas_fin=475, dropout_in=0.2, dropout_U=0.2, mi_loss=prvloss, mi_optimizador=prvoptimizador, mis_metrics=prvmetrics)
  leer_pesos_modelo(model, prvloss, prvoptimizador, prvmetrics, 'weights_bch-1024_dri-0.2_dru-0.2_reg-220_col-503_ini-475_mid-475_fin-475_seq-11__151-0.2879.hdf5')
  modelos.append(model)
  model = crear_modelo(seq_len=12, num_capas=3, num_cols=503, lstm_neuronas_ini=475, lstm_neuronas_mid=475, lstm_neuronas_fin=475, dropout_in=0.2, dropout_U=0.2, mi_loss=prvloss, mi_optimizador=prvoptimizador, mis_metrics=prvmetrics)
  leer_pesos_modelo(model, prvloss, prvoptimizador, prvmetrics, 'weights_bch-512_dri-0.2_dru-0.2_reg-10257_col-503_ini-475_mid-475_fin-475_seq-12__100-0.2718.hdf5')
  modelos.append(model)
  return [modelos, batchsizes]

def leer_y_entrenar(model, batchsize, totFichsMax = 2, totFichsPorNumAdMax = 2, X_train = None, y_train = None, X_valid = None, y_valid = None, X_test = None, y_test = None, MaxRegsAcum = 400000):
  totFichs = 0
  for numAds in range(12, 1, -1): # 2,...,12
    totFichsPorNumAd = 0
    for numFich in range(1, totFichsPorNumAdMax+1): # 1,...,(n-1)
      if not X_train is None:
        if X_train.shape[0] > MaxRegsAcum: # 300000 # 610000
          X_train,y_train,X_valid,y_valid,X_test,y_test = None,None,None,None,None,None
      # Leemos train_X_numad_numfich...
      X_ant, y_ant = X_train, y_train
      X_train, y_train = leer_y_reshape('train', seq_len, numAds, numFich, X_ant = X_ant, y_ant = y_ant)
      if X_train is None:
        X_train, y_train = X_ant, y_ant
        break
      X_valid, y_valid = leer_y_reshape('valid', seq_len, numAds, numFich, X_ant = X_valid, y_ant = y_valid)
      X_test, y_test = leer_y_reshape('test', seq_len, numAds, numFich, X_ant = X_test, y_ant = y_test)
      if X_valid.shape[0] > 0:
        # Entrenamos:
        num_reg_train = X_train.shape[0]
        num_cols = X_train.shape[2]
        entrenar_modelo(model, X_train[0:num_reg_train,:,:], y_train[0:num_reg_train,:], X_valid, y_valid, batchsize, iteraciones, mi_early_stop, mi_shuffle, b_guardar = True)
        evaluar_modelo(model, X_valid, y_valid, 'Valid', batchsize)
        evaluar_modelo(model, X_test, y_test, 'Test', batchsize)
        totFichs += 1
        totFichsPorNumAd += 1
        ## Predecimos en todos los testset de este numAds:
        #for numFich_red in range(1, 999999): # 1,...,(n-1)
        #  X_test_red, y_test_red = leer_y_reshape('test', seq_len, numAds, numFich)
        #  if X_test_red is None:
        #    break
        #  evaluar_modelo(model, X_test_red, y_test_red, 'Test', batchsize)
        #  guardar_preds(model, X_test_red, 0, b_csv = False, numAds = numAds, numAdsFich = numFich_red, num_reg_train=num_reg_train, num_cols=num_cols, iteraciones=iteraciones, batchsize=batchsize, dropout_in=dropout_in, dropout_U=dropout_U)
        X_test_red, y_test_red = leer_y_reshape('test', seq_len, numAds, numFich)
        evaluar_modelo(model, X_test_red, y_test_red, 'Test', batchsize)
        guardar_preds(model, X_test_red, 0, b_csv = False, numAds = numAds, numAdsFich = numFich, num_reg_train=num_reg_train, num_cols=num_cols, iteraciones=iteraciones, batchsize=batchsize, dropout_in=dropout_in, dropout_U=dropout_U)
      else:
        print('No hay registros en X_valid. No se puede entrenar...')
        totFichs += 1
        totFichsPorNumAd += 1
      if totFichs >= totFichsMax:
        break
    
    if totFichs >= totFichsMax:
      break
    
  print('Ok. ' + str(totFichs) + ' ficheros usados para entrenar (y otros tantos para validar y otros tantos para test)')
  return [X_train, y_train, X_valid, y_valid, X_test, y_test]

def predecir_testprobs(modelos, X, batchsizes, X_numAds = None):
  if X_numAds is None: # Un único modelo para todos
    model = modelos[0]
    X = mi_reshape(X, y = None, seq_len = 1)[0]
    probs = model.predict_proba(X, batch_size=batchsizes[0])
  else:
    # En modelos hay 11 modelos (un modelo para cada numAd):
    l_probs = list()
    if len(modelos) != 11:
      print('ERROR: No hay 11 modelos!')
      return(None)
    if len(batchsizes) != 11:
      print('ERROR: No hay 11 batchsizes!')
      return(None)
    X_indices = range(0, X.shape[0]) # [0,.....,X.shape[0]-1] (index original)
    X = cbind[X_indices, X] # Añadimos columna key para recuperar el orden original al final
    # Mejor no ordenar porque no se mantienen las secuencias en su lugar... X_numAds = cbind[X_indices, X_numAds] # Añadimos columna 'key' para recuperar el orden original al final
    # Mejor no ordenar porque no se mantienen las secuencias en su lugar... X_numAds = X_numAds[X_numAds[:,-1].argsort()] # Ordenamos por la última columna (numAd)
    # Mejor no ordenar porque no se mantienen las secuencias en su lugar... X = X[X[:,-1].argsort()] # Ordenamos por la última columna (numAd)
    for numAds in range(2, 13): # 2,...,12
      print('Prediciendo (numAds ' + str(numAds) + ')...')
      model = modelos[numAds - 2] # modelos[0] para numAds=2(i.e. seq_len=2) ... modelos[10] para numAds=12(i.e. seq_len=12)
      X2 = X[X_numAds[:,0] == numAds]
      if X2.shape[0] != 0:
        # Esto en realidad no hace falta... X2 = X2[X2[:,0].argsort()] # Ordenamos por la primera columna (index original) para recuperar las secuencias bien
        X_indices = X2[:,0] # 'key's de estos displays (index original)
        X2 = mi_reshape(X2[:,1:], y = None, seq_len = numAds)[0] # Quitamos esa primera columna (index original) para predecir
        probs = model.predict_proba(X2, batch_size=batchsize)
        probs = mi_reshape_probs(probs, seq_len = numAds) # Volvemos a dos dimensiones
        probs = cbind[X_indices, probs] # Añadimos columna 'key' (index original) para poder recuperar el orden al final
        l_probs.append(probs)
    probs = np_concat(tuple(l_probs), axis=0)
    probs = probs[probs[:,0].argsort()][:,1] # Ordenamos por la primera columna (index original) y nos quedamos con la segunda (las probs)
    # Esto en realidad no hace falta... X = X[X[:,0].argsort()] # Ordenamos por la primera columna (index original)
    # Esto en realidad no hace falta... X_numAds = X_numAds[X_numAds[:,0].argsort()] # Ordenamos por la primera columna (index original)
  
  return(probs)


### ------------------------------
### LEER DATOS:
### ------------------------------
if seq_len != 0:
  X_train, y_train = leer_y_reshape('train', seq_len, numAds = seq_len, numAdsFich = 1)
  X_valid, y_valid = leer_y_reshape('valid', seq_len, numAds = seq_len, numAdsFich = 1)
  X_test, y_test   = leer_y_reshape('test',  seq_len, numAds = seq_len, numAdsFich = 1)
  for numFich in range(2, maxNumFichs+1): # 1,...,(n-1)
    ##if not X_train is None:
    ##  if X_train.shape[0] > MaxRegsAcum: # 300000 # 610000
    ##    X_train,y_train,X_valid,y_valid,X_test,y_test = None,None,None,None,None,None
    # Leemos train_X_numad_numfich...
    X_ant, y_ant = X_train, y_train
    X_train, y_train = leer_y_reshape('train', seq_len, numAds = seq_len, numAdsFich = numFich, X_ant = X_ant,   y_ant = y_ant)
    if X_train is None:
      X_train, y_train = X_ant, y_ant # Ya no hay más ficheros!
      break
    X_valid, y_valid = leer_y_reshape('valid', seq_len, numAds = seq_len, numAdsFich = numFich, X_ant = X_valid, y_ant = y_valid)
    X_test, y_test   = leer_y_reshape('test',  seq_len, numAds = seq_len, numAdsFich = numFich, X_ant = X_test,  y_ant = y_test)
  num_reg_train = X_train.shape[0]
  num_cols = X_train.shape[2]
  print(X_train.shape)

### ------------------------------
### CREAR MODELO:
### ------------------------------
#embedding_vector_length = 32
dropout_in = 0.2
dropout_U = 0.2
num_capas = 3 # 1, 2 ó 3
lstm_neuronas_ini = 475 # 192
lstm_neuronas_mid = 475 # 48
lstm_neuronas_fin = 475 # 12
mi_loss = 'binary_crossentropy' # = mi_mapk12_tf # 'binary_crossentropy' # mapk12_tf # 'binary_crossentropy' # mi_loss = 'mean_absolute_error'
# mi_optimizador = 'adam'
# from keras.optimizers import SGD
# mi_optimizador = SGD(lr=0.01,momentum=0.0, decay=0.0, nesterov=False)
from keras.optimizers import RMSprop
mi_optimizador = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0) # This optimizer is usually a good choice for recurrent neural networks.
mis_metrics = ['accuracy'] # mis_metrics = ['precision']

if seq_len != 0:
  model = crear_modelo(seq_len, num_capas, num_cols, lstm_neuronas_ini, lstm_neuronas_mid, lstm_neuronas_fin, dropout_in, dropout_U, mi_loss, mi_optimizador, mis_metrics)

### ------------------------------
### ENTRENAR MODELO:
### ------------------------------
mi_shuffle = True
batchsize = 1024
mi_early_stop = 50 # max(25, int(iteraciones / 10)) # si no hay mejora (en val_loss) en N rounds seguidos, se detiene el fit (training)

if seq_len != 0:
  eval_ini = evaluar_modelo(model, X_test, y_test, 'Test BEFORE', batchsize)

#num_reg_train = 4096
iteraciones = iteraciones # 5 # 10

# totFichsPorNumAdMax = maxNumFichs
# totFichsMax = 44
# if seq_len != 0:
#   X_train, y_train, X_valid, y_valid, X_test, y_test = leer_y_entrenar(model, batchsize, totFichsMax, totFichsPorNumAdMax) # , b_Entrenar = False) # Para comprobar hasta donde podemos concatenar (sin OutOfMemory)
#   # Segunda pasada, más rápida, pero con el modelo ya entrenado:
#   del X_train, y_train, X_valid, y_valid, X_test, y_test # Vaciamos memoria
#   iteraciones = 2 # 10
#   totFichsPorNumAdMax = 1
#   X_train, y_train, X_valid, y_valid, X_test, y_test = leer_y_entrenar(model, batchsize, totFichsMax, totFichsPorNumAdMax)
#   num_reg_train = X_train.shape[0]
#   num_cols = X_train.shape[2]
# 
# else:
#   #model = leer_pesos_modelo(model, mi_loss, mi_optimizador, mis_metrics, 'weights_bch-1024_dri-0.2_dru-0.2_reg-40236_col-506_ini-361_mid-193_fin-49__00-0.3288.hdf5')
#   #model = leer_pesos_modelo(model, mi_loss, mi_optimizador, mis_metrics, 'weights_bch-2048_dri-0.3_dru-0.3_reg-40236_col-506_ini-475_mid-475_fin-475__02-0.4340.hdf5')
#   #model = leer_pesos_modelo(model, mi_loss, mi_optimizador, mis_metrics, 'weights_bch-2048_dri-0.3_dru-0.3_reg-40236_col-506_ini-475_mid-475_fin-475__02-0.4342.hdf5')
#   pass

#evaluar_modelo(model, X_valid, y_valid, 'Valid', batchsize)
#evaluar_modelo(model, X_test, y_test, 'Test', batchsize)
#guardar_preds(model, X_test) # Guardamos probs [python/test_probs_debug.csv]

# # Evaluamos:
# X_test, y_test = None, None
# for numAds in range(2, 13): # 2,...,12
#   for numFich in range(1, 2): # 1,...,(n-1)
#     X_ant, y_ant = X_test, y_test
#     X_test, y_test = leer_y_reshape('test', seq_len, numAds, numFich, X_ant = X_ant, y_ant = y_ant)
#     if X_test is None:
#       X_test, y_test = X_ant, y_ant
# 
# print(X_test.shape)
# 
# inicio = time.time()
# evaluar_modelo(model, X_test, y_test, 'Test', batchsize);   final = time.time(); print('{x:,.2f}'.format(x = final - inicio) + ' segs.')
# print('-----------------')

# # Finalmente, predecimos en todos los testset:
# for numAds in range(2, 13): # 2,...,12
#   for numFich_red in range(1, 999999): # 1,...,(n-1)
#     X_test_red, y_test_red = leer_y_reshape('test', seq_len, numAds, numFich_red)
#     if X_test_red is None:
#       break
#     evaluar_modelo(model, X_test_red, y_test_red, 'Test', batchsize)
#     guardar_preds(model, X_test_red, 0, b_csv = False, numAds = numAds, numAdsFich = numFich_red, num_reg_train=num_reg_train, num_cols=num_cols, iteraciones=iteraciones, batchsize=batchsize, dropout_in=dropout_in, dropout_U=dropout_U)

### ------------------------------
### ENTRENAR MODELO:
### ------------------------------
if seq_len != 0:
  entrenar_modelo(model, X_train, y_train, X_valid, y_valid, batchsize, iteraciones, mi_early_stop, mi_shuffle, b_guardar = b_guardarDatos)

# # Aumentamos poco a poco los registros y las iteraciones (para inicializar más rápido los pesos):
# for i in range(2, 5): # 0,1,2,3,4
#   num_reg_train = min(X_train.shape[0],  int(X_train.shape[0] ** ( 1.0/2 + (i+1.0)/10 )))
#   mi_early_stop = (1 if i < 4 else 15) # si no hay mejora (en val_loss) en N rounds seguidos, se detiene el fit (training)
#   iteraciones = 4 ** i # 1, 4, 16, 64, 256
#   entrenar_modelo(model, X_train[0:num_reg_train,:,:], y_train[0:num_reg_train,:], X_valid, y_valid, batchsize, iteraciones, mi_early_stop, mi_shuffle, b_guardar = True)
#   evaluar_modelo(model, X_valid, y_valid, 'Valid', batchsize)
#   evaluar_modelo(model, X_test, y_test, 'Test', batchsize)
#   guardar_preds(model, X_test, i + 1)
# 
# # Última pasada, con el triple de batchsize:
# batchsize = int(batchsize * 3)

## mi_early_stop = 25 # si no hay mejora (en val_loss) en N rounds seguidos, se detiene el fit (training)
## iteraciones = 2
## 
## folds = 100
## frac = np_float64(X_train.shape[0]) / folds
## fracv = np_float64(X_valid.shape[0]) / folds
## num_reg_train = int(frac)
## for i in range(0, folds): # 0,1,2,...,(folds-1)
##   ini=int(i * frac)
##   fin=int((i+1) * frac)
##   iniv=int(i * fracv)
##   finv=int((i+1) * fracv)
##   print(100.0*i/folds, ini, fin, num_reg_train)
##   entrenar_modelo(model, X_train[ini:fin,:,:], y_train[ini:fin,:], X_valid[iniv:finv,:,:], y_valid[iniv:finv,:], batchsize, iteraciones, mi_early_stop, mi_shuffle, b_guardar = True)

### ------------------------------
### EVALUAR Y GUARDAR MODELO:
### ------------------------------
if seq_len != 0:
  evaluar_modelo(model, X_valid, y_valid, 'Valid', batchsize)
  evaluar_modelo(model, X_test, y_test, 'Test', batchsize, eval_ini)
  # probs = model.predict_proba(X_test, batch_size=batchsize)
  # print(np_vstack((probs[0:10].T, y_test[0:10].T)).T)
  # # preds = model.predict_classes(X_test, batch_size=batchsize)
  # # print(np_vstack((preds[0:10].T, y_test[0:10].T)).T)
  # print('mapk12 BEFORE = ', mapk12_ini)
  # print('mapk12 AFTER  = ', mi_mapk12(y_test, probs))

#guardar_preds(model, X_test) # Guardamos probs [python/test_probs_debug.csv]

#model = leer_pesos_modelo(model, mi_loss, mi_optimizador, mis_metrics, 'weights_bch-1024_dri-0.1_dru-0.1_reg-436153_col-70_ini-500_mid-500_fin-150__110-0.4193.hdf5')
#evaluar_modelo(model, X_valid, y_valid, 'Valid', batchsize)
#evaluar_modelo(model, X_test, y_test, 'Test', batchsize)

#guardar_preds(model, X_test, indice = 0, b_csv = False, numAds = seq_len, numAdsFich = 0, num_reg_train = num_reg_train, num_cols = num_cols, iteraciones = iteraciones, batchsize = batchsize, dropout_in = dropout_in, dropout_U = dropout_U) # Guardamos probs [python/test_probs_debug.csv]

### ------------------------------
### CREAR TEST SUBMIT:
### ------------------------------
# # Crear ficheros CSV (32x32) de probs para hacer un submit (desde R):
if seq_len == 0:
  if iteraciones == 0:
    generar_test_submit_probs(None, 1, 32) # Predecimos con numAds (11 modelos)
print('Ok.')
