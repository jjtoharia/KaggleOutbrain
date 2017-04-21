### Inicialización (setwd() y rm() y packages):

# setwd(getwd())
try(setwd('C:/Users/jtoharia/Dropbox/AFI_JOSE/Kaggle/Outbrain'), silent=TRUE)
try(setwd('C:/Personal/Dropbox/AFI_JOSE/Kaggle/Outbrain'), silent=TRUE)
try(setwd('D:/Kaggle-Outbrain'), silent=TRUE)
rm(list = ls()) # Borra todos los elementos del entorno de R.

s_input_path <- "C:/Users/jtoharia/Downloads/Kaggle_Outbrain/"
# s_input_path <- "../input/"
if(file.exists("In/testset_001.RData")) s_input_path <- "In/"
s_output_path <- "Out/"

# options(echo = FALSE) # ECHO OFF
print('###########################################')
print('# Outbrain Click Prediction - JJTZ 2016')
print('###########################################')
# NOTA: XGBoost (Extreme Gradient Boosting: "mejor" que RandomForest...
# https://www.analyticsvidhya.com/blog/2016/01/xgboost-algorithm-easy-steps/
# install.packages("data.table")
suppressPackageStartupMessages(library(data.table))
# Process in parallel:
# install.packages("doParallel")
suppressPackageStartupMessages(library(foreach))
library(iterators)
library(parallel)
library(doParallel)
# # Process in parallel: Ejemplo de uso:
# cl <- makeCluster(detectCores(), type='PSOCK') # library(doParallel) [turn parallel processing on]
# registerDoParallel(cl) # library(doParallel) [turn parallel processing on]
# registerDoSEQ() # library(doParallel) [turn parallel processing off and run sequentially again]
# #

# install.packages("bit64")
suppressPackageStartupMessages(library(bit64))
# install.packages("stringr")
library(stringr)
# ##################################################
# ## Funciones:
# ##################################################

source("Outbrain_jjtz_funciones.R", encoding = "UTF-8")

# ##################################################
# ## Inicio:
# ##################################################
Proyecto <- "Outbrain Clicks Prediction"
print(paste0(Sys.time(), ' - ', 'Proyecto = ', Proyecto))

Proyecto.s <- str_replace_all(Proyecto, "\\(|\\)| |:", "_") # Quitamos espacios, paréntesis, etc.

# Inicializamos variables:
# NOTA: Dejamos un Core de la CPU "libre" para no "quemar" la máquina:
cl <- makeCluster(detectCores(), type='PSOCK') # library(doParallel) [turn parallel processing on]
registerDoParallel(cl) # library(doParallel) [turn parallel processing on]

memory.limit(size = 35000)

systime_ini <- proc.time()

# --------------------------------------------------------
G_b_DEBUG <- FALSE # Reducimos todo para hacer pruebas más rápido
G_b_REV <- FALSE # Empezamos por el final (numModelo <- NUM_MODELOS - forNumModelo + 1)
NUM_BLOQUES <- 32
NUM_MODELOS <- 11 # clustering por numAds (de 2 a 12)
minForNumModelo <- 1
maxForNumModelo <- NUM_MODELOS
maxTamFullTrainset <- 100000 # 200.000 para hacerlo "rápido" (y crear las importance_matrix) ó 3.000.000 que es lo máx. que he podido hasta ahora (con muy pocas vars.)
maxImportanceNumVars <- 0 # 150 # Seleccionamos las primeras variables por importancia (de ejecuciones anteriores con misma versión y numAd)
# --------------------------------------------------------
gc()
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# https://www.analyticsvidhya.com/blog/2016/01/xgboost-algorithm-easy-steps/
# install.packages("xgboost")
suppressPackageStartupMessages(library(xgboost))

# suppressPackageStartupMessages(library(lattice)) # Para xgb_prep_datos
# suppressPackageStartupMessages(library(ggplot2)) # Para xgb_prep_datos
# suppressPackageStartupMessages(library(caret))   # Para xgb_prep_datos
# -------------------------------
# Inicializamos:
# -------------------------------
# mi_sum_pos <- sum(full_trainset$clicked == 1) # 16874593
# mi_sum_neg <- sum(full_trainset$clicked == 0) # 70267138
mi_sum_pos <- 16874593
mi_sum_neg <- 70267138
# xgb_scale_pos_weight <- mi_sum_neg / mi_sum_pos # == 4.164079
xgb_reduc_seed <- 1234
xgb_train_porc <- 0.8
xgb_get_predictions <- FALSE # TRUE para hacer stacking de los modelos (Subsemble, SuperLearner)

print('-------------------------------')
print('Primero buscamos algún modelo entrenado pero SIN submit:')
print('-------------------------------')
xgb_tipo_modelo_XGB = "XGB"    # XGBoost
xgb_tipo_modelo_XGRF = "XGRF"  # Random Forest con XGBoost
xgb_modelos <- vector(mode = "character", length = NUM_MODELOS)
xgb_filenames <- vector(mode = "character")
for(mi_tipo in c(xgb_tipo_modelo_XGRF, xgb_tipo_modelo_XGB))
  xgb_filenames <- c(xgb_filenames, unique(str_replace(dir(path = s_output_path, pattern = paste0(mi_tipo, '.*.modelo')), "_[0-9]*\\.[0-9][0-9][0-9]\\.modelo$", "")))
for(mi_tipo in xgb_filenames)
{
  if(file.exists(file.path(s_output_path, paste0(mi_tipo, '_submit.csv'))))
  { xgb_filenames <- xgb_filenames[xgb_filenames != mi_tipo]
  } else {
    if(file.exists(file.path(s_output_path, paste0(mi_tipo, '_submit.zip'))))
      xgb_filenames <- xgb_filenames[xgb_filenames != mi_tipo]
  }
}
for(xgb_filename in xgb_filenames)
{
  print('Encontrado un modelo sin submit...')
  print(xgb_filename)
  tmp_xgb_modelos <- str_replace(dir(path = s_output_path, pattern = paste0(str_replace(xgb_filename, "_[0-9]*\\.[0-9][0-9][0-9]$", ""), '.*.modelo')), pattern = "\\.modelo", replacement = "")
  n_version <- as.integer(substring(str_extract(xgb_filename, pattern = "v[0-9]+"), first = 2))
  if(n_version > 500)
    next # Es de los de numAds
  if(length(tmp_xgb_modelos) == NUM_MODELOS)
  {
    # Ordenamos modelos por numModelo (aquí NO da igual, porque ya NO los vamos a promediar)
    xgb_modelos[as.integer(substr(tmp_xgb_modelos, nchar(tmp_xgb_modelos)-2, nchar(tmp_xgb_modelos)))] <- tmp_xgb_modelos
    break # Ok. Pasamos directamente a predecir con estos modelos.
  } else {
    print('NOTA: Lo descartamos porque no están todos entrenados')
  }
}
n_versiones <- as.integer(substring(str_extract(xgb_filenames, pattern = "v[0-9]+"), first = 2))
for(n_version in unique(n_versiones)[unique(n_versiones) > 500])
{
  nSubmits <- length(dir(path = s_output_path, pattern = paste0('.*', xgb_tipo_modelo_XGRF,'.*_v', n_version, '.*.submit.*')))
  nSubmits <- nSubmits + length(dir(path = s_output_path, pattern = paste0('.*', xgb_tipo_modelo_XGB,'.*_v', n_version, '.*.submit.*')))
  if(nSubmits != 0)
  {
    print(paste0('NOTA: Descartamos v', n_version, '. Encontrado(s) ', nSubmits, ' submit(s) con esta versión.'))
    break
  } else { print(paste0('Buscando modelos con versión ', n_version)) }
  tmp_xgb_modelos <- vector(mode = "character", length = NUM_MODELOS)
  xgb_filenames_version <- xgb_filenames[n_versiones == n_version]
  for(xgb_filename in xgb_filenames)
  {
    if(n_version != as.integer(substring(str_extract(xgb_filename, pattern = "v[0-9]+"), first = 2)))
      next
    mis_modelos <- dir(path = s_output_path, pattern = paste0(xgb_filename, '.*.modelo'))
    for(mi_modelo in mis_modelos)
    {
      numAds <- 1 + as.integer(substring(str_extract(mi_modelo, pattern = "\\.[0-9][0-9][0-9]\\.modelo"), first = 2, last = 4))
      if(tmp_xgb_modelos[numAds - 1] != ""){
        print(paste0('Nota: Encontrado más de un modelo ', str_pad(numAds-1,3,"left","0"), ' v', n_version, ' (numAds = ', numAds, '). Nos quedamos con el menor...'))
        if(tmp_xgb_modelos[numAds - 1] > str_replace(mi_modelo, pattern = "\\.modelo", replacement = ""))
          tmp_xgb_modelos[numAds - 1] <- str_replace(mi_modelo, pattern = "\\.modelo", replacement = "")
      } else {
        tmp_xgb_modelos[numAds - 1] <- str_replace(mi_modelo, pattern = "\\.modelo", replacement = "")
      }
    }
  }
  if(length(tmp_xgb_modelos[tmp_xgb_modelos != ""]) == NUM_MODELOS)
  {
    # Ordenamos modelos por numModelo (aquí NO da igual, porque ya NO los vamos a promediar)
    xgb_modelos[as.integer(substr(tmp_xgb_modelos, nchar(tmp_xgb_modelos)-2, nchar(tmp_xgb_modelos)))] <- tmp_xgb_modelos
    print(paste0('Hay ', NUM_MODELOS, ' modelos v', n_version, ' (sin submit). Pasamos directamente a predecir con ellos [', xgb_filename, '].'))
    print(xgb_modelos)
    break # Ok. Pasamos directamente a predecir con estos modelos.
  } else {
    print(paste0('NOTA: Descartamos v', n_version, ' porque no están todos entrenados (faltan algunos numAds)'))
    print(substring(str_extract(tmp_xgb_modelos, pattern = "\\.[0-9][0-9][0-9]$"), first = 2, last = 4))
  }
}
if(any(xgb_modelos == "") | anyNA(xgb_modelos))
{
  print('-------------------------------')
  print('Entrenamos:')
  print('-------------------------------')

  for(forNumModelo in minForNumModelo:maxForNumModelo)
  {
    if(!G_b_REV) numModelo <- forNumModelo else numModelo <- (NUM_MODELOS - forNumModelo + 1) # Empezamos por el final (numModelo <- NUM_MODELOS - forNumModelo + 1)

    if(!is.na(xgb_modelos[numModelo]) & xgb_modelos[numModelo] != "")
    {
      print(paste0('Warning: Ya hay un modelo ', str_pad(numModelo, 3, "left", "0"), ' (', xgb_modelos[numModelo], '). Pasamos al siguiente...'))
      next # el "continue" de C
    }
    miDescr <- paste0("XGB Training - [numAds cluster ", numModelo, "]")
    numAdsCluster <- numModelo + 1
    fich_name <- paste0("train_valid_", str_pad(numAdsCluster, 2, "left", "0"), "__", xgb_reduc_seed, "__", xgb_train_porc, ".RData") # "train_valid_nn__ssss__pppp.RData"
    
    if(file.exists(paste0(s_input_path, fich_name)))
    {
      print(paste0(miDescr, ' Batch trainset+validset ALL (numAds == ', numAdsCluster, ')'))
      load(file = paste0(s_input_path, fich_name))
      print(paste0(miDescr, ' Batch trainset+validset ALL (numAds == ', numAdsCluster, ')', ' Ok. ', format(nrow(trainset), scientific = F, decimal.mark = ',', big.mark = '.'), ' + ', format(nrow(validset), scientific = F, decimal.mark = ',', big.mark = '.'), ' regs.'))
    } else {
  		fich_name <- paste0("full_trainset_", str_pad(numAdsCluster, 2, "left", "0"), ".RData") # "full_trainset_nn.RData"
  		
  		if(file.exists(paste0(s_input_path, fich_name)))
  		{
  		  print(paste0(miDescr, ' Batch trainset ALL (numAds == ', numAdsCluster, ')'))
  		  load(file = paste0(s_input_path, fich_name))
  		  print(paste0(miDescr, ' Batch trainset ALL (numAds == ', numAdsCluster, ')', ' Ok. ', nrow(full_trainset), ' regs.'))
  		  numBatch <- NUM_BLOQUES # Para dejar claro que hemos cargado todos los ficheros!
  		} else {
  		  numBatch <- 1
  		  fich_name <- paste0("full_trainset_", str_pad(numAdsCluster, 2, "left", "0"), '_', str_pad(numBatch, 3, "left", "0"), ".RData") # "full_trainset_nn_001.RData"
  		  
  		  if(!file.exists(paste0(s_input_path, fich_name)))
  		  {
    			full_trainset <- leer_batch_train(numBatch, miDescr, s_input_path)[numAds == numAdsCluster,]
  		  } else {
    			# print(paste0(miDescr, ' Batch trainset ', numBatch, ' (numAds == ', numAdsCluster, ')'))
    			load(file = paste0(s_input_path, fich_name))
    			print(paste0(miDescr, ' Batch trainset ', numBatch, ' (numAds == ', numAdsCluster, ')', ' Ok. ', nrow(full_trainset), ' regs.'))
  		  }
  		  print(paste0(Sys.time(), ' - ', 'Ok. full_trainset: ', nrow(full_trainset), ' registros.'))
  		  nrowsPorBloqueEstim <- nrow(full_trainset)
  		  for(numBatch in 2:NUM_BLOQUES)
  		  {
    			# if(file.exists(paste0(s_input_path, "clicks_full_trainset_2x_debug.csv")))
    			# {
    			#   if(numBatch > 1)
    			#      break # Finished!
    			#   print(paste0(Sys.time(), ' - ', 'XGB Training (extreme gradient boosting) - PRUEBA CON clicks_full_trainset_2x_debug.csv...'))
    			#   full_trainset <- fread(paste0(s_input_path, "clicks_full_trainset_2x_debug.csv"))
    			# } else {
    				# if(nrow(full_trainset) + nrowsPorBloqueEstim > ifelse(G_b_DEBUG, 50000, maxTamFullTrainset))
    				# {
    				#   print(paste0("NOTA: Modelo ", str_pad(numModelo, 3, "left", "0")," (numAds == ", numAdsCluster, ") NO está completo! (Solo tiene ", numBatch-1, "/", NUM_BLOQUES, " batches)"))
    				#   break # Paramos antes de pasarnos...
    				# }
  				fich_name <- paste0("full_trainset_", str_pad(numAdsCluster, 2, "left", "0"), '_', str_pad(numBatch, 3, "left", "0"), ".RData")
  	  
  				full_trainset_tmp <- full_trainset
  				if(!file.exists(paste0(s_input_path, fich_name)))
  				{
  				  full_trainset_tmp <- rbindlist(list(full_trainset_tmp, leer_batch_train(numBatch, miDescr, s_input_path)[numAds == numAdsCluster,]))
  				} else {
  				  # print(paste0(miDescr, 'Batch trainset ', numBatch, ' (numAds == ', numAdsCluster, ')'))
  				  load(file = paste0(s_input_path, fich_name))
  				  print(paste0(miDescr, 'Batch trainset ', numBatch, ' (numAds == ', numAdsCluster, ')', ' Ok. ', nrow(full_trainset), ' regs.'))
  				  full_trainset_tmp <- rbindlist(list(full_trainset_tmp, full_trainset))
  				}
  				full_trainset <- full_trainset_tmp
  				print(paste0(Sys.time(), ' - ', 'Ok. full_trainset: ', nrow(full_trainset), ' registros.'))
  				if(nrow(full_trainset) > ifelse(G_b_DEBUG, 50000, maxTamFullTrainset))
  				{
  				  if(numBatch < NUM_BLOQUES)
  					print(paste0("NOTA: Modelo ", str_pad(numModelo, 3, "left", "0")," (numAds == ", numAdsCluster, ") NO está completo! (Solo tiene ", numBatch, "/", NUM_BLOQUES, " batches)"))
  				  break # Paramos antes de pasarnos...
  				}
  				
    			#   if(!G_b_DEBUG)
    			#   {
    			#     # Ampliamos con uno más:
    			#     if(numBatch > 1)
    			#       break # Finished!
    			#     full_trainset <- rbind(full_trainset, leer_batch_train(numBatch + 1, "XGB Training (extreme gradient boosting) 2x Blocks", s_input_path))
    			#   }
    			# }
    			# # sapply(trainset, uniqueN)[sapply(trainset, uniqueN)==1]
    			# # sapply(full_trainset, uniqueN)
    			# # print(100 * round(table(full_trainset$dia)) / nrow(full_trainset), digits = 3)
  		  }
  		  if(exists("full_trainset_tmp")) { rm(full_trainset_tmp); gc() }
  		  if(numBatch == NUM_BLOQUES)
  		  {
    			# Hemos cargado todos los ficheros, así que podemos guardarlo en uno solo:
    			fich_name <- paste0("full_trainset_", str_pad(numAdsCluster, 2, "left", "0"), ".RData") # "full_trainset_nn.RData"
    			save(full_trainset, file = file.path(s_input_path, fich_name))
  		  }
  		}
  		
  		if(G_b_DEBUG)
  		{
  		  print('Hacemos un sample para que vaya más rápido...')
  		  full_trainset <- reducir_trainset(mi_set = full_trainset, n_seed = xgb_reduc_seed, n_porc = 0.1)
  		}
  		if(nrow(full_trainset) > maxTamFullTrainset)
  		{
  		  print(paste0('Hacemos un sample (', format(maxTamFullTrainset, scientific = F, decimal.mark = ',', big.mark = '.'), ') para que vaya más rápido...'))
  		  full_trainset <- reducir_trainset(mi_set = full_trainset, n_seed = xgb_reduc_seed, n_porc = (maxTamFullTrainset /  nrow(full_trainset)))
  		}
  		# if(!file.exists(paste0(s_input_path, "clicks_full_trainset_2x_debug.csv")))
  		# {
  		#   print(paste0(Sys.time(), ' - ', 'Guardando full_trainset en ', s_input_path, 'clicks_full_trainset_2x_debug.csv...'))
  		#   write.table(full_trainset, file = paste0(s_input_path, "clicks_full_trainset_2x_debug.csv"), row.names=F, quote=F, sep=",")
  		# }
  		# print(paste0(Sys.time(), ' - ', 'Ok. full_trainset: ', nrow(full_trainset), ' registros.'))
  		reduc_list <- reducir_trainset_4(mi_set = full_trainset, n_seed = xgb_reduc_seed, n_porc = xgb_train_porc)
  		# Reducimos uso de memoria:
  		if(!G_b_DEBUG)  rm(full_trainset); gc()
  		trainset <- reduc_list[[1]]
  		validset <- reduc_list[[2]]
  		setkey(trainset, display_id)
  		# Reducimos uso de memoria:
  		if(!G_b_DEBUG)  rm(reduc_list); gc()
  		if(numBatch == NUM_BLOQUES)
  		{
  			# Hemos cargado todos los ficheros, así que podemos guardar trainset y validset directamente:
  			fich_name <- paste0("train_valid_", str_pad(numAdsCluster, 2, "left", "0"), "__", xgb_reduc_seed, "__", xgb_train_porc, ".RData") # "train_valid_nn__ssss__pppp.RData"
  			save(trainset, validset, file = file.path(s_input_path, fich_name))
  		}
  	}
    # print(100 * round(table(trainset$dia)) / nrow(trainset), digits = 3)
    # print(100 * round(table(validset$dia)) / nrow(validset), digits = 3)

    if(nrow(trainset) + nrow(validset) > maxTamFullTrainset)
    {
      print(paste0('Hacemos un sample (', format(maxTamFullTrainset, scientific = F, decimal.mark = ',', big.mark = '.'), ') para que vaya más rápido...'))
      mi_n_porc = maxTamFullTrainset / (nrow(trainset) + nrow(validset))
      trainset <- reducir_trainset(mi_set = trainset, n_seed = xgb_reduc_seed, n_porc = mi_n_porc)
      validset <- reducir_trainset(mi_set = validset, n_seed = xgb_reduc_seed, n_porc = mi_n_porc)
    }

    print(paste0(Sys.time(), ' - ', 'Ok. trainset: ', format(nrow(trainset), scientific = F, decimal.mark = ',', big.mark = '.'), ' registros.  validset: ', format(nrow(validset), scientific = F, decimal.mark = ',', big.mark = '.'), ' registros.'))
    
    # sapply(trainset, uniqueN)[sapply(trainset, uniqueN)==1]
    
    xgb_preps <- xgb_prep_datos(mi_dt = NULL, b_verbose = 1) # Obtenemos solamente la versión
    # ===============  XGB params - START  ===============
    # Añadimos 100 a la versión porque ahora tratamos con dia (1:11) y (12,13) por separado (Cf. reducir_trainset_2):
    # Añadimos 200 a la versión porque ahora tratamos con dia (1:11) y (12,13) al 50%, como en el testset (Cf. reducir_trainset_2):
    # Añadimos 300 a la versión porque ahora tratamos con topics_prob_1, etc. de nuevo (Cf. reducir_trainset_2):
    # NOTA: v.3xx - También hemos añadido nuevos parámetros: scale_pos_weight, min_child_weight y gamma
    # Añadimos 400 a la versión porque ahora quitamos lo de los días (1:11)(12:13) (volvemos a un sample aleatorio y ampliamos el Map_12_diario() para ver si hay diferencias) (Cf. reducir_trainset_4):
    # Añadimos 500 a la versión porque ahora usamos clustering por numAds
    # Añadimos 600 a la versión porque ahora usamos selección de variables (por importanceMatrix)
    xgb_mi_version <- 600 + xgb_preps[[1]] # Esto viene de xgb_prep_datos (indica cambios en las "features" seleccionadas)
    xgb_objective = "binary:logistic" # logistic regression for binary classification, output probability
    # xgb_objective = "rank:pairwise" # set XGBoost to do ranking task by minimizing the pairwise loss
    xgb_eta = 0.07 # 0.3 [0.01 - 0.2] step size of each boosting step (si baja, afina mejor pero habrá que subir nround)
    xgb_max_depth = 14 # maximum depth of the tree (demasiado alto => overfitting)
    xgb_nround = 200 # max number of iterations
    if(G_b_DEBUG & xgb_nround > 1)  xgb_nround = 20
    xgb_subsample = 1 # (muestreo aleatorio por filas si es < 1, para intentar evitar overfitting)
    xgb_colsample_bytree = 0.8 # (muestreo aleatorio por columnas si es < 1, para intentar evitar overfitting)
    # xgb_eval_metric = "rmse": root mean square error, "mae" = mean absolut error, "logloss": negative log-likelihood
    xgb_eval_metric = "map"
    # xgb_eval_metric = "error"
    # xgb_eval_metric = "mae"
    # xgb_eval_metric = "logloss"
    # xgb_eval_metric = "auc"
    # xgb_objective = "multi:softprob"
    # xgb_eval_metric = "merror"
    # xgb_num_class = 12
    xgb_scale_pos_weight = 1 # numAdsCluster # Cambio por numAds clustering # mi_sum_neg / mi_sum_pos # == 4.164079
    xgb_min_child_weight = 0.25 + (numAdsCluster/2 - 1) # 1 [0.1 - 10]
    xgb_gamma = 0.1 # 0 [0 - 2] the larger, the more conservative the algorithm will be.
    xgb_early.stop.round = 25 # stop if performance keeps getting worse consecutively for k rounds.
    xgb_num_parallel_tree = 1000 # Random Forests with XGBoost (si xgb_nround == 1 & xgb_subsample < 1 & xgb_colsample_bytree < 1, claro)
    if(xgb_nround > 1 | xgb_subsample == 1 | xgb_colsample_bytree == 1)
    {
      xgb_s_tipo_modelo = xgb_tipo_modelo_XGB   # "XGB" XGBoost
      # xgb_num_parallel_tree = 1
      xgb_num_parallel_tree = 4 # Prueba que, con pocos datos, parece mejorar (aunque tarda mucho más)
    } else
    {
      xgb_s_tipo_modelo = xgb_tipo_modelo_XGRF # "XGRF" Random Forest con XGBoost
      xgb_nround = 1
      if(G_b_DEBUG & xgb_num_parallel_tree > 1)  xgb_num_parallel_tree = 100
    }
    # ===============  XGB params - END  ===============
    if(G_b_DEBUG)  xgb_mi_version <- xgb_mi_version + 990000
    # Nombre del fichero para guardar estadísticas (de estos modelos) en un único fichero:
    # xgb_filename <- str_replace(xgb_modelos[numModelo], "_[0-9]*\\.[0-9][0-9][0-9]$", "")
    xgb_filename <- paste0(paste(xgb_s_tipo_modelo,
                                 str_replace(xgb_objective, "^(...).*:(...).*$", "\\1\\2"),
                                 xgb_eta, xgb_max_depth, xgb_subsample, xgb_colsample_bytree, xgb_eval_metric, xgb_early.stop.round, 0, xgb_min_child_weight, xgb_gamma, xgb_num_parallel_tree, #Quitamos xgb_scale_pos_weight (ponemos un cero)
                                 paste0('v', str_pad(xgb_mi_version, 3, "left", "0")),
                                 sep = "_"))
    # Verificamos que no exista el modelo ya entrenado (fichero):
    if(length(dir(path = s_output_path, pattern = paste0(xgb_filename, '.*', '.', str_pad(numModelo, 3, "left", "0"), '.modelo'))) != 0)
    {
      xgb_modelos[numModelo] <- dir(path = s_output_path, pattern = paste0(xgb_filename, '.*', '.', str_pad(numModelo, 3, "left", "0"), '.modelo'))[1]
      xgb_modelos[numModelo] <- str_replace(xgb_modelos[numModelo], pattern = "\\.modelo", replacement = "")
      print(paste0('Nota: Encontrado modelo ya entrenado. Lo cargamos ', str_pad(numModelo, 3, "left", "0"), ' (', xgb_modelos[numModelo], ') y pasamos al siguiente...'))
      next # el "continue" de C
    }
    
    xgb_preps <- xgb_prep_datos(mi_dt = trainset, b_verbose = 1, maxImportanceNumVars = maxImportanceNumVars)
    # Seleccionamos variables (si hay otros modelos previos del mismo numAd y misma versión y con importance_matrix):
    # importance_vars <- xgb_prep_datos_busca_imp_matrix(xgb_filename) # vector de variables ordenadas por importancia (las primeras son las más importantes)
    # xgb_preps[[2]] <- xgb_prep_datos_selec_vars(mi_dt = trainset, numVars = 50, importance_vars, b_verbose = 1)
    print(colnames(xgb_preps[[2]])) # Finalmente, mostramos las columnas elegidas
    
    # sapply(xgb_preps[[2]], uniqueN)[sapply(xgb_preps[[2]], uniqueN)==1]
    X <- data.matrix(xgb_preps[[2]][,-1, with=FALSE])
    y <- xgb_preps[[2]]$clicked
    # midata <- xgb.DMatrix(data.matrix(xgb_preps[[2]][,-1, with=FALSE]), label = xgb_preps[[2]]$clicked, missing = NA)

    # Reducimos memoria:
    if(!G_b_DEBUG)  rm(xgb_preps); gc()
    s_misDims <- paste0(format(dim(X)[1], scientific = F, decimal.mark = ',', big.mark = '.'), ' regs/',
                       format(dim(X)[2], scientific = F, decimal.mark = ',', big.mark = '.'), ' cols')
    # Entrenamos:
    print(paste0(Sys.time(), ' - ', 'Entrenando (CV) ', str_pad(numModelo, 3, "left", "0"), '...[', xgb_filename, '] [', s_misDims, ']'))
    
    mi_tiempo <- system.time({
    xgb_cv <- xgb.cv(data = X, label = y, missing = NA, prediction = xgb_get_predictions,
                     nfold = 4, nround = xgb_nround,
                     objective = xgb_objective, eta = xgb_eta, max_depth = xgb_max_depth, subsample = xgb_subsample, colsample_bytree = xgb_colsample_bytree,
                     metrics = list(xgb_eval_metric), # eval_metric = xgb_eval_metric,
                     early_stopping_rounds = xgb_early.stop.round,
                     scale_pos_weight = xgb_scale_pos_weight, min_child_weight = xgb_min_child_weight, gamma = xgb_gamma,
                     num_parallel_tree = xgb_num_parallel_tree,
                     nthread = (detectCores()-1),
                     verbose = T,
                     save_period = NULL
                    )
    })
    tiempo_cv <- paste0('Tiempo xgb_cv(): ', mi_tiempo['elapsed']/60, ' minutos')
    print(paste0(Sys.time(), ' - ', tiempo_cv, ' [', s_misDims, ']'))
    if(xgb_get_predictions)
    {
      xgb_preds <- xgb_cv$pred # Predicciones en los N-Folds no usados para entrenar (vector de tamaño nrow(trainset))
      xgb_cv <- xgb_cv$dt      # Este es el mismo data.table que devolvía xgb.cv con prediction=FALSE
    }
    if(xgb_eval_metric == "map")
    {
      xgb_nround <- which.max(xgb_cv$evaluation_log$test_map_mean)
      mi_xgb_cv_train_bestScore <- max(xgb_cv$evaluation_log$train_map_mean)
      mi_xgb_cv_test_bestScore <- max(xgb_cv$evaluation_log$test_map_mean)
    } else if(xgb_eval_metric == "error") {
      xgb_nround <- which.min(xgb_cv$evaluation_log$test_error_mean)
      mi_xgb_cv_train_bestScore <- min(xgb_cv$evaluation_log$train_error_mean)
      mi_xgb_cv_test_bestScore <- min(xgb_cv$evaluation_log$test_map_mean)
    } else if(xgb_eval_metric == "auc") {
      xgb_nround <- which.max(xgb_cv$evaluation_log$test_auc_mean)
      mi_xgb_cv_train_bestScore <- max(xgb_cv$evaluation_log$train_auc_mean)
      mi_xgb_cv_test_bestScore <- max(xgb_cv$evaluation_log$test_map_mean)
    } else {
      save(xgb_cv, file = paste0(s_output_path, "xgb_cv_temp.RData"))
      stop('eval_metric desconocida (Cf. xgb_cv_temp.RData!')
    }
    print(xgb_cv$evaluation_log[1:min((xgb_nround+2),nrow(xgb_cv))])
    print(paste0(Sys.time(), ' - ', 'Best Test eval Nround = ', xgb_nround))
    mi_tiempo <- system.time({
      xgb <- xgboost(data = X, label = y, missing = NA, nround = xgb_nround,
                     objective = xgb_objective, eta = xgb_eta, max_depth = xgb_max_depth, subsample = xgb_subsample, colsample_bytree = xgb_colsample_bytree, eval_metric = xgb_eval_metric,
                     # early_stopping_rounds = xgb_early.stop.round,
                     nthread = (detectCores()),
                     verbose = 1, # 0,1,2
                     num_parallel_tree = xgb_num_parallel_tree,
                     save_period = NULL
                     )
    })
    tiempo_train <- paste0('Tiempo xgb(): ', mi_tiempo['elapsed']/60, ' minutos')
    print(paste0(Sys.time(), ' - ', tiempo_train, ' [', s_misDims, ']'))
    
    print("Guardando modelo entrenado...")
    # Añadimos nround obtenido en la CV (y el numModelo) al nombre de fichero de cada modelo:
    xgb_modelos[numModelo] <- paste0(paste(xgb_filename,
                                           xgb_nround,
                                        sep = "_")
                                     , '.', str_pad(numModelo, 3, "left", "0")) # lo guardamos pata hacer ensemble luego
    print('Predecimos en trainset para medir nuestro map12:')
    mi_tiempo <- system.time({
      trainset[, prob := predict(xgb, newdata = X, missing = NA)]
    })
    tiempo_predict <- paste0('Tiempo predict(xgb, trainset): ', mi_tiempo['elapsed'], ' segundos')
    print(paste0(Sys.time(), ' - ', tiempo_predict, ' [', s_misDims, ']'))
    # # Tabla de confusión:
    # table(trainset[, .(clicked, prob>.5)])
    mi_xgb_map <- as.numeric(xgb$evaluation_log[xgb_nround, 2]) # NOTA: Si quitamos early.stop.round de xgboost(), esto es NULL
    print(paste0(Sys.time(), ' - ', 'xgb_map = ', mi_xgb_map, '. numAds = ', numAdsCluster))
    mi_map12 <- Map_12(tr_valid = trainset[, .(display_id, ad_id, clicked, prob)], b_verbose = 0)
    print(paste0(Sys.time(), ' - ', 'map12 = ', mi_map12))
    print(paste0(Sys.time(), ' - ', 'Guardando modelo entrenado (', xgb_modelos[numModelo] ,'.modelo', ')'))
    if(!G_b_DEBUG)
      xgboost::xgb.save(model = xgb, fname = paste0(s_output_path, xgb_modelos[numModelo] ,'.modelo'))
    # # Leemos modelo:
    # xgb <- xgb.load(paste0(s_output_path, str_replace(xgb_modelos[numModelo], pattern = "\\.modelo", replacement = ""), '.modelo'))
    write.table(x = t(as.data.frame(list(modelo = paste(numModelo, NUM_MODELOS, sep = "/"),
                        dim_x = t(dim(X)),
                        tiempo_cv = tiempo_cv,
                        tiempo_train = tiempo_train,
                        tiempo_predict = tiempo_predict,
                        cols = t(dimnames(X)[[2]]),
                        list(xgb_eta=xgb_eta, xgb_max_depth=xgb_max_depth, xgb_subsample=xgb_subsample, xgb_colsample_bytree=xgb_colsample_bytree,xgb_eval_metric=xgb_eval_metric,xgb_early.stop.round=xgb_early.stop.round),
                        xgb_map = mi_xgb_map,
                        reduccion_seed = xgb_reduc_seed,
                        reduccion_train_porcent = xgb_train_porc,
                        xgb_cv_train_bestScore = mi_xgb_cv_train_bestScore,
                        xgb_cv_test_bestScore = mi_xgb_cv_test_bestScore,
                        map12 = mi_map12))),
              file = paste0(s_output_path, xgb_modelos[numModelo] ,'.txt'), row.names = T, col.names = F, quote = F, sep = " = ")
  
    suppressWarnings( # "Warning: appending column names to file"
      write.table(x = as.data.frame(list(Modelo = xgb_modelos[numModelo])), file = paste0(s_output_path, xgb_filename ,'.txt'), row.names = T, col.names = F, quote = F, sep = " = ", append = TRUE)
    )
    
    # Como la importance_matrix tarda bastante en calcularse, no lo hacemos para todos los numModelo:
    # if(numModelo %% 3 == 1) # 1, 4, 7, 10
    # {
      print('Compute feature importance matrix...')
      mi_tiempo <- system.time({
        importance_matrix <- xgb.importance(feature_names = dimnames(X)[[2]], model = xgb)
      })
      tiempo_imp_matrix <- paste0('Tiempo xgb.importance(): ', mi_tiempo['elapsed'], ' segundos')
      print(paste0(Sys.time(), ' - ', tiempo_imp_matrix, ' [', s_misDims, ']'))
      suppressWarnings(
        write.table(x = as.data.frame(importance_matrix)[,c(1,2)], file = paste0(s_output_path, xgb_filename ,'.txt'), row.names = T, col.names = T, quote = F, sep = "\t", append = TRUE)
      )
      suppressWarnings(
        write.table(x = as.data.frame(importance_matrix)[,c(1,3)], file = paste0(s_output_path, xgb_filename ,'.txt'), row.names = T, col.names = T, quote = F, sep = "\t", append = TRUE)
      )
      suppressWarnings(
        write.table(x = as.data.frame(importance_matrix)[,c(1,4)], file = paste0(s_output_path, xgb_filename ,'.txt'), row.names = T, col.names = T, quote = F, sep = "\t", append = TRUE)
      )
      # if(G_b_DEBUG) # xgb.plot.importance(), barplot(), chisq.test()...
      # {
      #   print(dimnames(X)[[2]])
      #   # # Nice graph
      #   # # install.packages("Ckmeans.1d.dp")
      #   # library(Ckmeans.1d.dp)
      #   # x11(); xgb.plot.importance(importance_matrix)
      #   
      #   #In case last step does not work for you because of a version issue, you can try following :
      #   x11(); barplot(importance_matrix[1:8]$Gain, names.arg = importance_matrix[1:8]$Feature,
      #                  main = paste0("XGB varImp.Gain - ", str_pad(numModelo, 3, "left", "0")))
      #   # x11(); barplot(importance_matrix[1:8]$Cover, names.arg = importance_matrix[1:8]$Feature,
      #   #                main = paste0("XGB varImp.Cover - ", str_pad(numModelo, 3, "left", "0")))
      #   # x11(); barplot(importance_matrix[1:8]$Frequence, names.arg = importance_matrix[1:8]$Feature,
      #   #                main = paste0("XGB varImp.Frequence - ", str_pad(numModelo, 3, "left", "0")))
      #   
      #   # # install.packages("DiagrammeR")
      #   # library(DiagrammeR)
      #   # x11(); xgb.plot.tree(feature_names = dimnames(X)[[2]], model = xgb, n_first_tree = 3)
      #   
      #   # # To see whether the variable is actually important or not:
      #   # test <- chisq.test(y, as.numeric(trainset$ad_publish_timestamp))
      #   # test <- chisq.test(y, as.numeric(trainset$publish_timestamp))
      #   # test <- chisq.test(y, as.numeric(trainset$numAds))
      #   # print(test)
      # }
    # }
    
    print('-------------------------------')
    print('Ahora predecimos en validset, que son los que NO hemos usado para CV:')
    print('-------------------------------')
    print(paste0(Sys.time(), ' - ', 'trainset: ', format(nrow(trainset), scientific = F, decimal.mark = ',', big.mark = '.'), ' registros.  validset: ', format(nrow(validset), scientific = F, decimal.mark = ',', big.mark = '.'), ' registros.'))
    xgb_preps <- xgb_prep_datos(mi_dt = validset, b_verbose = 1, maxImportanceNumVars = maxImportanceNumVars)
    # Seleccionamos variables (si hay otros modelos previos del mismo numAd y misma versión y con importance_matrix):
    # importance_vars <- xgb_prep_datos_busca_imp_matrix(xgb_filename) # vector de variables ordenadas por importancia (las primeras son las más importantes)
    # xgb_preps[[2]] <- xgb_prep_datos_selec_vars(mi_dt = trainset, numVars = 50, importance_vars, b_verbose = 1)
    # print(colnames(xgb_preps[[2]]))
    # sapply(xgb_preps[[2]], uniqueN)
    X <- data.matrix(xgb_preps[[2]][,-1, with=FALSE])
    y <- xgb_preps[[2]]$clicked
    # midata <- xgb.DMatrix(data.matrix(xgb_preps[[2]][,-1, with=FALSE]), label = xgb_preps[[2]]$clicked, missing = NA)
    rm(xgb_preps); gc()
    print('Predecimos en validset para medir nuestro map12:')
    mi_tiempo <- system.time({
      validset[, prob := predict(xgb, newdata = X, missing = NA)]
    })
    tiempo_predict <- paste0('Tiempo predict(xgb, validset): ', mi_tiempo['elapsed'], ' segundos')
    print(tiempo_predict)
    # # Tabla de confusión:
    # table(validset[, .(clicked, prob>.5)])
    mi_xgb_map <- xgb$bestScore
    print(paste0(Sys.time(), ' - ', 'xgb_map = ', mi_xgb_map, '. numAds = ', numAdsCluster))
    mi_map12_valid <- Map_12_diario(tr_valid = validset[, .(display_id, ad_id, dia, clicked, prob)], b_verbose = 0,
                              dias_list = list(c(1:3), c(3:5), c(5:7), c(7:9), c(9:11), c(11:13), c(1:11), c(12:13), c(1:13)))
    colnames(mi_map12_valid) <- c('map12val_1-3', 'map12val_3-5', 'map12val_5-7', 'map12val_7-9', 'map12val_9-11', 'map12val_11-13', 'map12val_1-11', 'map12val_12-13', 'map12val_Total')
    map12val_Total <- mi_map12_valid[2, "map12val_Total"][[1]]
    p_esp <- 1 / numAdsCluster
    mejora <- 100 * (map12val_Total - p_esp) * (1 - p_esp)
    stats2 <- data.frame(numModelo = str_pad(numModelo, 3, "left", "0"), numAds = numAdsCluster, map12val_Total = map12val_Total, mejora = mejora)
    print('----------------------------------------------------------------------------')
    print(cbind(mi_map12_valid[2,], map12Train = mi_map12, map12cvTest = mi_xgb_cv_test_bestScore)) # Volvemos a imprimir mi_map12 (del Training) al lado.
    print(stats2)
    print('----------------------------------------------------------------------------')
    # Añadimos a los ficheros de cada training los 3 map_12 del validset:
    write.table(x = t(mi_map12_valid[2,]),
                file = paste0(s_output_path, xgb_modelos[numModelo] ,'.txt'), row.names = T, col.names = F, quote = F, sep = " = ", append = TRUE)
    # Añadimos a los ficheros de cada training un resumen con la mejora (porcentaje sobre el máximo posible teórico):
    write.table(x = t(stats2[,-3]),
                file = paste0(s_output_path, xgb_modelos[numModelo] ,'.txt'), row.names = T, col.names = F, quote = F, sep = " = ", append = TRUE)
    
    minutos <- as.double((proc.time() - systime_ini)['elapsed'])/60
    if(minutos < 60) print(paste0(Sys.time(), ' - ', minutos, ' minutos en total.', ' - ', 'trainset: ', format(nrow(trainset), scientific = F, decimal.mark = ',', big.mark = '.'), ' registros.  validset: ', format(nrow(validset), scientific = F, decimal.mark = ',', big.mark = '.'), ' registros.')) else  print(paste0(Sys.time(), ' - ', minutos/60, ' horas en total.', ' - ', 'trainset: ', format(nrow(trainset), scientific = F, decimal.mark = ',', big.mark = '.'), ' registros.  validset: ', format(nrow(validset), scientific = F, decimal.mark = ',', big.mark = '.'), ' registros.'))
    minutos_pend <- (as.double((proc.time() - systime_ini)['elapsed'])/60) * (NUM_MODELOS / numModelo  -  1)
    if(minutos_pend < 60) print(paste0(Sys.time(), ' - ', 'Faltan aprox. ', minutos_pend, ' minutos.')) else print(paste0(Sys.time(), ' - ', 'Faltan aprox. ', minutos_pend/60, ' horas.'))
  }
}

# length(xgb_modelos) <- 1 # Si tenemos menos modelos, podemos crear submit con menos...

print('-------------------------------')
print('Predecimos en testset y creamos submit:')
print('-------------------------------')
if(G_b_DEBUG)
{
  print('NOTA: G_b_DEBUG == TRUE -> No creamos submit.csv')
} else {
  if(!exists("xgb_modelos"))  xgb_modelos <- vector(mode = "character", length = NUM_MODELOS)
  predict_testset(nombres_modelos = xgb_modelos
                  , filename = xgb_filename, s_input_path = s_input_path, s_output_path = s_output_path
                  , i_sDescr = "XGB Predicting"
                  , FUN_prep_datos = xgb_prep_datos, prep_datos_b_verbose = 1
                  , FUN_X = data.matrix
                  , FUN_Predict = function(modelo, X){ return(predict(modelo, X, missing = NA)) }
                  , FUN_loadmodelo = xgb.load
                  , nombreModeloLoaded = ""
                  , NUM_MODELOS = length(xgb_modelos)
                  , b_ReducirFicheros = FALSE # De momento no hace falta reducir más los testset, pero podría hacer falta...
  )
}
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# M12 <- foreach(reg = c(0,3,4,5,6), .inorder=TRUE, .combine = c, .packages = c('data.table'),
#                .export = c('Map_12_diario', 'Map_12', 'mapk_nodup_only_first_12', 'add_tiempo')
#                ) %do%
# {
#   return(
#     basic_preds_m12(trainset = trainset, validset = validset, k = 2, reg = reg, b_verbose = 1)
#     ) # ret_val de la función "foreach() %do%" (o foreach( %dopar%))
# }
# # M12    <- basic_preds_m12(trainset = trainset, validset = validset, k = 1)
# # M12[2] <- basic_preds_m12(trainset = trainset, validset = validset, k = 2)
# # M12[3] <- basic_preds_m12(trainset = trainset, validset = validset, k = 3)
# print(M12)
# # sort(M12)
# sort(unlist(sapply(M12, function(x) {return(x$V1[2])})))
# sort(unlist(sapply(M12, function(x) {return(x$V2[2])})))
# sort(unlist(sapply(M12, function(x) {return(x$V3[2])})))

tot_mins <- as.double((proc.time() - systime_ini)['elapsed'])/60
if(tot_mins < 60){
  print(paste0(Sys.time(), ' - ', round(tot_mins,3), ' minutos en total.'))
} else if(tot_mins < 60 * 24) {
  print(paste0(Sys.time(), ' - ', round(tot_mins/60,3), ' horas en total.'))
} else {
  print(paste0(Sys.time(), ' - ', round(tot_mins/(60*24),3), ' días en total.'))
}

print('Ok.')

# cleanup:
try(registerDoSEQ(), silent = TRUE) # library(doParallel) [turn parallel processing off and run sequentially again]
try(stopImplicitCluster(), silent = TRUE)
try(stopCluster(cl), silent = TRUE)
