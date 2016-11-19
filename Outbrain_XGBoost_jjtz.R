### Inicialización (setwd() y rm() y packages):

# setwd(getwd())
try(setwd('C:/Users/jtoharia/Dropbox/AFI_JOSE/Kaggle/Outbrain'), silent=TRUE)
try(setwd('C:/Personal/Dropbox/AFI_JOSE/Kaggle/Outbrain'), silent=TRUE)
rm(list = ls()) # Borra todos los elementos del entorno de R.

s_input_path <- "C:/Users/jtoharia/Downloads/Kaggle_Outbrain/"
# s_input_path <- "../input/"
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

# memory.limit(size = 16000)

systime_ini <- proc.time()

# --------------------------------------------------------
G_b_DEBUG <- FALSE # Reducimos todo para hacer pruebas más rápido
NUM_BLOQUES <- 16
# --------------------------------------------------------
gc()
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# https://www.analyticsvidhya.com/blog/2016/01/xgboost-algorithm-easy-steps/
# install.packages("xgboost")
suppressPackageStartupMessages(library(xgboost))
# -------------------------------
# train model with trainset:
# -------------------------------
xgb_reduc_seed <- 1234
xgb_train_porc <- 0.8
xgb_modelos <- vector(mode = "character", length = NUM_BLOQUES)
# tmp_xgb_modelos <- str_replace(dir(path = s_output_path, pattern = paste0(xgb_filename, '.*.modelo')), pattern = "\\.modelo", replacement = "")
# xgb_modelos[as.integer(substr(tmp_xgb_modelos, nchar(tmp_xgb_modelos)-2, nchar(tmp_xgb_modelos)))] <- tmp_xgb_modelos
for(numBatch in 1:NUM_BLOQUES)
{
  if(!is.na(xgb_modelos[numBatch]) & xgb_modelos[numBatch] != "")
  {
    print(paste0('Warning: Ya hay un modelo ', str_pad(numBatch, 3, "left" ,"0"), ' (', xgb_modelos[numBatch], '). Pasamos al siguiente...'))
    next # el "continue" de C
  }
  full_trainset <- leer_batch_train(numBatch, "XGB Training (extreme gradient boosting)", s_input_path)
  # sapply(full_trainset, uniqueN)
  # print(100 * round(table(full_trainset$dia)) / nrow(full_trainset), digits = 3)
  
  if(G_b_DEBUG)
  {
    print('Hacemos un sample para que vaya más rápido...')
    full_trainset <- reducir_trainset(mi_set = full_trainset, n_seed = xgb_reduc_seed, n_porc = 0.25)
  }
  # NOTA: Ahora usaremos los días (1:11) y (12,13), i.e. reducimos en train y valid y, de paso, usamos los días:
  reduc_list <- reducir_trainset_2(mi_set = full_trainset, n_seed = xgb_reduc_seed, n_porc = xgb_train_porc)
  trainset <- reduc_list[[1]]
  validset <- reduc_list[[2]]
  # print(100 * round(table(trainset$dia)) / nrow(trainset), digits = 3)
  # print(100 * round(table(validset$dia)) / nrow(validset), digits = 3)
  
  setkey(trainset, display_id)
  sapply(trainset, uniqueN)
  
  xgb_preps <- xgb_prep_datos(mi_dt = trainset)
  # dt_all    <- xgb_preps[[2]]
  print(colnames(xgb_preps[[2]]))
  sapply(xgb_preps[[2]], uniqueN)
  X <- data.matrix(xgb_preps[[2]][,-1, with=FALSE])
  y <- data.matrix(xgb_preps[[2]][,1, with=FALSE])
  # ===============  XGB params - START  ===============
  # Añadimos 100 a la versión porque ahora tratamos con dia (1:11) y (12,13) por separado (Cf. reducir_trainset_2):
  # Añadimos 200 a la versión porque ahora tratamos con dia (1:11) y (12,13) al 50%, como en el testset (Cf. reducir_trainset_2):
  xgb_mi_version <- 200 + xgb_preps[[1]] # Esto viene de xgb_prep_datos (indica cambios en las "features" seleccionadas)
  xgb_objective = "binary:logistic" # logistic regression for binary classification, output probability
  # xgb_objective = "rank:pairwise" # set XGBoost to do ranking task by minimizing the pairwise loss
  xgb_eta = 0.1 # step size of each boosting step (si baja, afina mejor pero habrá que subir nround)
  xgb_max_depth = 15 # maximum depth of the tree (demasiado alto => overfitting)
  xgb_nround = 200 # max number of iterations
  xgb_subsample = 0.6 # (muestreo aleatorio por filas si es < 1, para intentar evitar overfitting)
  xgb_colsample_bytree = 0.6 # (muestreo aleatorio por columnas si es < 1, para intentar evitar overfitting)
  # xgb_eval_metric = "rmse": root mean square error, "mae" = mean absolut error, "logloss": negative log-likelihood
  xgb_eval_metric = "map"
  # xgb_eval_metric = "error"
  # xgb_eval_metric = "mae"
  # xgb_eval_metric = "logloss"
  # xgb_eval_metric = "auc"
  # xgb_objective = "multi:softprob"
  # xgb_eval_metric = "merror"
  # xgb_num_class = 12
  xgb_early.stop.round = 20 # stop if performance keeps getting worse consecutively for k rounds.
  # ===============  XGB params - END  ===============
  if(G_b_DEBUG)  xgb_mi_version <- xgb_mi_version + 990000
  # Nombre del fichero para guardar estadísticas (de estos modelos) en un único fichero:
  # xgb_filename <- str_replace(xgb_modelos[numBatch], "_[0-9]*\\.[0-9]*$", "")
  xgb_filename <- paste0(paste("XGB",
                               str_replace(xgb_objective, "^(...).*:(...).*$", "\\1\\2"),
                               xgb_eta, xgb_max_depth, xgb_subsample, xgb_colsample_bytree, xgb_eval_metric, xgb_early.stop.round,
                               paste0('v', str_pad(xgb_mi_version, 3, "left" ,"0")),
                               sep = "_"))
  # Entrenamos:
  rm(xgb_preps); gc()
  print(paste0('Entrenando (CV) ', str_pad(numBatch, 3, "left" ,"0"), '...'))
  mi_tiempo <- system.time({
    xgb_cv <- xgb.cv(data = X, label = y, missing = NA,
                   nfold = 4,
                   objective = xgb_objective, eta = xgb_eta, max_depth = xgb_max_depth, nround = xgb_nround, subsample = xgb_subsample, colsample_bytree = xgb_colsample_bytree, eval_metric = xgb_eval_metric, early.stop.round = xgb_early.stop.round,
                   nthread = 4, verbose = T
                  )
  })
  tiempo_cv <- paste0('Tiempo xgb_cv(): ', mi_tiempo['elapsed']/60, ' minutos')
  print(tiempo_cv)
  if("test.map.mean" %in% colnames(xgb_cv))
  {
    xgb_nround <- which.max(xgb_cv$test.map.mean)
  } else {
    xgb_nround <- which.min(xgb_cv$test.error.mean)
  }
  print(xgb_cv[1:min((xgb_nround+2),nrow(xgb_cv))])
  print(paste0('Best Test eval Nround = ', xgb_nround))
  mi_tiempo <- system.time({
    xgb <- xgboost(data = X, label = y, missing = NA,
                 objective = xgb_objective, eta = xgb_eta, max_depth = xgb_max_depth, nround = xgb_nround, subsample = xgb_subsample, colsample_bytree = xgb_colsample_bytree, eval_metric = xgb_eval_metric, early.stop.round = xgb_early.stop.round,
                 nthread = 4, verbose = 1 # 0,1,2
                )
  })
  tiempo_train <- paste0('Tiempo xgb(): ', mi_tiempo['elapsed']/60, ' minutos')
  print(tiempo_train)
  print("Guardando modelo entrenado...")
  # Añadimos nround obtenido en la CV (y el numBatch) al nombre de fichero de cada modelo:
  xgb_modelos[numBatch] <- paste0(paste(xgb_filename,
                                        xgb_nround,
                                     sep = "_")
                                  , '.', str_pad(numBatch, 3, "left" ,"0")) # lo guardamos pata hacer ensemble luego
  print('Predecimos en trainset para medir nuestro map12:')
  mi_tiempo <- system.time({
    trainset[, prob := predict(xgb, newdata = X, missing = NA)]
  })
  tiempo_predict <- paste0('Tiempo predict(xgb, trainset): ', mi_tiempo['elapsed'], ' segundos')
  print(tiempo_predict)
  # # Tabla de confusión:
  # table(trainset[, .(clicked, prob>.5)])
  mi_xgb_map <- xgb$bestScore
  print(paste0('xgb_map = ', mi_xgb_map))
  mi_map12 <- Map_12(tr_valid = trainset[, .(display_id, ad_id, clicked, prob)], b_verbose = 0)
  print(paste0('map12 = ', mi_map12))
  print(paste0('Guardando modelo entrenado (', xgb_modelos[numBatch] ,'.modelo', ')'))
  if(!G_b_DEBUG)
    xgboost::xgb.save(model = xgb, fname = paste0(s_output_path, xgb_modelos[numBatch] ,'.modelo'))
  # # Leemos modelo:
  # xgb <- xgb.load(paste0(xgb_modelos[numBatch] ,'.modelo'))
  write.table(x = t(as.data.frame(list(batch = paste(numBatch, NUM_BLOQUES, sep = "/"),
                      dim_x = t(dim(X)),
                      tiempo_cv = tiempo_cv,
                      tiempo_train = tiempo_train,
                      tiempo_predict = tiempo_predict,
                      cols = t(dimnames(X)[[2]]),
                      list(xgb_eta=xgb_eta, xgb_max_depth=xgb_max_depth, xgb_subsample=xgb_subsample, xgb_colsample_bytree=xgb_colsample_bytree,xgb_eval_metric=xgb_eval_metric,xgb_early.stop.round=xgb_early.stop.round),
                      xgb_map = mi_xgb_map,
                      reduccion_seed = xgb_reduc_seed,
                      reduccion_train_porcent = xgb_train_porc,
                      map12 = mi_map12))),
            file = paste0(s_output_path, xgb_modelos[numBatch] ,'.txt'), row.names = T, col.names = F, quote = F, sep = " = ")

  suppressWarnings( # "Warning: appending column names to file"
    write.table(x = as.data.frame(list(Modelo = xgb_modelos[numBatch])), file = paste0(s_output_path, xgb_filename ,'.txt'), row.names = T, col.names = F, quote = F, sep = " = ", append = TRUE)
  )
  # Como la importance_matrix tarda bastante en calcularse, no lo hacemos para todos los numBatch:
  if(numBatch %% 5 == 1) # 1, 6, 11, 16
  {
    print('Compute feature importance matrix...')
    importance_matrix <- xgb.importance(feature_names = dimnames(X)[[2]], model = xgb)
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
    #                  main = paste0("XGB varImp.Gain - ", str_pad(numBatch, 3, "left" ,"0")))
    #   # x11(); barplot(importance_matrix[1:8]$Cover, names.arg = importance_matrix[1:8]$Feature,
    #   #                main = paste0("XGB varImp.Cover - ", str_pad(numBatch, 3, "left" ,"0")))
    #   # x11(); barplot(importance_matrix[1:8]$Frequence, names.arg = importance_matrix[1:8]$Feature,
    #   #                main = paste0("XGB varImp.Frequence - ", str_pad(numBatch, 3, "left" ,"0")))
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
  }
  
  # -------------------------------
  # Ahora predecimos en validset, que son los que NO hemos usado para CV:
  # -------------------------------
  xgb_preps <- xgb_prep_datos(mi_dt = validset)
  # dt_all    <- xgb_preps[[2]]
  print(colnames(xgb_preps[[2]]))
  # sapply(xgb_preps[[2]], uniqueN)
  X <- data.matrix(xgb_preps[[2]][,-1, with=FALSE])
  y <- data.matrix(xgb_preps[[2]][,1, with=FALSE])
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
  print(paste0('xgb_map = ', mi_xgb_map))
  mi_map12_valid <- Map_12_diario(tr_valid = validset[, .(display_id, ad_id, dia, clicked, prob)], b_verbose = 0,
                            dias_list = list(c(1:11), c(12,13), c(1:13)))
  colnames(mi_map12_valid) <- c('mi_map12_valid_1-11', 'mi_map12_valid_12-13', 'mi_map12_valid_Total')
  print('----------------------------------------------------------------------------')
  print(cbind(mi_map12_valid[2,], Train = mi_map12)) # Volvemos a imprimir mi_map12 (del Training) al lado.
  print('----------------------------------------------------------------------------')
  # Añadimos a los ficheros de cada training los 3 map_12 del validset:
  write.table(x = t(mi_map12_valid[2,]),
              file = paste0(s_output_path, xgb_modelos[numBatch] ,'.txt'), row.names = T, col.names = F, quote = F, sep = " = ", append = TRUE)
}
# -------------------------------
# predict values in testset:
# -------------------------------
predict_testset <- function()
{
  # Leemos modelo(s):
  if(!exists("xgb_modelos"))  xgb_modelos <- vector(mode = "character", length = NUM_BLOQUES)
  if(length(xgb_modelos[xgb_modelos != ""]) == 0)
  {
    tmp_xgb_modelos <- str_replace(dir(path = s_output_path, pattern = paste0(xgb_filename, '.*.modelo')), pattern = "\\.modelo", replacement = "")
    # Ordenamos modelos por numBatch (aunque aquí da igual porque los vamos a promediar)
    xgb_modelos[as.integer(substr(tmp_xgb_modelos, nchar(tmp_xgb_modelos)-2, nchar(tmp_xgb_modelos)))] <- tmp_xgb_modelos
  }
  # # Ordenamos modelos por numBatch (aunque aquí da igual porque los vamos a promediar)
  # xgb_modelos <- xgb_modelos[order(substr(xgb_modelos, nchar(xgb_modelos) - 3, nchar(xgb_modelos)))]
  stopifnot(length(xgb_modelos[xgb_modelos != ""]) == NUM_BLOQUES)
  xgb <- list() # Inicializamos lista de modelos
  for(i in 1:NUM_BLOQUES)
  {
    print(paste0('Leyendo ', str_pad(numBatch, 3, "left" ,"0"), ' (modelo ', i, ' de ', NUM_BLOQUES, '): ', xgb_modelos[i],'...'))
    xgb[[i]] <- xgb.load(paste0(s_output_path, xgb_modelos[i] ,'.modelo'))
  }
  # s_Fichero_submit <- paste0(unique(str_replace(string = xgb_modelos, pattern = "_[0-9]*\\.[0-9]*$", replace = "")), '_submit.csv')
  s_Fichero_submit <- paste0(str_replace(xgb_filename, "_[0-9]*\\.[0-9]*$", ""), '_submit.csv')
  for(numBatch in 1:NUM_BLOQUES)
  {
    testset <- leer_batch_test(numBatch, "XGB Predicting (extreme gradient boosting)", s_input_path)
    if(G_b_DEBUG)
    {
      # Hacemos un sample para que vaya más rápido...
      testset <- reducir_trainset(mi_set = testset, n_seed = xgb_reduc_seed, n_porc = 0.01)
    }
    # Extreme Gradient Boosting:
    # --------------------------
    # if(!exists("testset"))
    #   load(paste0(s_input_path, "testset.RData"))
    dt_all <- xgb_prep_datos(mi_dt = testset)[[2]]
    print(colnames(dt_all))
    print('Preparando matrix y predicting...')
    X <- data.matrix(dt_all[,-1,with=FALSE])
    # y <- data.matrix(dt_all[,1,with=FALSE])
    y_pred <- vector(mode = "numeric", length = NUM_BLOQUES)
    for(i in 1:NUM_BLOQUES)
    {
      print(paste0('Predicting ', str_pad(numBatch, 3, "left" ,"0"), ' (modelo ', i, ' de ', NUM_BLOQUES, '): ', xgb_modelos[i],'...'))
      # xgb <- xgb.load(paste0(s_output_path, xgb_modelos[i] ,'.modelo'))
      y_pred[i] <- list(xgboost::predict(xgb[[i]], X, missing = NA))
    }
    mean_y_pred <- rowMeans(rbindlist(list(y_pred)))
    testset[, prob := mean_y_pred]
    print(paste0('Guardando ', s_Fichero_submit, ' (', numBatch, '/', NUM_BLOQUES, ')...'))
    guardar_submit(testset = testset, fichero = s_Fichero_submit, b_write_file_append = (numBatch>1), s_output_path = s_output_path)
  }
}
predict_testset()
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

print(paste0(as.double((proc.time() - systime_ini)['elapsed'])/60, ' minutos en total.'))

print('Ok.')

# cleanup:
try(registerDoSEQ(), silent = TRUE) # library(doParallel) [turn parallel processing off and run sequentially again]
try(stopImplicitCluster(), silent = TRUE)
try(stopCluster(cl), silent = TRUE)
