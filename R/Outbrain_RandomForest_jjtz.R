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

# memory.limit(size = 14000)

systime_ini <- proc.time()

# --------------------------------------------------------
G_b_DEBUG <- FALSE # Reducimos todo para hacer pruebas más rápido
NUM_BLOQUES <- 16
# --------------------------------------------------------
gc()
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# https://www.analyticsvidhya.com/blog/2016/01/xgboost-algorithm-easy-steps/
# # install.packages("randomForest")
# suppressPackageStartupMessages(library(randomForest))
# install.packages("xgboost")
suppressPackageStartupMessages(library(xgboost))
# -------------------------------
# train model with trainset:
# -------------------------------
# mi_sum_pos <- sum(full_trainset$clicked == 1) # 16874593
# mi_sum_neg <- sum(full_trainset$clicked == 0) # 70267138
mi_sum_pos <- 16874593
mi_sum_neg <- 70267138
# randfor_scale_pos_weight <- mi_sum_neg / mi_sum_pos # == 4.164079
randfor_reduc_seed <- 1234
randfor_train_porc <- 0.8
randfor_modelos <- vector(mode = "character", length = NUM_BLOQUES)
# tmp_randfor_modelos <- str_replace(dir(path = s_output_path, pattern = paste0(randfor_filename, '.*.modelo')), pattern = "\\.modelo", replacement = "")
# randfor_modelos[as.integer(substr(tmp_randfor_modelos, nchar(tmp_randfor_modelos)-2, nchar(tmp_randfor_modelos)))] <- tmp_randfor_modelos
for(numBatch in 1:NUM_BLOQUES)
{
  if(!is.na(randfor_modelos[numBatch]) & randfor_modelos[numBatch] != "")
  {
    print(paste0('Warning: Ya hay un modelo ', str_pad(numBatch, 3, "left" ,"0"), ' (', randfor_modelos[numBatch], '). Pasamos al siguiente...'))
    next # el "continue" de C
  }
  full_trainset <- leer_batch_train(numBatch, "RF Training (Random Forest)", s_input_path)
  # sapply(full_trainset, uniqueN)
  # print(100 * round(table(full_trainset$dia)) / nrow(full_trainset), digits = 3)
  
  if(G_b_DEBUG)
  {
    print('Hacemos un sample para que vaya más rápido...')
    full_trainset <- reducir_trainset(mi_set = full_trainset, n_seed = randfor_reduc_seed, n_porc = 0.05)
  }
  # NOTA: Ahora usaremos los días (1:11) y (12,13), i.e. reducimos en train y valid y, de paso, usamos los días:
  reduc_list <- reducir_trainset_2(mi_set = full_trainset, n_seed = randfor_reduc_seed, n_porc = randfor_train_porc)
  trainset <- reduc_list[[1]]
  validset <- reduc_list[[2]]
  # print(100 * round(table(trainset$dia)) / nrow(trainset), digits = 3)
  # print(100 * round(table(validset$dia)) / nrow(validset), digits = 3)
  
  setkey(trainset, display_id)
  sapply(trainset, uniqueN)
  
  randfor_preps <- randfor_prep_datos(mi_dt = trainset)
  # dt_all    <- randfor_preps[[2]]
  print(colnames(randfor_preps[[2]]))
  # sapply(randfor_preps[[2]], uniqueN)
  X <- data.matrix(randfor_preps[[2]][,-1, with=FALSE])
  y <- factor(as.data.frame(randfor_preps[[2]])[,1]) # factor para que RandomForest use defaults para clasificación y no regresión
  # ===============  RandomForest params - START  ===============
  # Añadimos 100 a la versión porque ahora tratamos con dia (1:11) y (12,13) por separado (Cf. reducir_trainset_2):
  # Añadimos 200 a la versión porque ahora tratamos con dia (1:11) y (12,13) al 50%, como en el testset (Cf. reducir_trainset_2):
  # Añadimos 300 a la versión porque ahora tratamos con topics_prob_1, etc. de nuevo (Cf. reducir_trainset_2):
  # NOTA: v.3xx - También hemos añadido nuevos parámetros: scale_pos_weight, min_child_weight y gamma
  randfor_mi_version <- 300 + randfor_preps[[1]] # Esto viene de randfor_prep_datos (indica cambios en las "features" seleccionadas)
  randfor_ntrees = 200 # 500 [1000 - 5000] Number of trees to grow
  if(G_b_DEBUG)  randfor_ntrees = 10
  randfor_replace = FALSE # Should sampling of cases be done with or without replacement?
  ranfor_classwt = c(mi_sum_neg, mi_sum_pos)/(mi_sum_neg + mi_sum_pos) # Priors of the classes.
  randfor_nodesize = 1 # 1 [1 - 10] Minimum size of terminal nodes. Larger causes smaller trees to be grown (and thus take less time).
  randfor_mtry = 8 # sqrt(p) Number of variables randomly sampled as candidates at each split. Note that the default values are different for classification (sqrt(p) where p is number of variables in x) and regression (p/3)
  randfor_do.trace = TRUE # If set to TRUE, give a more verbose output as randomForest is run. If set to some integer, then running output is printed for every do.trace trees.
  # randfor_do.trace = 1 # If set to TRUE, give a more verbose output as randomForest is run. If set to some integer, then running output is printed for every do.trace trees.
  # ===============  randfor params - END  ===============
  if(G_b_DEBUG)  randfor_mi_version <- randfor_mi_version + 990000
  # Nombre del fichero para guardar estadísticas (de estos modelos) en un único fichero:
  # randfor_filename <- str_replace(randfor_modelos[numBatch], "_[0-9]*\\.[0-9]*$", "")
  randfor_filename <- paste0(paste("randfor",
                               str_replace(randfor_objective, "^(...).*:(...).*$", "\\1\\2"),
                               randfor_nodesize, 
                               paste0('v', str_pad(randfor_mi_version, 3, "left" ,"0")),
                               sep = "_"))
  # Entrenamos:
  if(!G_b_DEBUG)  rm(randfor_preps); gc()
  print(paste0('Entrenando (CV) ', str_pad(numBatch, 3, "left" ,"0"), '...'))
  sapply(full_trainset, anyNA)
  mi_tiempo <- system.time({
  randfor_cv <- rfcv(trainx = X, trainy = y, cv.fold = 4, scale = "log", step = 0.25, recursive = TRUE,
                     mtry = function(p){return(max(randfor_mtry, sqrt(p)))} # Como mínimo quiero usar 8 variables
                    )
  })
  tiempo_cv <- paste0('Tiempo randfor_cv(): ', mi_tiempo['elapsed']/60, ' minutos')
  print(tiempo_cv)
  
  if("test.map.mean" %in% colnames(randfor_cv))
  {
    randfor_nround <- which.max(randfor_cv$test.map.mean)
  } else {
    randfor_nround <- which.min(randfor_cv$test.error.mean)
  }
  print(randfor_cv[1:min((randfor_nround+2),nrow(randfor_cv))])
  print(paste0('Best Test eval Nround = ', randfor_nround))
  mi_tiempo <- system.time({
    fit.RANDFOR <- randomForest(x = X, y = y, importance = TRUE, proximity = TRUE, keep.forest = TRUE,
                                ntree = randfor_ntrees, replace = randfor_replace, classwt = ranfor_classwt, nodesize = randfor_nodesize, mtry = randfor_mtry, do.trace = randfor_do.trace)
  })
  tiempo_train <- paste0('Tiempo randfor(): ', mi_tiempo['elapsed']/60, ' minutos')
  print(tiempo_train)
  print("Guardando modelo entrenado...")
  # Añadimos nround obtenido en la CV (y el numBatch) al nombre de fichero de cada modelo:
  randfor_modelos[numBatch] <- paste0(paste(randfor_filename,
                                        randfor_nround,
                                     sep = "_")
                                  , '.', str_pad(numBatch, 3, "left" ,"0")) # lo guardamos pata hacer ensemble luego

  
  
  
  
  # Gráfico de las variables más importantes:
  varImpPlot(fit.RANDFOR)
  
  # print("Tabla de clasificación (trainset):")
  # print(fit.RANDFOR$confusion)
  # print(table(trainset$P_C, fit.RANDFOR$predicted, deparse.level = 2))
  # # Error cometido (trainset):
  # err.TR_RANDFOR <- (1 - mean(fit.RANDFOR$predicted == trainset$P_C))
  # print(paste("Error cometido (trainset) =", 100 * round(err.TR_RANDFOR,4), "%"))
  # # Verificamos modelo (testset) (calculamos AUC, etc.):
  # predict.RANDFOR.class <- predict(fit.RANDFOR, testset[,-9], type = 'response')
  # print("Tabla de clasificación (testset):")
  # print(table(testset$P_C, predict.RANDFOR.class, deparse.level = 2))
  # # Error cometido (testset):
  # err.RANDFOR <- (1 - mean(predict.RANDFOR.class == testset$P_C))
  # print(paste("Error cometido (testset) =", 100 * round(err.RANDFOR,4), "%"))
  print('Predecimos en trainset para medir nuestro map12:')
  mi_tiempo <- system.time({
    trainset[, prob := predict(fit.RANDFOR, newdata = X, type = 'prob')[,"1"]] # devuelve las probabilidades de pertenencia
  })
  tiempo_predict <- paste0('Tiempo predict(randfor, trainset): ', mi_tiempo['elapsed'], ' segundos')
  print(tiempo_predict)
  # # Tabla de confusión:
  # table(trainset[, .(clicked, prob>.5)])
  mi_randfor_map <- randfor$bestScore
  print(paste0('randfor_map = ', mi_randfor_map))
  mi_map12 <- Map_12(tr_valid = trainset[, .(display_id, ad_id, clicked, prob)], b_verbose = 0)
  print(paste0('map12 = ', mi_map12))
  print(paste0('Guardando modelo entrenado (', randfor_modelos[numBatch] ,'.modelo', ')'))
  if(!G_b_DEBUG)
    randforoost::randfor.save(model = randfor, fname = paste0(s_output_path, randfor_modelos[numBatch] ,'.modelo'))
  # # Leemos modelo:
  # randfor <- randfor.load(paste0(randfor_modelos[numBatch] ,'.modelo'))
  write.table(x = t(as.data.frame(list(batch = paste(numBatch, NUM_BLOQUES, sep = "/"),
                      dim_x = t(dim(X)),
                      tiempo_cv = tiempo_cv,
                      tiempo_train = tiempo_train,
                      tiempo_predict = tiempo_predict,
                      cols = t(dimnames(X)[[2]]),
                      list(randfor_eta=randfor_eta, randfor_max_depth=randfor_max_depth, randfor_subsample=randfor_subsample, randfor_colsample_bytree=randfor_colsample_bytree,randfor_eval_metric=randfor_eval_metric,randfor_early.stop.round=randfor_early.stop.round),
                      randfor_map = mi_randfor_map,
                      reduccion_seed = randfor_reduc_seed,
                      reduccion_train_porcent = randfor_train_porc,
                      map12 = mi_map12))),
            file = paste0(s_output_path, randfor_modelos[numBatch] ,'.txt'), row.names = T, col.names = F, quote = F, sep = " = ")

  suppressWarnings( # "Warning: appending column names to file"
    write.table(x = as.data.frame(list(Modelo = randfor_modelos[numBatch])), file = paste0(s_output_path, randfor_filename ,'.txt'), row.names = T, col.names = F, quote = F, sep = " = ", append = TRUE)
  )
  # Como la importance_matrix tarda bastante en calcularse, no lo hacemos para todos los numBatch:
  if(numBatch %% 5 == 1) # 1, 6, 11, 16
  {
    print('Compute feature importance matrix...')
    importance_matrix <- randfor.importance(feature_names = dimnames(X)[[2]], model = randfor)
    suppressWarnings(
      write.table(x = as.data.frame(importance_matrix)[,c(1,2)], file = paste0(s_output_path, randfor_filename ,'.txt'), row.names = T, col.names = T, quote = F, sep = "\t", append = TRUE)
    )
    suppressWarnings(
      write.table(x = as.data.frame(importance_matrix)[,c(1,3)], file = paste0(s_output_path, randfor_filename ,'.txt'), row.names = T, col.names = T, quote = F, sep = "\t", append = TRUE)
    )
    suppressWarnings(
      write.table(x = as.data.frame(importance_matrix)[,c(1,4)], file = paste0(s_output_path, randfor_filename ,'.txt'), row.names = T, col.names = T, quote = F, sep = "\t", append = TRUE)
    )
    # if(G_b_DEBUG) # randfor.plot.importance(), barplot(), chisq.test()...
    # {
    #   print(dimnames(X)[[2]])
    #   # # Nice graph
    #   # # install.packages("Ckmeans.1d.dp")
    #   # library(Ckmeans.1d.dp)
    #   # x11(); randfor.plot.importance(importance_matrix)
    #   
    #   #In case last step does not work for you because of a version issue, you can try following :
    #   x11(); barplot(importance_matrix[1:8]$Gain, names.arg = importance_matrix[1:8]$Feature,
    #                  main = paste0("randfor varImp.Gain - ", str_pad(numBatch, 3, "left" ,"0")))
    #   # x11(); barplot(importance_matrix[1:8]$Cover, names.arg = importance_matrix[1:8]$Feature,
    #   #                main = paste0("randfor varImp.Cover - ", str_pad(numBatch, 3, "left" ,"0")))
    #   # x11(); barplot(importance_matrix[1:8]$Frequence, names.arg = importance_matrix[1:8]$Feature,
    #   #                main = paste0("randfor varImp.Frequence - ", str_pad(numBatch, 3, "left" ,"0")))
    #   
    #   # # install.packages("DiagrammeR")
    #   # library(DiagrammeR)
    #   # x11(); randfor.plot.tree(feature_names = dimnames(X)[[2]], model = randfor, n_first_tree = 3)
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
  randfor_preps <- randfor_prep_datos(mi_dt = validset)
  # dt_all    <- randfor_preps[[2]]
  print(colnames(randfor_preps[[2]]))
  # sapply(randfor_preps[[2]], uniqueN)
  X <- data.matrix(randfor_preps[[2]][,-1, with=FALSE])
  y <- data.matrix(randfor_preps[[2]][,1, with=FALSE])
  rm(randfor_preps); gc()
  print('Predecimos en validset para medir nuestro map12:')
  mi_tiempo <- system.time({
    validset[, prob := predict(fit.RANDFOR, newdata = X, type = 'prob')[,"1"]]
  })
  tiempo_predict <- paste0('Tiempo predict(randfor, validset): ', mi_tiempo['elapsed'], ' segundos')
  print(tiempo_predict)
  # # Tabla de confusión:
  # table(validset[, .(clicked, prob>.5)])
  mi_randfor_map <- randfor$bestScore
  print(paste0('randfor_map = ', mi_randfor_map))
  mi_map12_valid <- Map_12_diario(tr_valid = validset[, .(display_id, ad_id, dia, clicked, prob)], b_verbose = 0,
                            dias_list = list(c(1:11), c(12,13), c(1:13)))
  colnames(mi_map12_valid) <- c('mi_map12_valid_1-11', 'mi_map12_valid_12-13', 'mi_map12_valid_Total')
  print('----------------------------------------------------------------------------')
  print(cbind(mi_map12_valid[2,], Train = mi_map12)) # Volvemos a imprimir mi_map12 (del Training) al lado.
  print('----------------------------------------------------------------------------')
  # Añadimos a los ficheros de cada training los 3 map_12 del validset:
  write.table(x = t(mi_map12_valid[2,]),
              file = paste0(s_output_path, randfor_modelos[numBatch] ,'.txt'), row.names = T, col.names = F, quote = F, sep = " = ", append = TRUE)
  
  print(paste0(as.double((proc.time() - systime_ini)['elapsed'])/60, ' minutos en total.'))
  minutos_pend <- (as.double((proc.time() - systime_ini)['elapsed'])/60) * (NUM_BLOQUES / numBatch  -  1)
  if(minutos_pend < 60) print(paste0('Faltan aprox. ',minutos_pend, ' minutos.')) else print(paste0('Faltan aprox. ',minutos_pend/60, ' horas.'))
}
# -------------------------------
# predict values in testset:
# -------------------------------
predict_testset <- function()
{
  # Leemos modelo(s):
  if(!exists("randfor_modelos"))  randfor_modelos <- vector(mode = "character", length = NUM_BLOQUES)
  if(length(randfor_modelos[randfor_modelos != ""]) == 0)
  {
    tmp_randfor_modelos <- str_replace(dir(path = s_output_path, pattern = paste0(randfor_filename, '.*.modelo')), pattern = "\\.modelo", replacement = "")
    # Ordenamos modelos por numBatch (aunque aquí da igual porque los vamos a promediar)
    randfor_modelos[as.integer(substr(tmp_randfor_modelos, nchar(tmp_randfor_modelos)-2, nchar(tmp_randfor_modelos)))] <- tmp_randfor_modelos
  }
  # # Ordenamos modelos por numBatch (aunque aquí da igual porque los vamos a promediar)
  # randfor_modelos <- randfor_modelos[order(substr(randfor_modelos, nchar(randfor_modelos) - 3, nchar(randfor_modelos)))]
  stopifnot(length(randfor_modelos[randfor_modelos != ""]) == NUM_BLOQUES)
  randfor <- list() # Inicializamos lista de modelos
  for(i in 1:NUM_BLOQUES)
  {
    print(paste0('Leyendo ', str_pad(numBatch, 3, "left" ,"0"), ' (modelo ', i, ' de ', NUM_BLOQUES, '): ', randfor_modelos[i],'...'))
    randfor[[i]] <- randfor.load(paste0(s_output_path, randfor_modelos[i] ,'.modelo'))
  }
  # s_Fichero_submit <- paste0(unique(str_replace(string = randfor_modelos, pattern = "_[0-9]*\\.[0-9]*$", replace = "")), '_submit.csv')
  s_Fichero_submit <- paste0(str_replace(randfor_filename, "_[0-9]*\\.[0-9]*$", ""), '_submit.csv')
  systime_ini2 <- proc.time()
  for(numBatch in 1:NUM_BLOQUES)
  {
    testset <- leer_batch_test(numBatch, "randfor Predicting (extreme gradient boosting)", s_input_path)
    if(G_b_DEBUG)
    {
      # Hacemos un sample para que vaya más rápido...
      testset <- reducir_trainset(mi_set = testset, n_seed = randfor_reduc_seed, n_porc = 0.01)
    }
    # Extreme Gradient Boosting:
    # --------------------------
    # if(!exists("testset"))
    #   load(paste0(s_input_path, "testset.RData"))
    dt_all <- randfor_prep_datos(mi_dt = testset)[[2]]
    print(colnames(dt_all))
    print('Preparando matrix y predicting...')
    X <- data.matrix(dt_all[,-1,with=FALSE])
    # y <- data.matrix(dt_all[,1,with=FALSE])
    y_pred <- vector(mode = "numeric", length = NUM_BLOQUES)
    for(i in 1:NUM_BLOQUES)
    {
      print(paste0('Predicting ', str_pad(numBatch, 3, "left" ,"0"), ' (modelo ', i, ' de ', NUM_BLOQUES, '): ', randfor_modelos[i],'...'))
      # randfor <- randfor.load(paste0(s_output_path, randfor_modelos[i] ,'.modelo'))
      y_pred[i] <- list(predict(randfor[[i]], newdata = X, type = 'prob')[,"1"])
    }
    mean_y_pred <- rowMeans(rbindlist(list(y_pred)))
    testset[, prob := mean_y_pred]
    print(paste0('Guardando ', s_Fichero_submit, ' (', numBatch, '/', NUM_BLOQUES, ')...'))
    guardar_submit(testset = testset, fichero = s_Fichero_submit, b_write_file_append = (numBatch>1), s_output_path = s_output_path)
    
    print(paste0(as.double((proc.time() - systime_ini)['elapsed'])/60, ' minutos en total.'))
    minutos_pend <- (as.double((proc.time() - systime_ini2)['elapsed'])/60) * (NUM_BLOQUES / numBatch  -  1)
    if(minutos_pend < 60) print(paste0('Faltan aprox. ',minutos_pend, ' minutos.')) else print(paste0('Faltan aprox. ',minutos_pend/60, ' horas.'))
  }
}
if(!G_b_DEBUG)  predict_testset() else print('G_b_DEBUG == TRUE -> No creamos submit.csv')
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
