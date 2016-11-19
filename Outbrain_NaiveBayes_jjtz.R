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
# install.packages("data.table")
suppressMessages(library(data.table))
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
# install.packages("klaR")
suppressPackageStartupMessages(library(klaR))
# install.packages("caret")
suppressPackageStartupMessages(library(caret))
# model <- train(x, y, 'nb', metric = 'Accuracy', metric = 'Kappa', trControl = trainControl(method = 'cv', number = 10))
# predicted <- predict(model$finalModel, testData[,-50])
# -------------------------------
# train model with trainset:
# -------------------------------
nbay_reduc_seed <- 1234
nbay_train_porc <- 0.8
nbay_modelos <- vector(mode = "character", length = NUM_BLOQUES)
for(numBatch in 1:NUM_BLOQUES)
{
  full_trainset <- leer_batch_train(numBatch, "Naive Bayes Training", s_input_path)
  # sapply(full_trainset, uniqueN)
  # print(table(full_trainset$dia))
  
  if(G_b_DEBUG)
  {
    print('Hacemos un sample para que vaya más rápido...')
    full_trainset <- reducir_trainset(mi_set = full_trainset, n_seed = nbay_reduc_seed, n_porc = 0.01)
  }
  # NOTA: Ahora usaremos los días (1:11) y (12,13), i.e. reducimos en train y valid y, de paso, usamos los días:
  reduc_list <- reducir_trainset_2(mi_set = full_trainset, n_seed = nbay_reduc_seed, n_porc = nbay_train_porc)
  trainset <- reduc_list[[1]]
  validset <- reduc_list[[2]]
  # print(100 * round(table(trainset$dia)) / nrow(trainset), digits = 3)
  # print(100 * round(table(validset$dia)) / nrow(validset), digits = 3)
  
  setkey(trainset, display_id)
  # print(sapply(trainset, uniqueN))
  
  nbay_preps <- nbay_prep_datos(mi_dt = trainset)
  # dt_all    <- nbay_preps[[2]]
  print(colnames(nbay_preps[[2]]))
  # sapply(nbay_preps[[2]], uniqueN)
  X <- nbay_preps[[2]][,-1, with=FALSE]
  y <- as.data.frame(nbay_preps[[2]])[,1] # factor
  # ===============  NBAY params - START  ===============
  # Añadimos 100 a la versión porque ahora tratamos con dia (1:11) y (12,13) por separado (Cf. reducir_trainset_2):
  # Añadimos 200 a la versión porque ahora tratamos con dia (1:11) y (12,13) al 50%, como en el testset (Cf. reducir_trainset_2):
  nbay_mi_version <- 200 + nbay_preps[[1]] # Esto viene de nbay_prep_datos (indica cambios en las "features" seleccionadas)
  nbay_eval_metric = "Accuracy"
  # nbay_eval_metric = "Kappa"
  # nbay_laplace = 0
  # nbay_laplace = 3
  # nbay_usekernel = TRUE
  # nbay_adjust = 0.1
  # nbay_laplace = 0
  nbay_nfold = 4 # nFold Cross Validation
  nbay_tuneLength = 3 # tuning grid size for Cross Validation
  # ===============  NBAY params - END  ===============
  if(G_b_DEBUG)  nbay_mi_version <- nbay_mi_version + 990000
  rm(full_trainset, nbay_preps); gc()
  print(paste0('Entrenando (CV) ', str_pad(numBatch, 3, "left" ,"0"), '...'))
  mi_tiempo <- system.time({
    nbay_cv <- train(x = X, y = y, method = 'nb',
                     tuneLength = nbay_tuneLength,
                     tuneGrid = expand.grid(usekernel = c(TRUE),
                                            fL = c(2, 3),
                                            adjust = c(0.001, 0.01, 0.1)),
                     metric = nbay_eval_metric,
                     trControl = trainControl(method = 'cv', number = nbay_nfold, verboseIter = FALSE)
                     # NOTA: con foreach() y doParallel va mucho más rápido, pero no muestra nada a pesar de verboseIter = TRUE...
                     )
  })
  tiempo_cv <- paste0('Tiempo nbay_cv(): ', mi_tiempo['elapsed']/60, ' minutos')
  print(tiempo_cv)
  nbay_bestTune <- print(which(nbay_cv$results$fL == nbay_cv$bestTune$fL & nbay_cv$results$usekernel == nbay_cv$bestTune$usekernel & nbay_cv$results$adjust == nbay_cv$bestTune$adjust))
  print(nbay_cv$results)
  print(nbay_cv$bestTune)
  mi_tiempo <- system.time({
    nbay <- NaiveBayes(x = X, grouping = y,
                       usekernel = nbay_cv$bestTune$usekernel,
                       fL = nbay_cv$bestTune$fL,
                       adjust = nbay_cv$bestTune$adjust)
  })
  tiempo_train <- paste0('Tiempo nbay(): ', mi_tiempo['elapsed']/60, ' minutos')
  print(tiempo_train)
  print("Guardando modelo entrenado...")
  nbay_modelos[numBatch] <- paste0(paste("NBAY",
                                         nbay_eval_metric, nbay_nfold, nbay_tuneLength,
                                         paste0('v', str_pad(nbay_mi_version, 3, "left" ,"0")),
                                         nbay_bestTune,
                                         sep = "_")
                                   , '.', str_pad(numBatch, 3, "left" ,"0")) # lo guardamos pata hacer ensemble luego
  # Quitamos nbay_bestTune y numBatch al nombre de fichero para guardar estadísticas en un único fichero:
  if(!G_b_DEBUG)
    save(nbay, file = paste0(s_output_path, nbay_modelos[numBatch] ,'.modelo'))
  # # Leemos modelo:
  # load(paste0(s_output_path, nbay_modelos[numBatch] ,'.modelo'))
  nbay_filename <- str_replace(nbay_modelos[numBatch], "_[0-9]*\\.[0-9]*$", "")
  print('Predecimos en trainset para medir nuestro map12:')
  mi_tiempo <- system.time({
    trainset[, prob := predict(nbay, newdata = X, type = 'prob')$posterior[,"1"]]
  })
  tiempo_predict <- paste0('Tiempo predict(nbay, trainset): ', mi_tiempo['elapsed']/60, ' minutos')
  print(tiempo_predict)
  # # Tabla de confusión:
  # table(trainset[, .(clicked, prob>.2)])
  # mi_nbay_map <- nbay$bestScore
  # print(paste0('nbay_map = ', mi_nbay_map))
  print('Calculamos map12(trainset):')
  mi_map12 <- Map_12(tr_valid = trainset[, .(display_id, ad_id, clicked, prob)], b_verbose = 0)
  print(paste0('map12 = ', mi_map12))
  print(paste0('Guardando modelo entrenado (', nbay_modelos[numBatch] ,'.modelo', ')'))
  write.table(x = t(as.data.frame(list(batch = paste(numBatch, NUM_BLOQUES, sep = "/"),
                      dim_x = t(dim(X)),
                      tiempo_cv = tiempo_cv,
                      tiempo_train = tiempo_train,
                      tiempo_predict = tiempo_predict,
                      cols = t(dimnames(X)[[2]]),
                      list(nbay_eval_metric = nbay_eval_metric, nbay_nfold = nbay_nfold, nbay_tuneLength = nbay_tuneLength),
                      list(nbay_bestTune = nbay_bestTune, nbay_laplace = nbay_cv$bestTune$fL, nbay_usekernel = nbay_cv$bestTune$usekernel, nbay_adjust = nbay_cv$bestTune$adjust),
                      nbay_mi_version = nbay_mi_version,
                      reduccion_seed = nbay_reduc_seed,
                      reduccion_train_porcent = nbay_train_porc,
                      map12 = mi_map12))),
            file = paste0(s_output_path, nbay_modelos[numBatch] ,'.txt'), row.names = T, col.names = F, quote = F, sep = " = ")
  
  suppressWarnings( # "Warning: appending column names to file"
    write.table(x = as.data.frame(list(Modelo = nbay_modelos[numBatch])), file = paste0(s_output_path, nbay_filename ,'.txt'), row.names = T, col.names = F, quote = F, sep = " = ", append = TRUE)
  )
  print('Compute feature importance matrix...')
  importance_matrix <- varImp(nbay_cv, scale=T)[[1]] # scale=T para que ponga 100 a la más importante...
  importance_matrix <- importance_matrix[order(importance_matrix$X1, decreasing = T),]
  importance_matrix <- data.frame(Variable = rownames(importance_matrix), Importancia = importance_matrix$X1, stringsAsFactors=F)
  # suppressWarnings(
  #   write.table(x = as.data.frame(importance_matrix)[,c(1,2)], file = paste0(s_output_path, nbay_filename ,'.txt'), row.names = T, col.names = T, quote = F, sep = "\t", append = TRUE)
  # )
  suppressWarnings(
    write.table(x = as.data.frame(importance_matrix), file = paste0(s_output_path, nbay_filename ,'.txt'), row.names = F, col.names = T, quote = F, sep = "\t", append = TRUE)
  )
  # if(G_b_DEBUG) # nbay.plot.importance(), barplot(), chisq.test()...
  # {
  #   print(dimnames(X)[[2]])
  #   #In case last step does not work for you because of a version issue, you can try following :
  #   x11(); barplot(importance_matrix[1:8,]$Importancia, names.arg = importance_matrix[1:8,]$Variable,
  #                  main = paste0("N.Bayes varImp - ", str_pad(numBatch, 3, "left" ,"0")))
  #   
  #   # # To see whether the variable is actually important or not:
  #   # test <- chisq.test(y, as.numeric(trainset$ad_topic_prob_1))
  #   # test <- chisq.test(y, as.numeric(trainset$publish_timestamp))
  #   # test <- chisq.test(y, as.numeric(trainset$numAds))
  #   # print(test)
  # }
  
  # -------------------------------
  # Ahora predecimos en validset, que son los que NO hemos usado para CV:
  # -------------------------------
  nbay_preps <- nbay_prep_datos(mi_dt = validset)
  # dt_all    <- nbay_preps[[2]]
  print(colnames(nbay_preps[[2]]))
  # sapply(nbay_preps[[2]], uniqueN)
  X <- nbay_preps[[2]][,-1, with=FALSE]
  y <- as.data.frame(nbay_preps[[2]])[,1] # factor
  rm(nbay_preps); gc()
  print('Predecimos en validset para medir nuestro map12:')
  mi_tiempo <- system.time({
    validset[, prob := predict(nbay, newdata = X, type = 'prob')$posterior[,"1"]]
  })
  tiempo_predict <- paste0('Tiempo predict(nbay, validset): ', mi_tiempo['elapsed']/60, ' minutos')
  print(tiempo_predict)
  # # Tabla de confusión:
  # table(validset[, .(clicked, prob>.5)])
  mi_map12_valid <- Map_12_diario(tr_valid = validset[, .(display_id, ad_id, dia, clicked, prob)], b_verbose = 0,
                                  dias_list = list(c(1:11), c(12,13), c(1:13)))
  colnames(mi_map12_valid) <- c('mi_map12_valid_1-11', 'mi_map12_valid_12-13', 'mi_map12_valid_Total')
  print('----------------------------------------------------------------------------')
  print(cbind(mi_map12_valid[2,], Train = mi_map12)) # Volvemos a imprimir mi_map12 (del Training) al lado.
  print('----------------------------------------------------------------------------')
  # Añadimos a los ficheros de cada training los 3 map_12 del validset:
  write.table(x = t(mi_map12_valid[2,]),
              file = paste0(s_output_path, nbay_modelos[numBatch] ,'.txt'), row.names = T, col.names = F, quote = F, sep = " = ", append = TRUE)
}
# -------------------------------
# predict values in testset:
# -------------------------------
# Leemos modelo(s):
if(!exists("nbay_modelos"))    nbay_modelos <- str_replace(dir(path = s_output_path, pattern = paste0(nbay_filename, '.*.modelo')), pattern = "\\.modelo", replacement = "")
if(length(nbay_modelos) == 0)  nbay_modelos <- str_replace(dir(path = s_output_path, pattern = paste0(nbay_filename, '.*.modelo')), pattern = "\\.modelo", replacement = "")
stopifnot(length(nbay_modelos) != 0)
nbay <- list() # Inicializamos lista de modelos
for(i in 1:NUM_BLOQUES)
{
  print(paste0('Leyendo ', str_pad(numBatch, 3, "left" ,"0"), ' (modelo ', i, ' de ', NUM_BLOQUES, '): ', nbay_modelos[i],'...'))
  nbay[[i]] <- load(paste0(s_output_path, nbay_modelos[i] ,'.modelo'))
}
# s_Fichero_submit <- paste0(unique(str_replace(string = nbay_modelos, pattern = "_[0-9]*\\.[0-9]*$", replace = "")), '_submit.csv')
s_Fichero_submit <- paste0(str_replace(nbay_filename, "_[0-9]*\\.[0-9]*$", ""), '_submit.csv')
for(numBatch in 1:NUM_BLOQUES)
{
  testset <- leer_batch_test(numBatch, "NBAY Predicting (extreme gradient boosting)", s_input_path)
  if(G_b_DEBUG)
  {
    # Hacemos un sample para que vaya más rápido...
    testset <- reducir_trainset(mi_set = testset, n_seed = nbay_reduc_seed, n_porc = 0.01)
  }
  # Extreme Gradient Boosting:
  # --------------------------
  # if(!exists("testset"))
  #   load(paste0(s_input_path, "testset.RData"))
  dt_all <- nbay_prep_datos(mi_dt = testset)[[2]]
  print(colnames(dt_all))
  print('Preparando matrix y predicting...')
  X <- nbay_preps[[2]][,-1, with=FALSE]
  # y <- as.data.frame(nbay_preps[[2]])[,1] # factor
  y_pred <- vector(mode = "numeric", length = NUM_BLOQUES)
  for(i in 1:NUM_BLOQUES)
  {
    print(paste0('Predicting ', str_pad(numBatch, 3, "left" ,"0"), ' (modelo ', i, ' de ', NUM_BLOQUES, '): ', nbay_modelos[i],'...'))
    # nbay <- load(paste0(s_output_path, nbay_modelos[i] ,'.modelo'))
    y_pred[i] <- list(predict(nbay[[i]], newdata = X, type = 'prob')$posterior[,"1"])
  }
  mean_y_pred <- rowMeans(rbindlist(list(y_pred)))
  testset[, prob := mean_y_pred]
  print(paste0('Guardando ', s_Fichero_submit, ' (', numBatch, '/', NUM_BLOQUES, ')...'))
  guardar_submit(testset = testset, fichero = s_Fichero_submit, b_write_file_append = (numBatch>1), s_output_path = s_output_path)
}

# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# M12 <- foreach(reg = c(0,3,4,5,6), .inorder=TRUE, .combine = c, .packages=c('data.table'),
#                .export = c("Map_12", "mapk_nodup", "add_tiempo")
#                ) %do%
# {
#   return(
#     basic_preds_m12(trainset = trainset, validset = validset, k = 2, reg = reg, b_verbose = 0)
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

print(paste0(as.double((proc.time() - systime_ini)['elapsed']), ' segundos en total.'))

print('Ok.')

# cleanup:
try(registerDoSEQ(), silent = TRUE) # library(doParallel) [turn parallel processing off and run sequentially again]
try(stopImplicitCluster(), silent = TRUE)
try(stopCluster(cl), silent = TRUE)
