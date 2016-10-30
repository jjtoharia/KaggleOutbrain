### Inicialización (setwd() y rm() y packages):

# setwd(getwd())
try(setwd('C:/Users/jtoharia/Dropbox/AFI_JOSE/Kaggle/Outbrain'), silent=TRUE)
try(setwd('C:/Personal/Dropbox/AFI_JOSE/Kaggle/Outbrain'), silent=TRUE)
rm(list = ls()) # Borra todos los elementos del entorno de R.

s_input_path <- "C:/Users/jtoharia/Downloads/Kaggle_Outbrain/"
# s_input_path <- "../input/"

# options(echo = FALSE) # ECHO OFF
print('###########################################')
print('# Outbrain Click Prediction - JJTZ 2016')
print('###########################################')
# install.packages("data.table")
suppressMessages(library(data.table))
# Process in parallel:
# install.packages("doParallel")
suppressMessages(library(foreach))
library(iterators)
library(parallel)
library(doParallel)
# # Process in parallel: Ejemplo de uso:
# cl <- makeCluster(detectCores(), type='PSOCK') # library(doParallel) [turn parallel processing on]
# registerDoParallel(cl) # library(doParallel) [turn parallel processing on]
# registerDoSEQ() # library(doParallel) [turn parallel processing off and run sequentially again]
# #

# ##################################################
# ## Funciones útiles:
# ##################################################
source("../../funciones_utiles.R")
# ##################################################
# ## Funciones:
# ##################################################
basic_preds_m12 <- function(trainset = trainset, validtest = validset, k = 1, b_restore_key = TRUE) # k=1,2,3
{
  # --------------------------------------------------------
  # Crear primera predicción con las frecuencias como prob.:
  # --------------------------------------------------------
  stopifnot(k %in% 1:3)
  
  # Frecuencias de ad_id en trainset:
  setkey(trainset, ad_id)
  probs_ads <- trainset[, .(prob = mean(clicked) ), by = ad_id]
  # head(probs_ads)
  
  setkey(validtest, ad_id)
  
  # head(validtest)
  # str(validtest)
  # summary(validtest)
  
  print('Ads de validtest en trainset (orig: 0.17  0.83):')
  print(table(unique(validtest$ad_id) %in% unique(trainset$ad_id))/length(unique(validtest$ad_id))) # FALSE= 65.350 TRUE=316.035
  print('Ads de trainset en validtest (orig: 0.34 0.66):')
  print(table(unique(trainset$ad_id) %in% unique(validtest$ad_id))/length(unique(trainset$ad_id))) # FALSE=162.915 TRUE=316.035
  
  # Proporción de clicks (global):
  if(k == 1) prob_click_global_ads_en_test <- mean(trainset$clicked[trainset$ad_id %in% validtest$ad_id]) # 1 [0.61733]
  if(k == 2) prob_click_global_ads_no_en_test <- mean(trainset$clicked[!(trainset$ad_id %in% validtest$ad_id)]) # 2 [0.63551]
  if(k == 3) prob_click_global <- mean(trainset$clicked) # 3 [0.63529]
  
  if(k == 1) prob_na <- prob_click_global_ads_en_test
  if(k == 2) prob_na <- prob_click_global_ads_no_en_test
  if(k == 3) prob_na <- prob_click_global
  # # Free memory:
  # rm(trainset)
  # gc() # Garbage collector
  
  # Añadir las probs de los ads:
  validtest <- merge(validtest, probs_ads, all.x = T, by = "ad_id") # by ad_id
  
  # Predecir las probs de los ads que no están (i.e. los NAs) con "prob_na":
  # validtest[is.na(validtest$prob), prob := prob_na]
  ## Ligeramente más rápido:
  mi_j <- which(names(validtest)=="prob")
  set(validtest, which(is.na(validtest$prob)), mi_j, prob_na)
  # summary(validtest)

  return(Map_12(tr_valid = validtest, b_restore_key = b_restore_key))
}

Map_12 <- function(tr_valid, b_restore_key = TRUE)
{
  stopifnot(all(names(tr_valid) %in% c("display_id", "ad_id", "clicked", "prob")))
  # Este data.table es el tr_valid (que es el subconjunto de trainset que usamos para validar),
  #  - ad_id
  #  - display_id
  #  - clicked es 0 ó 1 (es la variable target)
  #  - prob es la probabilidad de click, predicha
  # Evaluamos (Mean Average Precision MAP@12):
  print('Calculando actual_lst...')
  mi_tiempo <- system.time({
    old.key <- key(tr_valid)
    # Para conseguir los Ads de mayor "prob" a menor "prob", ponemos "prob" en la key y usamos rev():
    if(!all(key(tr_valid) == c("display_id", "prob")))
      setkey(tr_valid, display_id, prob)
    actual_lst <- tr_valid[, .(ad_id.v = list(rev(ad_id))), by = display_id]$ad_id.v
  })
  print(mi_tiempo['elapsed'])
  
  print('Calculando predicted_lst...')
  mi_tiempo <- system.time({
    # Para conseguir los Ads de mayor "clicked" a menor "clicked", ponemos "clicked" en la key y usamos rev():
    if(!all(key(tr_valid) == c("display_id", "clicked")))
      setkey(tr_valid, display_id, clicked)
    predicted_lst <- tr_valid[, .(ad_id.v = list(rev(ad_id))), by = display_id]$ad_id.v
  })
  print(mi_tiempo['elapsed'])
  
  if(b_restore_key)
  {
    print('Restaurando key inicial...')
    mi_tiempo <- system.time({
      if(!all(key(tr_valid) == old.key))
        setkeyv(tr_valid, old.key)
    })
    print(mi_tiempo['elapsed'])
  }
  
  print('Calculando Map_12...')
  mi_tiempo <- system.time({
    MAP.12 <- mapk(k = 12, actual_list = actual_lst, predicted_list = predicted_lst)
  })
  print(mi_tiempo['elapsed'])
  print(paste0('MAP_12 = ', MAP.12))
  return(MAP.12)
}

basic_preds_guardar_submit <- function(trainset, k) # k=1,2,3
{
  # --------------------------------------------------------
  # Crear primera predicción con las frecuencias como prob.:
  # --------------------------------------------------------
  stopifnot(k %in% 1:3)
  
  # Frecuencias de ad_id en trainset:
  setkey(trainset, ad_id)
  probs_ads <- trainset[, .(prob = mean(clicked) ), by = ad_id]
  # head(probs_ads)
  
  # ------------------
  # Leer testset:
  # ------------------
  testset <- fread(paste0(s_input_path, "clicks_test.csv")) # testset <- fread( "../input/clicks_test.csv")
  setkey(testset, ad_id)
  
  # head(testset)
  # str(testset)
  # summary(testset)
  
  # print('Ads de testset en trainset:')
  # table(unique(testset$ad_id) %in% unique(trainset$ad_id)) # FALSE= 65.350 TRUE=316.035
  # print('Ads de trainset en testset:')
  # table(unique(trainset$ad_id) %in% unique(testset$ad_id)) # FALSE=162.915 TRUE=316.035
  
  # Proporción de clicks (global):
  if(k == 1) prob_click_global_ads_en_test <- mean(trainset$clicked[trainset$ad_id %in% testset$ad_id]) # 1 [0.61733]
  if(k == 2) prob_click_global_ads_no_en_test <- mean(trainset$clicked[!(trainset$ad_id %in% testset$ad_id)]) # 2 [0.63551]
  if(k == 3) prob_click_global <- mean(trainset$clicked) # 3 [0.63529]
  
  if(k == 1) prob_na <- prob_click_global_ads_en_test
  if(k == 2) prob_na <- prob_click_global_ads_no_en_test
  if(k == 3) prob_na <- prob_click_global
  # # Free memory:
  # rm(trainset)
  # gc() # Garbage collector
  
  # Añadir las probs de los ads:
  testset <- merge(testset, probs_ads, all.x = T, by = "ad_id") # by ad_id
  
  # Predecir las probs de los ads que no están (i.e. los NAs) con "prob_na":
  # testset[is.na(testset$prob), prob := prob_na]
  ## Ligeramente más rápido:
  mi_j <- which(names(testset)=="prob")
  set(testset, which(is.na(testset$prob)), mi_j, prob_na)
  # summary(testset)
  
  guardar_submit(testset = testset, fichero = paste0("submitset", k, ".csv"))
}

guardar_submit <- function(testset, fichero = "submit.csv")
{
  # -------------------------
  # PREPARACIÓN DEL SUBMIT:
  # -------------------------
  # NOTA: length(submitset) == length(unique(testset$display_id)) == 6.245.533 rows
  # Para conseguir los Ads de mayor "prob" a menor "prob", ponemos "prob" en la key y usamos rev():
  mi_tiempo <- system.time({
    old.key <- key(testset)
    setkey(testset, display_id, prob)
    submitset <- testset[, .(ad_id = paste(rev(ad_id), collapse=" ")), by = display_id] # 85 secs
    if(!all(key(testset) == old.key))
      setkeyv(testset, old.key)
  })
  print(mi_tiempo['elapsed'])
  # head(submitset)
  
  # Ordenamos por display_id:
  setkey(submitset, display_id)
  
  # Guardamos fichero:
  mi_tiempo <- system.time({
    write.table(submitset, file = fichero, row.names = F, quote = FALSE, sep = ",")
  })
  print(mi_tiempo['elapsed'])
}

# ##################################################
# ## Inicio:
# ##################################################
Proyecto <- "Outbrain Clicks Prediction"
print(paste0(Sys.time(), ' - ', 'Proyecto = ', Proyecto))

Proyecto.s <- str_replace_all(Proyecto, "\\(|\\)| |:", "_") # Quitamos espacios, paréntesis, etc.

# Inicializamos variables:
# NOTA: Dejamos un Core de la CPU "libre" para no "quemar" la máquina:
cl <- makeCluster(detectCores() - 1, type='PSOCK') # library(doParallel) [turn parallel processing on]
registerDoParallel(cl) # library(doParallel) [turn parallel processing on]

memory.size(max = 10000)

systime_ini <- proc.time()

# Leer trainset:
full_trainset <- fread(paste0(s_input_path, "clicks_train.csv")) # trainset <- fread("../input/clicks_train.csv")

# setkey(trainset, display_id)
# summary(trainset[, .(ads = sum(clicked)), by = display_id]$ads)
# # min == max == 1 ==> Hay un ad_id (y solamente uno) con clicked==1 en cada display_id.

# head(trainset)
# str(trainset)
# summary(trainset)

# Dividir muestra en entrenamiento y validación:
set.seed(1)
index <- 1:nrow(full_trainset)
porc_test <- 0.3
testindex <- sample(index, trunc(length(index)*porc_test))
validset <- full_trainset[testindex,]
trainset <- full_trainset[-testindex,]
# Free memory:
rm(full_trainset)
gc() # Garbage collector

M12 <- foreach(k = 1:3, .inorder=TRUE, .combine = c, .packages=c('data.table'),
               .export = c("procesar_tweets_csv", "add_tiempo", "procesar_campos_tweets", "procesa_lista_tweets", "procesa_contenido_campos")
               ) %do%
{
  return(basic_preds_m12(trainset = trainset, validtest = validset, k, b_restore_key = FALSE)) # ret_val de la función "foreach() %do%" (o foreach( %dopar%))
}
# M12    <- basic_preds_m12(trainset = trainset, validtest = validset, k = 1, b_restore_key = FALSE)
# M12[2] <- basic_preds_m12(trainset = trainset, validtest = validset, k = 2, b_restore_key = FALSE)
# M12[3] <- basic_preds_m12(trainset = trainset, validtest = validset, k = 3, b_restore_key = FALSE)
print(M12)

# basic_preds_guardar_submit(trainset = trainset, k = 2)

print(paste0(as.double((proc.time() - systime_ini)['elapsed']), ' segundos en total.'))

print('Ok.')

# cleanup:
try(registerDoSEQ(), silent = TRUE) # library(doParallel) [turn parallel processing off and run sequentially again]
try(stopImplicitCluster(), silent = TRUE)
try(stopCluster(cl), silent = TRUE)
