### Inicialización (setwd() y rm() y packages):

# # Para guardar como libsvm (para Spark):
# for(j in names(sapply(full_trainset, is.character)[sapply(full_trainset, is.character)]))
#   set(full_trainset, j = j, value = NULL) # Quitamos variables tipo char
# cols <- colnames(full_trainset)
# cols <- cols[!(cols %in% c("clicked", "display_id"))]
# setcolorder(full_trainset, c("clicked", "display_id", cols)) # Ponemos "clicked" en primer lugar
# # Luego con str_replace se quitan los ceros y los NAs (expresión regular sencillita)
# # Y se guarda cada string como una línea en un txt (cat(..., file='', append=T))
# pp<-function(full_trainset)
# {
#   gsub(x = gsub(x =
#       paste(full_trainset["clicked"], paste(seq(1, length(cols)), full_trainset[cols], sep = ':', collapse = ' '))
#     ,pattern = "[0-9]+:0 ", replacement = "")
#     ,pattern = " [0-9]+:0$", replacement = "")
# }
# for(j in cols)  set(full_trainset, which(is.na(full_trainset[[j]])), j, 0) # NAs a cero (para quitarlos luego)
# if(nrow(full_trainset) < 100000)
# {
#   file.remove("prueba.libsvm")
#   cat(apply(full_trainset, 1, pp), file="prueba.libsvm", append = TRUE, sep = "\n")
# } else {
#   file.remove(paste0(s_input_path, "trainset.libsvm"))
#   cat(apply(full_trainset, 1, pp), file=paste0(s_input_path, "trainset.libsvm"), append = TRUE, sep = "\n")
# }
# # 

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

memory.limit(size = 16000)

systime_ini <- proc.time()

# --------------------------------------------------------
G_b_DEBUG <- FALSE # Reducimos todo para hacer pruebas más rápido
NUM_BLOQUES <- 32
# --------------------------------------------------------
# CARGAMOS EL ÚLTIMO BATCH PARA EMPEZAR CON ALGO:
s_fich_train <- get_batch_train_filename(NUM_BLOQUES)
s_fich_test <- get_batch_test_filename(NUM_BLOQUES)
numAdsCluster <- 12
fich_name_numAds <- paste0("full_trainset_", str_pad(numAdsCluster, 2, "left", "0"), '_', str_pad(NUM_BLOQUES, 3, "left", "0"), ".RData")
b_con_num_modelo <- file.exists(file.path(s_input_path, fich_name_numAds))
if(!file.exists(file.path(s_input_path, s_fich_train)))
{
  if(b_con_num_modelo)
  {
    load(file = paste0(s_input_path, fich_name_numAds))
    save(full_trainset, file = file.path(s_input_path, s_fich_train))
  } else {
    # Si no existe full_trainset_016, los creamos todos:
    if(!G_b_DEBUG)
    {
      print('Leyendo full_trainset completo')
      full_trainset <- fread(paste0(s_input_path, "clicks_train.csv")) # trainset <- fread("../input/clicks_train.csv")
      print('Splitting full_trainset...')
      mi_split_train(NUM_BLOQUES)
      rm(full_trainset)
      gc()
    }
  }
}
# Si no existe testset_016 (o testset_016_016), los creamos todos:
if(!file.exists(file.path(s_input_path, s_fich_test)) & !file.exists(paste0(s_input_path, get_batch_test_filename(NUM_BLOQUES, NUM_BLOQUES))))
{
  if(!G_b_DEBUG)
  {
    print('Leyendo testset completo')
    testset <- fread(paste0(s_input_path, "clicks_test.csv"))
    print('Splitting testset...')
    mi_split_test(NUM_BLOQUES)
    rm(testset)
    gc()
  }
}
if(exists("full_trainset")) rm(full_trainset)
if(exists("testset"))       rm(testset)
gc()
full_trainset <- leer_batch_train(NUM_BLOQUES, "inicio", s_input_path)
if( file.exists(paste0(s_input_path, get_batch_test_filename(NUM_BLOQUES, NUM_BLOQUES))))
  testset <- leer_batch_test(NUM_BLOQUES, "inicio", s_input_path, numSubBatch = NUM_BLOQUES)
if(!file.exists(paste0(s_input_path, get_batch_test_filename(NUM_BLOQUES, NUM_BLOQUES))))
  testset <- leer_batch_test(NUM_BLOQUES, "inicio", s_input_path)
# numBatch = 2
# s_fich_train <- paste0("full_trainset_", str_pad(numBatch, 3, "left" ,"0"), ".RData")
# load(file.path(s_input_path, s_fich_train)) # Cuidado que se puede llamar trainset_batch!!!
# full_trainset[, (c("uuid.x", "document_id.x", "timestamp.x", "platform.x", "geo_location.x", "geo_loc.country.x")) := NULL]
# setnames(full_trainset, old = c("uuid.y", "document_id.y", "timestamp.y", "platform.y", "geo_location.y", "geo_loc.country.y"),
#   new = c("uuid", "document_id", "timestamp", "platform", "geo_location", "geo_loc.country"))
# save(full_trainset, file = file.path(s_input_path, s_fich_train))
# ------------------------------------------
# 1.- Leer trainset:
# ------------------------------------------
if(!exists("full_trainset"))
{
  if(!G_b_DEBUG)
  {
    if(file.exists(file.path(s_input_path, "full_trainset.RData")))
    {
      load(file.path(s_input_path, "full_trainset.RData"))
    } else 
    {
      full_trainset <- fread(paste0(s_input_path, "clicks_train.csv")) # trainset <- fread("../input/clicks_train.csv")
      sumwpos <- sum(full_trainset$clicked == 1)
      sumwneg <- sum(full_trainset$clicked == 0)
      mi_scale_pos_weight <- sumwneg / sumwpos # == 4.164079
      mi_split_train(NUM_BLOQUES)
    }
  } else
  {
    if(file.exists(paste0(s_input_path, "clicks_train_debug.csv")))
    {
      # 1.- Leer trainset (pequeño - DEBUG):
      full_trainset <- fread(paste0(s_input_path, "clicks_train_debug.csv")) # trainset <- fread("../input/clicks_train.csv")
    } else
    {
      # 1.- Leer trainset:
      full_trainset <- fread(paste0(s_input_path, "clicks_train.csv")) # trainset <- fread("../input/clicks_train.csv")
      full_trainset <- reducir_trainset(mi_set = full_trainset, n_seed = 1, n_porc = 0.01)
      write.table(full_trainset, file = paste0(s_input_path, "clicks_train_debug.csv"), row.names=F, quote=F, sep=",")
      rm(disps, index, smallset)
      gc() # Garbage collector
    }
  }
}
# numAds (por display_id):
if(!("numAds" %in% colnames(full_trainset)))
{
  for(numBatch in 1:NUM_BLOQUES)
  {
    full_trainset <- leer_batch_train(numBatch, "numAds", s_input_path)
    setkey(full_trainset, display_id)
    full_trainset[, numAds := .N, by = "display_id"]
    {
    # jj_queso('Cantidad de Tipos de Display (numAds):', table(full_trainset[, .(n=max(numAds)), by = "display_id"]$n))
    print('Cantidad de Tipos de Display (numAds):')
    print(jjfmt(table(full_trainset[, .(n=max(numAds)), by = "display_id"]$n)))
    # jj_queso('Cantidad de Registros por Tipo de Display (numAds):', table(full_trainset[, numAds]))
    print('Cantidad de Registros por Tipo de Display (numAds):')
    print(jjfmt(table(full_trainset[, numAds])))
    }
    # # # numDisps (por ad_id):
    # # setkey(full_trainset, ad_id)
    # # full_trainset[, numDisps := .N, by = "ad_id"]
    # # # table(full_trainset[, .(n=max(numDisps)), by = "ad_id"]$n)
    # # summary(full_trainset$numDisps)
    save(full_trainset, file = paste0(s_input_path, get_batch_train_filename(numBatch)))
  }
}
print("1.- Leer trainset - Ok.")
# ------------------------------------------
# 2.- Leer testset:
# ------------------------------------------
if(!exists("testset"))
{
  if(!G_b_DEBUG & file.exists(file.path(s_input_path, "testset.RData")))
  {
    load(paste0(s_input_path, "testset.RData"))
  } else 
  {
    testset <- fread(paste0(s_input_path, "clicks_test.csv"), nrows = ifelse(G_b_DEBUG, 2500L, -1L))
    mi_split_test(NUM_BLOQUES)
  }
}
# numAds (por display_id):
if(!G_b_DEBUG & !("numAds" %in% colnames(testset)))
{
  for(numBatch in 1:NUM_BLOQUES)
  {
    testset <- leer_batch_test(numBatch, "numAds", s_input_path)
    setkey(testset, display_id)
    testset[, numAds := .N, by = "display_id"]
    {
    # jj_queso('Cantidad de Tipos de Display (numAds):', table(testset[, .(n=max(numAds)), by = "display_id"]$n))
    print('Cantidad de Tipos de Display (numAds):')
    print(jjfmt(table(testset[, .(n=max(numAds)), by = "display_id"]$n)))
    # jj_queso('Cantidad de Registros por Tipo de Display (numAds):', table(testset[, numAds]))
    print('Cantidad de Registros por Tipo de Display (numAds):')
    print(jjfmt(table(testset[, numAds])))
    }
    print(summary(testset$numAds))
    # # # numDisps (por ad_id):
    # # setkey(testset, ad_id)
    # # testset[, numDisps := .N, by = "ad_id"]
    # # # table(testset[, .(n=max(numDisps)), by = "ad_id"]$n)
    # # summary(testset$numDisps)
    save(testset, file = paste0(s_input_path, get_batch_test_filename(numBatch)))
  }
}

# Leemos e integramos los datos de los demás ficheros:
print("2.- Leer testset - Ok.")
# ------------------------------------------
# 3.- "page_views_sample.csv" (uuid, document_id, timestamp, platform, geo_location, traffic_source)
# ------------------------------------------
# # NOTA: Movido al punto 14.1.-
# ------------------------------------------
# 4.- "events.csv": (display_id, uuid, document_id, timestamp, platform, geo_location) -> (document_id, uuid, timestamp, platform, geo_location, geo_loc.country, pais, hora, dia, horadia)
# ------------------------------------------
if(!("document_id" %in% colnames(full_trainset) & "document_id" %in% colnames(testset)))
{
  print('events.csv')
  evts <- fread(paste0(s_input_path, "events.csv"), colClasses = c("integer", "character", "integer", "integer64", "character", "character"), nrows = ifelse(G_b_DEBUG, 2500L, -1L))
  # setkey(evts, display_id, uuid, document_id)
  # # timestamp (ms since 1970-01-01 - 1465876799998)
  # # platform (desktop = 1, mobile = 2, tablet =3)
  # # geo_location (country>state>DMA)
  evts[platform == "\\N"]
  evts[platform == "\\N", platform := "2"] # Es el más frecuente (y solo hay 5 casos)
  evts[, platform := as.integer(platform)]
  setkey(evts, geo_location)
  evts[, geo_loc.country := str_split_fixed(geo_location, '>', Inf)[[1]], by = geo_location]
  # evts[geo_location!=geo_loc.country, geo_loc.state := str_split_fixed(geo_location, '>', Inf)[[2]], by = geo_location]
  # evts[str_detect(geo_location, pattern = '.*>.*>.*'), geo_loc.DMA := str_split_fixed(geo_location, '>', Inf)[[3]], by = geo_location]
  # print(jjfmt(sort(table(evts$geo_loc.state), decreasing = T)[1:10]))
  # print(jjfmt(sort(table(evts$geo_loc.DMA), decreasing = T)[1:10]))
  setkeyv(evts, c("display_id", "geo_loc.country"))
  print(jjfmt(sort(table(evts$geo_loc.country), decreasing = T)[1:10]))
  print(jjfmt(sort(table(evts[display_id < 16874594,]$geo_loc.country), decreasing = T)[1:10])) # trainset
  print(jjfmt(sort(table(evts[display_id > 16874593,]$geo_loc.country), decreasing = T)[1:10])) # testset
  # Agrupamos países: US, CA, GB, Resto:
  evts[!(geo_loc.country %in% c("US", "CA", "GB")), pais := "Resto"]
  evts[  geo_loc.country %in% c("US", "CA", "GB") , pais := geo_loc.country, by = geo_loc.country]
  print(sort(table(evts$pais), decreasing = T))
  print(jjfmt(sort(table(evts$pais,evts$geo_loc.state), decreasing = T)[1:10]))
  print(sort(table(evts[display_id < 16874594,]$pais), decreasing = T)) # trainset
  print(sort(table(evts[display_id > 16874593,]$pais), decreasing = T)) # testset
  evts[, hora := as.integer(1 + (timestamp %/% 3600000) %% 24)] # De 1 a 24
  evts[, dia  := as.integer(1 + timestamp %/% (3600000 * 24))]  # De 1 a 15
  evts[, horadia  := as.integer(1 + 24 * (dia-1) + (hora-1))]  # De 1 a 15*24
  print(jjfmt(table(evts$dia))); print(jjfmt(table(evts$hora))) #; print(table(evts$horadia))
  print(jjfmt(table(evts[display_id < 16874594,]$dia))); print(jjfmt(table(evts[display_id < 16874594,]$hora))) # trainset
  print(jjfmt(table(evts[display_id > 16874593,]$dia))); print(jjfmt(table(evts[display_id > 16874593,]$hora))) # testset
  x11(); smoothScatter(evts[,.(dia, display_id)]);abline(h=16874593)
  x11(); smoothScatter(evts[display_id < 16874594,.(dia, display_id)]) # trainset
  x11(); smoothScatter(evts[display_id > 16874593,.(dia, display_id)]) # testset (!!!)
  x11(); smoothScatter(evts[,.(hora, display_id)]);abline(h=16874593)
  x11(); smoothScatter(evts[,.(horadia, display_id)]);abline(h=16874593)
  # x11(); smoothScatter(evts[,.(timestamp=as.numeric(timestamp), display_id)]);abline(h=16874593)
  print(summary(evts))
  # dim(full_trainset)
  jjfmt(sapply(evts, uniqueN))
  jjfmt(sapply(evts[display_id < 16874594,], uniqueN)) # trainset
  jjfmt(sapply(evts[display_id > 16874593,], uniqueN)) # testset
  setkey(evts, display_id)
  x11(); smoothScatter(evts[,.(platform, display_id)]);abline(h=16874593)
  pp <- jjfmt(table(evts[,platform])); names(pp) <- c("1-desktop", "2-mobile", "3-tablet"); print(pp)
  pp <- jjfmt(table(evts[display_id < 16874594,platform])); names(pp) <- c("1-desktop", "2-mobile", "3-tablet"); print(pp) # trainset
  pp <- jjfmt(table(evts[display_id > 16874593,platform])); names(pp) <- c("1-desktop", "2-mobile", "3-tablet"); print(pp) # testset
  
  # Finalmente, metemos los datos de los eventos en trainset y en testset:
  if(!("document_id" %in% colnames(full_trainset)))
  {
    setkey(evts, display_id)
    for(numBatch in 1:NUM_BLOQUES)
    {
      full_trainset <- leer_batch_train(numBatch, "events.csv", s_input_path)
      setkey(full_trainset, display_id)
      stopifnot(unique(full_trainset$display_id) %in% unique(evts$display_id)) # Ok.
      full_trainset <- merge(full_trainset, evts, by = "display_id")
      save(full_trainset, file = paste0(s_input_path, get_batch_train_filename(numBatch)))
      print(paste0(as.double((proc.time() - systime_ini)['elapsed'])/60, ' minutos en total.'))
      minutos_pend <- (as.double((proc.time() - systime_ini)['elapsed'])/60) * 2 * (NUM_BLOQUES / numBatch  -  1)
      if(minutos_pend < 60) print(paste0('Faltan aprox. ',minutos_pend, ' minutos.')) else print(paste0('Faltan aprox. ',minutos_pend/60, ' horas.'))
    }
  }
  if(!("document_id" %in% colnames(testset)))
  {
    for(numBatch in 1:NUM_BLOQUES)
    {
      testset <- leer_batch_test(numBatch, "events.csv", s_input_path)
      setkey(testset, display_id)
      stopifnot(unique(testset$display_id) %in% unique(evts$display_id)) # Ok.
      testset <- merge(testset, evts, by = "display_id")
      save(testset, file = paste0(s_input_path, get_batch_test_filename(numBatch)))
      print(paste0(as.double((proc.time() - systime_ini)['elapsed'])/60, ' minutos en total.'))
      minutos_pend <- (as.double((proc.time() - systime_ini)['elapsed'])/60) * 2 * (NUM_BLOQUES / numBatch  -  1)
      if(minutos_pend < 60) print(paste0('Faltan aprox. ',minutos_pend, ' minutos.')) else print(paste0('Faltan aprox. ',minutos_pend/60, ' horas.'))
    }
  }
  rm(evts)
  gc()
}
print('4.- events.csv - Ok.')
# ------------------------------------------
# 5.- "promoted_content.csv": (ad_id, document_id, campaign_id, advertiser_id) -> (ad_document_id, ad_campaign_id, ad_advertiser_id)
# ------------------------------------------
if(!("ad_document_id" %in% colnames(full_trainset) & "ad_document_id" %in% colnames(testset)))
{
  print('promoted_content.csv')
  promcnt <- fread(paste0(s_input_path, "promoted_content.csv"), nrows = ifelse(G_b_DEBUG, 2500L, -1L))
  # setkey(promcnt, ad_id, document_id, campaign_id, advertiser_id)
  # sapply(promcnt, uniqueN)

  colnames(promcnt)[2:ncol(promcnt)] <- paste0('ad_', colnames(promcnt)[2:ncol(promcnt)])
  colnames(promcnt)
  setkey(promcnt, ad_id)
  
  # Finalmente, metemos los datos de los ads en trainset y en testset:
  if(!("ad_document_id" %in% colnames(full_trainset)))
  {
    for(numBatch in 1:NUM_BLOQUES)
    {
      full_trainset <- leer_batch_train(numBatch, "promoted_content.csv", s_input_path)
      setkey(full_trainset, ad_id)
      stopifnot(unique(full_trainset$ad_id) %in% unique(promcnt$ad_id)) # Ok.
      full_trainset <- merge(full_trainset, promcnt, by = "ad_id")
      save(full_trainset, file = paste0(s_input_path, get_batch_train_filename(numBatch)))
      print(paste0(as.double((proc.time() - systime_ini)['elapsed'])/60, ' minutos en total.'))
      minutos_pend <- (as.double((proc.time() - systime_ini)['elapsed'])/60) * 2 * (NUM_BLOQUES / numBatch  -  1)
      if(minutos_pend < 60) print(paste0('Faltan aprox. ',minutos_pend, ' minutos.')) else print(paste0('Faltan aprox. ',minutos_pend/60, ' horas.'))
    }
  }
  if(!("ad_document_id" %in% colnames(testset)))
  {
    for(numBatch in 1:NUM_BLOQUES)
    {
      testset <- leer_batch_test(numBatch, "promoted_content.csv", s_input_path)
      setkey(testset, ad_id)
      stopifnot(unique(testset$ad_id) %in% unique(promcnt$ad_id)) # Ok.
      testset <- merge(testset, promcnt, by = "ad_id")
      save(testset, file = paste0(s_input_path, get_batch_test_filename(numBatch)))
      print(paste0(as.double((proc.time() - systime_ini)['elapsed'])/60, ' minutos en total.'))
      minutos_pend <- (as.double((proc.time() - systime_ini)['elapsed'])/60) * 2 * (NUM_BLOQUES / numBatch  -  1)
      if(minutos_pend < 60) print(paste0('Faltan aprox. ',minutos_pend, ' minutos.')) else print(paste0('Faltan aprox. ',minutos_pend/60, ' horas.'))
    }    
  }
  rm(promcnt)
  gc()
}
print('5.- promoted_content.csv - Ok.')
# # ------------------------------------------
# # 5.1.- "doc_probs.RData": (document_id, ad_document_id) -> (docprob, docclicks, doctot, ad_docprob, ad_docclicks, ad_doctot)
# # ------------------------------------------
# if(!file.exists(file.path(s_output_path, "doc_probs.RData")))
# {
#   # Calculamos la probabilidad de cada document_id usando clicked de todos los full_trainset:
#   # Usamos solo clicks/no_clicks de cada topic en los los que aparecen sus documentos (para que los pesos sean mayores):
#   mis_docs <- data.table(document_id = integer(0), clicks = integer(0), tot = integer(0))
#   mis_ad_docs <- data.table(ad_document_id = integer(0), clicks = integer(0), tot = integer(0))
#   for(numBatch in 1:NUM_BLOQUES)
#   {
#     full_trainset <- leer_batch_train(numBatch, "doc_probs.RData - prob(document_id) y prob(ad_document_id)", s_input_path)
#     stopifnot("document_id" %in% colnames(full_trainset))
#     stopifnot("ad_document_id" %in% colnames(full_trainset))
# 
#     # Frecuencias (clicked) de cada doc_id:
#     setkey(full_trainset, document_id)
#     setkey(mis_docs, document_id)
#     mis_docs <- merge(mis_docs, full_trainset[,.(clicks2 = sum(clicked), tot2 = .N), by = "document_id"], all = TRUE, by = "document_id")
#     for(j in colnames(mis_docs))  set(mis_docs, which(is.na(mis_docs[[j]])), j, 0) # NAs a cero
#     setkey(mis_docs, document_id)
#     mis_docs[, clicks := clicks + clicks2, by = "document_id"]
#     mis_docs[, tot := tot + tot2, by = "document_id"]
#     mis_docs[, c("clicks2", "tot2"):= NULL]
#     
#     # Frecuencias (clicked) de cada ad_doc_id:
#     setkey(full_trainset, ad_document_id)
#     setkey(mis_ad_docs, document_id)
#     mis_ad_docs <- merge(mis_ad_docs, full_trainset[,.(clicks2 = sum(clicked), tot2 = .N), by = "ad_document_id"], all = TRUE, by = "ad_document_id")
#     for(j in colnames(mis_ad_docs))  set(mis_ad_docs, which(is.na(mis_ad_docs[[j]])), j, 0) # NAs a cero
#     setkey(mis_ad_docs, document_id)
#     mis_ad_docs[, clicks := clicks + clicks2, by = "ad_document_id"]
#     mis_ad_docs[, tot := tot + tot2, by = "ad_document_id"]
#     mis_ad_docs[, c("clicks2", "tot2"):= NULL]
#   }
#   # Ahora calculamos esas probs:
#   mis_docs[, prob := clicks / tot]
#   setnames(mis_docs, old = c("clicks", "tot", "prob"), new = c("docclicks", "doctot", "docprob"))
#   summary(mis_docs)
#   
#   mis_ad_docs[, prob := clicks / tot]
#   setnames(mis_ad_docs, old = c("clicks", "tot", "prob"), new = c("ad_docclicks", "ad_doctot", "ad_docprob"))
#   summary(mis_ad_docs)
#   # Fusionamos todo en all_misdocs:
#   all_misdocs <- merge(mis_docs, mis_ad_docs, all = TRUE, by.x = "document_id", by.y = "ad_document_id")
#   for(j in colnames(all_misdocs))  set(all_misdocs, which(is.na(all_misdocs[[j]])), j, 0) # NAs a cero
#   # Guardamos doc_probs:
#   save(all_misdocs, file = file.path(s_output_path, "doc_probs.RData"))
#   rm(all_misdocs, mis_docs, mis_ad_docs); gc()
# } # else load(file.path(s_output_path, "doc_probs.RData"))
# print('5.1.- doc_probs.RData - Ok.')
# ------------------------------------------
# 6.- "documents_meta.csv": (document_id, source_id, publisher_id, publish_time) -> (publish_timestamp, source_id, publisher_id, ad_publish_timestamp, ad_source_id, ad_publisher_id)
# ------------------------------------------
if(!("source_id"    %in% colnames(full_trainset) & "source_id"    %in% colnames(testset) &
     "ad_source_id" %in% colnames(full_trainset) & "ad_source_id" %in% colnames(testset)))
{
  print('documents_meta.csv')
  docmeta <- fread(paste0(s_input_path, "documents_meta.csv"), nrows = ifelse(G_b_DEBUG, 2500L, -1L))
  # setkey(docmeta, document_id, source_id, publisher_id, publish_time)
  # source_id (the part of the site on which the document is displayed, e.g. edition.cnn.com)
  setkey(docmeta, publish_time)
  docmeta[publish_time == "", publish_time := NA]
  setkey(docmeta, publish_time)
  docmeta[year(as.POSIXct(publish_time))>2000 & year(as.POSIXct(publish_time))<2017, publish_timestamp := as.numeric(1000) * as.numeric(as.POSIXct(publish_time)) - as.numeric(1465876799998), by = publish_time] # (ms since 1970-01-01 - 1465876799998)

  sapply(docmeta, uniqueN)
  
  setkey(docmeta, document_id)
  # Finalmente, metemos los datos de los documents (by display_id-document_id) en trainset y en testset:
  if(!("source_id"    %in% colnames(full_trainset)))
  {
    for(numBatch in 1:NUM_BLOQUES)
    {
      full_trainset <- leer_batch_train(numBatch, "documents_meta.csv - source_id", s_input_path)
      setkey(full_trainset, document_id)
      stopifnot(unique(full_trainset$document_id) %in% unique(docmeta$document_id)) # Ok.
      full_trainset <- merge(full_trainset, docmeta, by = "document_id")
      save(full_trainset, file = paste0(s_input_path, get_batch_train_filename(numBatch)))
      print(paste0(as.double((proc.time() - systime_ini)['elapsed'])/60, ' minutos en total.'))
      minutos_pend <- (as.double((proc.time() - systime_ini)['elapsed'])/60) * 4 * (NUM_BLOQUES / numBatch  -  1)
      if(minutos_pend < 60) print(paste0('Faltan aprox. ',minutos_pend, ' minutos.')) else print(paste0('Faltan aprox. ',minutos_pend/60, ' horas.'))
    }
  }
  if(!("source_id"    %in% colnames(testset)))
  {
    for(numBatch in 1:NUM_BLOQUES)
    {
      testset <- leer_batch_test(numBatch, "documents_meta.csv - source_id", s_input_path)
      setkey(testset, document_id)
      stopifnot(unique(testset$document_id) %in% unique(docmeta$document_id)) # Ok.
      testset <- merge(testset, docmeta, by = "document_id")
      save(testset, file = paste0(s_input_path, get_batch_test_filename(numBatch)))
      print(paste0(as.double((proc.time() - systime_ini)['elapsed'])/60, ' minutos en total.'))
      minutos_pend <- (as.double((proc.time() - systime_ini)['elapsed'])/60) * 4 * (NUM_BLOQUES / numBatch  -  1)
      if(minutos_pend < 60) print(paste0('Faltan aprox. ',minutos_pend, ' minutos.')) else print(paste0('Faltan aprox. ',minutos_pend/60, ' horas.'))
    }    
  }
  
  # ad_docmeta <- copy(docmeta)
  colnames(docmeta) <- paste0('ad_', colnames(docmeta))
  colnames(docmeta)
  
  # Finalmente, metemos los datos de los ad_documents (by ad_id-ad_document_id) en trainset y en testset:
  if(!("ad_source_id" %in% colnames(full_trainset)))
  {
    for(numBatch in 1:NUM_BLOQUES)
    {
      full_trainset <- leer_batch_train(numBatch, "documents_meta.csv - ad_source_id", s_input_path)
      setkey(full_trainset, ad_document_id)
      stopifnot(unique(full_trainset$ad_document_id) %in% unique(docmeta$ad_document_id)) # Ok.
      full_trainset <- merge(full_trainset, docmeta, by = "ad_document_id")
      save(full_trainset, file = paste0(s_input_path, get_batch_train_filename(numBatch)))
      print(paste0(as.double((proc.time() - systime_ini)['elapsed'])/60, ' minutos en total.'))
      minutos_pend <- (as.double((proc.time() - systime_ini)['elapsed'])/60) * 4 * (NUM_BLOQUES / numBatch  -  1)
      if(minutos_pend < 60) print(paste0('Faltan aprox. ',minutos_pend, ' minutos.')) else print(paste0('Faltan aprox. ',minutos_pend/60, ' horas.'))
    }
  }
  if(!("ad_source_id" %in% colnames(testset)))
  {
    for(numBatch in 1:NUM_BLOQUES)
    {
      testset <- leer_batch_test(numBatch, "documents_meta.csv - ad_source_id", s_input_path)
      setkey(testset, ad_document_id)
      stopifnot(unique(testset$ad_document_id) %in% unique(docmeta$ad_document_id)) # Ok.
      testset <- merge(testset, docmeta, by = "ad_document_id")
      save(testset, file = paste0(s_input_path, get_batch_test_filename(numBatch)))
      print(paste0(as.double((proc.time() - systime_ini)['elapsed'])/60, ' minutos en total.'))
      minutos_pend <- (as.double((proc.time() - systime_ini)['elapsed'])/60) * 4 * (NUM_BLOQUES / numBatch  -  1)
      if(minutos_pend < 60) print(paste0('Faltan aprox. ',minutos_pend, ' minutos.')) else print(paste0('Faltan aprox. ',minutos_pend/60, ' horas.'))
    }
  }
  rm(docmeta)
  gc()
}
print('6.- documents_meta.csv - Ok.')
# ------------------------------------------
# 7.- "documents_topics.csv": (document_id, topic_id, confidence_level) -> (topics_prob, ad_topics_prob)
# ------------------------------------------
incluir_docs_conf_tipos(doctipos_file = 'documents_topics',
                        tipo_id = 'topic_id',
                        from_name = 'topic_prob',
                        to_name = 'topics_prob',
                        full_trainset = full_trainset, testset = testset, s_input_path = s_input_path, s_output_path = s_output_path)
print('7.- documents_topics.csv - Ok.')
# ------------------------------------------
# 8.- "documents_entities.csv": (document_id, entity_id, confidence_level) -> (entities_prob, ad_entities_prob)
# ------------------------------------------
incluir_docs_conf_tipos(doctipos_file = 'documents_entities',
                        tipo_id = 'entity_id',
                        from_name = 'entity_prob',
                        to_name = 'entities_prob',
                        full_trainset = full_trainset, testset = testset, s_input_path = s_input_path, s_output_path = s_output_path)
print('8.- documents_entities.csv - Ok.')
# ------------------------------------------
# 9.- "documents_categories.csv": (document_id, category_id, confidence_level) -> (categories_prob, ad_categories_prob)
# ------------------------------------------
incluir_docs_conf_tipos(doctipos_file = 'documents_categories',
                        tipo_id = 'category_id',
                        from_name = 'category_prob',
                        to_name = 'categories_prob',
                        full_trainset = full_trainset, testset = testset, s_input_path = s_input_path, s_output_path = s_output_path)
print('9.- documents_categories.csv - Ok.')
# ------------------------------------------------------------------------------------
# 10.- "page_views.csv"
# ------------------------------------------
# Cf. Ver también el Punto 14.-
if(!file.exists(paste0(s_input_path, "uuid", "_tiempos_pgvw.RData")))
{
  if(!file.exists(paste0(s_input_path, "pgvw_uuids.csv")))
  {
    rm(full_trainset, testset); gc() # Liberamos memoria
    # pgvw <- fread(paste0(s_input_path, "page_views.csv"), nrows = ifelse(G_b_DEBUG, 2500L, -1L))
    # # setkey(pgvw, uuid, document_id)
    # # # timestamp (ms since 1970-01-01 - 1465876799998)
    # # # platform (desktop = 1, mobile = 2, tablet =3)
    # # # geo_location (country>state>DMA)
    # # # traffic_source (internal = 1, search = 2, social = 3)
    
    # Primero cargamos los uuid que nos interesan (los de trainset y testset):
    print(paste0(Sys.time(), ' - ', 'Leyendo mis_uuids...'))
    if(!file.exists(paste0(s_input_path, "mis_uuids.RData")))
    {
      print(paste0(Sys.time(), ' - ', 'events.csv'))
      evts <- fread(paste0(s_input_path, "events.csv"), colClasses = c("integer", "character", "integer", "integer64", "character", "character"), nrows = ifelse(G_b_DEBUG, 2500L, -1L))
      setkeyv(evts, "uuid")
      mis_uuids <- unique(evts[, .(uuid)])
      save(mis_uuids, file = paste0(s_input_path, "mis_uuids.RData"))
      rm(evts); gc()
    } else { load(paste0(s_input_path, "mis_uuids.RData")) }
    print(paste0(Sys.time(), ' - ', 'Ok. ', format(nrow(mis_uuids), digits = 3, decimal.mark = ',', big.mark = '.'),' distinct uuids.'))
    
    # Fichero page_views.csv (100 GB!)
    fich <- paste0(s_input_path, "page_views.csv")
    systime_ini_2 <- proc.time()
    # Leemos la cabecera:
    pgvw <- read.table(fich, nrows=1, header=T, fill=TRUE, sep=",", stringsAsFactors = F)
    colNames <- colnames(pgvw)
    # system(paste0("wc -l ", fich)) # 2.034.275.449 líneas! Para contar las líneas del fichero (pero tarda mucho!)
    # Así que hacemos una estimación:
    # uuid    document_id      timestamp       platform   geo_location traffic_source 
    # 14              4              8              1              9              1 
    fichsize_in_lines <- 2034275449L # file.size(fich) / (14+1+7+1+8+1+1+1+9+1+1)
    blqSize <- 2000000L
    NUM_CHUNKS <- 1L + as.integer(fichsize_in_lines / blqSize)
    numRegs <- 0
    
    con <- file(description=fich, open="r")
    scan(con, nlines = 1, what = character(0)) # Saltamos la cabecera...
    for(numChunk in 1:NUM_CHUNKS) # bucle para leer page_views a trozos!
    {
      print(paste('Processing rows:', format(numChunk * blqSize, scientific = F, decimal.mark = ',', big.mark = '.'), ' (', format(numChunk, decimal.mark = ',', big.mark = '.'), '/', format(NUM_CHUNKS, decimal.mark = ',', big.mark = '.')
                  , ' [', format(100L * numChunk/NUM_CHUNKS, digits = 3, decimal.mark = ',', big.mark = '.'), '%])'))
      # pgvw <- fread(paste0(s_input_path, "page_views_sample.csv"), colClasses = c("character", "integer", "numeric", "integer", "character", "integer"), nrows = ifelse(G_b_DEBUG, 2500L, -1L))
      # # dataChunk <- fread(con, header = F, col.names = colNames, colClasses = c("character", "integer", "numeric", "integer", "character", "character"), nrows = blqSize)
      # dataChunk <- read.table(con, nrows=blqSize, col.names = colNames, skip=0, header=FALSE, fill = TRUE, sep=",", stringsAsFactors = F)
      pgvw <- as.data.table(scan(con, nlines = blqSize, sep = ',', what = list(character(0), integer(0), numeric(0), integer(0), character(0), integer(0))))
      colnames(pgvw) <- colNames
      
      print(paste0(Sys.time(), ' - ', numChunk, '/' , NUM_CHUNKS, ' - ', 'Filtramos por uuid (los de trainset y testset):'))
      setkeyv(pgvw, "uuid")
      pgvw <- merge(pgvw, mis_uuids, by = "uuid")
      
      numRegs <- numRegs + nrow(pgvw)
      print(paste0(Sys.time(), ' - ', 'Ok. ', format(numChunk * blqSize / 1e06, scientific = F, decimal.mark = ',', big.mark = '.'),'M. regs.'
                   , ' => ', format(round(numRegs / 1e06, 3), scientific = F, decimal.mark = ',', big.mark = '.'),'M. regs.'
                   , ' [Reduc. = ', format(100L * (numRegs/numChunk) / blqSize, digits = 3, decimal.mark = ',', big.mark = '.'), '%])'))
      
      pgvw[, platform := as.integer(substr(platform,1,1))]
      pgvw[, traffic_source := as.integer(substr(traffic_source,1,1))]
      pgvw[, geo_loc.country := str_split_fixed(geo_location, '>', Inf)[[1]], by = "geo_location"]
      pgvw[ geo_loc.country == "US", idpais := 1L, by = "geo_loc.country"]
      pgvw[ geo_loc.country == "CA", idpais := 2L, by = "geo_loc.country"]
      pgvw[ geo_loc.country == "GB", idpais := 3L, by = "geo_loc.country"]
      pgvw[!(geo_loc.country %in% c("US", "CA", "GB")), idpais := 4L, by = "geo_loc.country"]
      # print(sort(table(pgvw$pais), decreasing = T))
      # print(sort(table(pgvw$idpais), decreasing = T))
      # pgvw[, hora := as.integer(1 + (timestamp %/% 3600000) %% 24)] # De 1 a 24
      # pgvw[, dia  := as.integer(1 + timestamp %/% (3600000 * 24))]  # De 1 a 15
      
      cols <- c("uuid", "document_id", "platform", "traffic_source", "idpais", "timestamp")
      pgvw <- pgvw[,cols,with=F]
      
      fwrite(x = pgvw, file = paste0(s_input_path, "pgvw_uuids.csv"), append = T, eol = "\n", col.names = F)
      
      print(paste0(Sys.time(), ' - ', numChunk, '/' , NUM_CHUNKS, ' - ', round(as.double((proc.time() - systime_ini)['elapsed'])/60, 1), ' minutos en total.'))
      minutos_pend <- (as.double((proc.time() - systime_ini_2)['elapsed'])/60) * ( NUM_CHUNKS / numChunk  -  1)
      if(minutos_pend < 60) print(paste0(Sys.time(), ' - ', numChunk, '/' , NUM_CHUNKS, ' - Faltan aprox. ',minutos_pend, ' minutos.')) else print(paste0(Sys.time(), ' - ', numChunk, '/' , NUM_CHUNKS, ' - Faltan aprox. ',minutos_pend/60, ' horas.'))
    } # for(numChunk in 1:NUM_CHUNKS)
    close(con)
    rm(pgvw, mis_uuids)
    gc()
    # Recargamos full_trainset y testset:
    full_trainset <- leer_batch_train(NUM_BLOQUES, "inicio", s_input_path)
    if( file.exists(paste0(s_input_path, get_batch_test_filename(NUM_BLOQUES, NUM_BLOQUES))))
      testset <- leer_batch_test(NUM_BLOQUES, "inicio", s_input_path, numSubBatch = NUM_BLOQUES)
    if(!file.exists(paste0(s_input_path, get_batch_test_filename(NUM_BLOQUES, NUM_BLOQUES))))
      testset <- leer_batch_test(NUM_BLOQUES, "inicio", s_input_path)
  }
  if(file.exists(paste0(s_input_path, "pgvw_uuids.csv")))
  {
    if(!file.exists(paste0(s_input_path, "uuid", "_tiempos_pgvw.RData")))
    {
      #
      # Preparamos uuid_tiempos_pgvw.RData:
      #
      rm(full_trainset, testset); gc() # Liberamos memoria
      # pgvw <- fread(paste0(s_input_path, "pgvw_uuids.csv"), colClasses = c("character", "integer", "numeric", "integer", "character", "integer"), nrows = ifelse(G_b_DEBUG, 2500L, -1L))
      # Cargamos docmeta para añadir "source_id", "publisher_id", "publish_timestamp":
      print(paste0(Sys.time(), ' - ', 'Leyendo docmeta...'))
      docmeta <- fread(paste0(s_input_path, "documents_meta.csv"), nrows = ifelse(G_b_DEBUG, 2500L, -1L))
      setkeyv(docmeta, "publish_time")
      docmeta[publish_time == "", publish_time := NA]
      setkeyv(docmeta, "publish_time")
      docmeta[, publish_timestamp := as.numeric(1000) * as.numeric(as.POSIXct(publish_time)) - as.numeric(1465876799998), by = "publish_time"] # (ms since 1970-01-01 - 1465876799998)
      docmeta[, publish_time := NULL]
      setkeyv(docmeta, "document_id")
      
      mi_dif_op <- function(FUN_op, mi_vector, na.rm = T, is.sorted = T)
      {
        if(!is.sorted)  mi_vector <- sort(mi_vector)
        # Devuelve la mínima diferencia entre un elemento y el siguiente:
        if(length(mi_vector) < 2) {
          return(mi_vector - mi_vector) # cero!
        } else {
          return(FUN_op(mi_vector[2:(length(mi_vector))] - mi_vector[1:(length(mi_vector)-1)], na.rm = na.rm))
        }
      }
      # Columnas por las que vamos a agrupar (en page_views):
      miscols <- c("uuid")
      # lista_pgvw_docs <- list()
      mivar <- "uuid"
      # for(mivar in miscols)
      # {
        # Creamos lista de esos data.tables (una para cada variable):
        eval(parse(text = paste0("pgvw_docs <- data.table(", mivar
                                 , " = ", ifelse(mivar == "uuid", "character(0)",ifelse(mivar %in% c("publish_timestamp", "ad_publish_timestamp"),"numeric(0)","integer(0)"))
                                 , ", tot = integer(0)"
                                 , ", timestamp_min = numeric(0), timestamp_max = numeric(0), timestamp_avg = numeric(0), timestamp_var = numeric(0)"
                                 , ", timestamp_difmin = numeric(0), timestamp_difmax = numeric(0), timestamp_difavg = numeric(0), timestamp_difvar = numeric(0)"
                                 , ", paisUS = numeric(0), paisCA = numeric(0), paisGB = numeric(0), paisResto = numeric(0)"
                                 , ", platform1 = numeric(0), platform2 = numeric(0), platform3 = numeric(0)"
                                 , ", trafsrc1 = numeric(0), trafsrc2 = numeric(0), trafsrc3 = numeric(0)"
                                 , ")")))
      # }
      # Empezamos con el data.table entero (con los 19M de usuarios), a ver si así tarda menos:
      print(paste0(Sys.time(), ' - ', 'Leyendo mis_uuids...'))
      load(paste0(s_input_path, "mis_uuids.RData"))
      mis_uuids[, colnames(pgvw_docs)[colnames(pgvw_docs) != "uuid"] := as.numeric(0)]
      pgvw_docs <- mis_uuids
      rm(mis_uuids); gc()
      print(paste0(Sys.time(), ' - ', 'Ok. Empezamos con pgvw_uuids.csv...'))
      # Fichero pgvw_uuids.csv (7 GB!)
      fich <- paste0(s_input_path, "pgvw_uuids.csv")
      systime_ini_2 <- proc.time()
      # Cabecera:
      # pgvw <- read.table(fich, nrows=1, header=T, fill=TRUE, sep=",", stringsAsFactors = F)
      colNames <- c("uuid", "document_id", "platform", "traffic_source", "idpais", "timestamp")
      # system(paste0("wc -l ", fich)) # 195.013.040 líneas! Para contar las líneas del fichero (pero tarda mucho!)
      fichsize_in_lines <- 195013040L
      blqSize <- 100000L
      NUM_CHUNKS <- 1L + as.integer(fichsize_in_lines / blqSize)
      
      con <- file(description=fich, open="r")
      scan(con, nlines = 1, what = character(0)) # Saltamos la cabecera...
      for(numChunk in 1:NUM_CHUNKS) # bucle para leer page_views a trozos!
      {
        print(paste('Processing rows:', numChunk * blqSize, ' (', format(numChunk, decimal.mark = ',', big.mark = '.'), '/', format(NUM_CHUNKS, decimal.mark = ',', big.mark = '.')
                    , ' [', format(100L * numChunk/NUM_CHUNKS, digits = 3, decimal.mark = ',', big.mark = '.'), '%])'))
        # pgvw <- fread(paste0(s_input_path, "page_views_sample.csv"), colClasses = c("character", "integer", "numeric", "integer", "character", "integer"), nrows = ifelse(G_b_DEBUG, 2500L, -1L))
        # # dataChunk <- fread(con, header = F, col.names = colNames, colClasses = c("character", "integer", "numeric", "integer", "character", "character"), nrows = blqSize)
        # dataChunk <- read.table(con, nrows=blqSize, col.names = colNames, skip=0, header=FALSE, fill = TRUE, sep=",", stringsAsFactors = F)
        pgvw <- as.data.table(scan(con, nlines = blqSize, sep = ',', what = list(character(0), integer(0), integer(0), integer(0), integer(0), numeric(0))))
        colnames(pgvw) <- colNames # c("uuid", "document_id", "platform", "traffic_source", "idpais", "timestamp")
        # Añadimos "source_id", "publisher_id", "publish_timestamp":
        setkeyv(pgvw, "document_id")
        pgvw <- merge(pgvw, docmeta, all.x = T, by = "document_id")
        for(mivar in miscols)
        {
          mivar_fich <- paste0(mivar, "_tiempos_pgvw.RData")
          if(!file.exists(paste0(s_input_path, mivar_fich)))
          {
            print(paste0(Sys.time(), ' - ', numChunk, '/' , NUM_CHUNKS, ' - (', mivar, ') - Creando pgvw_docs [', mivar_fich, ']...'))
            setkeyv(pgvw, c(mivar, "timestamp"))
            # pgvw_docs <- lista_pgvw_docs[[mivar]] # Esto no debería hacer una copia porque son data.tables
            pgvw <- pgvw[,.(tot2 = .N
                          , paisUS2 = sum(idpais == 1) # Cantidad de cada pais por mivar
                          , paisCA2 = sum(idpais == 2) # Cantidad de cada pais por mivar
                          , paisGB2 = sum(idpais == 3) # Cantidad de cada pais por mivar
                          , paisResto2 = sum(idpais == 4) # Cantidad de cada pais por mivar
                          , platform12 = sum(platform == 1) # Cantidad de cada plataforma por mivar
                          , platform22 = sum(platform == 2) # Cantidad de cada plataforma por mivar
                          , platform32 = sum(platform == 3) # Cantidad de cada plataforma por mivar
                          , trafsrc12 = sum(traffic_source == 1) # Cantidad de cada traffic_source por mivar
                          , trafsrc22 = sum(traffic_source == 2) # Cantidad de cada traffic_source por mivar
                          , trafsrc32 = sum(traffic_source == 3) # Cantidad de cada traffic_source por mivar
                          , timestamp_min2 = timestamp[1]
                          , timestamp_avg2 = mean(timestamp) # Habrá que recalcularla (usando tot2), para no tener que gestionar sumas con números tan grandes
                          , timestamp_var2 = var(timestamp) # Habrá que recalcularla (usando tot2), para no tener que gestionar sumas con números tan grandes
                          , timestamp_max2 = timestamp[.N]
                          , timestamp_difmin2 = mi_dif_op(min,  timestamp) # Mínimo   de diferencias entre timestamps consecutivos # Habrá que recalcularla (usando tot2)
                          , timestamp_difavg2 = mi_dif_op(mean, timestamp) # Media    de diferencias entre timestamps consecutivos # Habrá que recalcularla (usando tot2)
                          , timestamp_difvar2 = mi_dif_op(var,  timestamp) # Varianza de diferencias entre timestamps consecutivos # Habrá que recalcularla (usando tot2)
                          , timestamp_difmax2 = mi_dif_op(max,  timestamp) # Máximo   de diferencias entre timestamps consecutivos # Habrá que recalcularla (usando tot2)
                          ), by = mivar]
            pgvw_docs <- merge(pgvw_docs, pgvw, all.x = TRUE, by = mivar)
            # for(j in c("tot", "tot2", "paisUS", "paisCA", "paisGB", "paisResto", "platform1", "platform2", "platform3", "trafsrc1", "trafsrc2", "trafsrc3", "paisUS2", "paisCA2", "paisGB2", "paisResto2", "platform12", "platform22", "platform32", "trafsrc12", "trafsrc22", "trafsrc32"))
            for(j in c("tot2", "paisUS2", "paisCA2", "paisGB2", "paisResto2", "platform12", "platform22", "platform32", "trafsrc12", "trafsrc22", "trafsrc32"))
              set(pgvw_docs, which(is.na(pgvw_docs[[j]])), j, 0) # NAs a cero
            print(paste0(Sys.time(), ' - ', numChunk, '/' , NUM_CHUNKS, ' - (', mivar, ') - Acumulando datos pgvw_docs...'))
            setkeyv(pgvw_docs, mivar)
            # pgvw_docs[, timestamp_min := min(c(timestamp_min, timestamp_min2), na.rm = T), by = mivar]
            # pgvw_docs[, timestamp_max := max(c(timestamp_max, timestamp_max2), na.rm = T), by = mivar]
            # pgvw_docs[, timestamp_avg := mean(c(tot * timestamp_avg, tot2 * timestamp_avg2), na.rm = T)/sum(c(tot, tot2), na.rm = T), by = mivar]
            # pgvw_docs[, timestamp_var := mean(c(tot * timestamp_var, tot2 * timestamp_var2), na.rm = T)/sum(c(tot, tot2), na.rm = T), by = mivar]
            # pgvw_docs[, timestamp_difmin := min(c(timestamp_difmin, timestamp_difmin2), na.rm = T), by = mivar]
            # pgvw_docs[, timestamp_difmax := max(c(timestamp_difmax, timestamp_difmax2), na.rm = T), by = mivar]
            # pgvw_docs[, timestamp_difavg := mean(c(tot * timestamp_difavg, tot2 * timestamp_difavg2), na.rm = T)/sum(c(tot, tot2), na.rm = T), by = mivar]
            # pgvw_docs[, timestamp_difvar := mean(c(tot * timestamp_difvar, tot2 * timestamp_difvar2), na.rm = T)/sum(c(tot, tot2), na.rm = T), by = mivar]
            # pgvw_docs[, tot := tot + tot2, by = mivar]
            # pgvw_docs[, platform1 := platform1 + platform12, by = mivar]
            # pgvw_docs[, platform2 := platform2 + platform22, by = mivar]
            # pgvw_docs[, platform3 := platform3 + platform32, by = mivar]
            # pgvw_docs[, paisUS := paisUS + paisUS2, by = mivar]
            # pgvw_docs[, paisCA := paisCA + paisCA2, by = mivar]
            # pgvw_docs[, paisGB := paisGB + paisGB2, by = mivar]
            # pgvw_docs[, paisResto := paisResto + paisResto2, by = mivar]
            # pgvw_docs[, trafsrc1 := trafsrc1 + trafsrc12, by = mivar]
            # pgvw_docs[, trafsrc2 := trafsrc2 + trafsrc22, by = mivar]
            # pgvw_docs[, trafsrc3 := trafsrc3 + trafsrc32, by = mivar]
            pgvw_docs[uuid %in% pgvw$uuid, c("timestamp_min", "timestamp_max", "timestamp_avg", "timestamp_var", "timestamp_difmin", "timestamp_difmax", "timestamp_difavg", "timestamp_difvar",
                          "tot", "platform1", "platform2", "platform3", "paisUS", "paisCA", "paisGB", "paisResto", "trafsrc1", "trafsrc2", "trafsrc3") := 
                      list(
                        min(c(timestamp_min, timestamp_min2), na.rm = T)
                      , max(c(timestamp_max, timestamp_max2), na.rm = T)
                      , mean(c(tot * timestamp_avg, tot2 * timestamp_avg2), na.rm = T)/sum(c(tot, tot2), na.rm = T)
                      , mean(c(tot * timestamp_var, tot2 * timestamp_var2), na.rm = T)/sum(c(tot, tot2), na.rm = T)
                      , min(c(timestamp_difmin, timestamp_difmin2), na.rm = T)
                      , max(c(timestamp_difmax, timestamp_difmax2), na.rm = T)
                      , mean(c(tot * timestamp_difavg, tot2 * timestamp_difavg2), na.rm = T)/sum(c(tot, tot2), na.rm = T)
                      , mean(c(tot * timestamp_difvar, tot2 * timestamp_difvar2), na.rm = T)/sum(c(tot, tot2), na.rm = T)
                      , tot + tot2
                      , platform1 + platform12
                      , platform2 + platform22
                      , platform3 + platform32
                      , paisUS + paisUS2
                      , paisCA + paisCA2
                      , paisGB + paisGB2
                      , paisResto + paisResto2
                      , trafsrc1 + trafsrc12
                      , trafsrc2 + trafsrc22
                      , trafsrc3 + trafsrc32
                      ), by = mivar]
            pgvw_docs[, c("tot2", "platform12", "platform22", "platform32", "trafsrc12", "trafsrc22", "trafsrc32",
                          "paisUS2", "paisCA2", "paisGB2", "paisResto2",
                          "timestamp_min2", "timestamp_max2", "timestamp_avg2", "timestamp_var2",
                          "timestamp_difmin2", "timestamp_difmax2", "timestamp_difavg2", "timestamp_difvar2"):= NULL]
            # lista_pgvw_docs[[mivar]] <- pgvw_docs # El merge() hizo una copia, así que hay que devolverlo a la lista
            if(numChunk %% 10 == 0)
            {
              # Guardamos fichero (temporalmente):
              mivar_fich_tmp <- paste0(mivar, numChunk, "_", NUM_CHUNKS, "_tiempos_pgvw.RData")
              print(paste0('Guardando pgvw_docs (tmp) [', mivar_fich_tmp, '] (', nrow(pgvw_docs), ' registros)...'))
              save(pgvw_docs, file = paste0(s_input_path, mivar_fich_tmp)) # load(file = paste0(s_input_path, mivar_fich_tmp))
              # Borramos el anterior, si existe:
              mivar_fich_tmp <- paste0(mivar, numChunk - 10, "_", NUM_CHUNKS, "_tiempos_pgvw.RData")
              if(file.exists(paste0(s_input_path, mivar_fich_tmp))) try(file.remove(paste0(s_input_path, mivar_fich_tmp)), silent = T)
            }
            print(paste0(Sys.time(), ' - ', numChunk, '/' , NUM_CHUNKS, ' - (', mivar, ') - ', 'Ok. pgvw_docs - (', mivar, ')...'))
            print(paste0(Sys.time(), ' - ', numChunk, '/' , NUM_CHUNKS, ' - (', mivar, ') - ', as.double((proc.time() - systime_ini)['elapsed'])/60, ' minutos en total.'))
          }
        } # for(mivar in miscols)
        minutos_pend <- (as.double((proc.time() - systime_ini_2)['elapsed'])/60) * ( NUM_CHUNKS / numChunk  -  1)
        if(minutos_pend < 60) print(paste0(Sys.time(), ' - ', numChunk, '/' , NUM_CHUNKS, ' - Faltan aprox. ',minutos_pend, ' minutos.')) else print(paste0(Sys.time(), ' - ', numChunk, '/' , NUM_CHUNKS, ' - Faltan aprox. ',minutos_pend/60, ' horas.'))
      } # for(numChunk in 1:NUM_CHUNKS)
      close(con)
      # for(mivar in miscols)
      # {
        mivar_fich <- paste0(mivar, "_tiempos_pgvw.RData")
        if(!file.exists(paste0(s_input_path, mivar_fich)))
        {
          # Guardamos este fichero antes de meter los campos en trainset/testset:
          # pgvw_docs <- lista_pgvw_docs[[mivar]] # Esto no debería hacer una copia porque son data.tables
          pgvw_docs[, hora_min := as.integer(1 + (timestamp_min %/% 3600000) %% 24)] # De 1 a 24
          pgvw_docs[, dia_min  := as.integer(1 + timestamp_min %/% (3600000 * 24))]  # De 1 a 15
          pgvw_docs[, hora_max := as.integer(1 + (timestamp_max %/% 3600000) %% 24)] # De 1 a 24
          pgvw_docs[, dia_max  := as.integer(1 + timestamp_max %/% (3600000 * 24))]  # De 1 a 15
          pgvw_docs[, hora_avg := as.integer(1 + (timestamp_avg %/% 3600000) %% 24)] # De 1 a 24
          pgvw_docs[, dia_avg  := as.integer(1 + timestamp_avg %/% (3600000 * 24))]  # De 1 a 15
          if(anyNA(pgvw_docs[, mivar, with=F]))
            pgvw_docs <- pgvw_docs[!is.na(get(mivar)),] # Quitamos el registro donde pk == NA !!! (si lo hay)
          # Guardamos:
          print(paste0('Guardando pgvw_docs [', mivar_fich, '] (', nrow(pgvw_docs), ' registros)...'))
          save(pgvw_docs, file = paste0(s_input_path, mivar_fich)) # load(file = paste0(s_input_path, mivar_fich))
          # rm(pgvw_docs)
        }
      # }
      # summary(pgvw$dia) # NOTA: El fichero sample está todo en un único día (el primero), así que aquí no hay información del día...
      # ( max(pgvw$timestamp) - min(pgvw$timestamp) )%/% (3600000 * 24)
      rm(pgvw_docs, pgvw)
      gc()
      # Recargamos full_trainset y testset:
      full_trainset <- leer_batch_train(NUM_BLOQUES, "inicio", s_input_path)
      if( file.exists(paste0(s_input_path, get_batch_test_filename(NUM_BLOQUES, NUM_BLOQUES))))
        testset <- leer_batch_test(NUM_BLOQUES, "inicio", s_input_path, numSubBatch = NUM_BLOQUES)
      if(!file.exists(paste0(s_input_path, get_batch_test_filename(NUM_BLOQUES, NUM_BLOQUES))))
        testset <- leer_batch_test(NUM_BLOQUES, "inicio", s_input_path)
    }
  }
}
print("10.- page_views.csv - Ok.")

# ------------------------------------------------------------------------------------
# 11.- Preparamos ficheros "xxxx_tiempos_def.RData" con cálculos temporales (timestamp):
# ------------------------------------------
# "display_id" "ad_id" "clicked"
# "uuid" "document_id" ("source_id" "publisher_id" "publish_timestamp") "timestamp"
# "ad_document_id" "ad_source_id" "ad_publisher_id" "ad_publish_timestamp"
if("publish_time" %in% colnames(full_trainset) | "publish_time" %in% colnames(testset))
{
  # Columnas por las que vamos a agrupar (y, de paso, para las que calcularemos su prob -docprob, etc.-):
  miscols <- c("uuid", # "clicked", ("display_id" no tiene sentido porque no hay ninguno coincidente entre train y test)
               "document_id", "source_id", "publisher_id", "publish_timestamp" # , "timestamp"
  )
  ad_miscols <- c("ad_id", # "clicked",
                  "ad_document_id", "ad_source_id", "ad_publisher_id", "ad_publish_timestamp" # , "timestamp"
  )

  lista_mis_probs <- list() 
  # mis_probs <- data.table(xxxx_id = integer(0), clicks = integer(0), tot = integer(0))
  for(mivar in c(miscols, ad_miscols)){
    eval(parse(text = paste0("lista_mis_probs[['", mivar, "']] <- data.table(", mivar
                             , " = ", ifelse(mivar == "uuid", "character(0)",ifelse(mivar %in% c("publish_timestamp", "ad_publish_timestamp"),"numeric(0)","integer(0)"))
                             , ", clicks = integer(0), tot = integer(0)"
                             , ", timestamp_min = numeric(0), timestamp_max = numeric(0), timestamp_avg = numeric(0), timestamp_var = numeric(0)"
                             , ", timestamp_difmin = numeric(0), timestamp_difmax = numeric(0), timestamp_difavg = numeric(0), timestamp_difvar = numeric(0)"
                             , ", prob = numeric(0)"
                             , ")")))} # Lista de esos data.tables (una para cada variable)
    # class(lista_mis_probs[["document_id"]]); lista_mis_probs[["ad_publish_timestamp"]]

  mivar_ficheros <- paste0(c(miscols, ad_miscols), "_tiempos_def.RData")
  if(any(!file.exists(paste0(s_input_path, mivar_ficheros))))
  {
    # Falta algún fichero "_def": Iniciamos proceso:
    mi_dif_op <- function(FUN_op, mi_vector, na.rm = T, is.sorted = T)
    {
      if(!is.sorted)  mi_vector <- sort(mi_vector)
      # Devuelve la mínima diferencia entre un elemento y el siguiente:
      if(length(mi_vector) < 2) {
        return(mi_vector - mi_vector) # cero!
      } else {
        return(FUN_op(mi_vector[2:(length(mi_vector))] - mi_vector[1:(length(mi_vector)-1)], na.rm = na.rm))
      }
    }
    
    mivar_ficheros <- paste0(c(miscols, ad_miscols), "_tiempos_tmp.RData")
    if(any(!file.exists(paste0(s_input_path, mivar_ficheros))))
    {
      systime_ini_2 <- proc.time()
      for(numBatch in 1:NUM_BLOQUES)
      { 
        full_trainset <- leer_batch_train(numBatch, "Calculando timestamps", s_input_path)
        full_trainset[, timestamp := as.numeric(timestamp)] # Para evitar overflow (incluso con integer64)
        # if("publish_time" %in% colnames(full_trainset))  full_trainset[, publish_time:=NULL]
        # if("ad_publish_time" %in% colnames(full_trainset))  full_trainset[, ad_publish_time:=NULL]
        # if("docprob" %in% colnames(full_trainset))  full_trainset[, docprob:=NULL]
        # if("ad_docprob" %in% colnames(full_trainset))  full_trainset[, ad_docprob:=NULL]
        # gc()
        # # # NOTA: docprob va a ser recalculado, junto con otras variables más...
        # # setkey(full_trainset, document_id)
        # # stopifnot(unique(full_trainset$document_id) %in% all_misdocs$document_id) # Ok.
        # # full_trainset <- merge(full_trainset, all_misdocs, by = "document_id")
        # # stopifnot(unique(full_trainset$ad_document_id) %in% ad_all_misdocs$ad_document_id) # Ok.
        # # full_trainset <- merge(full_trainset, ad_all_misdocs, by = "ad_document_id")
        # print('Guardando full_trainset...')
        # save(full_trainset, file = paste0(s_input_path, get_batch_train_filename(numBatch)))
        
        # Ahora guardamos en ficheros los datos (acumulados) de tiempos:
        # uniqueN(full_trainset) # length(unique(full_trainset$ad_document_id))
        # Frecuencias (clicks) y cálculos con timestamp de cada doc_id, source_id, etc...:
        for(mivar in c(miscols, ad_miscols))
        {
          mivar_fich <- paste0(mivar, "_tiempos_tmp.RData")
          if(!file.exists(paste0(s_input_path, mivar_fich)))
          {
            print(paste0('Calculando timestamps full_trainset (', mivar, ')...'))
            mis_probs <- lista_mis_probs[[mivar]] # Esto no debería hacer una copia porque son data.tables
            setkeyv(mis_probs, mivar)
            setkeyv(full_trainset, c(mivar, "timestamp"))
            # setorderv(x = full_trainset, cols = c(mivar, "timestamp")) #Sort by (xxx, timestamp)
            print(paste0('Añadiendo datos - merge - (', mivar, ')...'))
            mis_probs <- merge(mis_probs, full_trainset[,.(  clicks2 = sum(clicked)
                                                             , tot2 = .N
                                                             , timestamp_min2 = timestamp[1]
                                                             , timestamp_avg2 = mean(timestamp) # Habrá que recalcularla (usando tot2), para no tener que gestionar sumas con números tan grandes
                                                             , timestamp_var2 = var(timestamp) # Habrá que recalcularla (usando tot2), para no tener que gestionar sumas con números tan grandes
                                                             , timestamp_max2 = timestamp[.N]
                                                             , timestamp_difmin2 = mi_dif_op(min,  timestamp) # Mínimo   de diferencias entre timestamps consecutivos # Habrá que recalcularla (usando tot2)
                                                             , timestamp_difavg2 = mi_dif_op(mean, timestamp) # Media    de diferencias entre timestamps consecutivos # Habrá que recalcularla (usando tot2)
                                                             , timestamp_difvar2 = mi_dif_op(var,  timestamp) # Varianza de diferencias entre timestamps consecutivos # Habrá que recalcularla (usando tot2)
                                                             , timestamp_difmax2 = mi_dif_op(max,  timestamp) # Máximo   de diferencias entre timestamps consecutivos # Habrá que recalcularla (usando tot2)
            ), by = mivar], all = TRUE, by = mivar)
            for(j in c("clicks", "tot", "clicks2", "tot2"))
              set(mis_probs, which(is.na(mis_probs[[j]])), j, 0) # NAs a cero
            print(paste0('Acumulando datos - (', mivar, ')...'))
            setkeyv(mis_probs, mivar)
            mis_probs[, timestamp_min := min(c(timestamp_min, timestamp_min2), na.rm = T), by = mivar]
            mis_probs[, timestamp_max := max(c(timestamp_max, timestamp_max2), na.rm = T), by = mivar]
            mis_probs[, timestamp_avg := mean(c(tot * timestamp_avg, tot2 * timestamp_avg2), na.rm = T)/sum(c(tot, tot2), na.rm = T), by = mivar]
            mis_probs[, timestamp_var := mean(c(tot * timestamp_var, tot2 * timestamp_var2), na.rm = T)/sum(c(tot, tot2), na.rm = T), by = mivar]
            mis_probs[, timestamp_difmin := min(c(timestamp_difmin, timestamp_difmin2), na.rm = T), by = mivar]
            mis_probs[, timestamp_difmax := max(c(timestamp_difmax, timestamp_difmax2), na.rm = T), by = mivar]
            mis_probs[, timestamp_difavg := mean(c(tot * timestamp_difavg, tot2 * timestamp_difavg2), na.rm = T)/sum(c(tot, tot2), na.rm = T), by = mivar]
            mis_probs[, timestamp_difvar := mean(c(tot * timestamp_difvar, tot2 * timestamp_difvar2), na.rm = T)/sum(c(tot, tot2), na.rm = T), by = mivar]
            mis_probs[, clicks := clicks + clicks2, by = mivar]
            mis_probs[, tot := tot + tot2, by = mivar]
            mis_probs[, c("clicks2", "tot2"):= NULL]
            mis_probs[, c("timestamp_min2", "timestamp_max2", "timestamp_avg2", "timestamp_var2"):= NULL]
            mis_probs[, c("timestamp_difmin2", "timestamp_difmax2", "timestamp_difavg2", "timestamp_difvar2"):= NULL]
            lista_mis_probs[[mivar]] <- mis_probs # El merge() hizo una copia, así que hay que devolverlo a la lista
            print(paste0('Ok. - (', mivar, ')...'))
            
            minutos_pend <- (as.double((proc.time() - systime_ini_2)['elapsed'])/60) * (NUM_BLOQUES / numBatch  -  1)
            if(minutos_pend < 60) print(paste0('Faltan aprox. ', minutos_pend, ' minutos.')) else print(paste0('Faltan aprox. ', minutos_pend/60, ' horas.'))
          }
        }
        
        print(paste0(as.double((proc.time() - systime_ini)['elapsed'])/60, ' minutos en total.'))
        minutos_pend <- (as.double((proc.time() - systime_ini_2)['elapsed'])/60) * (NUM_BLOQUES / numBatch  -  1)
        if(minutos_pend < 60) print(paste0('Faltan aprox. ', minutos_pend, ' minutos.')) else print(paste0('Faltan aprox. ', minutos_pend/60, ' horas.'))
        rm(full_trainset); gc()
      }
      # Guardamos todo, antes de continuar con testset:
      for(mivar in c(miscols, ad_miscols))
      {
        mivar_fich <- paste0(mivar, "_tiempos_tmp.RData")
        if(!file.exists(paste0(s_input_path, mivar_fich)))
        {
          # Calculamos xxxx_id_prob (a partir de aquí, "clicks" ya no se mueve, pero "tot" puede crecer en el testset):
          setkeyv(lista_mis_probs[[mivar]], mivar)
          lista_mis_probs[[mivar]][, prob := clicks/tot, by = mivar]
          mis_probs <- lista_mis_probs[[mivar]] # Esto no debería hacer una copia porque son data.tables
          # Guardamos:
          print(paste0('Guardando ', mivar_fich, ' (', nrow(mis_probs), ' registros)...'))
          save(mis_probs, file = paste0(s_input_path, mivar_fich)) # load(file = paste0(s_input_path, mivar_fich))
        }
      }
    }
    
    mivar_ficheros <- paste0(c(miscols, ad_miscols), "_tiempos_def.RData")
    if(any(!file.exists(paste0(s_input_path, mivar_ficheros))))
    {
      systime_ini_2 <- proc.time()
      for(numBatch in 1:NUM_BLOQUES)
      {
        for(numSubBatch in 1:NUM_BLOQUES)
        {
          if(!file.exists(paste0(s_input_path, get_batch_test_filename(NUM_BLOQUES, NUM_BLOQUES))))
          {
            # No hay subdivisión de testsets. Lo hacemos con los "grandes":
            if(numSubBatch != 1)  break # Sólo el primero!
            testset <- leer_batch_test(numBatch, "Calculando timestamps", s_input_path)
          } else {
            testset <- leer_batch_test(numBatch, numSubBatch = numSubBatch, s_descr = "Calculando timestamps", s_input_path = s_input_path)
          }
          testset[, timestamp := as.numeric(timestamp)] # Para evitar overflow (incluso con integer64)
          # if("publish_time" %in% colnames(testset))  testset[, publish_time:=NULL]
          # if("ad_publish_time" %in% colnames(testset))  testset[, ad_publish_time:=NULL]
          # if("docprob" %in% colnames(testset))  testset[, docprob:=NULL]
          # if("ad_docprob" %in% colnames(testset))  testset[, ad_docprob:=NULL]
          # # # NOTA: ad_docprob va a ser recalculado, junto con otras variables más...
          # # setkey(testset, document_id)
          # # stopifnot(unique(testset$document_id) %in% all_misdocs$document_id) # Ok.
          # # testset <- merge(testset, all_misdocs, by = "document_id")
          # # stopifnot(unique(testset$ad_document_id) %in% ad_all_misdocs$ad_document_id) # Ok.
          # # testset <- merge(testset, ad_all_misdocs, by = "ad_document_id")
          # print('Guardando testset...')
          # save(testset, file = paste0(s_input_path, get_batch_test_filename(numBatch, numSubBatch = numSubBatch)))
          
          for(mivar in c(miscols, ad_miscols))
          {
            mivar_fich <- paste0(mivar, "_tiempos_def.RData")
            if(!file.exists(paste0(s_input_path, mivar_fich)))
            {
              print(paste0(numBatch, '_', numSubBatch, '/' , NUM_BLOQUES, ' - Calculando timestamps testset (', mivar, ')...'))
              mis_probs <- lista_mis_probs[[mivar]] # Esto no debería hacer una copia porque son data.tables
              setkeyv(mis_probs, mivar)
              setkeyv(testset, c(mivar, "timestamp"))
              # setorderv(x = testset, cols = c(mivar, "timestamp")) #Sort by (xxx, timestamp)
              print(paste0(numBatch, '_', numSubBatch, '/' , NUM_BLOQUES, ' - Añadiendo datos - merge - (', mivar, ')...'))
              mis_probs <- merge(mis_probs, testset[,.(  tot2 = .N, clicks2 = 0 # No hay "clicked" (TESTSET!!!)
                                                         , timestamp_min2 = timestamp[1]
                                                         , timestamp_avg2 = mean(timestamp) # Habrá que recalcularla (usando tot2), para no tener que gestionar sumas con números tan grandes
                                                         , timestamp_var2 = var(timestamp) # Habrá que recalcularla (usando tot2), para no tener que gestionar sumas con números tan grandes
                                                         , timestamp_max2 = timestamp[.N]
                                                         , timestamp_difmin2 = mi_dif_op(min,  timestamp) # Mínimo   de diferencias entre timestamps consecutivos # Habrá que recalcularla (usando tot2)
                                                         , timestamp_difavg2 = mi_dif_op(mean, timestamp) # Media    de diferencias entre timestamps consecutivos # Habrá que recalcularla (usando tot2)
                                                         , timestamp_difvar2 = mi_dif_op(var,  timestamp) # Varianza de diferencias entre timestamps consecutivos # Habrá que recalcularla (usando tot2)
                                                         , timestamp_difmax2 = mi_dif_op(max,  timestamp) # Máximo   de diferencias entre timestamps consecutivos # Habrá que recalcularla (usando tot2)
              ), by = mivar], all = TRUE, by = mivar)
              for(j in c("clicks", "clicks2", "tot", "tot2", "prob")) # Añadimos "prob" por si acaso no había ninguno en trainset
                set(mis_probs, which(is.na(mis_probs[[j]])), j, 0) # NAs a cero
              print(paste0(numBatch, '_', numSubBatch, '/' , NUM_BLOQUES, ' - Acumulando datos - (', mivar, ')...'))
              setkeyv(mis_probs, mivar)
              mis_probs[, timestamp_min := min(c(timestamp_min, timestamp_min2), na.rm = T), by = mivar]
              mis_probs[, timestamp_max := max(c(timestamp_max, timestamp_max2), na.rm = T), by = mivar]
              mis_probs[, timestamp_avg := mean(c(tot * timestamp_avg, tot2 * timestamp_avg2), na.rm = T)/sum(c(tot, tot2), na.rm = T), by = mivar]
              mis_probs[, timestamp_var := mean(c(tot * timestamp_var, tot2 * timestamp_var2), na.rm = T)/sum(c(tot, tot2), na.rm = T), by = mivar]
              mis_probs[, timestamp_difmin := min(c(timestamp_difmin, timestamp_difmin2), na.rm = T), by = mivar]
              mis_probs[, timestamp_difmax := max(c(timestamp_difmax, timestamp_difmax2), na.rm = T), by = mivar]
              mis_probs[, timestamp_difavg := mean(c(tot * timestamp_difavg, tot2 * timestamp_difavg2), na.rm = T)/sum(c(tot, tot2), na.rm = T), by = mivar]
              mis_probs[, timestamp_difvar := mean(c(tot * timestamp_difvar, tot2 * timestamp_difvar2), na.rm = T)/sum(c(tot, tot2), na.rm = T), by = mivar]
              # mis_probs[, clicks := clicks + clicks2, by = mivar] # No hace falta (TESTSET!!!)
              mis_probs[, tot := tot + tot2, by = mivar]
              mis_probs[, c("tot2", "clicks2"):= NULL]
              mis_probs[, c("timestamp_min2", "timestamp_max2", "timestamp_avg2", "timestamp_var2"):= NULL]
              mis_probs[, c("timestamp_difmin2", "timestamp_difmax2", "timestamp_difavg2", "timestamp_difvar2"):= NULL]
              lista_mis_probs[[mivar]] <- mis_probs # El merge() hizo una copia, así que hay que devolverlo a la lista
              print(paste0(numBatch, '_', numSubBatch, '/' , NUM_BLOQUES, ' - Ok. - (', mivar, ')...'))
              
              # minutos_pend <- (as.double((proc.time() - systime_ini_2)['elapsed'])/60) * ( (NUM_BLOQUES^2) / ((numBatch-1) * NUM_BLOQUES + numSubBatch)  -  1)
              # if(minutos_pend < 60) print(paste0('Faltan aprox. ',minutos_pend, ' minutos.')) else print(paste0('Faltan aprox. ',minutos_pend/60, ' horas.'))
            }
          }
          # print(paste0(as.double((proc.time() - systime_ini)['elapsed'])/60, ' minutos en total.'))
          minutos_pend <- (as.double((proc.time() - systime_ini_2)['elapsed'])/60) * ( NUM_BLOQUES / numBatch  -  1)
          if(minutos_pend < 60) print(paste0('Faltan aprox. ',minutos_pend, ' minutos.')) else print(paste0('Faltan aprox. ',minutos_pend/60, ' horas.'))
        }
        print(paste0(numBatch, '_', numSubBatch, '/' , NUM_BLOQUES, ' - ', as.double((proc.time() - systime_ini)['elapsed'])/60, ' minutos en total.'))
        # minutos_pend <- (as.double((proc.time() - systime_ini_2)['elapsed'])/60) * ( NUM_BLOQUES / numBatch  -  1)
        # if(minutos_pend < 60) print(paste0('Faltan aprox. ',minutos_pend, ' minutos.')) else print(paste0('Faltan aprox. ',minutos_pend/60, ' horas.'))
      }
      # Guardamos todo:
      for(mivar in c(miscols, ad_miscols))
      {
        mivar_fich <- paste0(mivar, "_tiempos_def.RData")
        if(!file.exists(paste0(s_input_path, mivar_fich)))
        {
          mis_probs <- lista_mis_probs[[mivar]] # Esto no debería hacer una copia porque son data.tables
          # Guardamos:
          print(paste0('Guardando finalmente ', mivar_fich, ' (', nrow(mis_probs), ' registros)...'))
          save(mis_probs, file = paste0(s_input_path, mivar_fich)) # load(file = paste0(s_input_path, mivar_fich))
        }
      }
      rm(testset, lista_mis_probs)
      if(exists("mis_probs")) rm(mis_probs)
      gc()
    }
  }
}
print('11.- ficheros xxxx_tiempos_def.RData - Ok.')
# ------------------------------------------------------------------------------------
# 12.- Insertamos campos de "xxxx_tiempos_def.RData" (y quitamos variables) (para versión 7):
# ------------------------------------------
# Columnas por las que vamos a agrupar (y, de paso, para las que calcularemos su prob -docprob, etc.-):
miscols <- c("uuid", # "clicked", ("display_id" no tiene sentido porque no hay ninguno coincidente entre train y test)
             "document_id", "source_id", "publisher_id", "publish_timestamp" # , "timestamp"
)
ad_miscols <- c("ad_id", # "clicked",
                "ad_document_id", "ad_source_id", "ad_publisher_id", "ad_publish_timestamp" # , "timestamp"
)

# Nuevos campos (por cada variable), que serán renombrados tras cada merge():  
nuevos_campos <- c("tot", "clicks", "prob", "timestamp_min", "timestamp_max", "timestamp_avg", "timestamp_var", "timestamp_difmin", "timestamp_difmax", "timestamp_difavg", "timestamp_difvar")

for(mivar in c(miscols, ad_miscols))
{
  mivar_fich <- paste0(mivar, "_tiempos_def.RData")
  if(!file.exists(paste0(s_input_path, mivar_fich)))
  {
    print(paste0('ERROR: fichero de timestamps ', mivar_fich, ' NO encontrado.'))
    next
  }
  nuevos_campos_new_names <- paste0(mivar, '_', nuevos_campos) # prob => uuid_prob; timestamp_difmin => uuid_timestamp_difmin; etc.
  # Verificamos que los nuevos campos no estén ya dentro de full_trainset y/o testset:
  if(all(nuevos_campos_new_names %in% colnames(full_trainset)))
  {
    print(paste0('Full_trainset - Campo ', mivar, '(', which(c(miscols, ad_miscols) == mivar),'/', length(c(miscols, ad_miscols)),') Ok.'))
    # print(paste0('Full_trainset - Campo ', mivar, ' Ok.'))
    next
  }
  print(paste0('Full_trainset - Campo ', mivar, ' - Leyendo fichero de timestamps "', mivar_fich, '"...'))
  load(file = paste0(s_input_path, mivar_fich))
  
  systime_ini_2 <- proc.time()
  if(!all(nuevos_campos_new_names %in% colnames(full_trainset)))
  {
    for(numBatch in 1:NUM_BLOQUES)
    { 
      full_trainset <- leer_batch_train(numBatch, paste0("Insertar timestamps (", mivar, ")"), s_input_path)
      if(!all(nuevos_campos_new_names %in% colnames(full_trainset)))
      {
        full_trainset[, timestamp := as.numeric(timestamp)] # No habíamos guardado este cambio
        print('Quitamos variables que sobran...')
        if("publish_time" %in% colnames(full_trainset))  full_trainset[, publish_time:=NULL]
        if("ad_publish_time" %in% colnames(full_trainset))  full_trainset[, ad_publish_time:=NULL]
        # NOTA: docprob va a ser recalculado, junto con otras variables más...
        if("docprob" %in% colnames(full_trainset))  full_trainset[, docprob:=NULL]
        if("ad_docprob" %in% colnames(full_trainset))  full_trainset[, ad_docprob:=NULL]
        for(j in nuevos_campos_new_names)
        { if(j %in% colnames(full_trainset))  set(full_trainset, j = j, value = NULL) }
        gc()
        print(paste0('Cruzamos por ', mivar, '...'))
        setkeyv(full_trainset, mivar)
        setkeyv(mis_probs, mivar)
        full_trainset <- merge(full_trainset, mis_probs, by = mivar)
        print('NAs a cero...')
        for(j in nuevos_campos)
          set(full_trainset, which(is.na(full_trainset[[j]])), j, 0) # NAs a cero
        print('Renombramos nuevos campos...')
        setnames(x = full_trainset, old = nuevos_campos, new = nuevos_campos_new_names)
        print(paste0(numBatch, '/' , NUM_BLOQUES, ' - Guardando full_trainset...'))
        save(full_trainset, file = paste0(s_input_path, get_batch_train_filename(numBatch)))

        print(paste0(as.double((proc.time() - systime_ini)['elapsed'])/60, ' minutos en total.'))
        minutos_pend <- (as.double((proc.time() - systime_ini_2)['elapsed'])/60) * (NUM_BLOQUES / numBatch  -  1)
        if(minutos_pend < 60) print(paste0('Faltan aprox. ', minutos_pend, ' minutos.')) else print(paste0('Faltan aprox. ', minutos_pend/60, ' horas.'))
      }
    }
  }
  print(paste0('Full_trainset - Campo ', mivar, '(', which(c(miscols, ad_miscols) == mivar),'/', length(c(miscols, ad_miscols)),') Ok.'))
  # print(paste0('Full_trainset - Campo ', mivar, ' Ok.'))
  rm(mis_probs); gc()
}

for(mivar in c(miscols, ad_miscols))
{
  mivar_fich <- paste0(mivar, "_tiempos_def.RData")
  if(!file.exists(paste0(s_input_path, mivar_fich)))
  {
    print(paste0('ERROR: fichero de timestamps ', mivar_fich, ' NO encontrado.'))
    next
  }
  nuevos_campos_new_names <- paste0(mivar, '_', nuevos_campos) # prob => uuid_prob; timestamp_difmin => uuid_timestamp_difmin; etc.
  # Verificamos que los nuevos campos no estén ya dentro de full_trainset y/o testset:
  if(all(nuevos_campos_new_names %in% colnames(testset)))
  {
    print(paste0('Testset - Campo ', mivar, '(', which(c(miscols, ad_miscols) == mivar),'/', length(c(miscols, ad_miscols)),') Ok.'))
    # print(paste0('Testset - Campo ', mivar, ' Ok.'))
    next
  }
  print(paste0('Testset - Campo ', mivar, ' - Leyendo fichero de timestamps "', mivar_fich, '"...'))
  load(file = paste0(s_input_path, mivar_fich))
  
  systime_ini_2 <- proc.time()
  if(!all(nuevos_campos_new_names %in% colnames(testset)))
  {
    for(numBatch in 1:NUM_BLOQUES)
    {
      for(numSubBatch in 1:NUM_BLOQUES)
      {
        if(!file.exists(paste0(s_input_path, get_batch_test_filename(NUM_BLOQUES, NUM_BLOQUES))))
        {
          # No hay subdivisión de testsets. Lo hacemos con los "grandes":
          if(numSubBatch != 1)  break # Sólo el primero!
          testset <- leer_batch_test(numBatch, paste0("Insertar timestamps (", mivar, ")"), s_input_path)
        } else {
          testset <- leer_batch_test(numBatch, numSubBatch = numSubBatch, s_descr = paste0("Insertar timestamps (", mivar, ")"), s_input_path = s_input_path)
        }
        if(!all(nuevos_campos_new_names %in% colnames(testset)))
        {
          testset[, timestamp := as.numeric(timestamp)] # No habíamos guardado este cambio
          print('Quitamos variables que sobran...')
          if("publish_time" %in% colnames(testset))  testset[, publish_time:=NULL]
          if("ad_publish_time" %in% colnames(testset))  testset[, ad_publish_time:=NULL]
          # NOTA: docprob va a ser recalculado, junto con otras variables más...
          if("docprob" %in% colnames(testset))  testset[, docprob:=NULL]
          if("ad_docprob" %in% colnames(testset))  testset[, ad_docprob:=NULL]
          for(j in nuevos_campos_new_names)
          { if(j %in% colnames(testset))  set(testset, j = j, value = NULL) }
          gc()
          print(paste0('Cruzamos por ', mivar, '...'))
          setkeyv(testset, mivar)
          setkeyv(mis_probs, mivar)
          testset <- merge(testset, mis_probs, by = mivar)
          print('NAs a cero...')
          for(j in nuevos_campos)
            set(testset, which(is.na(testset[[j]])), j, 0) # NAs a cero
          print('Renombramos nuevos campos...')
          setnames(x = testset, old = nuevos_campos, new = nuevos_campos_new_names)
          print(paste0(numBatch, '/' , NUM_BLOQUES, ' - Guardando testset...'))
          save(testset, file = paste0(s_input_path, get_batch_test_filename(numBatch)))
          
          # print(paste0(as.double((proc.time() - systime_ini)['elapsed'])/60, ' minutos en total.'))
          minutos_pend <- (as.double((proc.time() - systime_ini_2)['elapsed'])/60) * ( NUM_BLOQUES / numBatch  -  1)
          if(minutos_pend < 60) print(paste0('Faltan aprox. ',minutos_pend, ' minutos.')) else print(paste0('Faltan aprox. ',minutos_pend/60, ' horas.'))
        }
      }
      print(paste0(numBatch, '_', numSubBatch, '/' , NUM_BLOQUES, ' - ', as.double((proc.time() - systime_ini)['elapsed'])/60, ' minutos en total.'))
      # minutos_pend <- (as.double((proc.time() - systime_ini_2)['elapsed'])/60) * ( NUM_BLOQUES / numBatch  -  1)
      # if(minutos_pend < 60) print(paste0('Faltan aprox. ',minutos_pend, ' minutos.')) else print(paste0('Faltan aprox. ',minutos_pend/60, ' horas.'))
    }
  }
  print(paste0('Testset - Campo ', mivar, '(', which(c(miscols, ad_miscols) == mivar),'/', length(c(miscols, ad_miscols)),') Ok.'))
  # print(paste0('Testset - Campo ', mivar, ' Ok.'))
  rm(mis_probs); gc()
}
print('12.- Insertar campos de xxxx_tiempos_def.RData - Ok.')
# ------------------------------------------
# 13.- Distrib. por numAds:
# ------------------------------------------
if(!b_con_num_modelo & !G_b_DEBUG)
{
  systime_ini_2 <- proc.time()
  for(numBatch in 1:NUM_BLOQUES)
  {
    full_trainset <- leer_batch_train(numBatch, "numAds (y 2)", s_input_path)
    setkeyv(full_trainset, "display_id")
    for(numAdsCluster in 2:12)
    {
      fich_name <- paste0("full_trainset_", str_pad(numAdsCluster, 2, "left" ,"0"), '_', str_pad(numBatch, 3, "left" ,"0"), ".RData")
      if(!file.exists(paste0(s_input_path, fich_name)))
      {
        tmp_full_trainset <- full_trainset
        full_trainset <- full_trainset[, numAds := .N, by = "display_id"][numAds == numAdsCluster,]
        save(full_trainset, file = paste0(s_input_path, fich_name))
        full_trainset <- tmp_full_trainset
      }
    }
    print(paste0("numAds (y 2): ", numBatch, ' - ', as.double((proc.time() - systime_ini)['elapsed'])/60, ' minutos en total.'))
    minutos_pend <- (as.double((proc.time() - systime_ini_2)['elapsed'])/60) * (NUM_BLOQUES / numBatch  -  1)
    if(minutos_pend < 60) print(paste0('Faltan aprox. ', minutos_pend, ' minutos.')) else print(paste0('Faltan aprox. ', minutos_pend/60, ' horas.'))
  }
}
print("13.- Distrib. por numAds - Ok.")
# ------------------------------------------
# 14.1.- page_views.csv - Ok. (uuid, document_id, timestamp, platform, geo_location, traffic_source)
# ------------------------------------------
if(!file.exists(paste0(s_input_path, "publish_timestamp", "_tiempos_pgvw.RData"))) # page_views.csv...
{
  rm(full_trainset, testset); gc() # Liberamos memoria
  # pgvw <- fread(paste0(s_input_path, "page_views_sample.csv"), colClasses = c("character", "integer", "numeric", "integer", "character", "integer"), nrows = ifelse(G_b_DEBUG, 2500L, -1L))
  # # # setkeyv(pgvw, c("uuid", "document_id", "timestamp", "platform", "geo_location", "traffic_source"))
  # # # # timestamp (ms since 1970-01-01 - 1465876799998)
  # # # # geo_location (country>state>DMA)
  # # # # platform (desktop = 1, mobile = 2, tablet =3)
  # # # table(pgvw$platform)
  # # # pgvw[, platform := as.integer(substr(platform,1,1))]
  # # # # traffic_source (internal = 1, search = 2, social = 3)
  # # # table(pgvw$traffic_source)
  # # # pgvw[, traffic_source := as.integer(substr(traffic_source,1,1))]
  # # summary(pgvw)
  # # sapply(pgvw, uniqueN)
  # # stopifnot(max(nchar(pgvw$uuid)) == 14 && min(nchar(pgvw$uuid)) == 14)
  # # pgvw[, user_id1 := as.integer(paste0('0x', substring(uuid, first = 1, last = 7)))]
  # # pgvw[, user_id2 := as.integer(paste0('0x', substring(uuid, first = 8, last = 14)))]
  # pgvw[, platform := as.integer(substr(platform,1,1))]
  # pgvw[, traffic_source := as.integer(substr(traffic_source,1,1))]
  # pgvw[, geo_loc.country := str_split_fixed(geo_location, '>', Inf)[[1]], by = "geo_location"]
  # # pgvw[geo_location!=geo_loc.country, geo_loc.state := str_split_fixed(geo_location, '>', Inf)[[2]], by = "geo_location"]
  # # # pgvw[, geo_loc.DMA := str_split_fixed(geo_location, '>', Inf)[[3]], by = geo_location]
  # # print(sort(table(pgvw$geo_loc.country), decreasing = T)[1:10])
  # # print(sort(table(pgvw$geo_loc.state), decreasing = T)[1:10])
  # # Agrupamos países: US, CA, GB, Resto:
  # # pgvw[!(geo_loc.country %in% c("US", "CA", "GB")), pais := "Resto"]
  # # pgvw[  geo_loc.country %in% c("US", "CA", "GB") , pais := geo_loc.country, by = "geo_loc.country"]
  # pgvw[ geo_loc.country == "US", idpais := 1L, by = "geo_loc.country"]
  # pgvw[ geo_loc.country == "CA", idpais := 2L, by = "geo_loc.country"]
  # pgvw[ geo_loc.country == "GB", idpais := 3L, by = "geo_loc.country"]
  # pgvw[!(geo_loc.country %in% c("US", "CA", "GB")), idpais := 4L, by = "geo_loc.country"]
  # # print(sort(table(pgvw$pais), decreasing = T))
  # # print(sort(table(pgvw$idpais), decreasing = T))
  # # pgvw[, hora := as.integer(1 + (timestamp %/% 3600000) %% 24)] # De 1 a 24
  # # pgvw[, dia  := as.integer(1 + timestamp %/% (3600000 * 24))]  # De 1 a 15
  # # pgvw[, horadia  := as.integer(1 + 24 * (dia-1) + (hora-1))]  # De 1 a 15*24
  # cols <- c("uuid", "document_id", "platform", "traffic_source", "idpais", "timestamp")
  # pgvw <- pgvw[,cols,with=F]
  # # save(pgvw, file = paste0(s_input_path, "page_views_sample2.RData"))
  # # write.table(pgvw, file = paste0(s_input_path, "page_views_sample2.csv"), row.names=F, quote=F, sep=",")
  
  # Cargamos docmeta para añadir "source_id", "publisher_id", "publish_timestamp":
  docmeta <- fread(paste0(s_input_path, "documents_meta.csv"), nrows = ifelse(G_b_DEBUG, 2500L, -1L))
  setkeyv(docmeta, "publish_time")
  docmeta[publish_time == "", publish_time := NA]
  setkeyv(docmeta, "publish_time")
  docmeta[, publish_timestamp := as.numeric(1000) * as.numeric(as.POSIXct(publish_time)) - as.numeric(1465876799998), by = "publish_time"] # (ms since 1970-01-01 - 1465876799998)
  docmeta[, publish_time := NULL]
  setkeyv(docmeta, "document_id")
  
  mi_dif_op <- function(FUN_op, mi_vector, na.rm = T, is.sorted = T)
  {
    if(!is.sorted)  mi_vector <- sort(mi_vector)
    # Devuelve la mínima diferencia entre un elemento y el siguiente:
    if(length(mi_vector) < 2) {
      return(mi_vector - mi_vector) # cero!
    } else {
      return(FUN_op(mi_vector[2:(length(mi_vector))] - mi_vector[1:(length(mi_vector)-1)], na.rm = na.rm))
    }
  }
  # Columnas por las que vamos a agrupar (en page_views):
  miscols <- c("document_id", "source_id", "publisher_id", "publish_timestamp")
  # ad_miscols <- c("ad_document_id", "ad_source_id", "ad_publisher_id", "ad_publish_timestamp")
  lista_pgvw_docs <- list() 
  for(mivar in miscols)
  {
    # Creamos lista de esos data.tables (una para cada variable):
    eval(parse(text = paste0("lista_pgvw_docs[['", mivar, "']] <- data.table(", mivar
                             , " = ", ifelse(mivar == "uuid", "character(0)",ifelse(mivar %in% c("publish_timestamp", "ad_publish_timestamp"),"numeric(0)","integer(0)"))
                             , ", tot = integer(0)"
                             , ", timestamp_min = numeric(0), timestamp_max = numeric(0), timestamp_avg = numeric(0), timestamp_var = numeric(0)"
                             , ", timestamp_difmin = numeric(0), timestamp_difmax = numeric(0), timestamp_difavg = numeric(0), timestamp_difvar = numeric(0)"
                             , ", paisUS = numeric(0), paisCA = numeric(0), paisGB = numeric(0), paisResto = numeric(0)"
                             , ", platform1 = numeric(0), platform2 = numeric(0), platform3 = numeric(0)"
                             , ", trafsrc1 = numeric(0), trafsrc2 = numeric(0), trafsrc3 = numeric(0)"
                             , ")")))
  }
  # Fichero page_views.csv (100 GB!)
  fich <- paste0(s_input_path, "page_views.csv")
  systime_ini_2 <- proc.time()
  # Leemos la cabecera:
  pgvw <- read.table(fich, nrows=15, header=T, fill=TRUE, sep=",", stringsAsFactors = F)
  colNames <- colnames(pgvw)
  # system(paste0("wc -l ", fich)) # 2.034.275.449 líneas! Para contar las líneas del fichero (pero tarda mucho!)
  # Así que hacemos una estimación:
  # uuid    document_id      timestamp       platform   geo_location traffic_source 
  # 14              4              8              1              9              1 
  fichsize_in_lines <- 2034275449L # file.size(fich) / (14+1+7+1+8+1+1+1+9+1+1)
  blqSize <- 2000000L
  NUM_CHUNKS <- 1L + as.integer(fichsize_in_lines / blqSize)
  
  con <- file(description=fich, open="r")
  scan(con, nlines = 1, what = character(0)) # Saltamos la cabecera...
  for(numChunk in 1:NUM_CHUNKS) # bucle para leer page_views a trozos!
  {
    print(paste('Processing rows:', numChunk * blqSize, ' (', format(numChunk, decimal.mark = ',', big.mark = '.'), '/', format(NUM_CHUNKS, decimal.mark = ',', big.mark = '.')
                , ' [', format(100L * numChunk/NUM_CHUNKS, digits = 3, decimal.mark = ',', big.mark = '.'), '%])'))
    # pgvw <- fread(paste0(s_input_path, "page_views_sample.csv"), colClasses = c("character", "integer", "numeric", "integer", "character", "integer"), nrows = ifelse(G_b_DEBUG, 2500L, -1L))
    # # dataChunk <- fread(con, header = F, col.names = colNames, colClasses = c("character", "integer", "numeric", "integer", "character", "character"), nrows = blqSize)
    # dataChunk <- read.table(con, nrows=blqSize, col.names = colNames, skip=0, header=FALSE, fill = TRUE, sep=",", stringsAsFactors = F)
    pgvw <- as.data.table(scan(con, nlines = blqSize, sep = ',', what = list(character(0), integer(0), numeric(0), integer(0), character(0), integer(0))))
    colnames(pgvw) <- colNames
    pgvw[, platform := as.integer(substr(platform,1,1))]
    pgvw[, traffic_source := as.integer(substr(traffic_source,1,1))]
    pgvw[, geo_loc.country := str_split_fixed(geo_location, '>', Inf)[[1]], by = "geo_location"]
    pgvw[ geo_loc.country == "US", idpais := 1L, by = "geo_loc.country"]
    pgvw[ geo_loc.country == "CA", idpais := 2L, by = "geo_loc.country"]
    pgvw[ geo_loc.country == "GB", idpais := 3L, by = "geo_loc.country"]
    pgvw[!(geo_loc.country %in% c("US", "CA", "GB")), idpais := 4L, by = "geo_loc.country"]
    # print(sort(table(pgvw$pais), decreasing = T))
    # print(sort(table(pgvw$idpais), decreasing = T))
    # pgvw[, hora := as.integer(1 + (timestamp %/% 3600000) %% 24)] # De 1 a 24
    # pgvw[, dia  := as.integer(1 + timestamp %/% (3600000 * 24))]  # De 1 a 15
    cols <- c("uuid", "document_id", "platform", "traffic_source", "idpais", "timestamp")
    pgvw <- pgvw[,cols,with=F]
    # Añadimos "source_id", "publisher_id", "publish_timestamp":
    setkeyv(pgvw, "document_id")
    pgvw <- merge(pgvw, docmeta, all.x = T, by = "document_id")
    for(mivar in miscols)
    {
      mivar_fich <- paste0(mivar, "_tiempos_pgvw.RData")
      if(!file.exists(paste0(s_input_path, mivar_fich)))
      {
        print(paste0(Sys.time(), ' - ', numChunk, '/' , NUM_CHUNKS, ' - (', mivar, ') - Creando pgvw_docs [', mivar_fich, ']...'))
        setkeyv(pgvw, c(mivar, "timestamp"))
        pgvw_docs <- lista_pgvw_docs[[mivar]] # Esto no debería hacer una copia porque son data.tables
        pgvw_docs <- merge(pgvw_docs, pgvw[,.(tot2 = .N
                                              , paisUS2 = sum(idpais == 1) # Cantidad de cada pais por mivar
                                              , paisCA2 = sum(idpais == 2) # Cantidad de cada pais por mivar
                                              , paisGB2 = sum(idpais == 3) # Cantidad de cada pais por mivar
                                              , paisResto2 = sum(idpais == 4) # Cantidad de cada pais por mivar
                                              , platform12 = sum(platform == 1) # Cantidad de cada plataforma por mivar
                                              , platform22 = sum(platform == 2) # Cantidad de cada plataforma por mivar
                                              , platform32 = sum(platform == 3) # Cantidad de cada plataforma por mivar
                                              , trafsrc12 = sum(traffic_source == 1) # Cantidad de cada traffic_source por mivar
                                              , trafsrc22 = sum(traffic_source == 2) # Cantidad de cada traffic_source por mivar
                                              , trafsrc32 = sum(traffic_source == 3) # Cantidad de cada traffic_source por mivar
                                              , timestamp_min2 = timestamp[1]
                                              , timestamp_avg2 = mean(timestamp) # Habrá que recalcularla (usando tot2), para no tener que gestionar sumas con números tan grandes
                                              , timestamp_var2 = var(timestamp) # Habrá que recalcularla (usando tot2), para no tener que gestionar sumas con números tan grandes
                                              , timestamp_max2 = timestamp[.N]
                                              , timestamp_difmin2 = mi_dif_op(min,  timestamp) # Mínimo   de diferencias entre timestamps consecutivos # Habrá que recalcularla (usando tot2)
                                              , timestamp_difavg2 = mi_dif_op(mean, timestamp) # Media    de diferencias entre timestamps consecutivos # Habrá que recalcularla (usando tot2)
                                              , timestamp_difvar2 = mi_dif_op(var,  timestamp) # Varianza de diferencias entre timestamps consecutivos # Habrá que recalcularla (usando tot2)
                                              , timestamp_difmax2 = mi_dif_op(max,  timestamp) # Máximo   de diferencias entre timestamps consecutivos # Habrá que recalcularla (usando tot2)
        ), by = mivar], all = TRUE, by = mivar)
        for(j in c("tot", "tot2", "paisUS", "paisCA", "paisGB", "paisResto", "platform1", "platform2", "platform3", "trafsrc1", "trafsrc2", "trafsrc3", "paisUS2", "paisCA2", "paisGB2", "paisResto2", "platform12", "platform22", "platform32", "trafsrc12", "trafsrc22", "trafsrc32"))
          set(pgvw_docs, which(is.na(pgvw_docs[[j]])), j, 0) # NAs a cero
        print(paste0(Sys.time(), ' - ', numChunk, '/' , NUM_CHUNKS, ' - (', mivar, ') - Acumulando datos pgvw_docs...'))
        setkeyv(pgvw_docs, mivar)
        pgvw_docs[, timestamp_min := min(c(timestamp_min, timestamp_min2), na.rm = T), by = mivar]
        pgvw_docs[, timestamp_max := max(c(timestamp_max, timestamp_max2), na.rm = T), by = mivar]
        pgvw_docs[, timestamp_avg := mean(c(tot * timestamp_avg, tot2 * timestamp_avg2), na.rm = T)/sum(c(tot, tot2), na.rm = T), by = mivar]
        pgvw_docs[, timestamp_var := mean(c(tot * timestamp_var, tot2 * timestamp_var2), na.rm = T)/sum(c(tot, tot2), na.rm = T), by = mivar]
        pgvw_docs[, timestamp_difmin := min(c(timestamp_difmin, timestamp_difmin2), na.rm = T), by = mivar]
        pgvw_docs[, timestamp_difmax := max(c(timestamp_difmax, timestamp_difmax2), na.rm = T), by = mivar]
        pgvw_docs[, timestamp_difavg := mean(c(tot * timestamp_difavg, tot2 * timestamp_difavg2), na.rm = T)/sum(c(tot, tot2), na.rm = T), by = mivar]
        pgvw_docs[, timestamp_difvar := mean(c(tot * timestamp_difvar, tot2 * timestamp_difvar2), na.rm = T)/sum(c(tot, tot2), na.rm = T), by = mivar]
        pgvw_docs[, tot := tot + tot2, by = mivar]
        pgvw_docs[, platform1 := platform1 + platform12, by = mivar]
        pgvw_docs[, platform2 := platform2 + platform22, by = mivar]
        pgvw_docs[, platform3 := platform3 + platform32, by = mivar]
        pgvw_docs[, paisUS := paisUS + paisUS2, by = mivar]
        pgvw_docs[, paisCA := paisCA + paisCA2, by = mivar]
        pgvw_docs[, paisGB := paisGB + paisGB2, by = mivar]
        pgvw_docs[, paisResto := paisResto + paisResto2, by = mivar]
        pgvw_docs[, trafsrc1 := trafsrc1 + trafsrc12, by = mivar]
        pgvw_docs[, trafsrc2 := trafsrc2 + trafsrc22, by = mivar]
        pgvw_docs[, trafsrc3 := trafsrc3 + trafsrc32, by = mivar]
        pgvw_docs[, c("tot2", "platform12", "platform22", "platform32", "trafsrc12", "trafsrc22", "trafsrc32") := NULL]
        pgvw_docs[, c("paisUS2", "paisCA2", "paisGB2", "paisResto2"):= NULL]
        pgvw_docs[, c("timestamp_min2", "timestamp_max2", "timestamp_avg2", "timestamp_var2"):= NULL]
        pgvw_docs[, c("timestamp_difmin2", "timestamp_difmax2", "timestamp_difavg2", "timestamp_difvar2"):= NULL]
        lista_pgvw_docs[[mivar]] <- pgvw_docs # El merge() hizo una copia, así que hay que devolverlo a la lista
        if(numChunk %% 10 == 0)
        {
          # Guardamos fichero (temporalmente):
          mivar_fich_tmp <- paste0(mivar, numChunk, "_", NUM_CHUNKS, "_tiempos_pgvw.RData")
          print(paste0('Guardando pgvw_docs (tmp) [', mivar_fich_tmp, '] (', nrow(pgvw_docs), ' registros)...'))
          save(pgvw_docs, file = paste0(s_input_path, mivar_fich_tmp)) # load(file = paste0(s_input_path, mivar_fich_tmp))
          # Borramos el anterior, si existe:
          mivar_fich_tmp <- paste0(mivar, numChunk - 10, "_", NUM_CHUNKS, "_tiempos_pgvw.RData")
          if(file.exists(paste0(s_input_path, mivar_fich_tmp))) try(file.remove(paste0(s_input_path, mivar_fich_tmp)), silent = T)
        }
        print(paste0(Sys.time(), ' - ', numChunk, '/' , NUM_CHUNKS, ' - (', mivar, ') - ', 'Ok. pgvw_docs - (', mivar, ')...'))
        print(paste0(Sys.time(), ' - ', numChunk, '/' , NUM_CHUNKS, ' - (', mivar, ') - ', as.double((proc.time() - systime_ini)['elapsed'])/60, ' minutos en total.'))
      }
    } # for(mivar in miscols)
    minutos_pend <- (as.double((proc.time() - systime_ini_2)['elapsed'])/60) * ( NUM_CHUNKS / numChunk  -  1)
    if(minutos_pend < 60) print(paste0(Sys.time(), ' - ', numChunk, '/' , NUM_CHUNKS, ' - Faltan aprox. ',minutos_pend, ' minutos.')) else print(paste0(Sys.time(), ' - ', numChunk, '/' , NUM_CHUNKS, ' - Faltan aprox. ',minutos_pend/60, ' horas.'))
  } # for(numChunk in 1:NUM_CHUNKS)
  close(con)
  for(mivar in miscols)
  {
    mivar_fich <- paste0(mivar, "_tiempos_pgvw.RData")
    if(!file.exists(paste0(s_input_path, mivar_fich)))
    {
      # Guardamos este fichero antes de meter los campos en trainset/testset:
      pgvw_docs <- lista_pgvw_docs[[mivar]] # Esto no debería hacer una copia porque son data.tables
      pgvw_docs[, hora_min := as.integer(1 + (timestamp_min %/% 3600000) %% 24)] # De 1 a 24
      pgvw_docs[, dia_min  := as.integer(1 + timestamp_min %/% (3600000 * 24))]  # De 1 a 15
      pgvw_docs[, hora_max := as.integer(1 + (timestamp_max %/% 3600000) %% 24)] # De 1 a 24
      pgvw_docs[, dia_max  := as.integer(1 + timestamp_max %/% (3600000 * 24))]  # De 1 a 15
      pgvw_docs[, hora_avg := as.integer(1 + (timestamp_avg %/% 3600000) %% 24)] # De 1 a 24
      pgvw_docs[, dia_avg  := as.integer(1 + timestamp_avg %/% (3600000 * 24))]  # De 1 a 15
      if(anyNA(pgvw_docs[, mivar, with=F]))
        pgvw_docs <- pgvw_docs[!is.na(get(mivar)),] # Quitamos el registro donde pk == NA !!! (si lo hay)
      # Guardamos:
      print(paste0('Guardando pgvw_docs [', mivar_fich, '] (', nrow(pgvw_docs), ' registros)...'))
      save(pgvw_docs, file = paste0(s_input_path, mivar_fich)) # load(file = paste0(s_input_path, mivar_fich))
      # rm(pgvw_docs)
    }
  }
  # summary(pgvw$dia) # NOTA: El fichero sample está todo en un único día (el primero), así que aquí no hay información del día...
  # ( max(pgvw$timestamp) - min(pgvw$timestamp) )%/% (3600000 * 24)
  rm(lista_pgvw_docs, pgvw_docs, pgvw)
  gc()
  # Recargamos full_trainset y testset:
  full_trainset <- leer_batch_train(NUM_BLOQUES, "inicio", s_input_path)
  if( file.exists(paste0(s_input_path, get_batch_test_filename(NUM_BLOQUES, NUM_BLOQUES))))
    testset <- leer_batch_test(NUM_BLOQUES, "inicio", s_input_path, numSubBatch = NUM_BLOQUES)
  if(!file.exists(paste0(s_input_path, get_batch_test_filename(NUM_BLOQUES, NUM_BLOQUES))))
    testset <- leer_batch_test(NUM_BLOQUES, "inicio", s_input_path)
}
print("14.1.- page_views.csv - Ok.")
# ------------------------------------------------------------------------------------
# 14.2.- page_views.csv - Insertamos campos de "xxxx_tiempos_pgvw.RData" (para versión 9):
# ------------------------------------------
if(file.exists(paste0(s_input_path, "publish_timestamp", "_tiempos_pgvw.RData"))) # page_views.csv...
{
  # Columnas por las que vamos a agrupar (en page_views):
  miscols <- c("document_id", "source_id", "publisher_id", "publish_timestamp")
  miscols <- c(miscols, "uuid") # creado en 10.-
  # ad_miscols <- c("ad_document_id", "ad_source_id", "ad_publisher_id", "ad_publish_timestamp")
  
  nuevos_campos_pgvw <- c("tot", "timestamp_min", "timestamp_max", "timestamp_avg", "timestamp_var", "timestamp_difmin", "timestamp_difmax", "timestamp_difavg", "timestamp_difvar")
  nuevos_campos_pgvw <- c(nuevos_campos_pgvw, "paisUS", "paisCA", "paisGB", "paisResto", "platform1", "platform2", "platform3", "trafsrc1", "trafsrc2", "trafsrc3")
  nuevos_campos_pgvw <- c(nuevos_campos_pgvw, "hora_min", "dia_min", "hora_max", "dia_max", "hora_avg", "dia_avg")
  for(mivar in miscols)
  {
    mivar_fich <- paste0(mivar, "_tiempos_pgvw.RData")
    if(!file.exists(paste0(s_input_path, mivar_fich)))
    {
      print(paste0('ERROR: fichero de pgvw timestamps [', mivar_fich, '] NO encontrado.'))
      next
    }
    nuevos_campos_pgvw_new_names <- paste0(mivar, '_pgvw_', nuevos_campos_pgvw) # tot => document_id_pgvw_tot; timestamp_difmin => document_id_pgvw_timestamp_difmin; etc.
    # Verificamos que los nuevos campos no estén ya dentro de full_trainset y/o testset:
    if(all(nuevos_campos_pgvw_new_names %in% colnames(full_trainset)))
    {
      print(paste0('Full_trainset - Campo ', mivar, '(', which(miscols == mivar),'/', length(miscols),') Ok.'))
      # print(paste0('Full_trainset - Campo ', mivar, ' Ok.'))
      next
    }
    if(!all(nuevos_campos_pgvw_new_names %in% colnames(full_trainset)))
    {
      systime_ini_2 <- proc.time()
      print(paste0('Full_trainset - Campo ', mivar, ' - Leyendo fichero pgvw de timestamps [', mivar_fich, ']...'))
      load(file = paste0(s_input_path, mivar_fich)) # pgvw_docs
      
      for(numBatch in 1:NUM_BLOQUES)
      {
        for(numAdsCluster in 2:12)
        {
          if(b_con_num_modelo)
          {
            fich_name <- paste0("full_trainset_", str_pad(numAdsCluster, 2, "left" ,"0"), '_', str_pad(numBatch, 3, "left" ,"0"), ".RData")
            load(file = paste0(s_input_path, fich_name))
            numChunk <- 11 * (numBatch - 1) + numAdsCluster - 1
            NUM_CHUNKS <- 11 * NUM_BLOQUES
          } else {
            if(numAdsCluster != 2)
              break
            fich_name <- get_batch_train_filename(numBatch)
            # No hay "sub_ficheros" con modelos, así que vamos con el full_trainset_batch completo
            full_trainset <- leer_batch_train(numBatch, paste0("Insertar pgvw timestamps (", mivar, ")"), s_input_path)
            numChunk <- numBatch
            NUM_CHUNKS <- NUM_BLOQUES
          }
          for(j in nuevos_campos_pgvw_new_names)
          { if(j %in% colnames(full_trainset))  set(full_trainset, j = j, value = NULL) }

          print(paste0(Sys.time(), ' - ', numChunk, '/' , NUM_CHUNKS, ' - (', mivar, ') - Cruzamos por ', mivar, '...'))
          setkeyv(full_trainset, mivar)
          setkeyv(pgvw_docs, mivar)
          full_trainset <- merge(full_trainset, pgvw_docs, by = mivar, all.x = T)
          print('NAs a cero...')
          for(j in nuevos_campos_pgvw)
            set(full_trainset, which(is.na(full_trainset[[j]])), j, 0) # NAs a cero
          print('Renombramos nuevos campos...')
          setnames(x = full_trainset, old = nuevos_campos_pgvw, new = nuevos_campos_pgvw_new_names)
          
          mivar_ad <- paste0('ad_', mivar)
          if(mivar_ad %in% colnames(full_trainset))
          {
            print(paste0(Sys.time(), ' - ', numChunk, '/' , NUM_CHUNKS, ' - (', mivar, ') - Cruzamos por ', mivar_ad, '...'))
            setkeyv(full_trainset, mivar_ad)
            setkeyv(pgvw_docs, mivar)
            full_trainset <- merge(full_trainset, pgvw_docs, by.x = mivar_ad, by.y = mivar, all.x = T)
            print('NAs a cero...')
            for(j in nuevos_campos_pgvw)
              set(full_trainset, which(is.na(full_trainset[[j]])), j, 0) # NAs a cero
            print('Renombramos nuevos campos...')
            setnames(x = full_trainset, old = nuevos_campos_pgvw, new = paste0('ad_', nuevos_campos_pgvw_new_names))
          }
          print(paste0(Sys.time(), ' - ', numChunk, '/' , NUM_CHUNKS, ' - (', mivar, ') - Guardando full_trainset...'))
          # Guardamos full_trainset actualizado:
          save(full_trainset, file = paste0(s_input_path, fich_name))
          print(paste0(Sys.time(), ' - ', numChunk, '/' , NUM_CHUNKS, ' - (', mivar, ') - ', 'Ok. ', fich_name, ' - (', mivar, ')...'))
          print(paste0(Sys.time(), ' - ', numChunk, '/' , NUM_CHUNKS, ' - (', mivar, ') - ', as.double((proc.time() - systime_ini)['elapsed'])/60, ' minutos en total.'))
          minutos_pend <- (as.double((proc.time() - systime_ini_2)['elapsed'])/60) * ( NUM_CHUNKS / numChunk  -  1)
          if(minutos_pend < 60) print(paste0('Faltan aprox. ',minutos_pend, ' minutos.')) else print(paste0('Faltan aprox. ',minutos_pend/60, ' horas.'))
        }
      }
    }
    if(!all(nuevos_campos_pgvw_new_names %in% colnames(testset)))
    {
      systime_ini_2 <- proc.time()
      for(numBatch in 1:NUM_BLOQUES)
      {
        numChunk <- numBatch
        NUM_CHUNKS <- NUM_BLOQUES
        testset <- leer_batch_test(numBatch, paste0("Insertar pgvw timestamps (", mivar, ")"), s_input_path)
        for(j in nuevos_campos_pgvw_new_names)
        { if(j %in% colnames(testset))  set(testset, j = j, value = NULL) }
        
        print(paste0(Sys.time(), ' - ', numChunk, '/' , NUM_CHUNKS, ' - (', mivar, ') - Cruzamos por ', mivar, '...'))
        setkeyv(testset, mivar)
        setkeyv(pgvw_docs, mivar)
        testset <- merge(testset, pgvw_docs, by = mivar, all.x = T)
        print('NAs a cero...')
        for(j in nuevos_campos_pgvw)
          set(testset, which(is.na(testset[[j]])), j, 0) # NAs a cero
        print('Renombramos nuevos campos...')
        setnames(x = testset, old = nuevos_campos_pgvw, new = nuevos_campos_pgvw_new_names)
        
        mivar_ad <- paste0('ad_', mivar)
        if(mivar_ad %in% colnames(full_trainset))
        {
          print(paste0(Sys.time(), ' - ', numChunk, '/' , NUM_CHUNKS, ' - (', mivar, ') - Cruzamos por ', mivar_ad, '...'))
          setkeyv(testset, mivar_ad)
          setkeyv(pgvw_docs, mivar)
          testset <- merge(testset, pgvw_docs, by.x = mivar_ad, by.y = mivar, all.x = T)
          print('NAs a cero...')
          for(j in nuevos_campos_pgvw)
            set(testset, which(is.na(testset[[j]])), j, 0) # NAs a cero
          print('Renombramos nuevos campos...')
          setnames(x = testset, old = nuevos_campos_pgvw, new = paste0('ad_', nuevos_campos_pgvw_new_names))
        }
        print(paste0(Sys.time(), ' - ', numChunk, '/' , NUM_CHUNKS, ' - (', mivar, ') - Guardando testset...'))
        save(testset, file = paste0(s_input_path, get_batch_test_filename(numBatch)))
        print(paste0(Sys.time(), ' - ', numChunk, '/' , NUM_CHUNKS, ' - (', mivar, ') - ', 'Ok. ', fich_name, ' - (', mivar, ')...'))
        print(paste0(Sys.time(), ' - ', numChunk, '/' , NUM_CHUNKS, ' - (', mivar, ') - ', as.double((proc.time() - systime_ini)['elapsed'])/60, ' minutos en total.'))
        minutos_pend <- (as.double((proc.time() - systime_ini)['elapsed'])/60) * 2 * (NUM_BLOQUES / numBatch  -  1)
        if(minutos_pend < 60) print(paste0('Faltan aprox. ',minutos_pend, ' minutos.')) else print(paste0('Faltan aprox. ',minutos_pend/60, ' horas.'))
      }
    }
  }
  
  # print('events.csv')
  # evts <- fread(paste0(s_input_path, "events.csv"), colClasses = c("integer", "character", "integer", "numeric", "character", "character"), nrows = ifelse(G_b_DEBUG, 2500L, -1L))
  # # # timestamp (ms since 1970-01-01 - 1465876799998)
  # # # platform (desktop = 1, mobile = 2, tablet =3)
  # # # geo_location (country>state>DMA)
  # evts[platform == "\\N"]
  # evts[platform == "\\N", platform := "2"] # Es el más frecuente (y solo hay 5 casos)
  # evts[, platform := as.integer(platform)]
  # evts[, timestamp := as.numeric(timestamp)]
  # # #
  # # # 1.- ¿Cuánto hay que reducir timestamp para cuadrarlos?
  # # # NOTA: Esto NO se puede hacer con "page_views_sample.csv" !!!
  # # #
  # # # timestamp_reducido = timestamp - timestamp % hh
  # # # 
  # # evts_disps_count <- uniqueN(evts$display_id)
  # # setkeyv(evts, c("uuid", "document_id", "platform", "geo_location", "timestamp"))
  # # setkeyv(pgvw, c("uuid", "document_id", "platform", "geo_location", "timestamp"))
  # # mis_evts <- merge(evts, pgvw, by = c("uuid", "document_id")) #, "platform", "geo_location"))
  # # # nrow(mis_evts)
  # # evts_disps_count <- uniqueN(mis_evts$display_id)
  # # setkeyv(mis_evts, c("uuid", "document_id", "platform", "geo_location", "timestamp.x", "timestamp.y"))
  # # for(i in 1:24)
  # # {
  # #   hh <- i * 24* 3600 * 1000
  # #   mis_evts_disps_count <- uniqueN(mis_evts[(timestamp.x - timestamp.x %% hh) == (timestamp.y - timestamp.y %% hh)]$display_id)
  # #   if(mis_evts_disps_count == evts_disps_count)
  # #   {
  # #     print(paste0('Ok. hh = ', hh/1000, ' horas.'))
  # #     break
  # #   }
  # #   print(paste0('hh = ', hh, ' horas. Diff = ', evts_disps_count - mis_evts_disps_count, ' (', round(100 - 100*(mis_evts_disps_count/evts_disps_count), 2), '%)'))
  # # }
  # 
  rm(pgvw_docs)
  gc()
}
print("14.2.- page_views.csv - Insertamos campos de [xxxx_tiempos_pgvw.RData] (para versión 9):")
# ------------------------------------------------------------------------------------
# # Dividir muestra en entrenamiento y validación: NOTA: Usamos display_id para el corte...
# set.seed(1)
# setkey(full_trainset, display_id)
# disps <- unique(full_trainset)[,.(display_id)] # data.table sin duplicados por clave (y de una columna)
# index <- 1:nrow(disps)
# porc_train <- 0.8
# trainindex <- sample(index, trunc(length(index) * porc_train))
# trainset <- disps[trainindex,]
# validset <- disps[-trainindex,]
# setkey(trainset, display_id)
# setkey(validset, display_id)
# trainset <- merge(full_trainset, trainset, by = "display_id")
# validset <- merge(full_trainset, validset, by = "display_id")
# 
# M12 <- foreach(reg = c(0,3,4,5,6), .inorder=TRUE, .combine = c, .packages = c('data.table'),
#                .export = c('Map_12_diario', 'Map_12', 'mapk_nodup_only_first_12', 'add_tiempo')
#                ) %do%
# {
#   return(
#     basic_preds_m12(trainset = trainset, validset = validset, k = 2, reg = reg, b_verbose = 1)
#     ) # ret_val de la función "foreach() %do%" (o foreach( %dopar%))
# }
# print(M12)
# # sort(M12)
# sort(unlist(sapply(M12, function(x) {return(x$V1[2])})))
# sort(unlist(sapply(M12, function(x) {return(x$V2[2])})))
# sort(unlist(sapply(M12, function(x) {return(x$V3[2])})))

# # basic_preds_guardar_submit(trainset = trainset, k = 2, reg = 4) # 0.63678 (0.64474 aquí)

minutos_acum <- as.double((proc.time() - systime_ini)['elapsed'])/60
print(paste0(minutos_acum, ' minutos en total.'))
print('Ok.')

# cleanup:
try(registerDoSEQ(), silent = TRUE) # library(doParallel) [turn parallel processing off and run sequentially again]
try(stopImplicitCluster(), silent = TRUE)
try(stopCluster(cl), silent = TRUE)
