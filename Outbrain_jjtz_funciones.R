# options(echo = FALSE) # ECHO OFF
###############################################################
#           Outbrain Click Prediction - JJTZ 2016
###############################################################
# ##################################################
# ## Funciones útiles:
# ##################################################
try(source("../../funciones_utiles.R", encoding = "UTF-8"), silent=TRUE)
try(source("funciones_utiles.R", encoding = "UTF-8"), silent=TRUE)
# ##################################################
# ## Funciones:
# ##################################################
basic_preds_m12 <- function(trainset = trainset, validset = validset, k = 1, b_restore_key = FALSE, reg = 0, b_verbose = 0) # k=1,2,3; b_verbose=0,1,2
{
  # -----------------------------------------------------------------------------
  # Crear primera predicción con las frecuencias como prob.:
  # -----------------------------------------------------------------------------
  stopifnot(k %in% 1:3, reg >= 0)
  
  if("prob" %in% names(trainset))
  {
    # Frecuencias de ad_id en trainset:
    setkey(trainset, ad_id)
    if(reg == 0)
    { # Original, sin regularización de las probs estimadas:
      probs_ads <- trainset[, .(prob = mean(prob) ), by = ad_id]
    } else
    {
      # Con regularización de las probs estimadas (para no sobre-estimar a los ad_id que aparecen poco):
      probs_ads <- trainset[, .(prob = (sum(prob))/(.N + reg) ), by = ad_id]
    }
  } else
  {
    # Frecuencias de ad_id en trainset:
    setkey(trainset, ad_id)
    if(reg == 0)
    { # Original, sin regularización de las probs estimadas:
      probs_ads <- trainset[, .(prob = mean(clicked) ), by = ad_id]
    } else
    {
      # Con regularización de las probs estimadas (para no sobre-estimar a los ad_id que aparecen poco):
      probs_ads <- trainset[, .(prob = (sum(clicked))/(.N + reg) ), by = ad_id]
    }
    # head(probs_ads)
  }
  
  setkey(validset, ad_id)
  
  if(b_verbose == 2)
  {
    # head(validset)
    # str(validset)
    # summary(validset)
    
    print('Ads de validset en trainset (orig: 0.17  0.83):')
    print(table(unique(validset$ad_id) %in% unique(trainset$ad_id))/uniqueN(validset$ad_id)) # FALSE= 65.350 TRUE=316.035
    print('Ads de trainset en validset (orig: 0.34 0.66):')
    print(table(unique(trainset$ad_id) %in% unique(validset$ad_id))/uniqueN(trainset$ad_id)) # FALSE=162.915 TRUE=316.035
  }
  # Proporción de clicks (global):
  if(reg == 0)
  { # Original, sin regularización de las probs estimadas:
    if(k == 1) prob_click_global_ads_en_test <- mean(trainset$clicked[trainset$ad_id %in% validset$ad_id]) # 1 [0.61733]
    if(k == 2) prob_click_global_ads_no_en_test <- mean(trainset$clicked[!(trainset$ad_id %in% validset$ad_id)]) # 2 [0.63551]
    if(k == 3) prob_click_global <- mean(trainset$clicked) # 3 [0.63529]
  } else
  { # Con regularización de las probs estimadas (para no sobre-estimar a los ad_id que aparecen poco):
    mean_reg <- function(vv, reg)  return( (sum(vv)) / (length(vv) + reg) )
    if(k == 1) prob_click_global_ads_en_test <- mean_reg(trainset$clicked[trainset$ad_id %in% validset$ad_id], reg)
    if(k == 2) prob_click_global_ads_no_en_test <- mean_reg(trainset$clicked[!(trainset$ad_id %in% validset$ad_id)], reg)
    if(k == 3) prob_click_global <- mean_reg(trainset$clicked, reg)
  }
  
  if(k == 1) prob_na <- prob_click_global_ads_en_test
  if(k == 2) prob_na <- prob_click_global_ads_no_en_test
  if(k == 3) prob_na <- prob_click_global
  # # Free memory:
  # rm(trainset)
  # gc() # Garbage collector
  
  # Añadir las probs de los ads:
  if("prob" %in% names(validset))  validset$prob <- NULL # Por si acaso...
  validset <- merge(validset, probs_ads, all.x = T, by = "ad_id") # by ad_id
  
  # Predecir las probs de los ads que no están (i.e. los NAs) con "prob_na":
  # validset[is.na(validset$prob), prob := prob_na]
  ## Ligeramente más rápido:
  mi_j <- which(names(validset)=="prob") # Nº de la columna "prob"
  set(validset, which(is.na(validset$prob)), mi_j, prob_na)
  # summary(validset)
  mi_tiempo <- system.time({
    retVal <- Map_12_diario(tr_valid = validset, b_restore_key = b_restore_key, b_verbose = b_verbose)
    retVal <- list(retVal) # Para poder ponerle un nombre al data.frame, lo metemos en una lista (de 1 elemento)
    names(retVal) <- paste0("k", k, "_Reg", reg)
  })
  if(b_verbose >= 1) print(paste0('Tiempo total Map12: ', mi_tiempo['elapsed']))
  if(b_verbose >= 1) print(retVal)
  return(retVal)
}

Map_12_diario <- function(tr_valid, b_restore_key = FALSE, b_verbose = 0, dias_list = list(c(1:11), c(12,13), c(1:13))) # dias_list = list(c(1:11), c(12,13), c(1:13))) # b_verbose=0,1,2
{
  if(!("dia" %in% names(tr_valid)))
    return(Map_12(tr_valid = tr_valid, b_restore_key = b_restore_key, b_verbose = b_verbose))
  Map_12_diario <- foreach(vector_dias = dias_list, .inorder = TRUE, # El orden es fundamental aquí!!!
                           .packages = c('data.table'), .export = c('Map_12_diario', 'Map_12', 'mapk_nodup_only_first_12', 'add_tiempo')) %do%
 {
   if('scaled:scale' %in% names(attributes(tr_valid$dia)))
   {
     # attr(validset$dia,'scaled:scale') * validset$dia + attr(validset$dia,'scaled:center')
     return(Map_12(tr_valid = tr_valid[(attr(dia,'scaled:scale') * dia + attr(dia,'scaled:center')) %in% vector_dias]
                   , b_restore_key = b_restore_key, b_verbose = b_verbose))
   } else
   {
    return(Map_12(tr_valid = tr_valid[dia %in% vector_dias]
                  , b_restore_key = b_restore_key, b_verbose = b_verbose))
   }
 }
  Map_12_diario <- rbind(as.data.frame(t(dias_list), row.names = "dias"),
                         as.data.frame(t(Map_12_diario), row.names = "map12"))
  # if(b_verbose >= 1)
  # {
  #   print("Map_12_diario:")
  #   print(Map_12_diario)
  # }
  return(Map_12_diario)
}

Map_12 <- function(tr_valid, b_restore_key = FALSE, b_verbose = 0) # b_verbose=0,1,2
{
  stopifnot(all(c("display_id", "ad_id", "clicked", "prob") %in% names(tr_valid)))
  if(nrow(tr_valid) == 0)  return(0)
  # Este data.table es el tr_valid (que es el subconjunto de trainset que usamos para validar),
  #  - ad_id
  #  - display_id
  #  - clicked es 0 ó 1 (es la variable target)
  #  - prob es la probabilidad de click, predicha
  # Evaluamos (Mean Average Precision MAP@12):
  old.key <- key(tr_valid)
  
  # if(b_verbose == 2)  print('Calculando predicted_lst (prob >= 0) (slower)...')
  # mi_tiempo <- system.time({
  #   # Para conseguir los Ads de mayor "prob" a menor "prob", ponemos "prob" en la key y usamos rev():
  #   setkey(tr_valid, display_id, prob)
  #   predicted_lst <- tr_valid[, .(ad_id.v = list(rev(ad_id))), by = display_id]$ad_id.v
  # })
  # if(b_verbose == 2)  print(mi_tiempo['elapsed'])
  
  if(b_verbose == 2)  print('Calculando predicted_lst (prob >= 0) (faster)...')
  mi_tiempo <- system.time({
    # Para conseguir los Ads de mayor "prob" a menor "prob", usamos setorderv():
    setkey(tr_valid, NULL) # Por si acaso
    setorderv(x = tr_valid, cols = c("display_id","prob"), order = c(1,-1)) #Sort by display_id, -prob
    predicted_lst <- tr_valid[, .(ad_id.v = list(ad_id)), by = display_id]$ad_id.v
  })
  if(b_verbose == 2)  print(mi_tiempo['elapsed'])
  
  # if(b_verbose == 2)  print('Calculando actual_lst (clicked == 1) (slower)...')
  # mi_tiempo <- system.time({
  #   # # Para conseguir los Ads de mayor "clicked" a menor "clicked", ponemos "clicked" en la key y usamos rev():
  #   # if(!all(key(tr_valid) == c("display_id", "clicked")))
  #   #   setkey(tr_valid, display_id, clicked)
  #   # actual_lst <- tr_valid[, .(ad_id.v = list(rev(ad_id))), by = display_id]$ad_id.v
  #   setkey(tr_valid, display_id, clicked)
  #   actual_lst <- tr_valid[clicked == 1, .(ad_id.v = list(ad_id)), by = display_id]$ad_id.v
  # })
  # if(b_verbose == 2)  print(mi_tiempo['elapsed'])
  
  if(b_verbose == 2)  print('Calculando actual_vec (clicked == 1) (faster)...')
  setkey(tr_valid, display_id, prob)
  mi_tiempo <- system.time({
    setkey(tr_valid, display_id, clicked)
    actual_vec <- tr_valid[clicked == 1, ad_id, by = display_id]$ad_id
  })
  if(b_verbose == 2)  print(mi_tiempo['elapsed'])
  
  if(b_restore_key)
  {
    if(b_verbose == 2)  print('Restaurando key inicial...')
    mi_tiempo <- system.time({
      if(!all(key(tr_valid) == old.key))
        setkeyv(tr_valid, old.key)
    })
    if(b_verbose == 2)  print(mi_tiempo['elapsed'])
  }
  
  # if(b_verbose == 2)  print('Calculando Map_12 (slower)...')
  # mi_tiempo <- system.time({
  #   MAP.12 <- mapk_nodup(k = 12, actual_list = actual_lst, predicted_list = predicted_lst)
  # })
  # if(b_verbose == 2)  print(mi_tiempo['elapsed'])
  # if(b_verbose == 2)  print(paste0('MAP_12 = ', MAP.12))
  
  if(b_verbose == 2)  print('Calculando Map_12 (faster)...')
  mi_len <- length(actual_vec)
  mi_tiempo <- system.time({
    MAP.12 <- mapk_nodup_only_first_12(actual_vector = actual_vec, predicted_list = predicted_lst,
                                       i_max = mi_len, j_max = mi_len)
  })
  if(b_verbose == 2)  print(mi_tiempo['elapsed'])
  if(b_verbose >= 1)  print(paste0('MAP_12 = ', MAP.12))
  return(MAP.12)
}

basic_preds_guardar_submit <- function(trainset, k, reg = 0, i_sFileSuffix = "", s_output_path = "") # k=1,2,3
{
  # -----------------------------------------------------------------------------
  # Crear primera predicción con las frecuencias como prob.:
  # -----------------------------------------------------------------------------
  stopifnot(k %in% 1:3, reg >= 0)
  
  # Frecuencias de ad_id en trainset:
  setkey(trainset, ad_id)
  if("prob" %in% names(trainset))
  {
    # Frecuencias de ad_id en trainset:
    setkey(trainset, ad_id)
    if(reg == 0)
    { # Original, sin regularización de las probs estimadas:
      probs_ads <- trainset[, .(prob = mean(prob) ), by = ad_id]
    } else
    {
      # Con regularización de las probs estimadas (para no sobre-estimar a los ad_id que aparecen poco):
      probs_ads <- trainset[, .(prob = (sum(prob))/(.N + reg) ), by = ad_id]
    }
  } else
  {
    if(reg == 0)
    { # Original, sin regularización de las probs estimadas:
      probs_ads <- trainset[, .(prob = mean(clicked) ), by = ad_id]
    } else
    { # Con regularización de las probs estimadas (para no sobre-estimar a los ad_id que aparecen poco):
      probs_ads <- trainset[, .(prob = (sum(clicked))/(.N + reg) ), by = ad_id]
    }
    # head(probs_ads)
  }
  
  # ------------------
  # 2.- Leer testset:
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
  if(reg == 0)
  { # Original, sin regularización de las probs estimadas:
    if(k == 1) prob_click_global_ads_en_test <- mean(trainset$clicked[trainset$ad_id %in% testset$ad_id]) # 1 [0.61733]
    if(k == 2) prob_click_global_ads_no_en_test <- mean(trainset$clicked[!(trainset$ad_id %in% testset$ad_id)]) # 2 [0.63551]
    if(k == 3) prob_click_global <- mean(trainset$clicked) # 3 [0.63529]
  } else
  { # Con regularización de las probs estimadas (para no sobre-estimar a los ad_id que aparecen poco):
    mean_reg <- function(vv, reg)  return( (sum(vv)) / (length(vv) + reg) )
    if(k == 1) prob_click_global_ads_en_test <- mean_reg(trainset$clicked[trainset$ad_id %in% testset$ad_id], reg)
    if(k == 2) prob_click_global_ads_no_en_test <- mean_reg(trainset$clicked[!(trainset$ad_id %in% testset$ad_id)], reg)
    if(k == 3) prob_click_global <- mean_reg(trainset$clicked, reg)
  }
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
  
  guardar_submit(testset = testset, fichero = paste0(s_output_path, "submitset_k", k, "_Reg", reg, i_sFileSuffix,".csv"))
}

guardar_submit <- function(testset, fichero = "submit.csv", b_restore_key = FALSE, b_write_file_append = FALSE, b_verbose = 1, s_output_path = "") # b_verbose=0,1,2
{
  if(!exists("G_b_DEBUG"))  G_b_DEBUG <- FALSE
  # ----------------------------------------------
  # PREPARACIÓN DEL SUBMIT:
  # ----------------------------------------------
  # NOTA: length(submitset) == length(unique(testset$display_id)) == 6.245.533 rows (excepto si b_write_file_append == TRUE, claro...)
  # Para conseguir los Ads de mayor "prob" a menor "prob", ponemos "prob" en la key y usamos rev():
  if(b_verbose == 2)  print('Calculando submitset...')
  mi_tiempo <- system.time({
    if(b_restore_key)  old.key <- key(testset)
    # Para conseguir los Ads de mayor "prob" a menor "prob", usamos setorderv():
    setkey(testset, NULL) # Por si acaso
    setorderv(x = testset, cols = c("display_id","prob"), order = c(1,-1)) #Sort by display_id, -prob
    submitset <- testset[, .(ad_id = paste(ad_id, collapse=" ")), by = display_id]
    if(b_restore_key)
      setkeyv(testset, old.key)
  })
  if(b_verbose == 2)  print(mi_tiempo['elapsed'])
  # head(submitset)
  
  if(G_b_DEBUG)  fichero <- paste0('Debug_', fichero)
  
  if(b_verbose == 2)  print(paste0('Guardamos fichero ', paste0(s_output_path, fichero), '...'))
  mi_tiempo <- system.time({
    # Ordenamos por display_id:
    setkey(submitset, display_id)
    suppressWarnings(
      write.table(submitset, file = paste0(s_output_path, fichero), append = b_write_file_append, row.names=F, col.names=!b_write_file_append, quote=F, sep=",")
    )
  })
  if(b_verbose == 2)  print(mi_tiempo['elapsed'])
  if(b_verbose >= 1)  print(paste0('Ok. Fichero (', paste0(s_output_path, fichero), ') guardado.'))
}
# ----------------------------------------------------------------
# incluir_docs_conf_tipos: Para añadir documents_topics.csv, documents_entities.csv y documents_categories.csv
# ----------------------------------------------------------------
incluir_docs_conf_tipos <- function(doctipos_file = 'documents_entities',
                                    tipo_id = 'entity_id',
                                    from_name = 'entity_prob',
                                    to_name = 'entities_prob',
                                    full_trainset, testset, s_input_path, s_output_path) # s_output_path para doc_probs.RData
{
  systime_ini2 <- proc.time()
  ad_to_name <- paste0('ad_', to_name)   # ad_entities_prob
  to_name2 <- paste0(from_name, '_1')    # entity_prob_1
  ad_to_name2 <- paste0('ad_', to_name2) # ad_entity_prob_1
  if(!(to_name    %in% colnames(full_trainset) & to_name    %in% colnames(testset) &
       ad_to_name %in% colnames(full_trainset) & ad_to_name %in% colnames(testset) &
       to_name2    %in% colnames(full_trainset) & to_name2    %in% colnames(testset) &
       ad_to_name2 %in% colnames(full_trainset) & ad_to_name2 %in% colnames(testset)))
  {
    print(paste0(doctipos_file, '.csv'))
    if(file.exists(file.path(s_input_path, paste0(doctipos_file, '3.RData'))))
    {
      print(paste0('Leyendo ', doctipos_file, '3.RData'))
      load(file.path(s_input_path, paste0(doctipos_file, '3.RData')))
    } else
    {
      if(file.exists(file.path(s_input_path, paste0(doctipos_file, '2.RData'))))
      {
        print(paste0('Leyendo ', doctipos_file, '2.RData'))
        load(file.path(s_input_path, paste0(doctipos_file, '2.RData')))
      } else
      {
        print(paste0('Leyendo ', doctipos_file, '.csv'))
        doctipos <- fread(file.path(s_input_path, paste0(doctipos_file, '.csv')))
        # setkeyv(doctipos_file, c('document_id', tipo_id))
        # confidence_level = Outbrain's confidence in each respective relationship
        print(sapply(doctipos, uniqueN))
        # head(doctipos, 25)
        
        #
        # A cada tipo_id le asignamos su probabilidad para no hacer One Hot Encoding (son uniqueN(doctipos[,tipo_id, with=F],by="tipo_id") == 300 topics, 1326009 entities y 97 categories!!):
        #
        print('Leemos las -probabilidades- de cada document_id (los de display_id) y ad_document_id (los de ad_id):')
        if(!exists("all_misdocs"))  load(file.path(s_output_path, "doc_probs.RData"))
        # Y las añadimos a doctipos: (Cuidado con los NAs, que hay docs en train/test que NO están en doctipos)
        setkey(all_misdocs, document_id)
        setkey(doctipos, document_id)
        # summary(all_misdocs)
        doctipos <- merge(doctipos, all_misdocs[,.(document_id, docprob, ad_docprob)], all.x = TRUE, by = "document_id")
        # sapply(doctipos, anyNA)
        # Y ponemos a cero las probabilidades de los NAs:
        for(j in colnames(doctipos))  set(doctipos, which(is.na(doctipos[[j]])), j, 0) # NAs a cero
        print('Calculamos la prob de cada tipo con el confidence_level y la tipo_prob final (MEDIA ponderada de las probs en cada documento en que aparece cada uno):')
        setkeyv(doctipos, tipo_id)
        doctipos[, tipo_prob    := sum(docprob    * confidence_level)/(.N), by = tipo_id]
        doctipos[, ad_tipo_prob := sum(ad_docprob * confidence_level)/(.N), by = tipo_id]
        # summary(doctipos$ad_tipo_prob)
        print('Calculamos la tipos_prob final de cada documento (SUMA ponderada de las probs de cada uno de sus tipos):')
        setkey(doctipos, document_id)
        doctipos[, tipos_prob    := sum(tipo_prob    * confidence_level), by = "document_id"]
        doctipos[, ad_tipos_prob := sum(ad_tipo_prob * confidence_level), by = "document_id"]
        # summary(doctipos)
        print(paste0('Guardamos ', doctipos_file, '2.RData', ' (doctipos) con sus documentos y sus tipos (y sus probs):'))
        save(doctipos, file = file.path(s_input_path, paste0(doctipos_file, '2.RData'))) # Definitivo (si todo ha ido bien, tendrá 9 variables por document_id)
        rm(all_misdocs); gc()
      }

      # RECUPERAMOS tipo_prob_1, ad_tipo_prob1, etc. de la versión 200...
      # NOTA: Como funcionó bien con 8 y como ya están los demás "incluidos" en tipos_prob y ad_tipos_prob,
      #       nos quedaremos con 8 siempre (los 8 mejores según su confidence_level, claro).
      #
      # setkey(doctipos, document_id)
      # doctipos_bydoc <- doctipos[, .(max_tipos = max(.N)), by = "document_id"]
      # print(summary(doctipos_bydoc$max_tipos))
      # hist(doctipos_bydoc$max_tipos)
      # # K = 8? Miramos en Full_trainset y en testset:
      # setkey(full_trainset, document_id)
      # pp <- merge(full_trainset[,.(display_id, ad_id, clicked, document_id, ad_document_id)], doctipos_bydoc, by = "document_id")
      # pp2 <- pp[, .(max_tipos = max(max_tipos)), by = "document_id"]
      # summary(pp2$max_tipos);  hist(pp2$max_tipos)
      # summary(pp$max_tipos);   hist(pp$max_tipos)
      # setkey(testset, document_id)
      # pp <- merge(testset[,.(display_id, ad_id, document_id, ad_document_id)], doctipos_bydoc, by = "document_id")
      # pp2 <- pp[, .(max_tipos = max(max_tipos)), by = "document_id"]
      # summary(pp2$max_tipos);  hist(pp2$max_tipos)
      # summary(pp$max_tipos);   hist(pp$max_tipos)
      # setkey(full_trainset, ad_document_id)
      # doctipos_bydoc[, ad_document_id := document_id]
      # setkey(doctipos_bydoc, ad_document_id)
      # pp <- merge(full_trainset[,.(display_id, ad_id, clicked, document_id, ad_document_id)], doctipos_bydoc, by = "ad_document_id")
      # pp2 <- pp[, .(max_tipos = max(max_tipos)), by = "ad_document_id"]
      # summary(pp2$max_tipos);  hist(pp2$max_tipos)
      # summary(pp$max_tipos);   hist(pp$max_tipos)
      # setkey(testset, ad_document_id)
      # pp <- merge(testset[,.(display_id, ad_id, document_id, ad_document_id)], doctipos_bydoc, by = "ad_document_id")
      # pp2 <- pp[, .(max_tipos = max(max_tipos)), by = "ad_document_id"]
      # summary(pp2$max_tipos);  hist(pp2$max_tipos)
      # summary(pp$max_tipos);   hist(pp$max_tipos)
      # rm(doctipos_bydoc, doctipos, pp, pp2); gc()
  
      # Finalmente, ponemos a cada document_id sus K = 10 tipos más fiables (y sus probabilidades) para luego incluirlas en train/test:
      # NOTA: Hay  2 categories (o menos) en cada doc en el 100% de full_trainset y de testset
      # NOTA: Hay 10 entities   (o menos) en cada doc en el 100% de full_trainset y de testset
      # NOTA: Hay  8 topics     (o menos) en cada doc en el  75% de full_trainset y de testset
      i <- 1
      tipo <- substring(tipo_id, 1, nchar(tipo_id) - 3)
      str_campos <- paste0(tipo_id, ' = ', tipo_id, ", confidence_level = confidence_level, tipo_prob = tipo_prob, ad_tipo_prob = ad_tipo_prob")
      str_campos <- paste0(str_campos, ", docprob = docprob, ad_docprob = ad_docprob, tipos_prob = tipos_prob, ad_tipos_prob = ad_tipos_prob")
      str_campos <- paste0(str_campos, ", ", tipo_id,i," = ",tipo_id,"[",i,"], ",tipo,"_prob_",i," = tipo_prob[",i,"], ad_",tipo,"_prob_",i," = ad_tipo_prob[",i,"]")
      if(doctipos_file == 'documents_categories')  col_max <- 2  else  col_max <- 10 # Sólo hay como máx. dos categorías por doc!
      for(i in 2:col_max)
        str_campos <- paste0(str_campos, ", ", tipo_id,i," = ",tipo_id,"[",i,"], ",tipo,"_prob_",i," = tipo_prob[",i,"], ad_",tipo,"_prob_",i," = ad_tipo_prob[",i,"]")
      str_campos <- paste0("doctipos10 <- doctipos[, .(", str_campos, "), by = 'document_id']")
      setkey(doctipos, NULL) # Por si acaso
      setorderv(x = doctipos, cols = c("document_id","confidence_level"), order = c(1,-1)) #Sort by document_id, -confidence_level
      # doctipos10 <- doctipos[, .(category_id = category_id, confidence_level = confidence_level, docprob = docprob, ad_docprob = ad_docprob, tipos_prob = tipos_prob, ad_tipos_prob = ad_tipos_prob,
      #                           tipo_id1 = tipo_id[1], tipo_prob_1 = tipo_prob[1], ad_tipo_prob_1 = ad_tipo_prob[1],
      #                           tipo_id2 = tipo_id[2], tipo_prob_2 = tipo_prob[2], ad_tipo_prob_2 = ad_tipo_prob[2],
      #                           tipo_id3 = tipo_id[3], tipo_prob_3 = tipo_prob[3], ad_tipo_prob_3 = ad_tipo_prob[3],
      #                           tipo_id4 = tipo_id[4], tipo_prob_4 = tipo_prob[4], ad_tipo_prob_4 = ad_tipo_prob[4],
      #                           tipo_id5 = tipo_id[5], tipo_prob_5 = tipo_prob[5], ad_tipo_prob_5 = ad_tipo_prob[5],
      #                           tipo_id6 = tipo_id[6], tipo_prob_6 = tipo_prob[6], ad_tipo_prob_6 = ad_tipo_prob[6],
      #                           tipo_id7 = tipo_id[7], tipo_prob_7 = tipo_prob[7], ad_tipo_prob_7 = ad_tipo_prob[7],
      #                           tipo_id8 = tipo_id[8], tipo_prob_8 = tipo_prob[8], ad_tipo_prob_8 = ad_tipo_prob[8]
      #                           ), by = "document_id"]
      eval(parse(text = str_campos))
      # head(doctipos); head(doctipos10)
      # summary(doctipos10)
      doctipos <- doctipos10; rm(doctipos10); gc()
      # for(j in colnames(doctipos))  set(doctipos, which(is.na(doctipos[[j]])), j, 0) # NAs a cero
      save(doctipos, file = file.path(s_input_path, paste0(doctipos_file, '3.RData'))) # Definitivo (si todo ha ido bien, tendrá 9 + 10*3 = 39 variables por document_id, excepto doc_categories que tendrá 9 + 2*3 = 15)
      if(file.exists(file = file.path(s_input_path, paste0(doctipos_file, '2.RData'))))
        file.remove(file = file.path(s_input_path, paste0(doctipos_file, '2.RData')))
      if(file.exists(file = file.path(s_input_path, paste0(doctipos_file, '.RData'))))
        file.remove(file = file.path(s_input_path, paste0(doctipos_file, '.RData')))
    }

    print(paste0('Hay ', uniqueN(doctipos[, tipo_id, with=F], by="tipo_id"), ' ', tipo_id, ' distintos. Actualizando full_trainset y testset...'))
    
    bor_campos <- colnames(doctipos)[grep(pattern = tipo_id, x = colnames(doctipos))]
    bor_campos <- c(bor_campos, "confidence_level", "tipo_prob", "ad_tipo_prob")
    print('Quitamos columnas de doctipos -confidence_level, tipo_prob, ad_tipo_prob, etc.-...')
    for(j in bor_campos)  set(doctipos, j = j, value = NULL)
    gc()
    print('Nos quedamos con los unique(doc)...')
    setkey(doctipos, document_id) # Por si acaso
    doctipos <- unique(doctipos, by = "document_id") # El unique de un data.table se hace por key (document_id)

    # Preparamos los dos data.tables para los dos joins (doc... y ad_doc...):
    setnames(doctipos, old = c('tipos_prob', 'ad_tipos_prob'), new = c(to_name, ad_to_name))
    doctipos[, ad_document_id := document_id]
    ad_doctipos <- doctipos[, colnames(doctipos)[grep("ad_", colnames(doctipos), invert = FALSE)], with=FALSE]
    doctipos <-    doctipos[, colnames(doctipos)[grep("ad_", colnames(doctipos), invert = TRUE)] , with=FALSE]
    # sapply(doctipos, function(x){return(mean(as.numeric(x), na.rm = T))})
    # sapply(ad_doctipos, function(x){return(mean(as.numeric(x), na.rm = T))})
    str_campos <- colnames(doctipos)[!(colnames(doctipos) %in% c("document_id", "ad_document_id"))]
    str_campos <- c(str_campos, colnames(ad_doctipos)[!(colnames(ad_doctipos) %in% c("document_id", "ad_document_id"))])
    # Ahora en doctipos hay un registro por document_id y los siguientes campos:
    # document_id, docprob, tipos_prob, [entity_prob_1, entity_prob_2, etc.]
    # Ahora en ad_doctipos hay un registro por ad_document_id y los siguientes campos:
    # ad_document_id, ad_docprob, ad_tipos_prob, [ad_entity_prob_1, ad_entity_prob_2, etc.]
    setkey(doctipos,    document_id)
    setkey(ad_doctipos, ad_document_id)
    
    # Finalmente, metemos los datos de los tipos de los documents (by display_id-document_id) en trainset y en testset:
    
    if(!(to_name    %in% colnames(full_trainset) & ad_to_name %in% colnames(full_trainset) & 
         to_name2    %in% colnames(full_trainset) & ad_to_name2 %in% colnames(full_trainset)))
    {
      for(numBatch in 1:NUM_BLOQUES)
      {
        rm(full_trainset); gc()
        full_trainset <- leer_batch_train(numBatch, paste0(doctipos_file, '.csv - ', to_name, ' y ', ad_to_name), s_input_path)
        # Aseguramos que no haya columnas repetidas, quitándolas primero:
        if(any(colnames(full_trainset) %in% str_campos))
        {
          mis_cols <- colnames(full_trainset)[colnames(full_trainset) %in% str_campos]
          print(paste0('Quitamos columnas -', paste0(mis_cols, collapse = ","), '-'))
          for(j in mis_cols)  set(full_trainset, j = j, value = NULL)
          gc()
        }
        print('Ponemos en cada document_id la suma ponderada por confidence_level de las probs de todos sus tipos_ids (i.e. tipos_prob), además de las probs de los hasta 10 tipo_id más fiables (por confidence_level):')
        setkey(full_trainset, document_id)
        full_trainset <- merge(full_trainset, doctipos, all.x = TRUE, by = "document_id")
        gc()
        print('Ponemos en cada ad_document_id la suma ponderada por confidence_level de las probs de todos sus ad_tipos_ids (i.e. ad_tipos_prob), además de las probs de los hasta 10 ad_tipo_id más fiables (por confidence_level)')
        setkey(full_trainset, ad_document_id)
        full_trainset <- merge(full_trainset, ad_doctipos, all.x = TRUE, by = "ad_document_id")
        gc()
        print(paste0('Ponemos a cero las probabilidades de los NAs:'))
        for(j in str_campos) set(full_trainset, which(is.na(full_trainset[,j,with=F])), j, 0)
        # summary(full_trainset[,str_campos,with=F])
        print(paste0('Guardando ', get_batch_train_filename(numBatch), '...'))
        save(full_trainset, file = paste0(s_input_path, get_batch_train_filename(numBatch)))
        print(paste0(as.double((proc.time() - systime_ini2)['elapsed'])/60, ' minutos en total (', doctipos_file,').'))
        minutos_pend <- (as.double((proc.time() - systime_ini2)['elapsed'])/60) * 2 * (NUM_BLOQUES / numBatch  -  1)
        if(minutos_pend < 60) print(paste0('Faltan aprox. ',minutos_pend, ' minutos.')) else print(paste0('Faltan aprox. ',minutos_pend/60, ' horas.'))
      }
    }
    if(!(to_name    %in% colnames(testset) & ad_to_name %in% colnames(testset) & 
         to_name2    %in% colnames(testset) & ad_to_name2 %in% colnames(testset)))
    {
      for(numBatch in 1:NUM_BLOQUES)
      {
        rm(testset); gc()
        testset <- leer_batch_test(numBatch, paste0(doctipos_file, '.csv - ', to_name, ' y ', ad_to_name), s_input_path)
        # Aseguramos que no haya columnas repetidas, quitándolas primero:
        if(any(colnames(testset) %in% str_campos))
        {
          mis_cols <- colnames(testset)[colnames(testset) %in% str_campos]
          print(paste0('Quitamos columnas -', paste0(mis_cols, collapse = ","), '-'))
          for(j in mis_cols)  set(testset, j = j, value = NULL)
          gc()
        }
        print('Ponemos en cada document_id la suma ponderada por confidence_level de las probs de todos sus tipos_ids (i.e. tipos_prob), además de las probs de los hasta 10 tipo_id más fiables (por confidence_level):')
        setkey(testset, document_id)
        testset <- merge(testset, doctipos, all.x = TRUE, by = "document_id")
        gc()
        print('Ponemos en cada ad_document_id la suma ponderada por confidence_level de las probs de todos sus ad_tipos_ids (i.e. ad_tipos_prob), además de las probs de los hasta 10 ad_tipo_id más fiables (por confidence_level)')
        setkey(testset, ad_document_id)
        testset <- merge(testset, ad_doctipos, all.x = TRUE, by = "ad_document_id")
        gc()
        print(paste0('Ponemos a cero las probabilidades de los NAs:'))
        for(j in str_campos) set(testset, which(is.na(testset[,j,with=F])), j, 0)
        # summary(testset[,str_campos,with=F])
        print(paste0('Guardando ', get_batch_test_filename(numBatch), '...'))
        save(testset, file = paste0(s_input_path, get_batch_test_filename(numBatch)))
        print(paste0(as.double((proc.time() - systime_ini2)['elapsed'])/60, ' minutos en total (', doctipos_file,').'))
        minutos_pend <- (as.double((proc.time() - systime_ini2)['elapsed'])/60) * 2 * (NUM_BLOQUES / numBatch  -  1)
        if(minutos_pend < 60) print(paste0('Faltan aprox. ',minutos_pend, ' minutos.')) else print(paste0('Faltan aprox. ',minutos_pend/60, ' horas.'))
      }
    }
    rm(doctipos, ad_doctipos); gc()
  }
  # print(paste0(doctipos_file, '.csv - Ok.'))
}
# ----------------------------------------------------------------
#
# CAMBIO DE ESTRATEGIA: DIVIDE Y VENCERÁS:
#
# ----------------------------------------------------------------
# IDEA: Split Train in small samples so that we can predict ad_id in Testset by several sampled batches (y de paso los comparamos)
# Para dividir de forma que luego se pueda fusionar (rbind), lo hacemos dividiendo por display_id en N bloques:
mi_split_train <- function(NUM_BLOQUES=16, s_path=s_input_path)
{
  s_fich <- paste0("full_trainset_", str_pad(NUM_BLOQUES, 3, "left" ,"0"), ".RData")
  if(exists(file.path(s_path, s_fich))) return(0);
  stopifnot(exists("full_trainset"))
  setkey(full_trainset, display_id)
  disps <- unique(full_trainset[,.(display_id)], by = "display_id") # data.table sin duplicados por clave (y de una columna)
  index <- 1:nrow(disps)
  # trainset primero ordenamos aleatoriamente:
  index <- sample(index, nrow(disps))
  disps <- disps[index,]
  # Ok.
  rows_per_block <- 1 + as.integer(nrow(disps) / NUM_BLOQUES)
  for(i in 1:NUM_BLOQUES)
  {
    s_fich <- paste0("full_trainset_", str_pad(i, 3, "left" ,"0"), ".RData")
    index_from <- 1 + (i-1) * rows_per_block
    index_to <- i * rows_per_block
    if(index_to > nrow(disps)) index_to <- nrow(disps)
    disps_set <- disps[index_from:index_to,]
    trainset_batch <- merge(full_trainset, disps_set, by = 'display_id')
    save(trainset_batch, file = file.path(s_path, s_fich)) # full_trainset_001 a full_trainset_016
    rm(trainset_batch)
    gc()
    print(paste0(s_fich, " Ok."))
  }
}
mi_split_test <- function(NUM_BLOQUES=16, s_path=s_input_path)
{
  s_fich <- paste0("testset_", str_pad(NUM_BLOQUES, 3, "left" ,"0"), ".RData")
  if(exists(file.path(s_path, s_fich))) return(0);
  stopifnot(exists("testset"))
  setkey(testset, display_id)
  disps <- unique(testset[,.(display_id)], by = "display_id") # data.table sin duplicados por clave (y de una columna)
  index <- 1:nrow(disps)
  # NOTA: testset se deja con el orden original para predecir y crear el submitset en orden.
  rows_per_block <- 1 + as.integer(nrow(disps) / NUM_BLOQUES)
  for(i in 1:NUM_BLOQUES)
  {
    s_fich <- paste0("testset_", str_pad(i, 3, "left" ,"0"), ".RData")
    index_from <- 1 + (i-1) * rows_per_block
    index_to <- i * rows_per_block
    if(index_to > nrow(disps)) index_to <- nrow(disps)
    disps_set <- disps[index_from:index_to,]
    testset_batch <- merge(testset, disps_set, by = 'display_id')
    save(testset_batch, file = file.path(s_path, s_fich)) # testset_001 a testset_016
    rm(testset_batch)
    gc()
    print(paste0(s_fich, " Ok."))
  }
}
reducir_trainset <- function(mi_set, n_seed = 1, n_porc = 0.1) # 10% por defecto
{
  if(n_porc == 1) return(mi_set)
  stopifnot(c("display_id", "ad_id") %in% colnames(mi_set))
  # Reducimos todo para hacer pruebas más rápido:
  set.seed(n_seed)
  setkey(mi_set, display_id)
  disps <- unique(mi_set, by = "display_id")[,.(display_id)] # data.table sin duplicados por clave (y de una columna)
  index <- sample(1:nrow(disps), trunc(n_porc * nrow(disps)))
  smallset <- disps[index,] # Random sample of display_ids
  return(merge(mi_set, smallset, by = "display_id"))
  # Ok.
}
reducir_trainset_4 <- function(mi_set, n_seed = 1, n_porc = 0.8) # 80% trainset / 20% validset por defecto
{
  # En este caso, sencillamente reducimos trainset y devolvemos ambos:
  # NOTA: Además devolvemos lista de dos data.table (trainset y validset)
  stopifnot(c("display_id", "ad_id", "dia") %in% colnames(mi_set))
  set.seed(n_seed)
  setkey(mi_set, display_id)
  disps <- unique(mi_set, by = "display_id")[,.(display_id)] # data.table sin duplicados por clave (y de una columna)
  # Dividimos disps en trainset y validset:
  index <- sample(1:nrow(disps), trunc(n_porc * nrow(disps)))
  return(list(merge(mi_set, disps[index,],  by = "display_id"),  # trainset
              merge(mi_set, disps[-index,], by = "display_id"))) # validset
  # Ok.
}
reducir_trainset_2 <- function(mi_set, n_seed = 1, n_porc = 0.8) # 80% trainset / 20% validset por defecto
{
  # En este caso, además preparamos trainset para que se parezca al testset definitivo
  # (i.e. con 50% de los datos en los dos últimos días)
  # NOTA: Además devolvemos lista de dos data.table (trainset y validset)
  stopifnot(c("display_id", "ad_id", "dia") %in% colnames(mi_set))
  # Primero reducimos c(1:11) al mismo tamaño que c(12,13) [y descartamos el resto]:
  set.seed(n_seed)
  setkey(mi_set, display_id)
  disps1 <- unique(mi_set[dia <  12], by = "display_id")[, .(display_id)] # data.table sin duplicados por clave (y de una columna)
  disps2 <- unique(mi_set[dia >= 12], by = "display_id")[, .(display_id)] # data.table sin duplicados por clave (y de una columna)
  # (1 - nporc) * nrow(disps1) == nrow(disps2):
  disps1_newsize <- nrow(disps2)  # ¡Esto era un error! No queremos desperdiciar los días 12 y 13... trunc(nrow(disps2) / (1 - n_porc))
  if(disps1_newsize < nrow(disps1))
  {
    index <- sample(x = 1:nrow(disps1), size = disps1_newsize) # Reducción de (1:11) para que su parte de validset sea del tamaño de (12,13)
    disps1 <- disps1[index,] # El resto se descarta
  }
  # Ahora nrow(disps1) == nrow(disps2).  # ¡Esto era un error! No queremos desperdiciar los días 12 y 13... (1 - nporc) * nrow(disps1) == nrow(disps2).
  
  disps <- rbindlist(list(disps1, disps2))
  # Dividimos disps en trainset y validset:
  setkey(disps, display_id)
  index <- sample(1:nrow(disps), trunc(n_porc * nrow(disps)))
  return(list(merge(mi_set, disps[index,],  by = "display_id"),  # trainset
              merge(mi_set, disps[-index,], by = "display_id"))) # validset
  # Ok.
}
# ----------------------------------------------------------------
get_batch_train_filename <- function(numBatch)
{
  return(paste0("full_trainset_", str_pad(numBatch, 3, "left" ,"0"), ".RData"))
}
get_batch_test_filename  <- function(numBatch, numSubBatch = 0)
{
  if(numSubBatch == 0){
    return(paste0("testset_",       str_pad(numBatch, 3, "left" ,"0"), ".RData"))
  } else {
    return(paste0("testset_", str_pad(numBatch, 3, "left" ,"0"), "_", str_pad(numSubBatch, 3, "left" ,"0"), ".RData"))
  }
}
leer_batch_train <- function(numBatch, s_descr = "", s_input_path, i_bVerbose = TRUE)
{
  if(i_bVerbose)  print(paste0(ifelse(s_descr != "", paste0(s_descr, ' - '), ""), 'Batch trainset ', numBatch))
  s_fich_train <- get_batch_train_filename(numBatch)
  stopifnot(file.exists(file.path(s_input_path, s_fich_train)))
  if(exists("full_trainset")) {rm(full_trainset, inherits = TRUE); gc()}
  load(file.path(s_input_path, s_fich_train), .GlobalEnv) # Cuidado que se puede llamar trainset_batch!!!
  if(exists("trainset_batch")) {full_trainset <- trainset_batch; rm(trainset_batch, inherits = TRUE); gc()}
  if(i_bVerbose)  print(paste0(ifelse(s_descr != "", paste0(s_descr, ' - '), ""), 'Batch trainset ', numBatch, ' Ok. ', nrow(full_trainset), ' regs.'))
  return(full_trainset)
}
leer_batch_test <- function(numBatch, s_descr = "", s_input_path, i_bVerbose = TRUE, numSubBatch = 0)
{
  if(i_bVerbose)  print(paste0(ifelse(s_descr != "", paste0(s_descr, ' - '), ""), 'Batch testset ', numBatch, ifelse(numSubBatch == 0, '', paste0('_', numSubBatch))))
  s_fich_test <- get_batch_test_filename(numBatch, numSubBatch)
  stopifnot(file.exists(file.path(s_input_path, s_fich_test)))
  if(exists("testset")) {rm(testset, inherits = TRUE); gc()}
  load(file.path(s_input_path, s_fich_test), .GlobalEnv) # Cuidado que se puede llamar testset_batch!!!
  if(exists("testset_batch")) {testset <- testset_batch; rm(testset_batch, inherits = TRUE); gc()}
  if(i_bVerbose)  print(paste0(ifelse(s_descr != "", paste0(s_descr, ' - '), ""), 'Batch testset ', numBatch, ifelse(numSubBatch == 0, '', paste0('_', numSubBatch)), ' Ok. ', nrow(testset), ' regs.'))
  return(testset)
}
# ----------------------------------------------------------------
xgb_leerVarsModelo <- function(fichModelo, s_output_path)
{
  fich <- str_replace(str_replace(fichModelo, pattern = "\\.modelo$", replacement = ""), pattern = "\\_[0-9]*\\.[0-9][0-9][0-9]$", replacement = "\\.txt")
  stopifnot(file.exists(paste0(s_output_path, fich)))
  vars <- read.delim(file = paste0(s_output_path, fich), header = FALSE, col.names = c("varNum", "VarName", "featGain"), quote = "", dec = ".",
                     colClasses = c("character", "character", "numeric"), skip = 2)
  ultF <- nrow(vars)
  if(anyNA(vars$featGain))
  {
    ultF <- min(which(is.na(vars$featGain))) - 1
    if(vars$varNum[(ultF+1)] == "Feature")
      vars <- vars[1:ultF,]
  }
  stopifnot(all(vars$varNum[1:ultF] == c(1:ultF)))
  # Ya tenemos TODAS las vars del fichero ordenadas por Gain (desc.).
  return(vars)
}
xgb_prep_datos <- function(mi_dt, b_verbose = 2, maxImportanceNumVars = 0, fichero_Modelo = NULL, s_output_path = "")
{
  if(!is.null(mi_dt))
  {
    # mi_dt_nombre <- deparse(substitute(mi_dt)) # Para los eval(parse())
    if(!("publish_timestamp" %in% colnames(mi_dt)))
    {
      setkey(mi_dt, publish_time)
      mi_dt[publish_time == "", publish_time := NA]
      setkey(mi_dt, publish_time)
      mi_dt[, publish_timestamp := 1000 * as.numeric(as.POSIXct(publish_time)) - as.numeric(1465876799998), by = publish_time] # (ms since 1970-01-01 - 1465876799998)
      setkey(mi_dt, NULL)
    }
    if(!("ad_publish_timestamp" %in% colnames(mi_dt)))
    {
      setkey(mi_dt, ad_publish_time)
      mi_dt[ad_publish_time == "", ad_publish_time := NA]
      setkey(mi_dt, ad_publish_time)
      mi_dt[, ad_publish_timestamp := 1000 * as.numeric(as.POSIXct(ad_publish_time)) - as.numeric(1465876799998), by = ad_publish_time] # (ms since 1970-01-01 - 1465876799998)
      setkey(mi_dt, NULL)
    }
    # mi_dt[, hora := as.integer(hora)]
    # mi_dt[, dia := as.integer(dia)]
    # sapply(mi_dt, uniqueN)
    # str(mi_dt)
    
    # one-hot-encoding categorical features:
    # if(exists("categs"))  rm(categs) # Por si acaso...
    categs = c('platform', 'pais')
    # categs = c('uuid', 'geo_location', 'geo_loc.country')
    # categs = c(categs, 'ad_campaign_id', 'ad_advertiser_id')
    # categs = c(categs, 'source_id', 'publisher_id')
    # categs = c(categs, 'ad_source_id', 'ad_publisher_id')
    
    # numerics = c('dia', 'hora', 'timestamp') # V.6
    numerics = c('dia', 'hora') # V.7
    numerics = c(numerics, 'publish_timestamp', 'ad_publish_timestamp') # NAs se ponen a cero y santas pascuas plin...
    numerics = c(numerics, 'topics_prob', 'ad_topics_prob')
    numerics = c(numerics, 'topic_prob_1', 'topic_prob_2', 'topic_prob_3', 'topic_prob_4', 'topic_prob_5', 'topic_prob_6', 'topic_prob_7', 'topic_prob_8')
    numerics = c(numerics, 'entities_prob', 'ad_entities_prob')
    numerics = c(numerics, 'entity_prob_1', 'entity_prob_2', 'entity_prob_3', 'entity_prob_4', 'entity_prob_5', 'entity_prob_6', 'entity_prob_7', 'entity_prob_8')
    numerics = c(numerics, 'categories_prob', 'ad_categories_prob')
    numerics = c(numerics, 'category_prob_1', 'category_prob_2')
    # V.7 - Inicio:
    miscols <- c("uuid", # "clicked", ("display_id" no tiene sentido porque no hay ninguno coincidente entre train y test)
                 "document_id", "source_id", "publisher_id", "publish_timestamp" # , "timestamp"
    )
    ad_miscols <- c("ad_id", # "clicked",
                    "ad_document_id", "ad_source_id", "ad_publisher_id", "ad_publish_timestamp" # , "timestamp"
    )
    # Nuevos campos (por cada variable), que serán renombrados tras cada merge():  
    nuevos_campos <- c("tot", "clicks", "prob", "timestamp_min", "timestamp_max", "timestamp_avg", "timestamp_var", "timestamp_difmin", "timestamp_difmax", "timestamp_difavg", "timestamp_difvar")
    nuevos_campos <- nuevos_campos[nuevos_campos != "clicks"] # Tenemos "prob". No hace falta "clicks"...
    nuevos_campos2 <- nuevos_campos[nuevos_campos != "timestamp_difmin"]
    for(mivar in c(miscols, ad_miscols))
    {
      if(mivar %in% c("uuid", "document_id", "source_id", "publisher_id", "publish_timestamp")){
        numerics = c(numerics, paste0(mivar, '_', nuevos_campos2)) # timestamp_difmin tiene un único valor en estos casos (== 0)
      } else {
        numerics = c(numerics, paste0(mivar, '_', nuevos_campos))
      }
    }
    # V.7 - Final
  
    # V.9 - Inicio:
    miscols <- c("document_id", "source_id", "publisher_id", "publish_timestamp")
    miscols <- c(miscols, "uuid") # creado en 10.-
    # Nuevos campos (por cada variable), que serán renombrados tras cada merge():
    nuevos_campos_pgvw <- c("tot", "timestamp_min", "timestamp_max", "timestamp_avg", "timestamp_var", "timestamp_difmin", "timestamp_difmax", "timestamp_difavg", "timestamp_difvar")
    nuevos_campos_pgvw <- c(nuevos_campos_pgvw, "paisUS", "paisCA", "paisGB", "paisResto", "platform1", "platform2", "platform3", "trafsrc1", "trafsrc2", "trafsrc3")
    nuevos_campos_pgvw <- c(nuevos_campos_pgvw, "hora_min", "dia_min", "hora_max", "dia_max", "hora_avg", "dia_avg")
    # nuevos_campos_pgvw2 <- nuevos_campos_pgvw[nuevos_campos_pgvw != "timestamp_difmin"]
    for(mivar in miscols)
    {
      # if(mivar %in% c("uuid", "document_id", "source_id", "publisher_id", "publish_timestamp")){
        # numerics = c(numerics, paste0(mivar, '_pgvw_', nuevos_campos_pgvw2)) # timestamp_difmin tiene un único valor en estos casos (== 0)
        # if(mivar != 'uuid')
        #   numerics = c(numerics, paste0('ad_', mivar, '_pgvw_', nuevos_campos_pgvw2)) # timestamp_difmin tiene un único valor en estos casos (== 0)
      # } else {
        numerics = c(numerics, paste0(mivar, '_pgvw_', nuevos_campos_pgvw))
        if(mivar != 'uuid')
          numerics = c(numerics, paste0('ad_', mivar, '_pgvw_', nuevos_campos_pgvw))
      # }
    }
    # V.9 - Final
    
    # V.10 - Inicio (variables del ad_id con mayor ad_id_prob de cada display_id):
    # Variables del mismo display_id del anuncio que corresponden al ad_id ganador (el del ad_id_prob más grande del display):
    # i.e., todas las variables ad_xxx
    numerics_clk_1 <- numerics[substr(numerics, 1, 3) == 'ad_']
    # categs_clk_1 <- ... [Cf. más abajo]
    # V.10 - Final
    
  } # if(!is.null(mi_dt))
  
  # xgb_mi_version <- 5 # Numéricas normalizadas
  # xgb_mi_version <- 6 # Numéricas normalizadas + recup. las de version 2 (topic_prob_1, etc.)
  # xgb_mi_version <- 7 # Quitamos timestamp pero incluimos las de xxx_tiempos_def (además quitamos normalización -sobra con árboles de decisión-)
  # xgb_mi_version <- 8 # Categóricas "bien" codificadas (i.e. siempre igual).
  # xgb_mi_version <- 9 # Añadimos variables provenientes de page_views.csv (100 GB!)
  # xgb_mi_version <- 10 # Añadimos variables del click==1
  xgb_mi_version <- 11 # Añadimos numAds (para lstm)
  
  if(is.null(mi_dt))
    return(list(xgb_mi_version,NULL)) # Sólo devolvemos la versión

  # v7: if(b_verbose >= 1)  print('Convertimos numéricas a numeric (double): (WHY?)')
  # v7: for(miconv in paste0("mi_dt[, ", numerics, " := scale(as.numeric(", numerics, "), center=T, scale=T)]"))
  # v7: {
  # v7:   if(b_verbose == 2)  print(miconv)
  # v7:   eval(parse(text = miconv))
  # v7: }
  
  # v8: Aseguramos que las categóricas se codifiquen siempre igual:
  # v8: if(b_verbose >= 1)  print('Convertimos categóricas a Factor, para crear One-Hot encoding features:')
  # v8: for(miconv in paste0("mi_dt[, ", categs, " := as.factor(", categs, ")]"))
  # v8: {
  # v8:   if(b_verbose == 2)  print(miconv)
  # v8:   eval(parse(text = miconv)) # Convertimos a Factor
  # v8: }
  # v8: if(b_verbose >= 1)  print('Añadimos un nivel NA a las categóricas (solo si hay NAs):')
  # v8: for(miconv in paste0("if(anyNA(mi_dt$", categs, "))  mi_dt$", categs, " <- addNA(mi_dt$", categs, ")"))
  # v8: {
  # v8:   if(b_verbose == 2)  print(miconv)
  # v8:   eval(parse(text = miconv)) # Insertamos xxx_NA, un level para los NA, si hay
  # v8: }
  # v8: if(b_verbose >= 1)  print('Binarizando categóricas... (One Hot Encoding)')

  # V.10 - NOTA: Hay que reordenar mi_dt ahora, para que mi_dt2 tenga el mismo orden!
  if(b_verbose >= 1)  print(paste0('Ordenando data.table...'))
  setkey(mi_dt, NULL)
  setorderv(x = mi_dt, cols = c("display_id","ad_id_prob"), order = c(1,-1)) #Sort by display_id, -ad_id_prob
  setkey(mi_dt, display_id)
  
  if(exists("categs"))
  {
    if(b_verbose >= 1)  print(paste0('Procesando categs...'))
    mi_dt2 <- mi_dt[, c("display_id", "numAds", categs), with=FALSE] # Copiamos categóricas en otro data.table (y añadimos "display_id" [V.10]) (y añadimos "numAds" [V.11])
    categsv8 <- vector(mode = "character")
    if("platform" %in% categs)
    {
      mi_dt2[, platform1 := ifelse(platform == 1, 1, 0)]
      mi_dt2[, platform2 := ifelse(platform == 2, 1, 0)]
      mi_dt2[, platform3 := ifelse(platform == 3, 1, 0)]
      categsv8 <- c(categsv8, "platform1", "platform2", "platform3")
      # mi_dt2[, platformNA := ifelse(is.na(platform), 1, 0)]
      # categsv8 <- c(categsv8, "platformNA")
    }
    if("pais" %in% categs)
    {
      mi_dt2[, paisUS := ifelse(pais == "US", 1, 0)]
      mi_dt2[, paisCA := ifelse(pais == "CA", 1, 0)]
      mi_dt2[, paisGB := ifelse(pais == "GB", 1, 0)]
      mi_dt2[, paisResto := ifelse(pais == "Resto", 1, 0)]
      categsv8 <- c(categsv8, "paisUS", "paisCA", "paisGB", "paisResto")
      # mi_dt2[, paisNA := ifelse(is.na(pais), 1, 0)]
      # categsv8 <- c(categsv8, "paisNA")
    }
    categs_clk_1 <- categsv8[substr(categsv8, 1, 3) == 'ad_'] # V.10
    
    # # sparse_matrix <- model.matrix(~ 0 + get(categs), data = campaign)
    # # # # install.packages("caret")
    # # # library(lattice)
    # # # library(ggplot2)
    # # # library(caret)
    # formula <- paste0("~ ", paste(categs, collapse = "+"))
    # dummies <- dummyVars(formula = formula, data = mi_dt)
  }
  importanceVars <- vector("character", 0)
  if(!is.null(fichero_Modelo))
  {
    if(b_verbose >= 1)  print('Seleccionando variables del modelo...')
    importanceVars <- xgb_leerVarsModelo(fichero_Modelo, s_output_path)$VarName
    if(b_verbose >= 1)  print(paste0('Seleccionadas las ', length(importanceVars),' variables del modelo...'))
  } else if(maxImportanceNumVars != 0)
  {
    if(b_verbose >= 1)  print('Seleccionando variables...')
    # Creamos la lista de las maxImportanceNumVars variables más importantes:
    fichs <- dir(path = s_output_path, pattern = paste0("XGB_.*v[0-9]", str_pad(xgb_mi_version, 2, "left","0"), "\\.txt$"))
    mis_vars <- data.table(mivar = character(0), mipos = integer(0))
    for(fich in fichs)
    {
      vars <- xgb_leerVarsModelo(fich, s_output_path)
      if(nrow(vars) < maxImportanceNumVars)
        next # Este fichero tiene menos variables (o no tiene ninguna), así que no lo usamos...
      # Nos quedaremos, después, con las maxImportanceNumVars más frecuentes en todos los ficheros que encontremos:
      mis_vars <- rbindlist(list(mis_vars, list(mivar = vars$VarName, mipos = as.integer(vars$varNum))), use.names = T, fill = F)
    }
    setkey(mis_vars, mivar)
    pp <- mis_vars[, .(mx=max(mipos),mn=min(mipos),av=mean(mipos),sd=sd(mipos),cnt=.N), by=mivar]
    # plot(x = 1:nrow(pp), y = pp$mx)
    # plot(x = 1:nrow(pp), y = pp$av)
    # plot(x = 1:nrow(pp), y = pp$mn)
    setorderv(pp, "mx") # ordenamos por máximo (i.e. la "peor" posición de cada variable)
    # Y nos quedamos con las mejores:
    importanceVars <- pp$mivar[1:maxImportanceNumVars]
    if(b_verbose >= 1)  print(paste0('Nos quedamos con las ', maxImportanceNumVars,' variables más importantes (XGB Feature Gain)...'))
  } # if(maxImportanceNumVars != 0)
  
  # V.10 (2) - Aquí añadimos los nombres de las variables numerics_clk_1 & categs_clk_1 a numerics & categsv8:
  if(length(numerics_clk_1) != 0)  numerics <- c(numerics, paste0('clk1_', numerics_clk_1))
  if(exists("categs"))
  { if(length(categs_clk_1) != 0)  categsv8 <- c(categsv8, paste0('clk1_', categs_clk_1)) }

  # Filtramos variables por importancia:
  if(length(importanceVars) != 0)
  {
    numerics <- numerics[numerics %in% importanceVars]
    if(exists("categs"))
    {
      categsv8 <- categsv8[categsv8 %in% importanceVars]
      if(length(categsv8) == 0)
      {
        if(b_verbose >= 1)  print("NOTA: Quitamos categóricas porque no ha 'sobrevivido' ninguna")
        rm(categs, categsv8, categs_clk_1) # Quitamos categóricas porque no ha 'sobrevivido' ninguna...
      }
    }
  } # if(length(importancevars) != 0)
  
  # V.10 (y 3) - Y finalmente. añadimos el contenido de las variables numerics_clk_1 & categs_clk_1 a mi_dt:
  if(length(numerics_clk_1) != 0)
    numerics_clk_1 <- numerics_clk_1[paste0('clk1_', numerics_clk_1) %in% numerics]
  if(length(numerics_clk_1) != 0)
  {
    if(b_verbose >= 1)  print(paste0(Sys.time(), ' - ', 'Creando variables numerics_clk_1 (', length(numerics_clk_1), ' vars.)...'))
    miconv_ini <- "mi_dt[,c("
    miconv_fin <- "), by='display_id']"
    for(i in seq.int(1,length(numerics_clk_1), 25))
    {
      j <- min(i+24, length(numerics_clk_1))
      if(b_verbose >= 1)  print(paste0(Sys.time(), ' - ', 'Creando variables numerics_clk_1[', i, ':', j, ']...'))
      miconv_mid_1 <- paste0("'clk1_", numerics_clk_1[i:j], "'", collapse = ",")
      miconv_mid_2 <- paste0(numerics_clk_1[i:j], "[1]", collapse = ",")
      if(b_verbose >= 2)  print(miconv_mid_1)
      miconv <- paste0(miconv_ini, miconv_mid_1, ") := list(", miconv_mid_2, miconv_fin)
      eval(parse(text = miconv))
    }
    # for(j in numerics_clk_1)
    # { mi_dt[, c(paste0('clk1_', j)) := get(j)[1], by = "display_id"]
    #   if(b_verbose >= 2)  print(paste0(Sys.time(), ' - ', 'Creando variable ', j, ' (', which(numerics_clk_1 == j),'/',length(numerics_clk_1),')...'))
    # }
  }
  if(exists("categs"))
  {
    if(length(categs_clk_1) != 0)
    {
      if(b_verbose >= 1)  print(paste0('Creando variables categs_clk_1 (', length(categs_clk_1), ' max.)...'))
      for(j in categs_clk_1)
      { if(paste0('clk1_', j) %in% categsv8)  mi_dt2[, c(paste0('clk1_', j)) := get(j)[1], by = "display_id"] }
    }
    # No hace falta! # Ya podemos quitar "display_id":
    # No hace falta! mi_dt2[, display_id := NULL]
    categsv8 = c(categsv8, 'numAds') # V.11 (numAds como última columna)
  } else {
    numerics = c(numerics, 'numAds') # V.11 (numAds como última columna)
  }
  
  if(b_verbose >= 1)  print('Preparando data.table...')
  if("clicked" %in% colnames(mi_dt))
  {
    clicked.col <- as.numeric(mi_dt$clicked)
  } else 
  {
    clicked.col <- as.numeric(rep.int(0, nrow(mi_dt)))
  }
  if(exists("categs"))
  {
    return(list(xgb_mi_version,
                cbind(clicked = clicked.col,
                      mi_dt[, numerics, with=FALSE],
                      # v8: predict(dummies, newdata = mi_dt)
                      mi_dt2[, categsv8, with=FALSE]
                      )
    )    )
  } else {
    return(list(xgb_mi_version,
                cbind(clicked = clicked.col,
                      mi_dt[, numerics, with=FALSE]
                )
    )    )
  }
}
# ----------------------------------------------------------------
nbay_prep_datos <- function(mi_dt, b_verbose = 2)
{
  # mi_dt_nombre <- deparse(substitute(mi_dt)) # Para los eval(parse())
  setkey(mi_dt, NULL)
  # mi_dt[, hora := as.integer(hora)]
  # mi_dt[, dia := as.integer(dia)]
  # sapply(mi_dt, uniqueN)
  # str(mi_dt)
  
  # Para Naive Bayes ponemos todo como categóricas:
  categs = c('platform', 'pais')
  categs = c(categs, 'uuid', 'geo_location', 'geo_loc.country')
  categs = c(categs, 'ad_campaign_id', 'ad_advertiser_id')
  categs = c(categs, 'source_id', 'publisher_id')
  categs = c(categs, 'ad_source_id', 'ad_publisher_id')
  categs = c(categs, 'publish_timestamp', 'ad_publish_timestamp')
  
  numerics = c('dia', 'hora', 'timestamp')
  # numerics = c(numerics, 'publish_timestamp', 'ad_publish_timestamp') # NAs ???
  numerics = c(numerics, 'topics_prob', 'ad_topics_prob')
  numerics = c(numerics, 'topic_prob_1', 'topic_prob_2', 'topic_prob_3', 'topic_prob_4', 'topic_prob_5', 'topic_prob_6', 'topic_prob_7', 'topic_prob_8')
  numerics = c(numerics, 'entities_prob', 'ad_entities_prob')
  numerics = c(numerics, 'entity_prob_1', 'entity_prob_2', 'entity_prob_3', 'entity_prob_4', 'entity_prob_5', 'entity_prob_6', 'entity_prob_7', 'entity_prob_8')
  numerics = c(numerics, 'categories_prob', 'ad_categories_prob')
  numerics = c(numerics, 'category_prob_1', 'category_prob_2')
  
  nbay_mi_version <- 5 # Numéricas normalizadas
  nbay_mi_version <- 6 # publish_timestamp y ad_publish_timestamp como categóricas
  
  if(b_verbose >= 1)  print('Convertimos numéricas a numeric (double) y normalizamos: (WHY?)')
  for(miconv in paste0("mi_dt[, ", numerics, " := scale(as.numeric(", numerics, "), center=T, scale=T)]"))
  {
    if(b_verbose == 2)  print(miconv)
    eval(parse(text = miconv))
  }
  if(b_verbose >= 1)  print('Convertimos categóricas a Factor:')
  for(miconv in paste0("mi_dt[, ", categs, " := as.factor(", categs, ")]"))
  {
    if(b_verbose == 2)  print(miconv)
    eval(parse(text = miconv)) # Convertimos a Factor
  }
  for(miconv in paste0("if(anyNA(mi_dt$", categs, "))  mi_dt$", categs, " <- addNA(mi_dt$", categs, ")"))
  {
    if(b_verbose == 2)  print(miconv)
    eval(parse(text = miconv)) # Insertamos xxx_NA, un level para los NA, si hay
  }
  if(b_verbose >= 1)  print('Preparando data.table...')
  if("clicked" %in% colnames(mi_dt))
  {
    clicked.col <- as.integer(mi_dt$clicked)
  } else 
  {
    clicked.col <- as.integer(rep.int(0, nrow(mi_dt)))
  }
  clicked.col <- as.factor(clicked.col)
  
  return(list(nbay_mi_version,
              cbind(clicked = clicked.col,
                    mi_dt[, numerics, with=FALSE],
                    mi_dt[, categs, with=FALSE])
  )    )
}
# ----------------------------------------------------------------
randfor_prep_datos <- function(mi_dt, b_verbose = 2)
{
  return(nbay_prep_datos(mi_dt = mi_dt, b_verbose = b_verbose))
}
# ----------------------------------------------------------------
RNN_prep_datos <- function(mi_dt, b_verbose = 2)
{
  # Basado en xgb_prep_datos (v.7)
  if("publish_timestamp" %in% colnames(mi_dt)) # QUITAMOS NAs
  {
    set(mi_dt, which(is.na(mi_dt[["publish_timestamp"]])), "publish_timestamp", 0) # NAs a cero
  }
  if("ad_publish_timestamp" %in% colnames(mi_dt)) # QUITAMOS NAs
  {
    set(mi_dt, which(is.na(mi_dt[["ad_publish_timestamp"]])), "ad_publish_timestamp", 0) # NAs a cero
  }
  
  print('Ordenamos datos por timestamp (para RNN y LSTM):')
  setkey(mi_dt, NULL)
  setorder(x = mi_dt, timestamp, na.last = T)

  # one-hot-encoding categorical features:
  categs = c('platform', 'pais')
  # categs = c('uuid', 'geo_location', 'geo_loc.country')
  # categs = c(categs, 'ad_campaign_id', 'ad_advertiser_id')
  # categs = c(categs, 'source_id', 'publisher_id')
  # categs = c(categs, 'ad_source_id', 'ad_publisher_id')
  
  # numerics = c('dia', 'hora', 'timestamp') # V.6
  numerics = c('dia', 'hora') # V.7
  # numerics = c(numerics, 'publish_timestamp', 'ad_publish_timestamp') # No queremos NAs, por si acaso...
  numerics = c(numerics, 'topics_prob', 'ad_topics_prob')
  numerics = c(numerics, 'topic_prob_1', 'topic_prob_2', 'topic_prob_3', 'topic_prob_4', 'topic_prob_5', 'topic_prob_6', 'topic_prob_7', 'topic_prob_8')
  numerics = c(numerics, 'entities_prob', 'ad_entities_prob')
  numerics = c(numerics, 'entity_prob_1', 'entity_prob_2', 'entity_prob_3', 'entity_prob_4', 'entity_prob_5', 'entity_prob_6', 'entity_prob_7', 'entity_prob_8')
  numerics = c(numerics, 'categories_prob', 'ad_categories_prob')
  numerics = c(numerics, 'category_prob_1', 'category_prob_2')
  # V.7 - Inicio:
  miscols <- c("uuid", # "clicked", ("display_id" no tiene sentido porque no hay ninguno coincidente entre train y test)
               "document_id", "source_id", "publisher_id", "publish_timestamp" # , "timestamp"
  )
  ad_miscols <- c("ad_id", # "clicked",
                  "ad_document_id", "ad_source_id", "ad_publisher_id", "ad_publish_timestamp" # , "timestamp"
  )
  # Nuevos campos (por cada variable), que serán renombrados tras cada merge():  
  nuevos_campos <- c("tot", "clicks", "prob", "timestamp_min", "timestamp_max", "timestamp_avg", "timestamp_var", "timestamp_difmin", "timestamp_difmax", "timestamp_difavg", "timestamp_difvar")
  nuevos_campos <- nuevos_campos[nuevos_campos != "clicks"] # Tenemos "prob". No hace falta "clicks"...
  nuevos_campos2 <- nuevos_campos[nuevos_campos != "timestamp_difmin"]
  for(mivar in c(miscols, ad_miscols))
  {
    if(mivar %in% c("uuid", "document_id", "source_id", "publisher_id", "publish_timestamp")){
      numerics = c(numerics, paste0(mivar, '_', nuevos_campos2)) # timestamp_difmin tiene un único valor en estos casos (== 0)
    } else {
      numerics = c(numerics, paste0(mivar, '_', nuevos_campos))
    }
  }
  # V.7 - Final
  
  # RNN_mi_version <- 7 # Datos ordenados por timestamp (para RNN y LSTM)
  RNN_mi_version <- 8 # Categóricas "bien" codificadas (i.e. siempre igual).
  
  if(b_verbose >= 1)  print('Normalizamos (max-min) numéricas para RNN:')
  mi_normaliza <- function(x) { return( (x - min(x)) / (max(x) - min(x)) )}
  for(miconv in paste0("mi_dt[, ", numerics, " := mi_normaliza(", numerics, ") ]"))
  {
    if(b_verbose == 2)  print(miconv)
    eval(parse(text = miconv))
  }

  # v8: if(b_verbose >= 1)  print('Convertimos categóricas a Factor, para crear One-Hot encoding features:')
  # v8: for(miconv in paste0("mi_dt[, ", categs, " := as.factor(", categs, ")]"))
  # v8: {
  # v8:   if(b_verbose == 2)  print(miconv)
  # v8:   eval(parse(text = miconv)) # Convertimos a Factor
  # v8: }
  # v8: if(b_verbose >= 1)  print('Añadimos un nivel NA a las categóricas (solo si hay NAs):')
  # v8: for(miconv in paste0("if(anyNA(mi_dt$", categs, "))  mi_dt$", categs, " <- addNA(mi_dt$", categs, ")"))
  # v8: {
  # v8:   if(b_verbose == 2)  print(miconv)
  # v8:   eval(parse(text = miconv)) # Insertamos xxx_NA, un level para los NA, si hay
  # v8: }
  # v8: 
  # v8: if(b_verbose >= 1)  print('Binarizando categóricas... (One Hot Encoding)')
  # v8: # sparse_matrix <- model.matrix(~ 0 + get(categs), data = campaign)
  # v8: # # # install.packages("caret")
  # v8: # # library(lattice)
  # v8: # # library(ggplot2)
  # v8: # # library(caret)
  # v8: formula <- paste0("~ ", paste(categs, collapse = "+"))
  # v8: dummies <- dummyVars(formula = formula, data = mi_dt)
  if(exists("categs"))
  {
    mi_dt2 <- mi_dt[, categs, with=FALSE] # Copiamos categóricas en otro data.table
    categsv8 <- vector(mode = "character")
    if("platform" %in% categs)
    {
      mi_dt2[, platform1 := ifelse(platform == 1, 1, 0)]
      mi_dt2[, platform2 := ifelse(platform == 2, 1, 0)]
      mi_dt2[, platform3 := ifelse(platform == 3, 1, 0)]
      categsv8 <- c(categsv8, "platform1", "platform2", "platform3")
      # mi_dt2[, platformNA := ifelse(is.na(platform), 1, 0)]
      # categsv8 <- c(categsv8, "platformNA")
    }
    if("pais" %in% categs)
    {
      mi_dt2[, paisUS := ifelse(pais == "US", 1, 0)]
      mi_dt2[, paisCA := ifelse(pais == "CA", 1, 0)]
      mi_dt2[, paisGB := ifelse(pais == "GB", 1, 0)]
      mi_dt2[, paisResto := ifelse(pais == "Resto", 1, 0)]
      categsv8 <- c(categsv8, "paisUS", "paisCA", "paisGB", "paisResto")
      # mi_dt2[, paisNA := ifelse(is.na(pais), 1, 0)]
      # categsv8 <- c(categsv8, "paisNA")
    }
    # # sparse_matrix <- model.matrix(~ 0 + get(categs), data = campaign)
    # # # # install.packages("caret")
    # # # library(lattice)
    # # # library(ggplot2)
    # # # library(caret)
    # formula <- paste0("~ ", paste(categs, collapse = "+"))
    # dummies <- dummyVars(formula = formula, data = mi_dt)
  }
  
  if(b_verbose >= 1)  print('Preparando data.table...')
  if("clicked" %in% colnames(mi_dt))
  {
    clicked.col <- as.numeric(mi_dt$clicked)
  } else 
  {
    clicked.col <- as.numeric(rep.int(0, nrow(mi_dt)))
  }
  if(exists("categs"))
  {
    return(list(RNN_mi_version,
                cbind(clicked = clicked.col,
                      mi_dt[, numerics, with=FALSE],
                      # v8: predict(dummies, newdata = mi_dt)
                      mi_dt2[, categsv8, with=FALSE]
                )
    )    )
  } else {
    return(list(RNN_mi_version,
                cbind(clicked = clicked.col,
                      mi_dt[, numerics, with=FALSE]
                )
    )    )
  }
}
# ----------------------------------------------------------------
predict_testset <- function(nombres_modelos
                            , filename, s_input_path, s_output_path
                            , i_sDescr = "" # "XGB Predicting (extreme gradient boosting)"
                            , FUN_prep_datos, prep_datos_b_verbose = 0
                            , FUN_X
                            , FUN_Predict
                            , FUN_loadmodelo, nombreModeloLoaded
                            # , i_sDescr = "NBAY Predicting (Naive Bayes)" # NaiveBayes
                            # , FUN_prep_datos = nbay_prep_datos # NaiveBayes
                            # , FUN_X = identity # FUN_X = function(x){ return(data.matrix(x)) } # NaiveBayes
                            # , FUN_Predict = function(modelo, X){ return(predict(modelo, newdata = X, type = 'prob')$posterior[,"1"]) } # NaiveBayes
                            # , FUN_loadmodelo = load, nombreModeloLoaded = "nbay" # NaiveBayes
                            # , i_sDescr = "XGB Predicting (extreme gradient boosting)" # XGBoost
                            # , FUN_prep_datos = xgb_prep_datos # XGBoost
                            # , FUN_X = data.matrix # FUN_X = function(x){ return(data.matrix(x)) } # XGBoost
                            # , FUN_Predict = function(modelo, X){ return(xgboost::predict(modelo, X, missing = NA)) } # XGBoost
                            # , FUN_loadmodelo = xgb.load, nombreModeloLoaded = "" # XGBoost
                            # , FUN_prep_datos = randfor_prep_datos # RandomForest
                            # , FUN_X = data.matrix # FUN_X = function(x){ return(data.matrix(x)) } # RandomForest
                            # , FUN_Predict = function(modelo, X){ return(predict(modelo, newdata = X, type = 'prob')[,"1"]) } # RandomForest
                            # , FUN_loadmodelo = load, nombreModeloLoaded = "randfor" # RandomForest
                            , NUM_MODELOS = NUM_BLOQUES
                            , b_ReducirFicheros = FALSE # De momento no hace falta, pero podría hacer falta...
                            , modelos_weights = NULL # Para mezclar modelos sin ton ni son (bueno, con estos pesos)...
                            , modelos_numAds = NULL # Es el numAd correspondiente a cada modelo, para mezclar modelos para cada numAd...
)
{
  # Leemos modelo(s):
  if(length(nombres_modelos[nombres_modelos != ""]) == 0)
  {
    tmp_nombres_modelos <- str_replace(dir(path = s_output_path, pattern = paste0(str_replace(filename, "_[0-9]*\\.[0-9][0-9][0-9]$", ""), '.*.modelo')), pattern = "\\.modelo", replacement = "")
    # Ordenamos modelos por numModelo (aquí NO da siempre igual, porque ya NO los vamos a promediar siempre)
    nombres_modelos[as.integer(substr(tmp_nombres_modelos, nchar(tmp_nombres_modelos)-2, nchar(tmp_nombres_modelos)))] <- tmp_nombres_modelos
  }
  stopifnot(length(nombres_modelos[nombres_modelos != ""]) == NUM_MODELOS)
  b_con_media_ponderada <- !is.null(modelos_weights)
  b_con_numAds <- !is.null(modelos_numAds)
  if(b_con_media_ponderada)
  {
    stopifnot(length(modelos_weights[!is.na(modelos_weights)]) == NUM_MODELOS)
    if(!b_con_numAds & substr(filename, 1, 2) != "w_")
      filename <- paste0("w_", filename)
  }
  if(b_con_numAds)
  {
    stopifnot(length(modelos_numAds[!is.na(modelos_numAds)]) == NUM_MODELOS)
    if(substr(filename, 1, 3) != "w2_")
      filename <- paste0("w2_", filename)
  }
  # s_Fichero_submit <- paste0(unique(str_replace(string = nombres_modelos, pattern = "_[0-9]*\\.[0-9][0-9][0-9]$", replace = "")), '_submit.csv')
  s_Fichero_submit <- paste0(str_replace(filename, "_[0-9]*\\.[0-9][0-9][0-9]$", ""), '_submit.csv')
  s_Fic_submit_log <- paste0(str_replace(filename, "_[0-9]*\\.[0-9][0-9][0-9]$", ""), '_submit.log')
  write('-------------------------------------------------------------------------------', file = s_Fic_submit_log, append = TRUE) # Línea separadora para el fichero...
  str_tmp <- paste0(Sys.time(), ' - ', 'Preparando submit [', s_Fichero_submit, ']...')
  print(str_tmp); write(str_tmp, file = s_Fic_submit_log, append = TRUE)

  if(b_ReducirFicheros)
  {
    systime_ini2 <- proc.time()
    # Creamos una división en cada bloque de otros 16 bloques (para acelerar y evitar problemas de memoria):
    if(!file.exists(file.path(s_input_path, get_batch_test_filename(NUM_BLOQUES, NUM_BLOQUES))))
    {
      for(numBatch in 1:NUM_BLOQUES)
      {
        str_tmp <- paste0(Sys.time(), ' - ', 'Creando sub_bloques de testset...')
        print(str_tmp); if(exists("s_Fic_submit_log")) write(str_tmp, file = s_Fic_submit_log, append = TRUE)
        testset_big <- leer_batch_test(numBatch, i_sDescr, s_input_path)
        setkey(testset_big, display_id)
        disps <- unique(testset_big[,.(display_id)], by = "display_id") # data.table sin duplicados por clave (y de una columna)
        index <- 1:nrow(disps)
        # NOTA: testset_big se deja con el orden original para predecir y crear el submitset en orden.
        rows_per_block <- 1 + as.integer(nrow(disps) / NUM_BLOQUES)
        for(numSubBatch in 1:NUM_BLOQUES)
        {
          s_fich <- get_batch_test_filename(numBatch, numSubBatch)
          index_from <- 1 + (numSubBatch-1) * rows_per_block
          index_to <- numSubBatch * rows_per_block
          if(index_to > nrow(disps)) index_to <- nrow(disps)
          disps_set <- disps[index_from:index_to,]
          testset <- merge(testset_big, disps_set, by = 'display_id')
          save(testset, file = file.path(s_input_path, s_fich)) # testset_XXX_001 a testset_XXX_016
          rm(testset); gc()
          str_tmp <- paste0(Sys.time(), ' - ', s_fich, " Ok.")
          print(str_tmp); if(exists("s_Fic_submit_log")) write(str_tmp, file = s_Fic_submit_log, append = TRUE)
          # print(paste0(as.double((proc.time() - systime_ini2)['elapsed'])/60, ' minutos.'))
          minutos_pend <- (as.double((proc.time() - systime_ini2)['elapsed'])/60) * ( (NUM_BLOQUES^2) / ((numBatch-1) * NUM_BLOQUES + numSubBatch)  -  1)
          if(minutos_pend < 60) print(paste0('Faltan aprox. ',minutos_pend, ' minutos.')) else print(paste0('Faltan aprox. ',minutos_pend/60, ' horas.'))
        }
        rm(testset_big, disps, disps_set); gc()
      }
    }
  }
  
  nombres_modelos <- str_replace(nombres_modelos, pattern = "\\.modelo$", replacement = "")
  modelos <- list() # Inicializamos lista de modelos y cargamos los modelos:
  for(i in 1:NUM_MODELOS)
  {
    str_tmp <- paste0(Sys.time(), ' - ', 'Leyendo ', str_pad(i, 3, "left" ,"0"), ' (modelo ', i, ' de ', NUM_MODELOS, '): [', nombres_modelos[i],']...')
    # print(str_tmp); write(str_tmp, file = s_Fic_submit_log, append = TRUE)
    ix_mod_ya_cargado <- ifelse(i == 1, integer(0), match(nombres_modelos[i], nombres_modelos[1:(i-1)]))
    if(!is.na(ix_mod_ya_cargado))
    { # El modelo ya está cargado:
      str_tmp <- paste0(str_tmp, ' Ok. (Ya estaba cargado)')
      print(str_tmp); write(str_tmp, file = s_Fic_submit_log, append = TRUE)
      ix_mod_ya_cargado <- min(ix_mod_ya_cargado)
      modelos[[i]] <- ix_mod_ya_cargado # Luego hay que verificar si el modelo es o no es un integer
    } else {
      # Leemos fichero del modelo:
      print(str_tmp); write(str_tmp, file = s_Fic_submit_log, append = TRUE)
      stopifnot(file.exists(paste0(s_output_path, nombres_modelos[i] ,'.modelo')))
      modelos[[i]] <- FUN_loadmodelo(paste0(s_output_path, nombres_modelos[i] ,'.modelo'))
      if(nombreModeloLoaded != "")
      {
        modelos[[i]] <- get(nombreModeloLoaded); rm(list = c(nombreModeloLoaded), inherits = TRUE); gc()
      } # deparse(nombreModeloLoaded) deparse(get(nombreModeloLoaded))
    }
    write(str_tmp, file = s_Fic_submit_log, append = TRUE)
  }
  
  # NOTA: A partir de la versión 500, cada modelo es para un numAdsCluster (=numModelo+1):
  n_versiones <- as.integer(substring(stringr::str_extract(nombres_modelos, pattern = "v[0-9]+"), first = 2))
  n_version <- n_versiones[1]
  str_tmp <- paste0(Sys.time(), ' - ', 'Versión ', n_version, '. Con_media_ponderada = ', b_con_media_ponderada, '. Con_numAds = ', b_con_numAds | (n_version > 500 & !b_con_media_ponderada))
  print(str_tmp); write(str_tmp, file = s_Fic_submit_log, append = TRUE)
  if(!b_con_media_ponderada)
    stopifnot(all(n_versiones == n_version)) # La versión debe coincidir si NO estamos haciendo medias ponderadas
  systime_ini2 <- proc.time()
  for(numBatch in 1:NUM_BLOQUES)
  {
    for(numSubBatch in 1:NUM_BLOQUES)
    {
      if(!file.exists(paste0(s_input_path, get_batch_test_filename(NUM_BLOQUES, NUM_BLOQUES))))
      {
        # No hay subdivisión de testsets. Lo hacemos con los "grandes":
        if(numSubBatch != 1)  break # Sólo el primero!
        testset <- leer_batch_test(numBatch, i_sDescr, s_input_path)
      } else {
        testset <- leer_batch_test(numBatch, numSubBatch = numSubBatch, s_descr = i_sDescr, s_input_path = s_input_path)
      }
      gc()
      if(b_con_media_ponderada | n_version < 500) # Versión < 500 o bien media ponderada:
      {
        dt_all <- FUN_prep_datos(mi_dt = testset, b_verbose = prep_datos_b_verbose)[[2]]
        if(b_con_numAds)  testset[, prob := as.numeric(NA)]
        # print(colnames(dt_all))
        str_tmp <- paste0(Sys.time(), ' - ', 'Preparando matrix y predicting...', ' Versión ', n_version, '. Con_media_ponderada = ', b_con_media_ponderada, '. Con_numAds = ', b_con_numAds | (n_version > 500 & !b_con_media_ponderada))
        print(str_tmp); write(str_tmp, file = s_Fic_submit_log, append = TRUE)
        X <- FUN_X(dt_all[,-1,with=FALSE])
        # y <- data.matrix(dt_all[,1,with=FALSE])
        y_pred <- vector(mode = "numeric", length = NUM_MODELOS)
        for(i in 1:NUM_MODELOS)
        {
          str_tmp <- paste0(Sys.time(), ' - ', 'Predicting ', str_pad(numBatch, 3, "left" ,"0"), '_', str_pad(numSubBatch, 3, "left" ,"0"), ' (modelo ', i, ' de ', NUM_MODELOS, '): ', nombres_modelos[i],'...', ' Versión ', n_version, '. Con_media_ponderada = ', b_con_media_ponderada, '. Con_numAds = ', b_con_numAds | (n_version > 500 & !b_con_media_ponderada))
          print(str_tmp); write(str_tmp, file = s_Fic_submit_log, append = TRUE)
          # modelos[[i]] <- xgb.load(paste0(s_output_path, nombres_modelos[i] ,'.modelo'))
          
          if(b_con_media_ponderada)
          {
            if(n_version > 500) # Hay selección de variables (que puede ser diferente para cada modelo)
            {
              if(b_con_numAds) {
                if(nrow(testset[numAds == modelos_numAds[i],]) != 0)
                {
                  X <- FUN_X(FUN_prep_datos(mi_dt = testset[numAds == modelos_numAds[i],], b_verbose = prep_datos_b_verbose, fichero_Modelo = nombres_modelos[i], s_output_path = s_output_path)[[2]][,-1, with=FALSE])
                  gc()
                  y_pred <- modelos_weights[i] * FUN_Predict( if(is.integer(modelos[[i]])) modelos[[(modelos[[i]])]] else modelos[[i]], X)
                  y_pred <- y_pred / sum(modelos_weights[modelos_numAds == modelos_numAds[i]]) # Suma de pesos de todos los modelos de este numAd
                  if(anyNA(testset[numAds == modelos_numAds[i], prob])) {
                    testset[numAds == modelos_numAds[i], prob := y_pred]
                  } else {
                    testset[numAds == modelos_numAds[i], prob := prob + y_pred]
                  }
                }
              } else {
                X <- FUN_X(FUN_prep_datos(mi_dt = testset, b_verbose = prep_datos_b_verbose, fichero_Modelo = nombres_modelos[i], s_output_path = s_output_path)[[2]][,-1, with=FALSE])
                gc()
              }
            }
            if(!b_con_numAds) y_pred[i] <- list( modelos_weights[i] * FUN_Predict( if(is.integer(modelos[[i]])) modelos[[(modelos[[i]])]] else modelos[[i]], X) )
          } else {
            y_pred[i] <- list(FUN_Predict( if(is.integer(modelos[[i]])) modelos[[(modelos[[i]])]] else modelos[[i]], X))
          }
        }
        if(!b_con_numAds)
        {
          if(NUM_MODELOS > 1)
          {
            mean_y_pred <- rowSums(rbindlist(list(y_pred)))
            if(b_con_media_ponderada) mean_y_pred <-  mean_y_pred / sum(modelos_weights[i]) else mean_y_pred <- mean_y_pred / NUM_MODELOS
          } else {
            mean_y_pred <- y_pred[1]
          }
          testset[, prob := mean_y_pred]
        }
      } else {
        # Versión 5xx o más: cada modelo corresponde con un numAdsCluster (=numModelo+1)
        # Primero dividimos cada testset en bloques por numAds, luego predecimos cada uno por separado y finalmente reordenamos testset:
        for(numModelo in 1:NUM_MODELOS)
        {
          numAdsCluster <- numModelo + 1
          if(any(testset$numAds == numAdsCluster))
          {
            dt_all <- FUN_prep_datos(mi_dt = testset[numAds == numAdsCluster,], b_verbose = prep_datos_b_verbose, fichero_Modelo = nombres_modelos[i], s_output_path = s_output_path)[[2]]
            # print(colnames(dt_all))
            str_tmp <- paste0(Sys.time(), ' - ', 'Preparando matrix y predicting...', ' Versión ', n_version, '. Con_media_ponderada = ', b_con_media_ponderada, '. Con_numAds = ', b_con_numAds | (n_version > 500 & !b_con_media_ponderada))
            print(str_tmp); write(str_tmp, file = s_Fic_submit_log, append = TRUE)
            X <- FUN_X(dt_all[,-1,with=FALSE]); rm(dt_all); gc()
            str_tmp <- paste0(Sys.time(), ' - ', 'Predicting ', str_pad(numBatch, 3, "left" ,"0"), '_', str_pad(numSubBatch, 3, "left" ,"0"), ' (modelo ', numModelo, ' de ', NUM_MODELOS, '): ', nombres_modelos[numModelo],'...', ' Versión ', n_version, '. Con_media_ponderada = ', b_con_media_ponderada, '. Con_numAds = ', b_con_numAds | (n_version > 500 & !b_con_media_ponderada))
            print(str_tmp); write(str_tmp, file = s_Fic_submit_log, append = TRUE)
            # modelos[[numModelo]] <- xgb.load(paste0(s_output_path, nombres_modelos[numModelo] ,'.modelo'))
            y_pred <- list(FUN_Predict( if(is.integer(modelos[[numModelo]])) modelos[[(modelos[[numModelo]])]] else modelos[[numModelo]], X))
            testset[numAds == numAdsCluster, prob := y_pred]
          }
        }
      }
      stopifnot(!anyNA(testset$prob))
      if(!file.exists(paste0(s_input_path, get_batch_test_filename(NUM_BLOQUES, NUM_BLOQUES))))
      {
        # No hay subdivisión de testsets. Lo hacemos con los "grandes":
        str_tmp <- paste0(Sys.time(), ' - ', 'Guardando ', s_Fichero_submit, ' (', (numBatch) , '/', NUM_BLOQUES, ')...')
        print(str_tmp); write(str_tmp, file = s_Fic_submit_log, append = TRUE)
      } else {
        str_tmp <- paste0(Sys.time(), ' - ', 'Guardando ', s_Fichero_submit, ' (', (numBatch-1) * NUM_BLOQUES + numSubBatch, '/', NUM_BLOQUES^2, ')...')
        print(str_tmp); write(str_tmp, file = s_Fic_submit_log, append = TRUE)
      }
      guardar_submit(testset = testset, fichero = s_Fichero_submit, s_output_path = s_output_path,
                     b_write_file_append = (numBatch>1 | numSubBatch>1)
                    )
      if(numSubBatch != 1)
      {
        minutos <- as.double((proc.time() - systime_ini2)['elapsed'])/60
        if(minutos < 60) str_tmp <- paste0(Sys.time(), ' - ', minutos, ' minutos.')  else  str_tmp <- paste0(Sys.time(), ' - ', minutos/60, ' horas.')
        print(str_tmp); write(str_tmp, file = s_Fic_submit_log, append = TRUE)
        minutos_pend <- (as.double((proc.time() - systime_ini2)['elapsed'])/60) * ( (NUM_BLOQUES^2) / ((numBatch-1) * NUM_BLOQUES + numSubBatch)  -  1)
        if(minutos_pend < 60) print(paste0('Faltan aprox. ', minutos_pend, ' minutos.')) else print(paste0('Faltan aprox. ', minutos_pend/60, ' horas.'))
      }
      if(exists("mean_y_pred")) rm(mean_y_pred)
      rm(testset, dt_all, X, y_pred); gc()
    }
    minutos <- as.double((proc.time() - systime_ini2)['elapsed'])/60
    if(minutos < 60) str_tmp <- paste0(Sys.time(), ' - (', (numBatch) , '/', NUM_BLOQUES, ')', minutos, ' minutos.')  else  str_tmp <- paste0(Sys.time(), ' - (', (numBatch) , '/', NUM_BLOQUES, ')', minutos/60, ' horas.')
    print(str_tmp); write(str_tmp, file = s_Fic_submit_log, append = TRUE)
    minutos_pend <- (as.double((proc.time() - systime_ini2)['elapsed'])/60) * ( NUM_BLOQUES / numBatch  -  1)
    if(minutos_pend < 60) print(paste0('Faltan aprox. ', minutos_pend, ' minutos.')) else print(paste0('Faltan aprox. ', minutos_pend/60, ' horas.'))
  }
}
# ----------------------------------------------------------------
