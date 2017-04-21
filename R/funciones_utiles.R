####################################################
#### Funciones útiles:
####################################################
#----------------------------------------------------------------------------
# ### jjfmt(): puntos para millares y comas para decimales
jjfmt <- function(minum)
{
  return(format(minum, scientific = F, decimal.mark = ',', big.mark = '.'))
}
#----------------------------------------------------------------------------
# ### check_Go(): stop() si no existe el fichero Proyecto.s & '.GO'
# ###   - Proyecto.s (string): Nombre del proyecto/fichero (sin ".GO")
check_Go <- function(Proyecto.s)
{
  if(!file.exists(paste0(Proyecto.s, '.GO')))
  {
    try(write("Buh!", file = paste0(Proyecto.s, '.stopped')), silent = TRUE)
    stop(paste0('Fichero <', Proyecto.s, '.GO', '> NO encontrado. Detenemos el proceso...'))
  }
}
#----------------------------------------------------------------------------
# ### init_tiempo(): Inicializar campos de la variable mi_tiempo_total
# ### NOTA: Esta función no crea ningún registro.
# ###       Sólo añade campos, si hace falta, y devuelve df vacío, si hace falta.
# ###   - mi_tiempo_tot (data.frame): puede ser mi_tiempo_total o NULL para crearla nueva
# ###   - i_bResetTime (boolean): TRUE para vaciar mi_tiempo_total
init_tiempo <- function(mi_tiempo_tot = NULL, i_bResetTime = FALSE)
{
  # Usamos la variable global "mi_tiempo_total", si existe:
  if(is.null(mi_tiempo_tot) & exists('mi_tiempo_total'))
  {
    mi_tiempo_tot <- mi_tiempo_total
  }
  bTmp <- i_bResetTime
  if(!bTmp)    bTmp <- is.null(mi_tiempo_tot)
  if(!bTmp)    bTmp <- (nrow(mi_tiempo_tot) == 0)
  if(bTmp) # if(i_bResetTime | is.null(mi_tiempo_tot) | nrow(mi_tiempo_tot) == 0)
  {
    # Creamos mi_tiempo_tot, vacío, con más campos:
    mi_tiempo_tot <- data.frame(proc = character(), segundos = numeric(), to_str = character(),
                                fich = character(), fecha = numeric(),
                                ret.val = numeric(), num.errs = numeric(),
                                stringsAsFactors = FALSE)
    # Cambiamos la clase del campo fecha (no sé crear ese objeto vacío):
    mi_tiempo_tot$fecha <- as.POSIXct(mi_tiempo_tot$fecha, origin = "1970-01-01 00:00.00 UTC")
  } else
  {
    # Añadimos más campos, si no están, a mi_tiempo_tot:
    if(!('fich' %in% colnames(mi_tiempo_tot)))
      mi_tiempo_tot[, 'fich'] <- ''
    if(!('fecha' %in% colnames(mi_tiempo_tot)))
      mi_tiempo_tot[, 'fecha'] <- Sys.time()
    if(!('ret.val' %in% colnames(mi_tiempo_tot)))
      mi_tiempo_tot[, 'ret.val'] <- 0
    if(!('num.errs' %in% colnames(mi_tiempo_tot)))
      mi_tiempo_tot[, 'num.errs'] <- 0
  }
  if(exists('mi_tiempo_total'))
  {
    mi_tiempo_total <- mi_tiempo_tot
  }
  # Y devolvemos data.frame:
  return(mi_tiempo_tot)
}
#----------------------------------------------------------------------------
# ### add_tiempo(): Devuelve data.frame con los tiempos de cada llamada:
# ###               data.frame( proc = character(), segundos = numeric(), to_str = character(), stringsAsFactors = FALSE)
add_tiempo <- function(mi_tiempo, proc_txt = "", mi_tiempo_tot = NULL,
                       ret_val = 0, num_errs = 0, fich = "", fecha = Sys.time(),
                       b_print = FALSE, b_print_acum = FALSE, b_print_una_linea = TRUE)
{
  # Creamos data.frame "mi_tiempo_tot", vacío, si no se ha creado todavía:
  if(is.null(mi_tiempo_tot))
    mi_tiempo_tot <- init_tiempo()
  if(is.null(mi_tiempo))
    return(mi_tiempo_tot)
  # Creamos campo "segundos":
  mi_reg.s <- mi_tiempo[3]
  # Creamos campo "to_str":
  if(mi_reg.s < 60)
  { to_str <- paste0(round(mi_reg.s, 1), ' segundos')
  } else { if(mi_reg.s < 3600)
    { to_str <- paste0(round(mi_reg.s/60, 1), ' minutos')
    } else { if(mi_reg.s < 86400) { to_str <- paste0(round(mi_reg.s/3600, 1), ' horas') # 3600 * 24 = 86400
      } else { to_str <- paste0(round(mi_reg.s/86400, 2), ' días') } } }
  to_str <- paste0(trimws(paste0("Tiempo ", proc_txt), "right"), ": ", trimws(to_str, "left"))
  
  # Añadimos el registro usando la estructura de mi_tiempo_tot, por si tuviera otros campos:
  # Insertamos nuevo registro:
  mi_tiempo_tot[nrow(mi_tiempo_tot) + 1, 'proc'] <- proc_txt
  # Actualizamos último registro:
  mi_tiempo_tot[nrow(mi_tiempo_tot), 'segundos'] <- as.numeric(mi_reg.s)
  mi_tiempo_tot[nrow(mi_tiempo_tot), 'to_str']   <- to_str
  mi_tiempo_tot[nrow(mi_tiempo_tot), 'ret.val']  <- ret_val
  mi_tiempo_tot[nrow(mi_tiempo_tot), 'num.errs'] <- num_errs
  mi_tiempo_tot[nrow(mi_tiempo_tot), 'fich'] <- fich
  mi_tiempo_tot[nrow(mi_tiempo_tot), 'fecha'] <- fecha
  
  # Mostramos tiempo(s):
  print.s <- ""
  print_acc.s <- ""
  if(b_print)
  {
    print.s <- mi_tiempo_tot[nrow(mi_tiempo_tot),]$to_str
  }
  if(b_print_acum & nrow(mi_tiempo_tot) > 1)
  {
    mi_reg.s <- sum(mi_tiempo_tot$segundos)
    # Creamos campo "to_str":
    if(mi_reg.s < 60)
    { to_str <- paste0(round(mi_reg.s, 1), ' segundos')
    } else { if(mi_reg.s < 3600)
      { to_str <- paste0(round(mi_reg.s/60, 1), ' minutos')
      } else { if(mi_reg.s < 86400) { to_str <- paste0(round(mi_reg.s/3600, 1), ' horas') # 3600 * 24 = 86400
        } else { to_str <- paste0(round(mi_reg.s/86400, 2), ' días') } } }
    to_str <- paste0("Tiempo acumulado: ", trimws(to_str, "left"))
    if(b_print & b_print_una_linea)
      print.s <- paste0(print.s, "  -  ", to_str)
    else
      print_acc.s <- to_str
  }
  if(nchar(print.s) != 0) print(print.s)
  if(nchar(print_acc.s) != 0) print(print_acc.s)
  return(mi_tiempo_tot)
}
#----------------------------------------------------------------------------
# ### relevancia(): Relevancia de una variable categórica respecto al target (Test Chi2)
relevancia <- function(VariableTarget, VariableCategorica)
{
  levels <- levels(VariableCategorica)
  colors <- c()
  for (i in 1:(length(levels)))
  {
    TABLA <- table(VariableTarget,VariableCategorica==levels[i])
    chi <- chisq.test(TABLA)
    if(chi$p.value < 0.05)
    {
      colors <- c(colors, "green")
    }else
    {
      colors <- c(colors, "gray")
    }
  }
  TABLA <- table(VariableTarget, VariableCategorica)
  plot <- barplot(100 * TABLA[2,] / (TABLA[1,] + TABLA[2,]), ylim = c(0,100), col = colors, cex.names = 0.6)
  text(x = plot, y = 5 + 100 * TABLA[2,] / (TABLA[1,] + TABLA[2,]), labels = paste(round(100 * TABLA[2,] / (TABLA[1,] + TABLA[2,]), 2), "%", sep = ""))
  abline(h = 100 * prior, col = "red")
  return(TABLA)
}
#----------------------------------------------------------------------------
# ### woe_iv(): Análisis de la capacidad predictiva de las variables
# ###           Weigth of Evidence (WOE) and  Information Value (IV)
woe_iv <- function(VariableTarget, VariableCategorica)
{
  Tabla_WOE <- table(VariableCategorica, VariableTarget)
  DF_WOE <- data.frame(FRACASOS = Tabla_WOE[,1], EXITOS = Tabla_WOE[,2])
  DF_WOE$EXITOS_PORC <- DF_WOE$EXITOS / sum(DF_WOE$EXITOS)
  DF_WOE$FRACASOS_PORC <- DF_WOE$FRACASOS / sum(DF_WOE$FRACASOS)
  DF_WOE$WOE <- log(DF_WOE$EXITOS_PORC / DF_WOE$FRACASOS_PORC)
  DF_WOE$IV <- (DF_WOE$EXITOS_PORC - DF_WOE$FRACASOS_PORC) * DF_WOE$WOE
  DF_WOE
  return(DF_WOE)
}
#' #----------------------------------------------------------------------------
#' #' Compute the average precision at k
#' #' 
#' #' This function computes the average precision at k
#' #' between two sequences
#' #' 
#' #' @param k max length of predicted sequence     (12  en  Kaggle-Outbrain)
#' #' @param actual ground truth set (vector)       (ad_id_target  en  Kaggle-Outbrain)
#' #' @param predicted predicted sequence (vector)  (ad_id_predicted  en  Kaggle-Outbrain)
#' #' @export
#' apk_slow <- function(k, actual, predicted)
#' {
#'   score <- 0.0
#'   cnt <- 0.0
#'   i_max <- min(k,length(predicted))
#'   for (i in 1:i_max)
#'   {
#'     if (predicted[i] %in% actual && !(predicted[i] %in% predicted[0:(i-1)]))
#'     {
#'       cnt <- cnt + 1
#'       score <- score + cnt/i
#'     }
#'   }
#'   score <- score / min(length(actual), k)
#'   return(score)
#' }
#' #----------------------------------------------------------------------------
#' #' Compute the mean average precision at k
#' #' 
#' #' This function computes the mean average precision at k
#' #' of two lists of sequences.
#' #' 
#' #' @param k max length of predicted sequence                   (12  en  Kaggle-Outbrain)
#' #' @param actual_list list of ground truth sets (vectors)      (ad_id_targets_list  en  Kaggle-Outbrain)
#' #' @param predicted_list list of predicted sequences (vectors) (ad_id_predicteds_list  en  Kaggle-Outbrain)
#' #' @export
#' mapk_slow <- function (k, actual_list, predicted_list)
#' {
#'   if( length(actual_list)==0 || length(predicted_list)==0 )
#'     return(0.0)
#'   scores <- rep(0, length(actual_list))
#'   i_max <- length(scores)
#'   for (i in 1:i_max)
#'   {
#'     scores[i] <- apk(k, actual_list[[i]], predicted_list[[i]])
#'   }
#'   return(mean(scores))
#' }
#----------------------------------------------------------------------------
#' #' Compute the average precision at k
#' #'
#' #' This function computes the average precision at k
#' #' between two sequences
#' #'
#' #' @param k max length of predicted sequence     (12  en  Kaggle-Outbrain)
#' #' @param actual ground truth set (vector)       (ad_id_target  en  Kaggle-Outbrain)
#' #' @param predicted predicted sequence (vector)  (ad_id_predicted  en  Kaggle-Outbrain)
#' #' @export
#' apk <- function (k, actual, predicted)
#' {
#'   predicted <- head(predicted, k)
#'   is.new <- !duplicated(predicted)
#'   is.relevant <- predicted %in% actual & is.new
#'   score <- sum(cumsum(is.relevant) * is.relevant / seq_along(predicted)) / min(length(actual), k)
#'   return(score)
#' }
#' apk_nodup <- function (k, actual, predicted)
#' {
#'   predicted <- head(predicted, k)
#'   # is.new <- !duplicated(predicted) # JJTZ - No hay ad_id repetidos
#'   is.relevant <- predicted %in% actual # & is.new # JJTZ - No hay ad_id repetidos
#'   score <- sum(cumsum(is.relevant) * is.relevant / seq_along(predicted)) / min(length(actual), k)
#'   return(score)
#' }
#' #----------------------------------------------------------------------------
#' #' Compute the mean average precision at k
#' #' 
#' #' This function computes the mean average precision at k
#' #' of two lists of sequences.
#' #' 
#' #' @param k max length of predicted sequence                   (12  en  Kaggle-Outbrain)
#' #' @param actual_list list of ground truth sets (vectors)      (ad_id_targets_list  en  Kaggle-Outbrain)
#' #' @param predicted_list list of predicted sequences (vectors) (ad_id_predicteds_list  en  Kaggle-Outbrain)
#' #' @export
#' mapk <- function (k, actual_list, predicted_list)
#' {
#'   if( length(actual_list)==0 || length(predicted_list)==0 )
#'     return(0.0)
#'   scores <- vector(mode="numeric",  length(actual_list)) # <- rep(0, length(actual_list))
#'   i_max <- length(actual_list)
#'   for (i in 1:i_max)
#'   {
#'     scores[i] <- apk(k, actual_list[[i]], predicted_list[[i]])
#'   }
#'   return(mean(scores))
#' }
mapk_nodup <- function (k, actual_list, predicted_list, i_max = length(actual_list), j_max = length(predicted_list))
{
  if( i_max==0 || j_max==0 )
    return(0.0)
  # i_max <- length(actual_list)
  scores <- vector(mode="numeric",  i_max) # <- rep(0, i_max)
  for (i in 1:i_max)
  {
    actual <- actual_list[[i]]
    predicted <- head(predicted_list[[i]], k)
    # is.new <- !duplicated(predicted) # JJTZ - No hay ad_id repetidos
    if(length(actual) == 1)
    {
      is.relevant <- predicted == actual # & is.new # JJTZ - No hay ad_id repetidos
      scores[i] <- sum(cumsum(is.relevant) * is.relevant / seq_along(predicted)) # / min(length(actual), k)
    } else
    {
      is.relevant <- predicted %in% actual # & is.new # JJTZ - No hay ad_id repetidos
      scores[i] <- sum(cumsum(is.relevant) * is.relevant / seq_along(predicted)) / min(length(actual), k)
    }
  }
  return(mean(scores))
}
# Versión algo más rápida para Kaggle_Outbrain:
mapk_nodup_only_first_12 <- function (actual_vector, predicted_list, i_max = length(actual_vector), j_max = length(predicted_list))
{ # - El vector actual_vector es un vector de elementos individuales (i.e. sólo se valida la primera recomendación)
  # if( i_max==0 || j_max==0 )  return(0.0)
  # i_max <- length(actual_vector)
  scores <- vector(mode="numeric",  i_max) # <- rep(0, i_max)
  for (i in 1:i_max)
  {
    # actual <- actual_vector[i]
    # predicted <- head(predicted_list[[i]], k)
    predicted <- predicted_list[[i]] # Nunca tienen más de 12 ad_id (Outbrain testset)
    # is.new <- !duplicated(predicted) # JJTZ - No hay ad_id repetidos

    is.relevant <- predicted == actual_vector[i] # & is.new # JJTZ - No hay ad_id repetidos
    scores[i] <- sum(cumsum(is.relevant) * is.relevant / seq_along(predicted)) # / min(length(actual), k)
  }
  return(mean(scores))
}
# # install.packages("Metrics")
# library('Metrics')
# # install.packages("rbenchmark")
# library('rbenchmark')
# 
# actual <- 1:20000
# predicted <- c(1:20, 200:600, 900:1522, 14000:32955)
# benchmark(replications=10,
#           apk(5000, actual, predicted),
#           Metrics::apk(5000, actual, predicted),
#           columns= c("test", "replications", "elapsed", "relative"))
# # actual_list <- vector(mode="list", 5)
# actual_list <- rep(list(actual), 50)
# predicted_list <- rep(list(predicted), 50)
# benchmark(replications=10,
#           # apk_slow(5000, actual, predicted),
#           # apk(5000, actual, predicted),
#           # apk2(5000, actual, predicted),
#           # mapk_slow(25, actual_list, predicted_list),
#           # mapk(25, actual_list, predicted_list),
#           # mapk2(25, actual_list, predicted_list),
#           mapk(12, actual_list, predicted_list),
#           mapk_nodup(12, actual_list, predicted_list),
#           columns= c("test", "replications", "elapsed", "relative"))
# # identical(apk(5000, actual, predicted),
# #           apk2(5000, actual, predicted))
# identical(mapk(12, actual_list, predicted_list),
#           mapk_12(actual_list, predicted_list))
# identical(mapk(12, actual_list, predicted_list),
#           mapk_nodup(12, actual_list, predicted_list))
#----------------------------------------------------------------------------
