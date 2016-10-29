# options(echo = FALSE) # ECHO OFF
print('#######################################################################################################')
print('# Outbrain Click Prediction - JJTZ 2016')
print('#######################################################################################################')
### Inicialización (setwd() y rm() y packages):

# setwd(getwd())
try(setwd('C:/Users/jtoharia/Dropbox/AFI_JOSE/Kaggle/Outbrain'), silent=TRUE)
try(setwd('C:/Personal/Dropbox/AFI_JOSE/Kaggle/Outbrain'), silent=TRUE)
rm(list = ls()) # Borra todos los elementos del entorno de R.

# install.packages("data.table")
suppressMessages(library(data.table))



# Leer trainset:
trainset <- fread("C:/Users/jtoharia/Downloads/Kaggle_Outbrain/clicks_train.csv")
# trainset <- fread("../input/clicks_train.csv")
setkey(trainset, "ad_id")

# head(trainset)
# str(trainset)
# summary(trainset)

# Proporción de clicks (global):
prob_click_global <- mean(trainset$clicked)

# Crear primer submit con las frecuencias como prob.:
probs_ads <- trainset[, .(prob = mean(clicked) ), by = ad_id]

# head(probs_ads)

# Leer testset:
testset <- fread( "C:/Users/jtoharia/Downloads/Kaggle_Outbrain/clicks_test.csv")
# testset <- fread( "../input/clicks_test.csv")
setkey(testset, "ad_id")

# head(testset)
# str(testset)
# summary(testset)

prob_click_global_ads_no_en_test <- mean(trainset$clicked[!(trainset$ad_id %in% testset$ad_id)])
prob_click_global_ads_en_test <- mean(trainset$clicked[trainset$ad_id %in% testset$ad_id])

# # Free memory:
# rm(trainset)
# gc() # Garbage collector

# Añadir las probs de los ads:
testset <- merge(testset, probs_ads, all.x = T, by = "ad_id") # by ad_id

# Predecir los NAs (con prob_click_global_ads_en_test) :
testset[is.na(testset$prob), prob := prob_click_global_ads_en_test]

# summary(testset)

# PREPARACIÓN DEL SUBMIT:
# length(unique(testset$display_id)) == 6.245.533 rows

# Para conseguir los Ads de mayor prob a menor prob, se pone la "key" al revés:
testset[!is.na(testset$prob), prob := 1 - prob] # Ahora prob es la prob de "no click"!
# Ahora usamos ese orden para crear el submit:
setkey(testset, "prob")
mi_tiempo <- system.time({
  submitset <- testset[, .(ad_id = paste(ad_id, collapse=" ")), by = display_id] # 85 secs
})
print(mi_tiempo['elapsed'])

# head(submitset)

# Ordenamos por display_id:
setkey(submitset,"display_id")

# Guardamos fichero:
mi_tiempo <- system.time({
  write.table(submitset, file = "submitset.csv", row.names = F, quote = FALSE, sep = ",")
})
print(mi_tiempo['elapsed'])
