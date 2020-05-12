#CHESS####
# install.packages("xgboost")
# installed.packages("seqinr")
#Librerias####
require(xgboost)
require(dplyr)
datos <- read.csv(file = "games.csv", header = T, sep = ",")
datos$moves <- as.character(datos$moves)

#Particionando la base de datos####
ld <- length(datos$id)
set.seed(2)
obs <- sample(x = ld, size = round(ld*0.75), replace = F)
train <- datos[obs,-c(1,9,11)]
test <- datos[-obs,-c(1,9,11)]

#TRAIN AND TEST####
#Parte TRAIN

train_data <- as.matrix(train[,-c(6)])
train_data <- as(object = train_data, Class = 'dgCMatrix')
train_label <- train[,6]
train_dgC <- xgb.DMatrix(data = train_data, label = train_label)

#Parte TEST
test_data <- as.matrix(test[,-c(6)])
test_data <- as(object = test_data, Class = 'dgCMatrix')
test_label <- test[,6]
test_dgc <- xgb.DMatrix(data = test_data, label = test_label)

#Parametros
param <- list(eval_metric = "error")
watchlist <- list(train = train_dgC, eval = test_dgc)
modelo2 <- xgb.train(data = train_dgC, nrounds = 200,
                     watchlist = watchlist)
modelo2$raw
#MODELO
modelo1 <- xgboost(data = train_data, label = train_label, nrounds = 200,
                   params = param)
pred <- predict(modelo1, test_data)
pred_r <- round(pred)
reales <- as.numeric(test_label)
table(pred_r,reales)

