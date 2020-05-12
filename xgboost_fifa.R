#Librería####
library(xgboost)
library(stringr)
#Lectura de datos####
datos <- read.csv(file = "data.csv", header = T, sep = ",", encoding = "UTF-8")

#Variables importantes####
#Analisis para que prediga que hace el costo de un jugador
datos2 <- datos[,c(4,6,8:10,12,14:20,22,27:88)]
datos2 <- na.omit(datos2)

#Modificación de la variable respuesta####
datos2$Value <- as.character(datos2$Value)
Cantidad <- str_sub(datos2$Value, start = -1L)
Cantidad <- str_replace_all(Cantidad,"M","1000000")
Cantidad <- str_replace_all(Cantidad,"K","1000")
Cantidad <- as.numeric(Cantidad)
datos2$Value <- as.numeric(str_sub(datos2$Value, start = 2, end = -2L))*Cantidad/1000000

#TRAIN AND TEST####
n <- length(datos$Value)
set.seed(3)
obs <- sample(x = 1:n, size = round(n*0.7), replace = F)
train <- na.omit(datos2[obs,])
td <- as(object = as.matrix(train[,-6]), Class = 'dgCMatrix')
tl <- train[,6]
dtrain <- xgb.DMatrix(data = td, label = tl)

test <- na.omit(datos2[-obs,])
ed <- as(object = as.matrix(test[,-6]), Class = 'dgCMatrix')
el <- test[,6]
dtest <- xgb.DMatrix(data = ed, label = el)

params <- list(booster = "gblinear", lambda = 0, lambda_bias = 0,
               alpha = 0, objective = "reg:squarederror",
               base_score = 0.5, eval_metric = "rmse")
watchlist <- list(train = dtrain, eval = dtest)

modelo1 <- xgb.train(params = params, data = dtrain, nrounds = 5000,
                     watchlist = watchlist, verbose = 1)
pred_mod1 <- predict(object = modelo1,dtrain)
sum(sqrt((tl-pred_mod1)^2))
cor(tl,pred_mod1)
#graficos#####
grafico <- data.frame(modelo1$evaluation_log)
plot(grafico$iter, grafico$train_rmse, col = 'blue',type = 'l')
lines(grafico$iter, grafico$eval_rmse, col = 'red',type = 'l')

#Importancia de las variables####
imp_m <- xgb.importance(feature_names = colnames(td), model = modelo1)
xgb.ggplot.importance(importance_matrix = imp_m[2:10], xlab = "Relative importance")

#Si solo aplico xgboost♣###
modelo2 <- xgboost(data = td, label = tl, params = params, nrounds = 1000)
predichos <- predict(modelo2, ed)
cor(el,predichos)
