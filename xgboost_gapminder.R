
# paquetes ----------------------------------------------------------------

require(Matrix)
require(xgboost)
require(gapminder)

# Datas -----------------------------------------------------------------------

gap <- gapminder[,c(3,4,6)]
gap <- as.matrix(gap)
gap <- as(as.matrix(gapminder) ,"dgCMatrix")

gap2 <- as.numeric(gapminder$lifeExp)


# modelamiento ------------------------------------------------------------

lista <- list(data = gap,label = gap2)

param <- list(max_depth = 2, eta = 1, verbose = 0, nthread = 2,
              objective = "reg:squarederror")

dgap <- xgb.DMatrix(lista$data, label = lista$label)

watchlist <- list(train = dgap, eval = dgap)

bst <- xgb.train(params = param, dgap, nrounds = 2, watchlist =  watchlist)


