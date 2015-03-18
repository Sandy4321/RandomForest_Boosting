library(ISLR)
attach(Carseats)
set.seed(1)

train = sample(dim(Carseats)[1], dim(Carseats)[1]/2)
Carseats.train = Carseats[train, ]
Carseats.test = Carseats[-train, ]

library(tree)
tree.carseats = tree(Sales ~ ., data = Carseats.train)
summary(tree.carseats)

plot(tree.carseats)
text(tree.carseats, pretty = 0)

pred.carseats = predict(tree.carseats, Carseats.test)
mean((Carseats.test$Sales - pred.carseats)^2)


#(c) - determine cross-validation
cv.carseats = cv.tree(tree.carseats, FUN = prune.tree)
par(mfrow = c(1, 2))
plot(cv.carseats$size, cv.carseats$dev, type = "b")
plot(cv.carseats$k, cv.carseats$dev, type = "b")

# Best size = 9
pruned.carseats = prune.tree(tree.carseats, best = 9)
par(mfrow = c(1, 1))
plot(pruned.carseats)
text(pruned.carseats, pretty = 0)

pred.pruned = predict(pruned.carseats, Carseats.test)
mean((Carseats.test$Sales - pred.pruned)^2)
#Pruning the tree increases the test MSE to 4.99

#(d) - use Bagging

bag.carseats = randomForest(Sales ~ ., data = Carseats.train, mtry = 10, ntree = 500, 
    importance = T)
#even though we use randomForest function, this is bagging NOT random forest 
#because mtry=10, the same as the number of predictors
#bagging is just a special case of randomForest with mtry=p
bag.pred = predict(bag.carseats, Carseats.test)
mean((Carseats.test$Sales - bag.pred)^2)

importance(bag.carseats)

#By using bagging, the test MSE increases to 2.58. 
#From the importance results, Price, ShelveLoc and Age are the
#three most important predictors of Sales

#(e) 
rf.carseats = randomForest(Sales ~ ., data = Carseats.train, mtry = 5, ntree = 500, 
    importance = T)
#this is random forest, not bagging, since mtry=5, half the total
#number of predictors
rf.pred = predict(rf.carseats, Carseats.test)
mean((Carseats.test$Sales - rf.pred)^2)

importance(rf.carseats)
#Random forest has a MSE of 2.87, lower than the MSE of 2.58 
#with bagging in (c)

#if in collection of bagged trees, if most/all trees use the same
#strong predictor in the top, all the bagged trees will look similar
#and hence the predictions from bagged trees will be similar, be
#highly correlated. But averaging highly correlated quantities does not
#reduce the variance as averaging many UNcorrelated quantities 