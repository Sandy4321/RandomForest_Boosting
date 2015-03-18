library(ISLR)
head(Hitters)
Hitters = Hitters[-which(is.na(Hitters$Salary)), ]
head(Hitters)
#Andy Allanson  is now removed since he had 'na' for Salary
Hitters$Salary = log(Hitters$Salary)

#(b)
#if we were to use random sample for training set, we do
#set.seed(1)
#train = sample(dim(Hitters)[1], 200)
#Hitters.train = Hitters[train, ]
#Hitters.test = Hitters[-train, ]
#however, in this problem we want first 200 observations
#to be the training set
train = 1:200
Hitters.train = Hitters[train, ]
Hitters.test = Hitters[-train, ]

#(c)
library(gbm)
set.seed(103)
pows = seq(-10, -0.2, by = 0.1)
#pows = -10, -9.9, ..., -.3, -.2
lambdas = 10^pows
train.errors = rep(NA, length(lambdas))
test.errors = rep(NA, length(lambdas))
for (i in 1:length(lambdas)) {
    boost.hitters = gbm(Salary ~ ., data = Hitters.train, distribution = "gaussian", 
        n.trees = 1000, shrinkage = lambdas[i])
    train.pred = predict(boost.hitters, Hitters.train, n.trees = 1000)
    test.pred = predict(boost.hitters, Hitters.test, n.trees = 1000)
    train.errors[i] = mean((Hitters.train$Salary - train.pred)^2)
    test.errors[i] = mean((Hitters.test$Salary - test.pred)^2)
}
#distribution="gaussian" since this is a regression problem
# if it were a bi-nary classification problem, we would use
#distribution="bernoulli"

plot(lambdas, train.errors, type = "b", xlab = "Shrinkage", ylab = "Train MSE", 
    col = "blue", pch = 20)


#(d)
plot(lambdas, test.errors, type = "b", xlab = "Shrinkage", ylab = "Test MSE", 
    col = "red", pch = 20)
> min(test.errors)
#[1] 0.2500707
> lambdas[which.min(test.errors)]
#[1] 0.1258925
#Minimum test error is obtained at lambda=.126


#(f)
boost.best = gbm(Salary ~ ., data = Hitters.train, distribution = "gaussian", 
    n.trees = 1000, shrinkage = lambdas[which.min(test.errors)])
summary(boost.best)

#    rel.inf
#CAtBat       CAtBat 19.59396208
#CWalks       CWalks  8.78206751
#PutOuts     PutOuts  8.10276136
#CRBI           CRBI  6.89795983
#Walks         Walks  6.58887185
#CHits         CHits  6.14738269
#Assists     Assists  6.01405476
#AtBat         AtBat  5.46299056
#CHmRun       CHmRun  5.31417533
#Years         Years  4.69574683
#HmRun         HmRun  4.08804336
#Hits           Hits  3.92001397
#RBI             RBI  3.84376214
#CRuns         CRuns  3.81326364
##Errors       Errors  2.89766152
#Runs           Runs  2.76330522
#Division   Division  0.56418149
#NewLeague NewLeague  0.45882406
#League       League  0.05097179

#CAtBat, CWalks, and PutOuts are the three most important variables



#(g)
library(randomForest)
set.seed(2)
rf.hitters = randomForest(Salary ~ ., data = Hitters.train, ntree = 500, mtry = 19)
rf.pred = predict(rf.hitters, Hitters.test)
mean((Hitters.test$Salary - rf.pred)^2)
#.22978 is the test MSE for bagging, which is lower than
#0.2500707, which is the lowest value for boosting
