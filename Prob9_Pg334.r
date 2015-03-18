library(ISLR)
attach(OJ)
set.seed(1)
train = sample(dim(OJ)[1], 800)
#dim(OJ) outputs 
#[1] 1070   18
#so dim(OJ)[1]=1070, dim(OJ)[2]=18 since there's 18 variables

OJ.train = OJ[train, ]
OJ.test = OJ[-train, ]

#(b)
oj.tree = tree(Purchase ~ ., data = OJ.train)
summary(oj.tree)
#output
#Classification tree:
#tree(formula = Purchase ~ ., data = OJ.train)
#Variables actually used in tree construction:
#[1] "LoyalCH"       "PriceDiff"     "SpecialCH"    
#[4] "ListPriceDiff"
#Number of terminal nodes:  8 
#Residual mean deviance:  0.7305 = 578.6 / 792 
#Misclassification error rate: 0.165 = 132 / 800 

Error rate is .165 and has 8 terminal nodes

#(c)
oj.tree

#output1) root 800 1064.00 CH ( 0.61750 0.38250 )  
#   2) LoyalCH < 0.508643 350  409.30 MM ( 0.27143 0.72857 )  
#     4) LoyalCH < 0.264232 166  122.10 MM ( 0.12048 0.87952 )  
 #      8) LoyalCH < 0.0356415 57   10.07 MM ( 0.01754 0.98246 ) *
#       9) LoyalCH > 0.0356415 109  100.90 MM ( 0.17431 0.82569 ) *
#     5) LoyalCH > 0.264232 184  248.80 MM ( 0.40761 0.59239 )  
#      10) PriceDiff < 0.195 83   91.66 MM ( 0.24096 0.75904 )  
 #       20) SpecialCH < 0.5 70   60.89 MM ( 0.15714 0.84286 ) *
#        21) SpecialCH > 0.5 13   16.05 CH ( 0.69231 0.30769 ) *
#      11) PriceDiff > 0.195 101  139.20 CH ( 0.54455 0.45545 ) *
#   3) LoyalCH > 0.508643 450  318.10 CH ( 0.88667 0.11333 )  
#     6) LoyalCH < 0.764572 172  188.90 CH ( 0.76163 0.23837 )  
 #     12) ListPriceDiff < 0.235 70   95.61 CH ( 0.57143 0.42857 ) *
#      13) ListPriceDiff > 0.235 102   69.76 CH ( 0.89216 0.10784 ) *
#     7) LoyalCH > 0.764572 278   86.14 CH ( 0.96403 0.03597 ) *

#At terminal node 10, splitting variable is PriceDiff with splitting value
#.195. There are 83 points in the subtree below this node. The deviance for all
#points contained in the region below this node is 91.66. It is not a terminal
#node because it does not have an * next to it. The prediction at this node is 
#Sales=MM. About 24.1% of points in this node have Sales value of CH, and
#75.9% have Sales value of MM

#(d)
plot(oj.tree)
text(oj.tree, pretty = 0)

#LoyalCH is the most important variable of the tree, in fact top 3 nodes 
#contain LoyalCH. If LoyalCH<0.264, the tree predicts MM. If LoyalCH>0.508, 
#the tree predicts CH. For intermediate values of LoyalCH, 
#the decision also depends on the value of PriceDiff and SpecialCH.

#(e)
oj.pred = predict(oj.tree, OJ.test, type = "class")
#predicted test labels
table(OJ.test$Purchase, oj.pred)
#confusion matrix

#    oj.pred
#      CH  MM
#  CH 147  12
#  MM  49  62


#(f)
cv.oj = cv.tree(oj.tree, FUN = prune.tree)
cv.oj

#output:
#$size
#[1] 8 7 6 5 4 3 2 1

#$dev
#[1]  689.1001  685.8030  654.9314  653.7774  666.8890
#[6]  721.2494  733.6936 1066.6499

#$k
#[1]      -Inf  11.20965  14.72877  17.88334  23.55203
#[6]  38.37537  43.02529 337.08200

#$method
#[1] "deviance"

#attr(,"class")
#[1] "prune"         "tree.sequence"

#optimal size is 5 since that has the lowest dev value



#(g)
plot(cv.oj$size, cv.oj$dev, type = "b", xlab = "Tree Size", ylab = "Deviance")

#(h)
#size of 5 gives lowest cross-validation error

#(i)
oj.pruned = prune.tree(oj.tree, best = 5)

#(j)
summary(oj.pruned)

#Classification tree:
#snip.tree(tree = oj.tree, nodes = 4:5)
#Variables actually used in tree construction:
#[1] "LoyalCH"       "ListPriceDiff"
#Number of terminal nodes:  5 
#Residual mean deviance:  0.7829 = 622.4 / 795 
#Misclassification error rate: 0.1825 = 146 / 800 

#Training error of .1825 is greater than .165 of the original tree


#(k)
pred.unpruned = predict(oj.tree, OJ.test, type = "class")
misclass.unpruned = sum(OJ.test$Purchase != pred.unpruned)
misclass.unpruned/length(pred.unpruned)
#output is .226

> pred.pruned = predict(oj.pruned, OJ.test, type = "class")
> misclass.pruned = sum(OJ.test$Purchase != pred.pruned)
> misclass.pruned/length(pred.pruned)
#output is .259
#pruned tree has higher test error of .259 than unpruned's .226

