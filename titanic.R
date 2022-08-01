library(dplyr)

data <- read.csv('/Users/namithamariacherian/Projects_R/titanic/train.csv')
View(data)
str(data)

#change columns to factor 
data[,'Sex'] <- as.factor(data[,'Sex'])
data[,'Embarked'] <- as.factor(data[,'Embarked'])
data[, 'Survived'] <- as.factor(data[,'Survived'])
data[,'Pclass'] <- as.factor(data[,'Pclass'])

#counting the values in each column
table(data['Embarked']) #There are 2 rows with missing values
table(data['SibSp'])
table(data['Parch'])

#finding NA values

sum(is.na(data$PassengerId))
sum(is.na(data$Name))
sum(is.na(data$Age)) #177 out 891 have missing values
sum(is.na(data$SibSp))
sum(is.na(data$Parch))
sum(is.na(data$Fare))

#Manipulating data to adjust for missing and Na values
data[data$Embarked == '',]$Embarked <- 'S' # Added the value as S as it has the highest frequency
summary(data$Age)
data[is.na(data$Age),]$Age <- 28 #Replaced NA value with the median value of 28

#Visualizing data to find outliers
hist(data$Age)
hist(data$SibSp)
hist(data$Parch)
hist(data$Fare)

#Quality control checks
xtabs(~ Survived + Sex, data=data)
xtabs(~ Survived + Pclass, data= data)
xtabs(~ Survived + Embarked, data= data)


#Training using Logistic Regression
logistic <- glm(Survived ~ Sex , data=data , family = 'binomial')
summary(logistic)


#Log of odds of survival for Female is 1.0566
#Log of odds of survival for Male / Log of odds of survival for Female is -2.5137
#i.e If the passenger is male he is -2.5 times less likely to survive
#Good model as residual are distributed symmetrically around 0
#Very small p-value, hence the results are statistically significant

logistic_1 <- glm(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data=data, family=binomial())
summary(logistic_1)

#We can see that the features Parch, Fare and Embared have results with lof p-value. Hence they are not statistically significant

logistic_2 <- glm(Survived ~ Pclass + Sex + Age + SibSp, data=data, family=binomial())
summary(logistic_2)


#Calculating R-square for the model logistic_2
ll.null <- (logistic_2$null.deviance) /-2
ll.proposed <- (logistic_2$deviance) /-2

r_square <- (ll.null-ll.proposed)/ll.null
r_square

p_value <- 1 - pchisq(2*(ll.proposed - ll.null), df=(length(logistic_2$coefficients)-1))
p_value

#r_square is 0.33 is not the best fit model and as p-value is 0 and is hence significant

#Predicting using logistic_2 model

prob_survive <- logistic_2$fitted.values
actual_survive <- data$Survived

data_1 <- data.frame(prob_survive, actual_survive)
data_1 <- data_1[order(data_1$prob_survive, decreasing = FALSE),]
data_1$rank <- 1:nrow(data_1)
head(data_1)

#Plotting
library(ggplot2)

ggplot(data=data_1, aes(x=rank, y=prob_survive))+
  geom_point(aes(color=actual_survive), alpha=1, shape=4, stroke=2)+
  xlab("Index") +
  ylab("Predicted probability of Survival")

#Testing using logistic regression model

data_2 <- read.csv('/Users/namithamariacherian/Projects_R/titanic/test.csv')
View(data_2)

#Pre processing testing data
data_2[,'Sex'] <- as.factor(data_2[,'Sex'])
data_2[,'Embarked'] <- as.factor(data_2[,'Embarked'])
data_2[, 'Survived'] <- as.factor(data_2[,'Survived'])
data_2[,'Pclass'] <- as.factor(data_2[,'Pclass'])


sum(is.na(data_2$PassengerId))
sum(is.na(data_2$Name))
sum(is.na(data_2$Age)) #86 out 418 have missing values
sum(is.na(data_2$SibSp))
sum(is.na(data_2$Parch))
sum(is.na(data_2$Fare)) #1 NA value, But we are not using this feature for the model


summary(data_2$Age)
data_2[is.na(data_2$Age),]$Age <- 27 #Replaced NA value with the median value of 27

str(data_2)

prediction_1 <- predict(logistic_2, data_2, type='response') #Change numeric variables to fators. aka data preprocessing
prediction_1
data_2$prediction_1 <- prediction_1

Survived_1 <- list()

Pred_survive <- c(ifelse(data_2$prediction_1 >  0.5, Survived_1 <-1, Survived_1 <-0))

Pred_survive
data_2$Survived<- Pred_survive

data_2

#Saving my prediction
lr_data <- data.frame(PassengerId = data_2$PassengerId, Survived = data_2$Survived)
View(lr_data)
write.csv(lr_data, '/Users/namithamariacherian/Projects_R/titanic/lr_prediction.csv')



