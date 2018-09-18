library(dplyr)
library(ggplot2)

train = read.csv("train.csv",stringsAsFactors = F)
test = read.csv("test.csv",stringsAsFactors = F)
test$renewal = NA
data = rbind(train,test)

summary(data)

data %>% ggplot(aes(x = Income )) +
  scale_y_continuous(label=scales::comma) +
  scale_x_continuous(labels = scales::comma) +
  geom_histogram(bins = 50) 

data %>% ggplot(aes(x = NA,y = Income )) +
  scale_y_continuous(label=scales::comma) +
  geom_boxplot()
