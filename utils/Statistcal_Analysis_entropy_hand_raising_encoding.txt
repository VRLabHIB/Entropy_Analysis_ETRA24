library(Matrix)
library(nlme)

setwd('C:/Users/Stark/Documents/Promotion/Entropy_Analysis/data/big_hand_raising/')

df0 <- read.csv2('2024-01-20_big_entropy_allAOIs_30sec_noNA.csv', sep=',')

df0$ID <- as.numeric(df0$ID)
df0$transition_entropy <- as.numeric(df0$transition_entropy)
df0$stationary_entropy <- as.numeric(df0$stationary_entropy)
df0$event_new <- as.numeric(df0$event_new)

df0$hand20 <- factor(ifelse(df0$Condition_hand==20,1,0))
df0$hand35 <- factor(ifelse(df0$Condition_hand==35,1,0))
df0$hand65 <- factor(ifelse(df0$Condition_hand==65,1,0))
df0$hand80 <- factor(ifelse(df0$Condition_hand==80,1,0))

df0$sitting <- factor(ifelse(df0$Condition== "A" | 
                               df0$Condition== "B" |
                               df0$Condition== "C" |
                               df0$Condition== "D" |
                               df0$Condition== "I" |
                               df0$Condition== "J" |
                               df0$Condition== "K" |
                               df0$Condition== "L" , 1,0 ))

df0$style <- factor(ifelse(df0$Condition== "A" | 
                             df0$Condition== "B" |
                             df0$Condition== "C" |
                             df0$Condition== "D" |
                             df0$Condition== "E" |
                             df0$Condition== "F" |
                             df0$Condition== "G" |
                             df0$Condition== "H" , 1,0 ))



library(Matrix)
library(nlme)
library(MuMIn)
library(dplyr)
library(xtable)

fit1 <- lme(scale(transition_entropy)~event_new, random=~1|ID, df0)
r.squaredGLMM(fit1)

fit2 <- lme(scale(transition_entropy) ~hand20 + hand80 + style + sitting ,random=~1|ID, df0)
r.squaredGLMM(fit2)


fit3 <- lme(scale(transition_entropy) ~ event_new +  hand20 + hand80 + style + sitting, random=~1|ID, df0)
r.squaredGLMM(fit3)
summary(fit3)

res.table <- as.data.frame(coef(summary(fit3)) %>% round(2))
xtable(res.table, type = "latex")

################################################################################
fit4 <- lme(scale(stationary_entropy)~event_new, random=~1|ID, df0)
r.squaredGLMM(fit4)

fit5 <- lme(scale(stationary_entropy) ~hand20 + hand80 + style + sitting ,random=~1|ID, df0)
r.squaredGLMM(fit5)


fit5 <- lme(scale(stationary_entropy) ~ event_new +  hand20 + hand80 + style + sitting, random=~1|ID, df0)
r.squaredGLMM(fit5)
summary(fit4)

res.table2 <- as.data.frame(coef(summary(fit4)) %>% round(2))
xtable(res.table2, type = "latex")