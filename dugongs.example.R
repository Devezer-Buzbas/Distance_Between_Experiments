library(R2jags)
library(runjags)
library(minpack.lm)
runjags.options(silent.jags=TRUE, silent.runjags = TRUE)
#-----------------------------------------------------------------
#-----------------------------------------------------------------
# Abbreviations and specifications
#-----------------------------------------------------------------
#
# Study 1
#---------
#   Spre1 := Pre-data methods 
#     variables measured 
#         y  := length of Dugongs (in meters), 
#         x1 := age of Dugongs (in years, discrete with interval of 0.5), 
#         x2 := weight of Dugongs (in kg*(1e-2)). 
#   M1 := Model
#         = log transformed x1, and x2 is used as predictor in the linear
#         regression model:
#         y = beta0 + beta1*log(x1) + beta2*x2 + epsilon
#         epsilon ~ Nor(0, tau), where tau is the precision parameter
#         beta0, beta1, beta2 regression parameters
#   Spost1 := Post-data methods
#         Ordinary Least Squares estimates
#         Point estimates used for prediction
#   Ds1 := Data structure
#         Small data set, data with bias in x1 and x2 recording
#         n1= 20
#
# Study 2
#---------
#   Spre2 := Pre-data methods 
#     variables measured 
#         y  := length of Dugongs (in meters), 
#         x1 := age of Dugongs (in years, discrete with interval of 0.5), 
#   M2 := Model
#         non-linear regression model:
#         y = alpha - beta3*gam^(x1) + epsilon
#         epsilon ~ Nor(0, tau), where tau is the precision parameter
#         alpha, beta, gam regression parameters
#   Spost2 := Post-data methods
#         Bayesian (non-conjugate) analysis with vague priors:
#         alpha ~ Nor(0, 1e-3), beta1 ~ Nor(0,1e-3), tau ~ Gamma(1e-3,1e-3) 
#         Posterior predictive distribution is used for prediction
#   Ds2 := Data structure
#         Large data set, with no error in recording
#         n2 = 100
#-------------------------------------------------------
#-------------------------------------------------------
# Index matrix for permutation studies with last column 
# for reproducibility rate of each permutation
# nob-permutable studies are assigned reproducibility rate
# by convention
# (1:=Study 1, 2:=Study 2, Order: Spre, M, Spost, Ds)
# 
study.ind = matrix(c(rep(c(rep(1,2^3),rep(2,2^3)),2^0),
               rep(c(rep(1,2^2),rep(2,2^2)),2^1),
               rep(c(rep(1,2^1),rep(2,2^1)),2^2),
               rep(c(rep(1,2^0),rep(2,2^0)),2^3),
               rep(0, 2^4)), ncol = 5, nrow = (2^4))
colnames(study.ind) = c("Spre","M", "Spost", "Ds", "Rep")
#-----------------------------------------------------------------
#-----------------------------------------------------------------
# Simulation parameters
#-----------------------------------------------------------------
# MCMC (Gibbs sampler) parameters
# Number of Monte Carlo simulations
m = 1e4
# equal to the number of model parameter vectors simulated with
# one data set is simulated under each parameter vector
# MCMC chain and sampler parameters for JAGS
nChains=1 # number of chains for each data set
burnInSteps=1000 # number o burn-in steps
thinSteps=1  # number of thinning steps 
nIter=2000 # number of total steps to run the chain
#----------------------------------------------------------
#----------------------------------------------------------
# Model parameter spaces (for true parameter values)
min.beta0 = min.beta1 = min.beta2 = 0
max.beta1 = max.beta2 = 1
max.beta0 = 3
min.beta3 = min.alpha = 1
min.gam = 0 
max.alpha = 3 
max.beta3 = 3
max.gam = 1
min.tau = 50; max.tau = 250
min.sigma = 1/sqrt(max.tau)
max.sigma= 1/sqrt(min.tau)
#
# Model parameter spaces are chosen to accommodate 
# results from previous real data analyses and growth curve properties. 
# Structurally, all parameters are greater than zero for growth, and 
# minimum length = 0, maximum length = 3
# Classical example of Carlin and Gelfand (1991) 
# based on a data set from Ratkowsky (1983) estimates alpha = 2.65, beta3 = 0.97
# gam in [0,1] for nonlinear model
#----------------------------------------------------------------
# Sample size
n1 = 20 # small sample size, part of Ds1 
n2 = 100 # large sample size, part of Ds2
#-----------------------------------------------------------------
# Predictor variable bounds 
max.age = 70 # based on life-span (maximum) Dugong age observed in wild
max.weight = 9 # >900kg rarely seen (max observed in wild 1,106kg)
min.weight = 0.2 # based on minimum weight of 20 kg. at birth
max.length = 3 # >3 m rarely seen (maximum observed in wild 4.06m.)
min.length = 1 # based on minimum length of 1 m at birth
#-----------------------------------------------------------------
# Result of interest
#-----------------------------------------------------------------
# Dugongs reach breeding age at maximum age 17.
# Individuals <2.20m are unlikely to breed,
# those >2.50m are likely to breed,
# while the status of individuals (2.20m, 2.50m) is unclear.
# At age 17 years, we take the minimum mean weight to reproduce as 250kg.
# We are interested whether the true model predicts the mean length of 
# a Dugong in >2.50 m at age 17 years, weight 250 kg,
# assigning the "clear to breed" status. If the result from a data analysis
# matches the classification of the true model then 
# we take the result = 1, otherwise result = 0.
length.upper = 2.50
age = 17
weight = 250*1e-2
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
# Simulate parameters and data (predictors x1, x2, and response y) under M1 and M2
#-----------------------------------------------------------------
#-----------------------------------------------------------------
beta0 = beta1 = beta2= alpha = beta3 = gam = rep(NA,m)
data.M1 = data.M2 = vector("list", m)
for ( i in 1:m){
#----------------------------------------------------
# Simulate age uniformly on (0.5, 70)
  x1 = sample(seq(0.5,max.age,0.5), n2, replace = T) # n2 for large sample unbiased x1, Ds2
# Simulate weights  
  # Weight and age in Dugongs are correlated. For illustration we use 
  # correlation ~ 0.6. The following is a logistic relationship
  # between age and weight (from Cherdsukjai et al. 2020) 
  # with adjusted coefficients for desired correlation ~0.6
 x2= 3/(0.9+exp(1-0.28*x1)) # large sample unbiased x2, Ds2
#--------------------------------------------
 # Simulate parameters uniformly and y
  # M1
  y1 = rep(-1,n2)
  while(any(y1<min.length) || any(y1>max.length)){
  beta0.sim  = runif(1, min.beta0, max.beta0) 
  beta1.sim  = runif(1, min.beta1, max.beta1)   
  beta2.sim  = runif(1, min.beta2, max.beta2)    
  sigma.sim = runif(1, min.sigma, max.sigma)
  y1 = beta0.sim + beta1.sim*log(x1) + beta2.sim*x2 + rnorm(n2, 0, sigma.sim)
  # large sample, unbiased data, Ds2
  }
  data.M1[[i]] = data.frame(x1=x1, x2=x2, y=y1) # large sample, unbiased data, Ds2
  beta0[i] = beta0.sim
  beta1[i] = beta1.sim
  beta2[i] = beta2.sim
#
  # M2
  y2 = rep(-1,n2)
  while(any(y2<min.length) || any(y2>max.length)){
    alpha.sim  = runif(1, min.alpha, max.alpha) 
    beta3.sim  = runif(1, min.beta3, max.beta3)   
    gam.sim  = runif(1, min.gam, max.gam)    
    sigma.sim = runif(1, min.sigma, max.sigma)
  y2 = alpha.sim - (beta3.sim)*((gam.sim)^(x1)) + rnorm(n2, 0, sigma.sim)
  # large sample, unbiased data, Ds2
  }
  data.M2[[i]] = data.frame(x1=x1, y=y2) # large sample, unbiased data, Ds2
  alpha[i] = alpha.sim
  beta3[i] = beta3.sim
  gam[i] = gam.sim
}
#----------------------------------------------------------------
# True result (under true simulated parameter values)
true.result.M1 = true.result.M2 = rep(0,m)
true.pred.M1 = beta0 + beta1*(log(age)) + beta2*(weight)
true.pred.M2 = alpha - beta3*(gam^(age))
for(i in 1:m){
if(true.pred.M1[i]>length.upper){true.result.M1[i]=1}
if(true.pred.M2[i]>length.upper){true.result.M2[i]=1}
}
#----------------------------------------------------------------
data.M1.Ds2 = data.M1 # large sample, unbiased data, Ds2
data.M2.Ds2 = data.M2 # large sample, unbiased data, Ds2
#-----------------------------------------------------------------
# Incorporate small sample, biased data, Ds1. 
# 
# data.M1.Ds1
data.M1.Ds1 = vector("list", m) # large unbiased data, Ds2
for(i in 1:m){
data.M1.Ds1[[i]] = data.M1[[i]][1:n1,] # small sample, unbiased data, n1 part of Ds1
}
# Incorporate bias to age and weight, part of Ds1
for ( i in 1:m){
  data.M1.Ds1[[i]]$x1 = data.M1.Ds1[[i]]$x1 - round(0.2*data.M1.Ds1[[i]]$x1) # age recorded 20% less  
  data.M1.Ds1[[i]]$x2 = data.M1.Ds1[[i]]$x2 + 0.2*data.M1.Ds1[[i]]$x1 # weight recorded 20 % more
}

# data.M2.Ds1
data.M2.Ds1 = vector("list", m) # large sample, unbiased data, Ds2
for (i in 1:m){
data.M2.Ds1[[i]] = data.M2[[i]][1:n1,] # small sample, unbiased data, n1 part of Ds1
}
# Incorporate bias to age, part of Ds1
for ( i in 1:m){
  data.M2.Ds1[[i]]$x1 = data.M2.Ds1[[i]]$x1 - round(0.2*data.M2.Ds1[[i]]$x1) # age recorded 20% less  
}
#----------------------------------------------------------------
# Spost specifications
#----------------------------------------------------------------
# Model specifications for Spost1 is done on the fly when calling lm function
# to get OLS estimates
# Model specifications for Spost2 (Bayesian) for JAGS
M1 <- "
   model{
   for (i in 1:N){
   y[i]~dnorm(mu[i],tau)
   mu[i] <- beta0+beta1*log(x1[i])+beta2*x2[i]
   }
   beta0 ~ dnorm(0,1E-06)
   beta1 ~ dnorm(0,1E-06)
   beta2 ~ dnorm(0,1E-06)
   tau ~ dgamma(1E-03,1E-03)
   sigma <- 1.0/sqrt(tau)
   }
 "

M2 <- "
  model{
  for( i in 1 : N ){
  y[i] ~ dnorm(mu[i], tau)
  mu[i] <- alpha - beta3 * pow(gam,x1[i])
  }
  alpha ~ dnorm(0.0, 1.0E-6) 
  beta3 ~ dnorm(0.0, 1.0E-6) 
  gam ~ dunif(0, 1)
  tau ~ dgamma(1.0E-3, 1.0E-3)
  sigma <- 1.0/sqrt(tau)
  }
"
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
#-----------------------------------------------------------------
# Simulations, data analyses, and reproducibility 
# rates for each study permutation
#-----------------------------------------------------------------
# Studies (see study.ind matrix) indexed by:
# Spre2, M1 are non-permutable due to Spre2 not having predictor 
# x2 (which M1 has) therefore in study.ind rows 8-12 labeled as
# (2,1,1,1), (2,1,1,2), (2,1,2,1), (2,1,2,2)
# are assigned reproducibility rate zero.
# Reproducibility rates for other permutations are
# obtained case by case below.
#-----------------------------------------------------------------
# Reproducibility rate of result for permutable studies
pred.all = vector("list", 16) # save also numerical predicted value
# of length from each analysis (not just whether categorical result
# matches the true value)
#-----------------------------------------------------------------
# Study permutation  1,1,1,1
 pred = rep(NA,m)
result = rep(0, m)
 for ( i in 1: m){
   reg<-lm(y~log(x1)+x2, data=data.M1.Ds1[[i]])
   pred[i] = sum(reg$coef*c(1, log(age), weight))
   if((pred[i]>length.upper)==true.result.M1[i]){result[i] = 1}
 }
pred.all[[1]] = pred
study.ind[1,5] = mean(result, na.rm=TRUE)
#-----------------------------------------------------------------
# Study permutation  1,1,1,2
pred = rep(NA,m)
result = rep(0, m)
for ( i in 1: m){
  reg<-lm(y~log(x1)+x2, data=data.M1.Ds2[[i]])
  pred[i] = sum(reg$coef*c(1, log(age), weight))
  if((pred[i]>length.upper)==true.result.M1[i]){result[i] = 1}
}
pred.all[[2]] = pred
study.ind[2,5] = mean(result, na.rm=TRUE)
#-----------------------------------------------------------------
# Study permutation  1,1,2,1
pred = rep(NA,m)
result = rep(0, m)
for( i in 1:m){
  data=list(x1 = data.M1.Ds1[[i]]$x1, x2 = data.M1.Ds1[[i]]$x2,
                 y = data.M1.Ds1[[i]]$y, 
                 N = n1) 
inits <- vector("list",nChains)
for(j in 1:nChains){
  inits[[j]]=list(beta0  = runif(1, 0, max.beta0), 
                  beta1  = runif(1, 0,max.beta1), 
                  beta2  = runif(1, 0, max.beta2),  
                  tau = runif(1, min.tau, max.tau)
  )
}
parameters.to.save<-c("beta0","beta1", "beta2")
reg <- jags(data = data, inits = inits,
                  parameters.to.save = parameters.to.save, 
                  n.chains = nChains, n.iter = nIter,
                  n.burnin = burnInSteps,n.thin = thinSteps,
                  model.file =textConnection(M1), quiet = TRUE, progress.bar = "none")                
pred[i] = mean(reg$BUGSoutput$sims.list$beta0 + 
             reg$BUGSoutput$sims.list$beta1*log(age) + 
             reg$BUGSoutput$sims.list$beta2*weight)
if((pred[i]>length.upper)==true.result.M1[i]){result[i] = 1}
}
pred.all[[3]] = pred
study.ind[3,5] = mean(result, na.rm=TRUE)
#-----------------------------------------------------------------
#-----------------------------------------------------------------
# Study permutation  1,1,2,2
pred = rep(NA, m)
result = rep(0, m)
for( i in 1:m){
  data=list(x1 = data.M1.Ds2[[i]]$x1, x2 = data.M1.Ds2[[i]]$x2,
            y = data.M1.Ds2[[i]]$y, 
            N = n2) 
  inits <- vector("list",nChains)
  for(j in 1:nChains){
    inits[[j]]=list(beta0  = runif(1, 0, max.beta0), 
                    beta1  = runif(1, 0,max.beta1), 
                    beta2  = runif(1, 0, max.beta2),  
                    tau = runif(1, min.tau, max.tau)
    )
  }
  parameters.to.save<-c("beta0","beta1", "beta2")
  reg <- jags(data = data, inits = inits,
              parameters.to.save = parameters.to.save, 
              n.chains = nChains, n.iter = nIter,
              n.burnin = burnInSteps,n.thin = thinSteps,
              model.file =textConnection(M1), quiet=TRUE, progress.bar = "none")                
  pred[i] = mean(reg$BUGSoutput$sims.list$beta0 + 
                   reg$BUGSoutput$sims.list$beta1*log(age) + 
                   reg$BUGSoutput$sims.list$beta2*weight)
  if((pred[i]>length.upper)==true.result.M1[i]){result[i] = 1}
}
pred.all[[4]] = pred
study.ind[4,5] = mean(result, na.rm=TRUE)
#-----------------------------------------------------------------
# Study permutation  1,2,1,1
pred = rep(NA,m)
result = rep(0, m)
for ( i in 1: m){
  #----------------------------
  skip_to_next <- FALSE
  alpha.start = runif(1,0, max.alpha)
  beta3.start = runif(1,0, max.beta3)
  gam.start = runif(1, 0, max.gam)
  tryCatch(nlsLM(y~alpha - beta3 * ((gam)^(x1)), 
                 start=list(alpha=alpha.start,beta3=beta3.start, gam=gam.start),
                 data=data.M2.Ds1[[i]]), error = function(e) { skip_to_next <<- TRUE})
  if(skip_to_next) { next }
  reg <- nlsLM(y ~ alpha - beta3 * ((gam)^(x1)),
               start=list(alpha=alpha.start,beta3=beta3.start, gam=gam.start),
               data=data.M2.Ds1[[i]])
  #----------------------------
  pred[i] = coef(reg)[1] - coef(reg)[2]*(coef(reg)[3]^age)
  if((pred[i]>length.upper)==true.result.M2[i]){result[i] = 1}
}
pred.all[[5]] = pred
study.ind[5,5] = mean(result, na.rm=TRUE)
#-----------------------------------------------------------------
# Study permutation  1,2,1,2 
pred = rep(NA, m)
result = rep(0, m)
for ( i in 1: m){
  skip_to_next <- FALSE
  alpha.start = runif(1,0, max.alpha)
  beta3.start = runif(1,0, max.beta3)
  gam.start = runif(1, 0, max.gam)
  tryCatch(nlsLM(y~alpha - beta3 * ((gam)^(x1)), 
                 start=list(alpha=alpha.start,beta3=beta3.start, gam=gam.start),
                 data=data.M2.Ds2[[i]]), error = function(e) { skip_to_next <<- TRUE})
  if(skip_to_next) { next }
  reg <- nlsLM(y ~ alpha - beta3 * ((gam)^(x1)),
        start=list(alpha=alpha.start,beta3=beta3.start, gam=gam.start),
        data=data.M2.Ds2[[i]])
  pred[i] = coef(reg)[1] - coef(reg)[2]*(coef(reg)[3]^age)
  if((pred[i]>length.upper)==true.result.M2[i]){result[i] = 1}
}
pred.all[[6]] = pred
study.ind[6,5] = mean(result, na.rm=TRUE)
#-----------------------------------------------------------------
# Study permutation  1,2,2,1
pred = rep(NA, m)
result = rep(0, m)
for ( i in 1: m){
  data=list(x1 = data.M2.Ds1[[i]]$x1,
            y = data.M2.Ds1[[i]]$y, 
            N = n1) 
  inits <- vector("list",nChains)
  for(j in 1:nChains){
    inits[[j]]=list(alpha  = runif(1, 0, max.alpha), 
                    beta3  = runif(1, 0, max.beta3), 
                    gam  = runif(1, 0, max.gam),  
                    tau = runif(1, min.tau, max.tau)
    )
  }
  parameters.to.save<-c("alpha","beta3", "gam")
  reg <- jags(data = data, inits = inits,
              parameters.to.save = parameters.to.save, 
              n.chains = nChains, n.iter = nIter,
              n.burnin = burnInSteps,n.thin = thinSteps,
              model.file =textConnection(M2), quiet = TRUE, progress.bar = "none")                
  pred[i] = mean(reg$BUGSoutput$sims.list$alpha - 
                   (reg$BUGSoutput$sims.list$beta3)*(reg$BUGSoutput$sims.list$gam^(age)))  
  if((pred[i]>length.upper)==true.result.M2[i]){result[i] = 1}
}
pred.all[[7]] = pred
study.ind[7,5] = mean(result, na.rm=TRUE)
#-----------------------------------------------------------------
# Study permutation  1,2,2,2
pred = rep(NA, m)
result = rep(0, m)
for ( i in 1: m){
  data=list(x1 = data.M2.Ds2[[i]]$x1,
            y = data.M2.Ds2[[i]]$y, 
            N = n2) 
  inits <- vector("list",nChains)
  for(j in 1:nChains){
    inits[[j]]=list(alpha  = runif(1, 0, max.alpha), 
                    beta3  = runif(1, 0, max.beta3), 
                    gam  = runif(1, 0, max.gam),  
                    tau = runif(1, min.tau, max.tau)
    )
  }
  parameters.to.save<-c("alpha","beta3", "gam")
  reg <- jags(data = data, inits = inits,
              parameters.to.save = parameters.to.save, 
              n.chains = nChains, n.iter = nIter,
              n.burnin = burnInSteps,n.thin = thinSteps,
              model.file =textConnection(M2), quiet = TRUE, progress.bar = "none")                
  pred[i] = mean(reg$BUGSoutput$sims.list$alpha - 
                   (reg$BUGSoutput$sims.list$beta3)*(reg$BUGSoutput$sims.list$gam^(age)))  
  if((pred[i]>length.upper)==true.result.M2[i]){result[i] = 1}
}
pred.all[[8]] = pred
study.ind[8,5] = mean(result, na.rm=TRUE)
#-----------------------------------------------------------------
# Study permutation 2,2,1,1
pred = rep(NA, m)
result = rep(0, m)
for ( i in 1: m){
  skip_to_next <- FALSE
    alpha.start = runif(1,0, max.alpha)
    beta3.start = runif(1,0, max.beta3)
    gam.start = runif(1, 0, max.gam)
     tryCatch(nlsLM(y~alpha - beta3 * ((gam)^(x1)), 
                    start=list(alpha=alpha.start,beta3=beta3.start, gam=gam.start),
                    data=data.M2.Ds1[[i]]), error = function(e) { skip_to_next <<- TRUE})
    if(skip_to_next) { next }
  reg<-nlsLM(y~alpha - beta3 * ((gam)^(x1)), 
             start=list(alpha=alpha.start,beta3=beta3.start, gam=gam.start),
           data=data.M2.Ds1[[i]])
  pred[i] = coef(reg)[1] - coef(reg)[2]*(coef(reg)[3]^age)
  if((pred[i]>length.upper)==true.result.M2[i]){result[i] = 1}
}
pred.all[[13]] = pred
study.ind[13,5] = mean(result, na.rm=TRUE)
#-----------------------------------------------------------------
# Study permutation 2,2,1,2
pred = rep(NA, m)
result = rep(0, m)
for ( i in 1: m){
  skip_to_next <- FALSE
  alpha.start = runif(1,0, max.alpha)
  beta3.start = runif(1,0, max.beta3)
  gam.start = runif(1, 0, max.gam)
  tryCatch(nlsLM(y~alpha - beta3 * ((gam)^(x1)), 
                 start=list(alpha=alpha.start,beta3=beta3.start, gam=gam.start),
                 data=data.M2.Ds2[[i]]), error = function(e) { skip_to_next <<- TRUE})
  if(skip_to_next) { next }
  reg<-nlsLM(y~alpha - beta3 * ((gam)^(x1)), 
             start=list(alpha=alpha.start,beta3=beta3.start, gam=gam.start),
           data=data.M2.Ds2[[i]])
  pred[i] = coef(reg)[1] - coef(reg)[2]*(coef(reg)[3]^age)
  if((pred[i]>length.upper)==true.result.M2[i]){result[i] = 1}
}
pred.all[[14]] = pred
study.ind[14,5] = mean(result, na.rm=TRUE)
#-----------------------------------------------------------------
# Study permutation  2,2,2,1
pred= rep(NA ,m)
result = rep(0, m)
for ( i in 1: m){
  data=list(x1 = data.M2.Ds1[[i]]$x1,
            y = data.M2.Ds1[[i]]$y, 
            N = n1) 
  inits <- vector("list",nChains)
  for(j in 1:nChains){
    inits[[j]]=list(alpha  = runif(1, 0, max.alpha), 
                    beta3  = runif(1, 0, max.beta3), 
                    gam  = runif(1, 0, max.gam),  
                    tau = runif(1, min.tau, max.tau)
    )
  }
  parameters.to.save<-c("alpha","beta3", "gam")
  reg <- jags(data = data, inits = inits,
              parameters.to.save = parameters.to.save, 
              n.chains = nChains, n.iter = nIter,
              n.burnin = burnInSteps,n.thin = thinSteps,
              model.file =textConnection(M2), quiet=TRUE, progress.bar = "none")                
  pred[i] = mean(reg$BUGSoutput$sims.list$alpha - 
                   (reg$BUGSoutput$sims.list$beta3)*(reg$BUGSoutput$sims.list$gam^(age)))  
  if((pred[i]>length.upper)==true.result.M2[i]){result[i] = 1}
}
pred.all[[15]] = pred
study.ind[15,5] = mean(result, na.rm=TRUE)
#-----------------------------------------------------------------
# Study permutation  2,2,2,2
pred = rep(NA, m)
result = rep(0, m)
for ( i in 1: m){
  data=list(x1 = data.M2.Ds2[[i]]$x1,
            y = data.M2.Ds2[[i]]$y, 
            N = n2) 
  inits <- vector("list",nChains)
  for(j in 1:nChains){
    inits[[j]]=list(alpha  = runif(1, 0, max.alpha), 
                    beta3  = runif(1, 0, max.beta3), 
                    gam  = runif(1, 0, max.gam),  
                    tau = runif(1, min.tau, max.tau)
    )
  }
  parameters.to.save<-c("alpha","beta3", "gam")
  reg <- jags(data = data, inits = inits,
              parameters.to.save = parameters.to.save, 
              n.chains = nChains, n.iter = nIter,
              n.burnin = burnInSteps,n.thin = thinSteps,
              model.file =textConnection(M2), quiet=TRUE, progress.bar = "none")                
  pred[i] = mean(reg$BUGSoutput$sims.list$alpha - 
                   (reg$BUGSoutput$sims.list$beta3)*(reg$BUGSoutput$sims.list$gam^(age)))  
   if((pred[i]>length.upper)==true.result.M2[i]){result[i] = 1}
}
pred.all[[16]] = pred
study.ind[16,5] = mean(result, na.rm=TRUE)
#-----------------------------------------------------------------
#----------------------------------------------------------------
study.ind # print study.ind to check reproducibility rates 
#----------------------------------------------------------------
#----------------------------------------------------------------
#----------------------------------------------------------------
# Calculate distances between elements of studies
#----------------------------------------------------------------
# Distance between Spre1 and Spre2
#----------------------------------------------------------------
dist.Spre = abs(mean(study.ind[1:8,5], na.rm=TRUE) 
                - mean(study.ind[9:16,5], na.rm=TRUE)) 
#----------------------------------------------------------------
# Distance between M1 and M2
#----------------------------------------------------------------
dist.M = abs(mean(study.ind[c(1:4,9:12),5], na.rm=TRUE) 
             - mean(study.ind[c(5:8,13:16),5], na.rm=TRUE)) 
#----------------------------------------------------------------
# Distance between Spost1 and Spost2
#----------------------------------------------------------------
dist.Spost = abs(mean(study.ind[c(1:2, 5:6, 9:10, 13:14),5], na.rm=TRUE) 
            - mean(study.ind[c(3:4, 7:8, 11:12, 15:16),5], na.rm=TRUE))
#----------------------------------------------------------------
# Distance between Ds1 and Ds2
dist.Ds = abs(mean(study.ind[c(1,3,5,7,9,11,13,15),5], na.rm=TRUE) 
                 - mean(study.ind[c(2,4,6,8,10,12,14,16),5], na.rm=TRUE))
#----------------------------------------------------------------
#----------------------------------------------------------------
#----------------------------------------------------------------
# end file
#----------------------------------------------------------------