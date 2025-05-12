library(deSolve)


ode_system <- function(t, state, params, matrix_data) {
  with(as.list(c(state, params)), {
    x <- state[1]
    y <- matrix_data[t, ]
    dx_dt <- alpha * x + sum(mu * y)
    dind_dt <- alpha * x
    list(c(dx_dt, dind_dt))
  })
}


qdODE_ls <- function(par, data){
  params = data$params
  params$alpha = par[1]
  params$mu = par[-1]
  
  derivatives <- ode(y = data$initial_state, times = data$solution_df$time, func = ode_system, parms = params, matrix_data = data$matrix_data)
  
  
  # actual_x <- data$solution_df$x
  # predicted_x <- as.data.frame(derivatives)[, "x"]
  # mse_x <- mean((actual_x - predicted_x)^2)
  # return(mse_x)
  
  actual_x <- data$solution_df$x
  predicted_x <- as.data.frame(derivatives)[, "x"]
  ind <- as.data.frame(derivatives)[, "ind"]
  mse_x <- sum(crossprod(actual_x-predicted_x),sum((ind[ind<0])^2))
  return(mse_x)
}

data <- readRDS("D:/Desktop/第四章/ode_solution_15p.rds")

  
pars_int <- c(data$params$alpha, data$params$mu)
print(pars_int)
len <- length(pars_int)

a <- rnorm(1, mean = data$params$alpha, sd = 0.2)
b <- rnorm(1, mean = data$params$mu[1], sd = 0.2)
c <- rnorm(1, mean = data$params$mu[2], sd = 0.2)
d <- rnorm(1, mean = data$params$mu[3], sd = 0.2)
e <- rnorm(1, mean = data$params$mu[4], sd = 0.2)
f <- rnorm(1, mean = data$params$mu[5], sd = 0.2)
g <- rnorm(1, mean = data$params$mu[6], sd = 0.2)
h <- rnorm(1, mean = data$params$mu[7], sd = 0.2)
i <- rnorm(1, mean = data$params$mu[8], sd = 0.2)
j <- rnorm(1, mean = data$params$mu[9], sd = 0.2)
k <- rnorm(1, mean = data$params$mu[10], sd = 0.2)
l <- rnorm(1, mean = data$params$mu[11], sd = 0.2)
m <- rnorm(1, mean = data$params$mu[12], sd = 0.2)
n <- rnorm(1, mean = data$params$mu[13], sd = 0.2)
o <- rnorm(1, mean = data$params$mu[14], sd = 0.2)
# p <- rnorm(1, mean = data$params$mu[15], sd = 0.2)
# q <- rnorm(1, mean = data$params$mu[16], sd = 0.2)
# r <- rnorm(1, mean = data$params$mu[17], sd = 0.2)
# s <- rnorm(1, mean = data$params$mu[18], sd = 0.2)
# t <- rnorm(1, mean = data$params$mu[19], sd = 0.2)

par <- c(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o)

length(par)

par <- runif(len, min = -10, max = 10)

qdODE.est <- optim(par = par, 
                   fn = qdODE_ls, 
                   data = data,
                   method = "L-BFGS-B",
                   control = list(trace = TRUE, maxit = 1e3))

cat("OPTIM_param: ", qdODE.est$par, "\n")
cat("OPTIM_fitness: ", qdODE.est$value, "\n")

diff <- qdODE.est$par - c(data$params$alpha,data$params$mu)
squared_diff <- diff^2
mse <- mean(squared_diff)
cat("Param-MSE:", mse, "\n")

file_path <- "D:/Desktop/第四章/derlpso_15p.csv"
write.table(t(data.frame(qdODE.est$par)), file = file_path, sep = ",", col.names = FALSE, row.names = FALSE, append = TRUE)
write.table(qdODE.est$value, file = file_path, sep = ",", col.names = FALSE, row.names = FALSE, append = TRUE)

print(data$params$alpha)
print(data$params$mu)


