library(deSolve)
library(ggplot2)


n <- 14 

sl <- 10

matrix_data <- matrix(runif(sl * n, min = 0, max = 2), nrow = sl, ncol = n)
colnames(matrix_data) <- paste0("y", 1:n)



alpha <- rnorm(1, mean = 0.1, sd = sqrt(0.1))  
mu <- rnorm(n, mean = 0.1, sd = sqrt(0.1))     


# ode_system <- function(t, state, params) {
#   with(as.list(c(state, params)), {
#     
#     x <- state[1]
#     y <- state[2:(n + 1)]
#     
#     
#     dy_dt <- a + b * t^(b - 1)
#     
#     
#     dx_dt <- alpha * x + sum(mu * y)
#     
#     
#     dind_dt <- alpha * x
#     
#     
#     list(c(dx_dt, dy_dt, dind_dt))
#   })
# }
ode_system <- function(t, state, params, matrix_data) {
  with(as.list(c(state, params)), {

    x <- state[1]
    
    y <- matrix_data[t, ]
    
    dx_dt <- alpha * x + sum(mu * y)
    
    dind_dt <- alpha * x
    
    list(c(dx_dt, dind_dt))
  })
}

t <-  seq_len(sl)


x0 <- 1.0
y0 <- rep(0, n)
ind0 <- 1
initial_state <- c(x = x0, ind = ind0)


params <- list(alpha = alpha, mu = mu, n = n)


solution <- ode(y = initial_state, times = t, func = ode_system, parms = params, matrix_data = matrix_data)

print(head(solution))

solution_df <- as.data.frame(solution)

print(colnames(solution_df))

ggplot(solution_df, aes(x = time, y = x)) +
  geom_line(color = "blue") +
  labs(x = "Time", y = "x(t)", title = "Simulated ODE Data: x(t)") +
  theme_minimal()

ggplot(solution_df, aes(x = time, y = y2)) +
  geom_line(color = "red") +
  labs(x = "Time", y = "y2(t)", title = "Simulated ODE Data: y2(t)") +
  theme_minimal()


data_to_save <- list(
  params = params,
  initial_state = initial_state,
  solution_df = solution_df,
  matrix_data = matrix_data
)

saveRDS(data_to_save, file = "D:/Desktop/第四章/ode_solution_15p.rds")



orig <- readRDS("D:/Desktop/第四章/ode_solution_5.rds")
para = orig$params

para$alpha = 0.5225444
para$mu = c(-0.2562885,0.7584430,-0.1343711)
solution <- ode(y = orig$initial_state, times = t, func = ode_system, parms = para,matrix_data=orig$matrix_data )
head(solution)
head(orig$solution_df)