library(deSolve)

random_small_value <- function() {
  lower_bound <- -0.1
  upper_bound <- 0.1
  return(runif(1, lower_bound, upper_bound))
}

level_competition <- function(max_iter, lec, FECount) {
  prob <- (FECount / max_iter)^2
  exemplar_levels <- rep(0, 2)
  for (i in 1:2) {
    if (runif(1) < prob) {
      lec1 <- sample(c(1:(lec-1)), 1)
      lec2 <- sample(c(1:(lec-1)), 1)
      exemplar_levels[i] <- min(lec1, lec2)
    } else {
      exemplar_levels[i] <- sample(c(1:(lec-1)), 1)
    }
  }
  if (exemplar_levels[1] > exemplar_levels[2]) {
    exemplar_levels <- rev(exemplar_levels)
  }
  return(exemplar_levels)
}

pso <- function(fitness_function,
                Time,
                data,
                lower = -10,
                upper = 10,
                max_iter,
                pN,
                numberOfLayers_list){
  
  dim <- length(c(data$params$alpha,data$params$mu))
  
  X <- matrix(0, nrow = pN, ncol = dim)
  V <- matrix(0, nrow = pN, ncol = dim)
  
  pBest <- matrix(0, nrow = pN, ncol = dim)
  gBest <- rep(0, dim)
  p_fit <- rep(0, pN)
  g_fit <- Inf
  
  currentState <- 1
  preState <- 1
  layers <- list()
  qTable <- matrix(0, nrow = length(numberOfLayers_list), ncol = length(numberOfLayers_list))
  
  fitness <- numeric(pN)
  
  phi <- 0.4
  epsilon <- 0.9
  alpha <- 0.4
  gamma <- 0.8
  
  # phi <- 0.6
  # epsilon <- 0.9
  # alpha <- 0.6
  # gamma <- 0.6
  
  rate <- 1
  count <- 0
  
  for (i in 1:pN) {
    # for (j in 1:dim) {
    #     X[i, j] <- rnorm(1, mean = pars[j], sd = 0.01)
    # }
    if(i <pN / 2){
      X[i,] <- exp(log(1e-10) + log(1 / 1e-10) * runif(dim)) * sample(c(-1, 1), dim, replace = TRUE)
      V[i,] <- exp(log(1e-10) + log(1 / 1e-10) * runif(dim)) * sample(c(-1, 1), dim, replace = TRUE)
    }else{
      X[i, ] <- runif(dim, -1, 1)
      V[i, ] <- runif(dim, -1, 1)
    }
    
    # X[i,] <- exp(log(1e-10) + log(10 / 1e-10) * runif(dim)) * sample(c(-1, 1), dim, replace = TRUE)
    # V[i,] <- exp(log(1e-10) + log(10 / 1e-10) * runif(dim)) * sample(c(-1, 1), dim, replace = TRUE)
    
    param = data$params
    param$alpha = X[i,1]
    param$mu = X[i,-1]
    tmp <- fitness_function(Time, data$initial_state, param, data$solution_df, data$matrix_data)
    fitness[i] <- tmp
    pBest[i, ] <- X[i, ]
    p_fit[i] <- tmp
    if (tmp < g_fit) {
      g_fit <- tmp
      gBest <- X[i, ]
    }
  }

  pregFit <- g_fit
  
  start <- 0.9
  end <- 0.4
  reduction_factor <- (start - end) / (max_iter - 1)
  values <- numeric(max_iter)
  for (i in 1:max_iter) {
    current_value <- start - reduction_factor * (i - 1)
    values[i] <- current_value
  }
  
  
  for (t in 1:max_iter) {
    
    if (runif(1) < epsilon) {
      next_action <- which.max(qTable[currentState, ])
    } else {
      next_action <- sample(c(1:length(numberOfLayers_list)), 1)
    }
    preState <- currentState
    currentState <- next_action
    currentTotalLayer <- numberOfLayers_list[next_action]
    

    # fitness <- sapply(1:pN, function(x) fitness_function(X[x, ], data, Time, power_par))
    
    baseCount <- floor(pN / currentTotalLayer)
    remainder <- pN %% currentTotalLayer
    layerCounts <- c(rep(baseCount, currentTotalLayer - 1), baseCount + remainder)
    sortedParticles <- order(fitness)
    start_idx <- 1
    layers <- list()
    for (i in seq_along(layerCounts)) {
      end_idx <- start_idx + layerCounts[i] - 1
      layers[[i]] <- sortedParticles[start_idx:end_idx]
      start_idx <- end_idx + 1
    }
    
    for (i in currentTotalLayer:3) {
      for (j in layers[[i]]) {
        exemplar_levels <- level_competition(max_iter, i, t)
        # id1 <- sample(layers[[exemplar_levels[1]]], 1)
        # id2 <- sample(layers[[exemplar_levels[2]]], 1)
        # X1 <- X[id1, ]
        # X2 <- X[id2, ]
        
        if (exemplar_levels[1] == exemplar_levels[2]) {
          index1 <- sample(c(1:(length(layers[[exemplar_levels[1]]]) - 1)), 1)
          index2 <- sample(c((index1 + 1):(length(layers[[exemplar_levels[1]]]))), 1)
          id1 <- layers[[exemplar_levels[1]]][index1]
          id2 <- layers[[exemplar_levels[1]]][index2]
        } else {
          id1 <- sample(layers[[exemplar_levels[1]]], 1)
          id2 <- sample(layers[[exemplar_levels[2]]], 1)
        }
        X1 <- X[id1, ]
        X2 <- X[id2, ]
        
        
        
        
        r1 <- runif(1)
        r2 <- runif(1)
        r3 <- runif(1)
        
        V[j, ] <- r1 * V[j, ] + r2 * (X1 - X[j, ]) + r3 * phi * (X2 - X[j, ])
        X[j, ] <- X[j, ] + V[j, ]
        
        # for (m in 1:dim) {
        #     if (X[j, m] > upper) {
        #       X[j, m] <- upper - (X[j, m] - upper)
        #       V[j, m] <- -V[j, m] + random_small_value()
        #     } else if (X[j, m] < lower) {
        #       X[j, m] <- lower + (lower - X[j, m])
        #       V[j, m] <- -V[j, m] + random_small_value()
        #     }
        # }
        
        param = data$params
        param$alpha = X[j,1]
        param$mu = X[j,-1]
        temp <- fitness_function(Time, data$initial_state, param, data$solution_df, data$matrix_data)

        fitness[j] <- temp
        if (temp < p_fit[j]) {
          p_fit[j] <- temp
          pBest[j, ] <- X[j, ]
          if (temp < g_fit) {
            gBest <- X[j, ]
            g_fit <- temp
          }
        }
      }
    }
    
    for (k in layers[[2]]) {
      
      index1 <- sample(c(1:(length(layers[[1]]) - 1)), 1)
      index2 <- sample(c((index1 + 1):(length(layers[[1]]))), 1)
      id1 = layers[[1]][index1]
      id2 = layers[[1]][index2]
      X1 = X[id1,]
      X2 = X[id2,]
      
      # X1 <- X[sample(layers[[1]], 1), ]
      # X2 <- X[sample(layers[[1]], 1), ]
      
      r1 <- runif(1)
      r2 <- runif(1)
      r3 <- runif(1)
      #  V[i, ] <- w * V[i, ] + c1 * r1 * (pBest[i, ] - X[i, ]) + c2 * r2 * (gBest - X[i, ])
      V[k, ] <- r1 * V[k, ] + r2 * (X1 - X[k, ]) + r3 * phi * (X2 - X[k, ])
      X[k, ] <- X[k, ] + V[k, ]
      
      # for (m in 1:dim) {
      #     if (X[k, m] > upper) {
      #       X[k, m] <- upper - (X[k, m] - upper)
      #       V[k, m] <- -V[k, m] + random_small_value()
      #     } else if (X[k, m] < lower) {
      #       X[k, m] <- lower + (lower - X[k, m])
      #       V[k, m] <- -V[k, m] + random_small_value()
      #     }
      # }
      param = data$params
      param$alpha = X[k,1]
      param$mu = X[k,-1]
      
      # cat("第", t, "次迭代 param$alpha : ", param$alpha, " param$mu : ", param$mu ,"\n")
      
      temp <- fitness_function(Time, data$initial_state, param, data$solution_df, data$matrix_data)
      
      # temp <- fitness_function(X[k, ], data, Time, power_par)
      fitness[k] <- temp
      if (temp < p_fit[k]) {
        p_fit[k] <- temp
        pBest[k, ] <- X[k, ]
        if (temp < g_fit) {
          gBest <- X[k, ]
          g_fit <- temp
        }
      }
    }
    
    
    if (t == as.integer(max_iter / 2) && g_fit > 1e-03) {
      for (i in 1:pN) {
        if(i <pN / 2){
          X[i] <- exp(log(1e-10) + log(10 / 1e-10) * runif(dim)) * sample(c(-1, 1), dim, replace = TRUE)
          V[i] <- exp(log(1e-10) + log(10 / 1e-10) * runif(dim)) * sample(c(-1, 1), dim, replace = TRUE)
        }else{
          X[i, ] <- runif(dim, lower, upper)
          V[i, ] <- runif(dim, lower, upper)
        }
      }
    }
    
    # for (i in 1:pN) {
    #     temp <- fitness_function(X[i, ], data, Time, power_par)
    #     if (temp < p_fit[i]) {
    #         p_fit[i] <- temp
    #         pBest[i, ] <- X[i, ]
    #         if (temp < g_fit) {
    #             gBest <- X[i, ]
    #             g_fit <- temp
    #         }
    #     }
    # }
    
    # pre_fitness <- fitness_function(pregBest, data, Time, power_par)
    pre_fitness <- pregFit
    
    param = data$params
    param$alpha = gBest[1]
    # cat("第", t, "次迭代 param$alpha : ", gBest[1] ,"\n")
    param$mu = gBest[-1]
    cur_fitness <- fitness_function(Time, data$initial_state, param, data$solution_df, data$matrix_data)
    
    # cur_fitness <- fitness_function(gBest, data, Time, power_par)
    pregFit <- cur_fitness
    
    reward <- abs(cur_fitness - pre_fitness) / abs(max(cur_fitness, 1e-10))
    
    newQ <- qTable[preState, currentState] +
      alpha * (reward + gamma * max(qTable[currentState, ]) - qTable[preState, currentState])
    qTable[preState, currentState] <- newQ
    
    cat("第", t, "次迭代 rate: ", rate, "---> param: ", gBest, " fitness: ", g_fit, "\n")
    
    if(abs(cur_fitness - pre_fitness) == 0){
      count <- count + 1
      if (count >= 5){
        rate <- rate + 0.1
      }
    }else{
      count <- 0
      rate <- 1
    }
    
  }
  
  return(list(gBest = gBest, g_fit = g_fit))
  
}


# qdODEmod <- function(Time, State, Pars, power_par) {
#   nn <- length(Pars)
#   ind_effect <- paste0("alpha","*",names(State)[1])
#   dep_effect <- sapply(2:nn, function(c) paste0(paste0("beta",c-1),"*",names(State)[c]))
#   dep_effect <- paste0(dep_effect, collapse = "+")
#   all_effect <- paste0(ind_effect, "+", dep_effect)
#   expr <- parse(text = all_effect)
#   
#   with(as.list(c(State, Pars)), {
#     dx <- eval(expr)
#     dy <- power_par[,1]*power_par[,2]*Time^(power_par[,2]-1)
#     dind <- alpha*x
#     for(i in c(1:(nn-1))){
#       tmp <- paste0(paste0("beta",i),"*",paste0("y",i))
#       expr2 <- parse(text = tmp)
#       assign(paste0("ddep",i),eval(expr2))
#     }
#     return(list(c(dx, dy, dind, mget(paste0("ddep",1:(nn-1))))))
#   })
# }


ode_system <- function(t, state, params, matrix_data) {
  # params = data$params
  with(as.list(c(state, params)), {
    x <- state[1]
    
    y <- matrix_data[t, ] 
    
    dx_dt <- alpha * x + sum(mu * y)
    
    dind_dt <- alpha * x
    
    list(c(dx_dt, dind_dt))
  })
}


qdODE_ls <- function(t, state, params,solution, matrix_data){

  derivatives <- ode(y = state, times = t, func = ode_system, parms = params, matrix_data=matrix_data)
  
  # derivatives_df <- as.data.frame(derivatives)
  # 
  # 
  # solution_df = as.data.frame(solution)
  # 
  # diff <- derivatives_df - solution_df
  # squared_diff <- diff^2
  # mse <- mean(as.matrix(squared_diff))
  # 
  # # ind <- as.numeric(derivatives_df$ind)
  # # sse <- sum(crossprod(solution_df-derivatives_df),sum((ind[ind<0])^2)) # + 0.0001 * sum(abs(pars))
  # return(mse)
  
  actual_x <- solution$x
  predicted_x <- as.data.frame(derivatives)[, "x"]
  ind <- as.data.frame(derivatives)[, "ind"]
  mse_x <- sum(crossprod(actual_x-predicted_x),sum((ind[ind<0])^2))
  return(mse_x)
}


data <- readRDS("D:/Desktop/第四章/ode_solution_15p.rds")


file_path <- "D:/Desktop/第四章/derlpso_15p.csv"
write.table(t(c(data$params$alpha,data$params$mu)), file = file_path, sep = ",", col.names = FALSE, row.names = FALSE, append = TRUE)


max_iter <- 200
numberOfLayers_list <- c(4,6,8,10)
pN <- 200


psoResult <- pso(qdODE_ls,
                 data$solution_df$time,
                 data,
                 lower = -10,
                 upper = 10,
                 max_iter = max_iter,
                 pN = pN,
                 numberOfLayers_list = numberOfLayers_list)

cat("PSO_param: ", round(psoResult$gBest, 16), "\n")
cat("PSO_fitness: ", round(psoResult$g_fit, 16), "\n")

diff <- psoResult$gBest - c(data$params$alpha,data$params$mu)
squared_diff <- diff^2
mse <- mean(squared_diff)
cat("Param-MSE:", mse, "\n")


write.table(t(data.frame(psoResult$gBest)), file = file_path, sep = ",", col.names = FALSE, row.names = FALSE, append = TRUE)
write.table(psoResult$g_fit, file = file_path, sep = ",", col.names = FALSE, row.names = FALSE, append = TRUE)

print(data$params$alpha)
print(data$params$mu)
