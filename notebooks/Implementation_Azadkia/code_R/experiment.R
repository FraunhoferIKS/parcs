source('~/Documents/simulator/notebooks/Implementation_Azadkia/code_R/DAG_FOCI.R')
set.seed(1997)

attacks <- c("U12", "U14", "U47")
for (run in 30:99) {
  print(paste(Sys.time(), ":", run))
  for (attack in attacks) {
    data <- read_csv(paste("Documents/simulator/notebooks/Implementation_Azadkia/data5000/", 
                           attack, "/", run, ".csv", sep = ""))[-1]
    data <- as.matrix(data)
    strength <- data[1, "U"]
    data <- data[, which(colnames(data) != "U")]
    parents <- DAG_FOCI(data, "X5", 0.005)
    out <- list(strength, parents)
    saveRDS(out, file = paste("Documents/simulator/notebooks/Implementation_Azadkia/data5000/", 
                              attack, "/", run, ".RDS", sep = ""))
  }
}

l <- list()
for (run in 0:4) {
  print(paste(Sys.time(), ":", run))
    data <- read_csv(paste("Documents/simulator/notebooks/Implementation_Azadkia/data5000/", 
                           "ones", "/", run, ".csv", sep = ""))[-1]
    data <- as.matrix(data)
    strength <- data[1, "U"]
    data <- data[, which(colnames(data) != "U")]
    parents <- DAG_FOCI(data, "X5", 0.05)
    l[[run + 1]] <- list(strength, parents)
}

##############################################################################
# plots
##############################################################################

runs <- 100
jaccard <- function(A, B) {
  as.numeric(setequal(A, B)) #/ length(union(A, B))
}
mat <- array(data = NA, dim = c(runs, 3))
colnames(mat) <- attacks
strength <- mat
for (attack in attacks) {
  for (run in 0:(runs - 1)) {
    result <- readRDS(paste("Documents/simulator/notebooks/Implementation_Azadkia/data5000/", 
                           attack, "/", run, ".RDS", sep = ""))
    strength[run + 1, attack] <- result[[1]]
    mat[run + 1, attack] <- jaccard(result[[2]], c("X2", "X3"))
    print(paste(attack, result[[2]]))
  }
}

library(ggplot2)
df <- data.frame('jaccard' = c(mat[, 1], mat[, 2], mat[, 3]),
                 'strength' = c(strength[, 1], strength[, 2], strength[, 3]),
                 'attack' = as.factor(rep(attacks, each = runs)))
ggplot(data = df, aes(x = strength, y = jaccard, color = attack)) + 
  geom_point() + 
  geom_smooth()



e# test, whether we can recover the underlying graph
attack <- "U57"
run <- 66
data <- read_csv(paste("Documents/simulator/notebooks/Implementation_Azadkia/data5000/", 
                       attack, "/", run, ".csv", sep = ""))[-1]
data <- as.matrix(data)
data[, "X7"] <- rnorm(2000)
data <- data[, which(colnames(data) != "U")]
obj <- "X5"
level <- 0.005
