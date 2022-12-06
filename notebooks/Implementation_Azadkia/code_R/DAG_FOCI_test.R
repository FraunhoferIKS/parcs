source('~/Documents/simulator/notebooks/Implementation_Azadkia/code_R/DAG_FOCI.R')

set.seed(1997)
data <- read_csv("Documents/simulator/notebooks/Implementation_Azadkia/data/azadkia_1.csv")[, 2:17]
data <- as.matrix(data)

DAG_FOCI(data, "X11", 0.005)
DAG_FOCI(data, "X6", 0.005)
