library(readr)
library(FOCI)
library(foreach)
library(doParallel)
library(igraph)
library(independence)

.markov_blanket <- function(data, name) {
  foci(data[, name], data[, colnames(data) != name], 
       numCores = parallel::detectCores())
} 

DAG_FOCI <- function(data, obj, level) {
  # Algorithm 1, Step 1
  mb <- setNames(list(.markov_blanket(data, obj)), obj)
  v <- mb[[obj]]$selectedVar$names
  mb <- append(mb, lapply(setNames(v, v), 
                          function(name) .markov_blanket(data, name)))

  # Algorithm 1, Step 2, i.e. Algorithm 3
  adj <- matrix(data = NA, ncol = length(v), nrow = length(v), 
                dimnames = list(v,v))
  for (i in v) {
    for (j in v) {
      membership_i <- i %in% mb[[j]]$selectedVar$names
      membership_j <- j %in% mb[[i]]$selectedVar$names
      if (membership_i & membership_j) {
        adj[j, i] <- adj[i, j] <- 1
      }
    }
  }
  graph <- graph_from_adjacency_matrix(adj)
  components <- decompose(graph)
  S_n <- list()
  for (index in 1:length(components)) {
    g <- components[[index]] 
    if (length(V(g)) == 1) {
      S_n[[index]] <- V(g)
    } else {
      # triangle enumeration to loop over {i,j \in [n]^2: j < i}
      triangle <- choose(length(V(g)), 2)
      triangles <- choose(seq(length(V(g)) - 1), 2)
      registerDoParallel(cores = detectCores())
      p_vals <- foreach(k = 1:triangle, .combine = "c", .inorder = FALSE) %dopar% {
        i <-  which(triangles == max(triangles[k > triangles])) + 1
        j <- k - triangles[i - 1]
        hoeffding.refined.test(data[, names(V(g))[i]], 
                               data[, names(V(g))[j]])$p.value
      }
      if (all(p_vals > level)) {
        S_n[[index]] <- V(g)
      }
    }
  }
  
  # Algorithm 1, Step 3
  magnitude <- unlist(lapply(S_n, length))
  if (sum(magnitude > 1) == 1) {
    S <- names(S_n[[which(magnitude > 1)]])
  } else if (all(magnitude %in% c(0,1))) {
    S <- names(unlist(S_n))
  } else {
    S <- "ERROR: A parent set could not be found"
  }
  S
}

