library(readr)
library(ggplot2)
library(patchwork)
d <- read_csv("Documents/simulator/notebooks/Implementation_Shimizu/data/d.csv")[, -1]
s <- read_csv("Documents/simulator/notebooks/Implementation_Shimizu/data/strength.csv")[, -1]
d <- as.matrix(d)
s <- as.matrix(s)

names <- c("Experiment 1", "Experiment 2", "LiNGAM (paper)")

df <- data.frame("x" = c(s), "y" = c(d), 
                 Scenario = factor(rep(names, each = dim(s)[1]), levels = names[c(3,1,2)]))

logical_experiment <- df[, 3] != "LiNGAM (paper)"
p1 <- ggplot(data = df[logical_experiment, ], 
             aes(x = x, y = y, col = Scenario)) + 
  geom_point(size = 1.8, alpha = 0.2) + 
  scale_color_manual(values=c("#4C72B0", "#DD8452")) +
  geom_smooth() +
  ylim(c(0,10)) +
  xlab(expression(varphi)) + 
  ylab(expression("||"*hat(B) - B*"||"[F])) + 
  ylim(0,10) + 
  theme(legend.position="none",
        text = element_text(size = 20))
p2 <- ggplot(data = df, aes(x = Scenario, y = y, col = Scenario)) + 
  ylim(c(0,10)) +
  geom_boxplot(alpha = 0.2) + scale_color_manual(values=c("#55A868", "#4C72B0", "#DD8452")) + 
  theme(axis.line=element_blank(),
        axis.text.y=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks=element_blank(),
        axis.title.y=element_blank(),
        axis.title.x=element_blank(),
        text = element_text(size = 20)) + 
  labs(color = "")
p <- p1 + p2 + plot_layout(widths = c(5,1))
ggsave("Documents/simulator/notebooks/Implementation_Shimizu/code_R/plot.pdf", 
       width = 15, height = 5)
