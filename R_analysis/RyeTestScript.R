library("readxl")
library("dplyr")
library("ggplot2")
library("ggforce")

setwd("E:/KA/temp/RyeTest")

# add data xlsx output from stryke script
sheet <- read_excel("E:\\KA\\temp\\RyeTest\\Rye_Stryke_Charleroi_test2.xlsx", sheet = "summary")

head(sheet)
summary(sheet)

# pull unique valuesfor species, scenarios
species <- unique(sheet$species)
scenarios <- unique(sheet$scenario)

# fetch number of units by looking for columns of num_entrained
columns <- grep("entrained|killed", colnames(sheet), value = TRUE)

stat_group <- sheet %>% group_by(species, scenario) %>%
  summarize_at(columns, mean)

#list <- c("species", "scenario", columns)

# use group_by to pull relevant stats - add min, max, median
#stats <- sheet %>% group_by(iteration)# %>% 
#  summarize(population = sum(pop_size),
#            entrained = sum(`num_entrained_Charleroi U1`),
#            killed = sum(`num_killed_Charleroi U1`))

# group entire sheet by iteration/scenario/species
stats <- sheet %>% group_by(iteration, scenario, species) %>%
summarize(population = (pop_size),
            entrained = (`num_entrained_Charleroi U1`),
            killed = (`num_killed_Charleroi U1`))

# fetches number of scenarios for formatting on page
length(unique(stat_group$scenario))

# plot daily count of entrained fish by iteration
plot <- ggplot(data=stats, aes(stats$entrained)) + # pulls the entrained values
  geom_histogram() +
  labs(title = "Entrained Fish by Species, Scenario",
       #subtitle = "",
       y = "Count", x = "Daily Number of Fish Entrained (log10)") + 
  #facet_wrap(~species+scenario, labeller = label_both) + # sorts plots by species and scenario
  facet_wrap_paginate(~ species + scenario, ncol = 4, nrow = 3)
  
# give the x axis a log10 transformation to make it the plots useful
plot +
  scale_x_continuous(trans = "log10") +
  theme_bw()





# print 
for(i in 1:n_pages(plot)){
  p_save <-  output + 
    facet_wrap_paginate(~ species + iteration, ncol = 4, nrow = 4, page = i)
  ggsave(plot = p_save, filename = paste0('~/page_', i, '.jpg'))
}

