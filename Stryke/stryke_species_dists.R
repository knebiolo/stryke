# Script by Andrew Yoder | github.com/yoderra
# Last updated 10/26/2021

library("readxl")
library("dplyr")
library("ggplot2")
library("ggforce")

#setwd("E:/KA/temp/RyeTest")

# data from \\kleinschmidtusa.com\Condor\Jobs\4287\001\Calcs\Entrainment\Data

# add data xlsx output from stryke script
#sheet <- read_excel("J:/4287/001/Calcs/Entrainment/Data/Rye_Stryke_Montgomery.xlsx", sheet = "summary")

# if there are duplicate rows in your data, run this instead
sheet <- read_excel("J:/4287/001/Calcs/Entrainment/Data/Rye_Stryke_Allegheny.xlsx", sheet = "summary")
drop_index <- sheet[-c(1)]
sheet <- drop_index %>% distinct()

head(sheet)
summary(sheet)

# pull unique valuesfor species, scenarios
species <- unique(sheet$species)
scenarios <- unique(sheet$scenario)

# fetch number of units by looking for columns of num_entrained
columns <- grep("entrained|killed", colnames(sheet), value = TRUE)
entrained <- grep("entrained", colnames(sheet), value = TRUE)

# use group_by to pull relevant stats - add min, max, median

# table of MEDIAN daily entrainment by species and scenario
median_entrained <- sheet %>% group_by(species, scenario) %>%
  summarize_at(entrained, median)

# table of MAX daily entrainment by species and scenario
max_entrained <- sheet %>% group_by(species, scenario) %>%
  summarize_at(entrained, max)

# table of sum entrained and sum killed by species and scenario
sum_entrained_killed <- sheet %>% group_by(species, scenario) %>% 
  summarise_at(columns, funs(sum = sum(.,na.rm=TRUE),
                             median = median(.,na.rm=TRUE),
                             max = max(.,na.rm=TRUE)))

# table of sum entrained and sum killed by ONLY scenario
sum_scenario_only <- sheet %>% group_by(scenario) %>% 
  summarise_at(columns, funs(sum = sum(.,na.rm=TRUE),
                             median = median(.,na.rm=TRUE),
                             max = max(.,na.rm=TRUE)))

# # export tables to csv
# directory <-getwd()
# write.csv(median_entrained, "E:\\KA\\temp\\RyeTest\\Emsworth\\median_entrained.csv", row.names = TRUE)
# write.csv(max_entrained, "E:\\KA\\temp\\RyeTest\\Emsworth\\max_entrained.csv", row.names = TRUE)
# write.csv(sum_entrained_killed, "E:\\KA\\temp\\RyeTest\\Emsworth\\sum_entrained_killed.csv", row.names = TRUE)
# write.csv(sum_scenario_only, "E:\\KA\\temp\\RyeTest\\Emsworth\\sum_scenario_only.csv", row.names = TRUE)

