library("readxl")
library("dplyr")
library("ggplot2")
library("ggforce")

setwd("E:/KA/temp/RyeTest")

# data from \\kleinschmidtusa.com\Condor\Jobs\4287\001\Calcs\Entrainment\Data

# add data xlsx output from stryke script
sheet <- read_excel("E:\\KA\\temp\\RyeTest\\Rye_Stryke_Montgomery.xlsx", sheet = "summary")

# if there are duplicate rows in your data, run this instead
#sheet_import <- read_excel("E:\\KA\\temp\\RyeTest\\Rye_Stryke_Emsworth.xlsx", sheet = "summary")
#drop_index <- sheet[-c(1)]
#sheet <- drop_index %>% distinct()

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


# create variable for killed columns
killed <- grep("killed", colnames(sheet), value = TRUE)

# pull a single species - Emerald Shiner as example
species_of_interest <- filter(sheet, species == species[1])

# create table of summary killed per scenario
fit_values <- species_of_interest %>% group_by(species, scenario) %>%
  summarise_at(killed, sum)

fit_gev <- fevd(x = fit_values$'num_killed_Montgomery U1',data = fit_values, method = "MLE", type="GEV")

# looping through sum killed tables for each species
for (i in 1:length(species)) {
  species_of_interest <- filter(sheet, species == species[i])
  fit_values <- species_of_interest %>% group_by(species, scenario) %>%
    summarise_at(killed, sum)
  print(fit_values)
  
}




# for each species, make a table of sum killed
for (i in 1:length(species)) {
  species_of_interest <- filter(sheet, species == species[i])
  fit_values <- species_of_interest %>% group_by(species, scenario) %>%
    summarise_at(killed, sum)
  
  # for each column of sum killed, fit it to a distribution
  for (j in fit_values(3:(length(fit_values)))) {
    fit <- gev.fit(xdat = fit_values, j)
    }
}

plot.gev(fitted)


library("tidyverse")
map(scenarios, keep, str_detect, '_\\d{2}$')
read.table(text = scenarios, sep = "_", as.is = TRUE)

  
# export tables to csv
directory <-getwd()
write.csv(median_entrained, "E:\\KA\\temp\\RyeTest\\Emsworth\\median_entrained.csv", row.names = TRUE)
write.csv(max_entrained, "E:\\KA\\temp\\RyeTest\\Emsworth\\max_entrained.csv", row.names = TRUE)
write.csv(sum_entrained_killed, "E:\\KA\\temp\\RyeTest\\Emsworth\\sum_entrained_killed.csv", row.names = TRUE)
write.csv(sum_scenario_only, "E:\\KA\\temp\\RyeTest\\Emsworth\\sum_scenario_only.csv", row.names = TRUE)
  
  