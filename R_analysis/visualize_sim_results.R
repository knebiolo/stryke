# script intent: Visualize simulation results for Winter SNEC AFS 2023

library ("tidyverse")
library("readxl")

# get data
dat <- read_excel("C:/Users/knebiolo/Desktop/Beaver_Falls_Production/Chgannel Catfish.xlsx", sheet = 'daily summary')

summer <- filter(dat, season == 'Summer')

hists <- ggplot(summer,aes(x = total_entrained)) + 
  geom_histogram() + 
  facet_wrap(~ flow_scen) + 
  theme_bw()  
hists