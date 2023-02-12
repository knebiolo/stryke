#

library (tidyverse)
library (ParetoPosStable)
library (lubridate)

# script intent:
# import historical Bad Creek entrainment data, calculated fish per Million cubic feet
# plot and profit - does it look like a Pareto?

dat <- data.frame(read.csv('J:/405/036/Calcs/Data/historical.csv'))%>%
  mutate(hours = round(Hour * 24,0))%>%
  mutate(Time = paste(hours, "00", sep = ":"))%>%
  mutate(DateTime = as.POSIXct(paste(Date,Time), format = "%m/%d/%Y %H:%M"))%>%
  mutate(MCF = (Obs/Obs * 3019 * 3600)/1000000)%>%
  mutate(FishPerMCF = Obs/MCF)%>%
  mutate(Month = month.name[month(DateTime)])

ent_rate <- ggplot(dat,aes(x = FishPerMCF)) + 
  geom_histogram()

ent_rate # oh snap, looks mighty pareto!

# import species count by month from Parr Fairfield, calculate month sum by species
spc_dat <- data.frame(read.csv('J:/405/036/Calcs/Data/parr_spc_data.csv'))%>%
  group_by(Species,Month)%>%
  summarize(species_count = sum(n))

xtable <- spc_dat %>%
  group_by(Species, Month) %>%
  summarise(entrained = sum(`species_count`)) %>%
  spread(Month, entrained)

write.csv(xtable,'J:/405/036/Calcs/Output/parr_xtable.csv',row.names = FALSE)

# calculate monthly sum
month_sum <- group_by(spc_dat,Month)%>%
  summarize(month_count = sum(species_count))

# calculate proportion of monthly count made up of each species
spc_dat <- left_join(spc_dat,month_sum,by = 'Month')%>%
  mutate(proportion = species_count/month_count)

# Create big daddy data frame and generate pareto fit statistics
bad_creek_spc <- left_join(dat,spc_dat, by = 'Month')%>%
  mutate(spc_FishPerMCF = FishPerMCF * proportion)%>%
  filter(MCF > 0.0)

brown_trout_may <- filter(bad_creek_spc,Species == 'Brown trout' & Month == 'May')

btm_hist <- ggplot(brown_trout_may,aes(x = spc_FishPerMCF)) + 
  geom_histogram() + 
  xlab("Brown Trout per MCF") + 
  theme_bw()
btm_hist

fit <- pareto.fit(brown_trout_may$spc_FishPerMCF)

write.csv(bad_creek_spc,'J:/405/036/Calcs/Data/bad_creek_spc.csv')