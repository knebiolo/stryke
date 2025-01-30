# Author: KPN
# Script Intent: can we estimate entrainment rate as a function of watershed size?
# Can these models be transferred elsewhere in North America?

# libraries
library (tidyverse)

# get data
dat <- read.csv('C:/Users/knebiolo/OneDrive - Kleinschmidt Associates, Inc/Software/stryke/Data/epri1997.csv')

#  add a column for season
dat <- dat %>%
  mutate(season = ifelse(Month == 12 | Month == 1 | Month == 2, 'winter',
                         ifelse(Month == 3 | Month == 4 | Month == 5, 'spring',
                                ifelse(Month == 6 | Month == 7 | Month == 8, 'summer', 'fall'))))

# filter out by important species we will find at almost every project
fishes <- filter(dat, Family == 'Centrarchidae'| Family == 'Catostomidae' | Family == 'Ictaluridae' | Family == 'Percidae')

#----
# summarize and view histograms of response and predictors
summary <- group_by(fishes, season, Family)%>%
  summarise(mean_ent_rate = mean(log(FishPerMft3), na.rm =  TRUE),
            sd_ent_rate = sd(log(FishPerMft3), na.rm = TRUE),
            mean_lat = mean(latitude, na.rm = TRUE),
            sd_lat = sd(latitude, na.rm = TRUE),
            mean_drainageArea = mean(drainageArea, na.rm = TRUE),
            sd_drainageArea = sd(drainageArea, na.rm = TRUE),
            mean_MCF = mean(MCF_Fill, na.rm = TRUE),
            sd_MCF = sd(MCF_Fill, na.rm = TRUE))

ent_rate_hist <- ggplot(fishes, aes(x = log(FishPerMft3))) +
  geom_histogram(bins = 10, fill = "blue", alpha = 0.7) +
  facet_grid(Family ~ season) +  # Rows: group1, Columns: group2
  theme_minimal() +
  labs(title = "Entrainment Rates by Family and Season",
       x = "Value",
       y = "Frequency")
ent_rate_hist

latitude_hist <- ggplot(fishes, aes(x = latitude)) +
  geom_histogram(bins = 10, fill = "green", alpha = 0.7) +
  facet_grid(Family ~ season) +  # Rows: group1, Columns: group2
  theme_minimal() +
  labs(title = "Latitude by Family and Season",
       x = "Value",
       y = "Frequency")
latitude_hist

drainage_hist <- ggplot(fishes, aes(x = log(drainageArea))) +
  geom_histogram(bins = 10, fill = "orange", alpha = 0.7) +
  facet_grid(Family ~ season) +  # Rows: group1, Columns: group2
  theme_minimal() +
  labs(title = "Drainage Area by Family and Season",
       x = "Value",
       y = "Frequency")
drainage_hist

mcf_hist <- ggplot(fishes, aes(x = log(MCF_Fill))) +
  geom_histogram(bins = 10, fill = "red", alpha = 0.7) +
  facet_grid(Family ~ season) +  # Rows: group1, Columns: group2
  theme_minimal() +
  labs(title = "Discharge by Family and Season",
       x = "Value",
       y = "Frequency")
mcf_hist

#----
# Filter by family, plot variables, fit models
catastomidae <-filter (dat, FishPerMft3 > 0, Family == 'Catostomidae')

#----
# Catastomidae Predictor: Drainage Area
# Scatterplot - Replace 'x_column' and 'y_column' with the column names
cat_drain_v_ent <- ggplot(catastomidae, aes(x = log(drainageArea), y = log(FishPerMft3))) +
  geom_point() +
  theme_minimal() +
  labs(x = "drainage area",
       y = "entrainment rate")
cat_drain_v_ent

mod_drain_v_ent_n1 <- lm(log(FishPerMft3) ~ log(drainageArea) + factor(season) + factor(HUC02), data = catastomidae)
summary(mod_drain_v_ent_1) # model significant, but R squared low (0.20), Ohio River basin low catastomidae
#plot(mod_drain_v_ent_1) # QQ plot is troubling

mod_drain_v_ent_2 <- lm(log(FishPerMft3) ~ log(drainageArea) + factor(season), data = catastomidae)
summary(mod_drain_v_ent_2) # model significant, no season different from fall, R squared very low 0.17
#plot(mod_drain_v_ent_2) # QQ plot is troubling

anova(mod_drain_v_ent_1, mod_drain_v_ent_2) # anova significant, more complex model better

mod_drain_v_ent_3 <- lm(log(FishPerMft3) ~ log(drainageArea) + factor(HUC02), data = catastomidae)
summary(mod_drain_v_ent_3) # model significant, Ohio basin low, R squared only 0.2
#plot(mod_drain_v_ent_3) # same

anova(mod_drain_v_ent_1, mod_drain_v_ent_3) # model is not significant, parsiminous model better

mod_drain_v_ent_4 <- lm(log(FishPerMft3) ~ log(drainageArea), data = catastomidae)
summary(mod_drain_v_ent_4) # model significant, R squared only 0.17
#plot(mod_drain_v_ent_4) # same

anova(mod_drain_v_ent_3, mod_drain_v_ent_4) # signficant, more complex model better 

# summary: the best model was model 3, which included HUC02 as a factor, 
#         adding season to the model did not improve predictive capabilities of the model

#----
# latitude
cat_lat_v_ent <- ggplot(catastomidae, aes(x = latitude, y = log(FishPerMft3))) +
  geom_point() +
  theme_minimal() +
  labs(x = "latitude",
       y = "entrainment rate")
cat_lat_v_ent

mod_lat_v_ent_1 <- lm(log(FishPerMft3) ~ latitude + factor(season) + factor(HUC02), data = catastomidae)
summary(mod_lat_v_ent_1) # model barely significant, latitude barely signficant, R squared crap 0.02
#plot(mod_lat_v_ent_1)

mod_lat_v_ent_2 <- lm(log(FishPerMft3) ~ latitude + factor(HUC02), data = catastomidae)
summary(mod_lat_v_ent_2) # model barely significant, latitude barely signficant, R squared crap 0.02, Great Lakes lower rates?
#plot(mod_lat_v_ent_2)

anova(mod_lat_v_ent_1,mod_lat_v_ent_2) # not significant, parsiminous model better

mod_lat_v_ent_3 <- lm(log(FishPerMft3) ~ latitude + factor(season), data = catastomidae)
summary(mod_lat_v_ent_3) # model barely significant, latitude barely signficant, R squared crap 0.02, no season signficantly different
#plot(mod_lat_v_ent_3)

anova(mod_lat_v_ent_1,mod_lat_v_ent_3) # significant, complex model better

mod_lat_v_ent_4 <- lm(log(FishPerMft3) ~ latitude, data = catastomidae)
summary(mod_lat_v_ent_4) # model barely significant, latitude barely signficant, R squared shit 0.0006, no season signficantly different
#plot(mod_lat_v_ent_4)

anova(mod_lat_v_ent_2, mod_lat_v_ent_4) # signficant more complex model better

# summary - latitude with HUC02 the best model however R squared value is in the gutter is this is good predictor?

#----
# Discharge
cat_mcf_v_ent <- ggplot(catastomidae, aes(x = log(MCF_Fill), y = log(FishPerMft3))) +
  geom_point() +
  theme_minimal() +
  labs(x = "million cubic feet",
       y = "entrainment rate")
cat_mcf_v_ent

mod_mcf_v_ent_1 <- lm(log(FishPerMft3) ~ log(MCF_Fill) + factor(season) + factor(HUC02), data = catastomidae)
summary(mod_mcf_v_ent_1) # model barely significant, MCF very signficant, R squared crap 0.21, SE much lower effect, summer significant
#plot(mod_mcf_v_ent_1)

mod_mcf_v_ent_2 <- lm(log(FishPerMft3) ~ log(MCF_Fill) + factor(HUC02), data = catastomidae)
summary(mod_mcf_v_ent_2) # model barely significant, MCF very signficant, R squared crap 0.19, SE much lower effect
#plot(mod_mcf_v_ent_2)

anova(mod_mcf_v_ent_1,mod_mcf_v_ent_2) # significant, more complex model better

mod_mcf_v_ent_3 <- lm(log(FishPerMft3) ~ log(MCF_Fill) + factor(season), data = catastomidae)
summary(mod_mcf_v_ent_3) # model barely significant, MCF very signficant, R squared crap 0.19, SE much lower effect
#plot(mod_mcf_v_ent_3)

anova(mod_mcf_v_ent_1,mod_mcf_v_ent_3) # signficant, more complex model better

mod_mcf_v_ent_4 <- lm(log(FishPerMft3) ~ log(MCF_Fill), data = catastomidae)
summary(mod_mcf_v_ent_4) # model barely significant, MCF very signficant, R squared crap 0.12
#plot(mod_mcf_v_ent_4)

anova(mod_mcf_v_ent_1,mod_mcf_v_ent_4) # significant, more complex model better

#----
# Combined predictors
mod_comb_v_ent_1 <- lm(log(FishPerMft3) ~ log(drainageArea) + log(MCF_Fill) + latitude + factor(season) + factor(HUC02), data = catastomidae)
summary(mod_comb_v_ent_1) # model significant, but latitude not
#plot(mod_comb_v_ent_1) 

mod_comb_v_ent_2 <- lm(log(FishPerMft3) ~ log(drainageArea) + log(MCF_Fill) + factor(season) + factor(HUC02), data = catastomidae)
summary(mod_comb_v_ent_2) # model significant, but latitude not
#plot(mod_comb_v_ent_2) 

mod_comb_v_ent_3 <- lm(log(FishPerMft3) ~ log(drainageArea) + log(MCF_Fill) + factor(season), data = catastomidae)
summary(mod_comb_v_ent_3) # model significant, but latitude not
#plot(mod_comb_v_ent_3) 

anova(mod_comb_v_ent_2,mod_comb_v_ent_3)