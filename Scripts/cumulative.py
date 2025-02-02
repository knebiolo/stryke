# -*- coding: utf-8 -*-

''' Script Intent: Iterate over projects within a system and generate cumulative
statistics.  Fit to distribution and calculate probability of an event > X occuring'''

# import modules
import os
import pandas as pd
from scipy.stats import pareto, weibull_min


# declare workspaces
inputWS = r"C:\Users\knebiolo\Desktop\FERA Paper Data"

cum_dat = pd.DataFrame()

# iterate over files in workspace and summarize
for file in os.listdir(inputWS):
    if file.endswith('.xlsx'):
        # get data
        dat = pd.read_excel(os.path.join(inputWS,file),sheet_name = 'daily summary')
        
        #calculate total killed
        dat['total_killed'] = dat.filter(regex = 'num_killed', axis = 'columns').sum(axis = 1)
        
        # calculate total entrained
        dat['total_entrained'] = dat.filter(regex = 'num_entrained', axis = 'columns').sum(axis = 1)
        
        # append to cumulative data
        cum_dat = cum_dat.append(dat[['species','scenario','iteration','day','pop_size','total_entrained','total_killed']])
        
print ('read and appended summary data')

# extract flow scenario        
cum_dat['flow_scen'] = cum_dat['scenario'].str.split(' ').str[0]
cum_dat['season'] = cum_dat['scenario'].str.split(' ').str[-1]
            

cum_sum_dict = {'species':[],
                'season':[],
                'scenario':[],
                'iteration':[],
                'population':[],
                'entrained':[], 
                'dead':[],
                'prob_gt_10_entrained':[],
                'prob_gt_100_entrained':[],
                'prob_gt_1000_entrained':[],
                'LL':[],
                'UL':[]}
        
# iterate over projects
for fishy in cum_dat.species.unique():
    fish_dat = cum_dat[cum_dat.species == fishy]
    #iterate over scenarios
    for scen in fish_dat.scenario.unique():
        scen_dat = fish_dat[fish_dat.scenario == scen]
        #iterate over scenarios
        for season in fish_dat.season.unique():
            seas_dat = scen_dat[scen_dat.season == season]
            for i in seas_dat.iteration.unique():
                # get data
                idat = cum_dat[(cum_dat.species == fishy) & 
                               (cum_dat.scenario == scen) & 
                               (cum_dat.season == season) & 
                               (cum_dat.iteration == i)]
                
                # get cumulative sums and append to dictionary
                cum_sum_dict['species'].append(fishy)
                cum_sum_dict['season'].append(season)
                cum_sum_dict['scenario'].append(scen)
                cum_sum_dict['iteration'].append(i)
                cum_sum_dict['population'].append(idat.pop_size.sum())
                cum_sum_dict['entrained'].append(idat.total_entrained.sum())
                cum_sum_dict['dead'].append(idat.total_killed.sum())
                
                # fit distribution to pareto, plot, calculate probability > 10, 100, and 1000 and append
                dist = weibull_min.fit(idat.total_entrained)
                probs = weibull_min.sf([10,100,1000],dist[0],dist[1])
                cum_sum_dict['prob_gt_10_entrained'].append(probs[0])
                cum_sum_dict['prob_gt_100_entrained'].append(probs[1])
                cum_sum_dict['prob_gt_1000_entrained'].append(probs[2])
                cum_sum_dict['LL'].append(weibull_min.interval(0.05,dist[0],dist[1],dist[2])[0])
                cum_sum_dict['UL'].append(weibull_min.interval(0.05,dist[0],dist[1],dist[2])[1])
             
             
print ("Summarized data by project, species, and scenario")        
# convert to pandas dataframe
sum_by_iter = pd.DataFrame.from_dict(cum_sum_dict,orient = 'columns')
cum_sum = sum_by_iter.groupby(['scenario','species'])['population','entrained','dead'].describe()


cum_sum.to_csv(os.path.join(inputWS,'cum_sum_byProjectSpeciesFlowSeason.csv'))

print ('created cumulative sum dataframe')
        

        
        

