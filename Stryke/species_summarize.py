# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 19:27:31 2022

Script Intent: Iterate over a directory of Stryke species specific workbooks and 
summarize for report

@author: KNebiolo
"""
# import modules
import pandas as pd
import stryke
import os

# connect to workspace
inputWS = r"C:\Users\knebiolo\Desktop\Beaver_Falls_Production"

# get list of files
files = os.listdir(inputWS)
#files.remove('~$Stryke Franklin Falls Banded Killifish.xlsx')

# beta distribution 
beta = pd.DataFrame()

# daily estimate
daily = pd.DataFrame()

# yearly estimate
yearly = pd.DataFrame()

# for f in files
for f in files:
    if f[-4:] == 'xlsx':
       # get beta dataframe
       df = pd.read_excel(os.path.join(inputWS,f),'beta fit', header = 0, index_col = 0, engine = 'openpyxl')
       beta = beta.append(df)
       
       # get daily dataframe 
       df = pd.read_excel(os.path.join(inputWS,f),'daily summary',header = 0,index_col = 0, engine = 'openpyxl')
       daily = daily.append(df)
       
       # get daily dataframe 
       df = pd.read_excel(os.path.join(inputWS,f),'yearly summary',header = 0,index_col = 0, engine = 'openpyxl')
       yearly = yearly.append(df)
        
# with pd.ExcelWriter(os.path.join(inputWS,'summary.xlsx'),engine = 'openpyxl', mode = 'a') as writer:
#     beta.to_excel(writer,sheet_name = 'beta fit')
#     daily.to_excel(writer,sheet_name = 'daily summary')    
#     yearly.to_excel(writer,sheet_name = 'yearly summary')
        
beta.to_csv(os.path.join(inputWS,'beta.csv'))
daily.to_csv(os.path.join(inputWS,'daily.csv'))
yearly.to_csv(os.path.join(inputWS,'yearly.csv'))


# statistics
# 1: average daily expected survival rate (beta distribution of daily observations)

# 2: expected daily entrainment by season and flow scenario 

# 3: expected daily mortality by season and flow scenario 

# 4: average length by season - flow scenario effects magnitude of event not size of fish

