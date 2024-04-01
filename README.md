# stryke

Fish entrained through hydroelectric facilities are exposed to turbine passage mortality stressors. Mortality through hydroelectric turbines has been well studied, with mathematical models able to predict the probability fish will get struck by a turbine blade (Von Raben 1957, Franke et al. 1997). The rate at which fish are entrained (fish per million [M] cubic feet [ft3] of water) through hydroelectric facilities is also a well-studied phenomenon, with results from field trials contributing to an entrainment database compiled by the Electric Power Research Institute (EPRI 1997). The 1997 EPRI database contains observations of 70 species at 43 facilities east of the Mississippi River. The EPRI dataset is particularly useful for quantitative analysis based on the assumption that when entrainment counts are standardized by discharge across facilities and holistically observed, the database will provide a reasonable estimate of entrainment rates for a watershed of a given size that are suitable for decision making purposes.  Also, by describing entrainment rates with statistical distributions and simulating with Monte Carlo methods, it is possible to estimate average daily entrainment and mortality with measures of certainty, as well as estimating the likelihood an event of a given size will occur. 

Simulated fish migrate through a hydroelectric project where passage routes are described with a directed acyclic graph. We assume all simulated fish will move downstream as they approach the project. If fish survive their current node, they can move to the next one. If there is more than one node available at their current location, then Monte-Carlo role of the dice and a priori determined transition probabilities control their movement. The simulation ends for a fish when it arrives at the last node in the network or dies.

For fish passing via entrainment, individuals are exposed to turbine blade strike, which is modeled with the Franke et al. (1997) equations. For fish that pass via passage structures or spill, mortality is assessed with a roll of the dice using survival metrics determined a priori, sourced from similar studies, or from expert opinion. The Franke et al. (1997) equations calculate the probability a fish of a given length will get struck by a turbine runner blade. With these equations, if we know how long a given fish is, the amount of discharge of through the turbine, the type of turbine, how many blades, and how fast it is rotating, then we can calculate with certainty the probability of being struck. Therefore, the only morphometric parameter needed to assess blade strike is length. All other input parameters are sourced from technical drawings of the facility.   

This README will guide the end user through an assessment of entrainment effects at a hydroelectric facility.  The README has directions for completing the input spreadsheet and setting up a project, while the Project Notebook guides users through project implmentation.  Stryke has two modes of operation for two different types of desktop entrainment studies: (1) it can be used to assess survival of anadromous species as they migrate past a facility (or series of facilities), and (2) estimate entrainment rates and survival of native species to assess population level impacts attributed to entrainment through a hydroelectric station.  

Stryke was developed on a 64 bit Windows operating system and utilizes Microsoft Excel spreadsheets as an interface.  Users are encouraged to use the table of contents for navigation within the README and Project Notebook.

# Project Creation 

To create a project, first create folder in the directory of your choice. Next, clone the repository (https://github.com/knebiolo/stryke) into your new directory using git commands or with GitHub Desktop.  After cloning, open up the ‘stryke’ folder that you will now find in the project directory.  Stryke will directly read from and write results to the spreadsheet interface found in the spreadsheet interface folder.  You don't have to keep the spreadsheet here because one of the first steps when creating a model is to point to the spreadsheet's directory. The following directions will guide the end user through setting up the project spreadsheet and with parameter entry.

# Spreadsheet Interface

To implement a desktop entrainment study, Stryke will need data describing river discharge scenarios, project operating scenarios, seasonal entrainment events, turbine parameters, and migratory routes.  There is a tab for each major study component, and you can find directions to complete them below.  It is possible to configure a Stryke project (spreadhsheet) for resident or anadromous species impact assessments, for different operating configurations (run of river, peaking, and pumped storage operations), and it allows for an expanded migratory network with multiple dependent facilities.  

## Flow Scenarios

This section contains instruction on setting up the ‘Flow Scenarios’ tab on the spreadsheet interface.  There are different setups for native and anadromous species, therefore it is advised to analyze them separately.  For native species, entrainment events often occur on a seasonal cycle and are a function of the amount of water discharged through a facility. The intent of stryke is to simulate over the range of potential river discharges and realistic plant operating scenarios. For facilities with multiple units, it is assumed that a single unit would be operated up until its most efficient flow. At that point, water will then begin to flow through other units up until their most efficient flow, or until the hydrologic capacity of the facility is met.  Any more discharge is then spilled over the dam. If we assume fish proportionally follow the flow, we can estimate the rates at which fish will pass via each passage route. Thus, if we know the river discharge and unit capacities, we can simulate passage through the facility. 

On the **Flow Scenarios** tab, you will note 10 columns: Scenario Number, Scenario, Flow, Min_Op_Flow, Env_Flow, Gage, FlowYear, Prorate, Season, Months.  An explanation of the columns, expected data types, and strategies for native species is in the table below.  


| Field           | Data Type |                                             Comment                                           |
|-----------------|-----------|-----------------------------------------------------------------------------------------------|
|Scenario Number  |Integer    |(required) scenario number.  **must be unique**                                                |
|Scenario         |String     |(required) name of scenario, not more than 50 characters                                       |
|Flow             |String     |(required) value must be 'hydrograph'                                                          |
|Min_Op_Flow      |Float      |(required) minimum operating discharge, if unknown enter 0                                     |
|Env_Flow         |Float      |(required) minimum release discharge, if unknown enter 0                                       |
|Gage             |String     |(required) USGS Gage Number 8 character length                                                 |
|Prorate          |Floatg     |(required) project watershed: USGS watershed ratio                                             |
|Season           |String     |(required) hydrologic season, e.g. winter                                                      |
|Months           |List       |(required) list of calendar months that make up a hydrologic season, values seperated by comma |

The **Flow Scenarios** tab should look like the following image when setup properly.  Note, we are using meteorological seasons as our hydraulic seasons.  However, this isn't required, you can use any season as long as it is a list of integers separated by a comma. 

![native flow scenarios](https://github.com/knebiolo/stryke/assets/61742537/2ac59f67-d3fd-45c6-93bc-59a6f3aa80e7)



