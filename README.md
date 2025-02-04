# stryke

Fish entrained through hydroelectric facilities are exposed to turbine passage mortality stressors. Mortality through hydroelectric turbines has been well studied, with mathematical models able to predict the probability fish will get struck by a turbine blade (Von Raben 1957, Franke et al. 1997). The rate at which fish are entrained (fish per million [M] cubic feet [ft3] of water) through hydroelectric facilities is also a well-studied phenomenon, with results from field trials contributing to an entrainment database compiled by the Electric Power Research Institute (EPRI 1997). The 1997 EPRI database contains observations of 70 species at 43 facilities east of the Mississippi River. The EPRI dataset is particularly useful for quantitative analysis based on the assumption that when entrainment counts are standardized by discharge across facilities and holistically observed, the database will reasonably estimate entrainment rates for a watershed of a given size suitable for decision-making purposes.  Also, by describing entrainment rates with statistical distributions and simulating them with Monte Carlo methods, it is possible to estimate average daily entrainment and mortality with measures of certainty, as well as estimate the likelihood that an event of a given size will occur. 

Simulated fish migrate through a hydroelectric project where passage routes are described with a directed acyclic graph. We assume all simulated fish will move downstream as they approach the project. If fish survive their current node, they can move to the next one. If there is more than one node available at their current location, then Monte-Carlo role of the dice and a priori-determined transition probabilities control their movement. The simulation ends for a fish when it arrives at the last node in the network or dies.

For fish passing via entrainment, individuals are exposed to turbine blade strike, which is modeled with the Franke et al. (1997) equations. For fish that pass via passage structures or spill, mortality is assessed with a roll of the dice using survival metrics determined a priori, sourced from similar studies, or from expert opinion. The Franke et al. (1997) equations calculate the probability a fish of a given length will get struck by a turbine runner blade. With these equations, if we know the length of a given fish, the amount of discharge through the turbine, the type of turbine, the number of blades, and rotation speed, then we can calculate with certainty the probability of the fish being struck. As such, the only morphometric parameter needed to assess blade strike is length. All other input parameters are sourced from the technical specifications of the facility and its turbines.   

This README will guide the end user through an assessment of the effects of entrainment at a hydroelectric facility.  The README has directions for completing the input spreadsheet and setting up a project, while the Project Notebook guides users through project implementation.  Stryke has two modes of operation for two different types of desktop entrainment studies. It can be used to (1) assess survival of anadromous species as they migrate past a facility (or series of facilities), and (2) estimate entrainment rates and survival of native species to assess population-level impacts attributed to entrainment through a hydroelectric station.  

Stryke was developed on a 64-bit Windows operating system and utilizes Microsoft Excel spreadsheets as an interface.  Users are encouraged to use the table of contents for navigation within the README and Project Notebook.

# Installation instructions
Thank you for using this open-source software! Unlike traditional programs that come with simple `.exe` installers, open-source projects often require a bit more setup to get started. Don’t worry though—this guide will walk you through everything step-by-step.

## What You'll Need:
To run this software, you'll need to install **GitHub Desktop** and certain Python libraries that the project depends on. But don’t worry, we'll use **Anaconda Navigator** to manage everything for you. Anaconda Navigator provides an easy, graphical way to handle all the software dependencies without needing to use the command line. **By following the steps below, you'll be able to recreate the necessary environment, ensuring the software runs smoothly on your system.**

**Anaconda Navigator:**
https://www.anaconda.com/download
**GitHub Desktop App:**
https://desktop.github.com/download/

---
## How to Set Up the Environment

This project requires specific Python libraries to run properly. To make it easy, you can recreate the environment using Anaconda Navigator after cloning this repository from GitHub. Follow these steps:

### 1. Clone the Repository Using GitHub Desktop
1. Download and install **GitHub Desktop**.
2. Open **GitHub Desktop**.
3. Click on `File > Clone Repository > URL`.
![file_clone_repository](https://github.com/knebiolo/stryke/blob/master/pics/file_clone_repository.jpg)
4. Copy the URL for this project **https://github.com/knebiolo/stryke**
5. Select this repository from the list or paste the repository URL.
6. Choose the folder where you want to save the repository locally. e.g. documents or your desktop, make note of this folder as it is where you will access the excel interface.
7. Click `Clone` to download the repository to your machine.
![cloning](https://github.com/knebiolo/stryke/blob/master/pics/cloning.jpg)

### 2. Open Anaconda Navigator
1. Launch **Anaconda Navigator**.
2. Go to the **"Environments"** tab on the left side of the interface.
3. In the bottom-left corner, click the **"Import"** button.
![environment_import](https://github.com/knebiolo/stryke/blob/master/pics/environment_import_v2.jpg)
4. **"Stryke"** will show up as an environment, click on it and a green arrow will appear.
![environment_load](https://github.com/knebiolo/stryke/blob/master/pics/environment_load.jpg)

### 3. Import the `environment.yml` File
1. In the **Import** dialog:
   - Give the environment a name (e.g., `stryke`).
   - Click the **"Browse"** button next to **"Specification File"**.
   - Navigate to the folder where you cloned the repository and select the `environment.yml` file.
   - Click **"Import"** to create the environment.
![setting_up_env](https://github.com/knebiolo/stryke/blob/master/pics/setting_up_environment.jpg)   
Anaconda Navigator will install all the required packages listed in the `environment.yml` file.

### 4. Activate the Environment
1. Once the environment is created, it will appear in the **Environments** list in Anaconda Navigator.
2. Click on the environment name (e.g., `stryke`) to activate it.
3. You can now launch applications like **Jupyter Notebook** or **Spyder** from within the activated environment.
4. Within the Anaconda interface, go to Home. Here you can click **Jupyter Notebook**, **Spyder** or **JupyterLab**. To start, launch **JupyterLab**.
![launch_jupyter_lab](https://github.com/knebiolo/stryke/blob/master/pics/launch_jupyter_lab.jpg)
5. **JupyterLab** will launch as a window in your default browser.
6. Within **JupyterLab** you will need to import the file `stryke_project_notebook.ipynb` from the `GitHub > stryke` folder (located in the same place as defined in step 1.6 above) into the area below the list of file folders in JupyterLab. Then click on this file within **JupyterLab** to launch Stryke.
![import_button](https://github.com/knebiolo/stryke/blob/master/pics/import_button.jpg)
![load_notebook](https://github.com/knebiolo/stryke/blob/master/pics/load_notebook.jpg)
### 5. Simplified User Interface  
To launch the simplified user interface, open the folder where Stryke is located and double click the `RUN_STRYKE.bat` file. This will automatically launch a command window, find the Anaconda environment, and open a browser tab with a codeless interface. Use the file explorer in the tab to select the input spreadsheet and click the 'Run Stryke' button. After clicking run, text output will show that either Stryke finished running or encountered an error. 

Please note, this file may need to be tailored to point to the user’s Anaconda installation if it is not a common installation location.

If you are getting an error `ValueError: Sheet 'beta fit' already exists and if_sheet_exists is set to 'error'`, the output Excel sheets are still in the file and will not let you overwrite them. Please delete the output sheets (last sheets in the file, starting with lowercase letters), save and close the file, and click run again.

---

### Additional Notes
- **Standard Libraries**: Packages like `os` and `math` are part of Python's standard library and do not need to be installed separately.
- If you encounter any issues during setup, feel free to [reach out for assistance](mailto:kevin.nebiolo@kleinschmidtgroup.com).

# Project Creation 

To create a project, first create a folder in the directory of your choice. Next, clone the repository (https://github.com/knebiolo/stryke) into your new directory using git commands or with GitHub Desktop.  After cloning, open the ‘stryke’ folder, which you will now find in the project directory.  Stryke will directly read from and write results to the spreadsheet interface found in the spreadsheet interface folder.  You don't have to keep the spreadsheet here because one of the first steps when creating a model is to point to the spreadsheet's directory. The following directions will guide the end user through setting up the project spreadsheet and with parameter entry.

# Spreadsheet Interface

To implement a desktop entrainment study, Stryke will need data describing river discharge scenarios, project operating scenarios, seasonal entrainment events, turbine parameters, and migratory routes.  There is a tab for each major study component; you can find directions to complete them below.  It is possible to configure a Stryke project (spreadsheet) for resident or anadromous species impact assessments for different operating configurations (run of river, peaking, and pumped storage operations), and it allows for an expanded migratory network with multiple dependent facilities.  

## Flow Scenarios

This section contains instructions for setting up the ‘Flow Scenarios’ tab on the spreadsheet interface.  There are different setups for native and anadromous species; therefore, it is advised that they be analyzed separately.  For native species, entrainment events often occur on a seasonal cycle and are a function of the amount of water discharged through a facility. The intent of Stryke is to simulate over the range of potential river discharges and realistic plant operating scenarios. For facilities with multiple units, it is assumed that a single unit would be operated until its most efficient flow. At that point, water will then begin to flow through the other units until their most efficient flow or until the hydraulic capacity of the facility is met.  Additional discharge beyond the facility's capacity is then spilled over the dam. Assuming fish proportionally follow the flow, we can estimate the rates at which fish will pass via each passage route. Thus, if we know the river discharge and hydraulic capacities of the unit(s), we can simulate passage through the facility. 

On the **Flow Scenarios** tab, you will note 10 columns: Scenario Number, Scenario, Flow, Min_Op_Flow, Env_Flow, Gage, FlowYear, Prorate, Season, Months.  An explanation of the columns, expected data types, and strategies for native species is in the table below.  


| Field           | Data Type |                                             Comment                                           |
|-----------------|-----------|-----------------------------------------------------------------------------------------------|
|Scenario Number  |Integer    |(required) scenario number.  **must be unique**                                                |
|Scenario         |String     |(required) name of scenario, not more than 50 characters                                       |
|Flow             |String     |(required) value must be 'hydrograph'                                                          |
|Gage             |String     |(not required) USGS Gage Number 8 character length, if blank user provides hydrograph          |
|Prorate          |Float      |(required) project watershed: USGS watershed ratio                                             |
|Season           |String     |(required) hydrologic season, e.g. winter                                                      |
|Months           |List       |(required) list of calendar months that make up a hydrologic season, values separated by commas |

When set up properly, the **Flow Scenarios** tab should look like the following image.  Note, we are using meteorological seasons as our hydrologic seasons.  However, this isn't required. You can use any season as long as it is a list of integers separated by commas. 

![native flow scenarios](https://github.com/knebiolo/stryke/blob/master/pics/flow_scenarios_tab.jpg)

## Hydrology

For projects in the United States, Stryke utilizes the Python library 'Hydrofunctions' to fetch stream gage data from the United States Geological Service.  However, there is no such library that fetches Canadian stream gage data.  Therefore, the Canadian end user must provide their own hydrograph.  There may also be projects within the United States that want to provide their own simulated hydrograph to study climate change scenarios.  The default default unit is cubic meters per second, but this can be switched to cubic feet per second.  

| Field           | Data Type |                                             Comment                                           |
|-----------------|-----------|-----------------------------------------------------------------------------------------------|
|Date             |DateTime   |(required) Excel formatted data, on import converted to YYYY-MM-DD format                      |
|Discahrge        |Float      |(required) daily average discharge in CMS or CFS                                               |

![hydrology tab](https://github.com/knebiolo/stryke/blob/master/pics/hydrology_tab.jpg)

## Facilities
Stryke is capable of simulating survival and movement through a complex migratory network that can include multiple dependent facilities.  The Facilities tab contains information describing operations and seasonal minimum flow releases for each simulated powerhouse.

| Field           | Data Type |                                             Comment                                           |
|-----------------|-----------|-----------------------------------------------------------------------------------------------|
|Facility         |String     |(required) Facility/Powerhouse name.  **must be unique**                                       |
|Season           |String     |(required) hydrologic season, must be listed on Flow Scenarios tab                             |
|Operations       |String     |(required) pull down, choose between 'run-of-river', 'pumped storage', or 'peaking'            |
|Min_Op_Flow      |Float      |(required) minimum operating discharge, if unknown enter 0                                     |
|Env_Flow         |Float      |(required) minimum release discharge, if unknown enter 0                                       |
|Bypass_Flow      |Float      |(required) minimum discharge through downstream bypass structure, if unknown enter 0           |
|Spillway         |String     |(required) spillway that services a particular facility                                        |

![facilities_tab](https://github.com/knebiolo/stryke/blob/master/pics/facilities_tab.jpg)

## Operating Scenarios

The **Operating Scenarios** tab tells Stryke how to simulate powerhouse operations.  Every season and unit combination must be represented in this table.  For run-of-river facilities, it is assumed that the facility will run 24/7, whereas a peaking facility or pumped storage faclity will run for a different amount of hours every day as demand dictates.  

| Field           | Data Type |                                             Comment                                           |
|-----------------|-----------|-----------------------------------------------------------------------------------------------|
|Scenario Number  |Integer    |(required) scenario number.  **must be related to scenario number of Flow Scenario tab**       |
|Season           |String     |(required) name of hydrologic season                                                           |
|Unit             |String     |(required) turbine unit ID, every turbine in the study gets a unique ID                        |
|Hours            |Integer    |(not required) number of hours facility runs every day; if pumped storage, then leave blank    |
|Prob_Not_Op      |Float      |(not required) Binomial probability facility is not operating                                 |
|shape            |String     |(not required) Scipy Log Normal shape parameter                                                |
|location         |Floatg     |(not required) Scipy Log Normal location parameter                                              |
|scale            |String     |(not required) Scipy Log Normal scale parameter                                                  |
|Months           |List       |(required) list of calendar months that make up a hydrologic season, values separated by commas |

The setup for run-of-river facilities is below:
![run of river op scen](https://github.com/knebiolo/stryke/assets/61742537/4b81099f-8d9c-428f-a56e-73be2e2189ee)

**Note, every season and unit combination represented**

The following image depicts **Operation Scenarios** for peaking and pumped storage projects, which are assumed to operate on demand.  Operations at these facilities can be described with two probabilities: the probability that a facility will or will not operate, and given that the facility is operating, the probability that a facility will operate for n hours.  These parameters are modeled with a binomial and log normal respectively.  Note, Stryke requires shape parameters to be fitted with [Scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html).  

![pump store op scen](https://github.com/knebiolo/stryke/assets/61742537/62bd0c8b-fba5-4c26-9ac9-b575a1772596)

## Population

The population tab is the most complex and can be set up for anadromous or resident species.  When assessing impact for resident species, entrainment is expressed as a rate (fish per million cubic feet), where the number of fish simulated per day is a function of the river discharge.  You can define entrainment rates with your own empirical data, or you can fit them to observations from the EPRI entrainment database, which is inlcuded with Stryke.  Entrainment rates can be simulated with a Log Normal, Weibull, or Pareto distribution.  For more information and tips for fitting distributions, see the documentation.  The maximum entrainment rate (max_ent_rate) is the largest entrainment rate observed.  Given that each of these distributions are heavy tailed, the maximum simulated entrainment rate can be very large.  Stryke limits the maximum simulated entrainment rate to 1 magnitude larger than the largest observation.  Entrainment events are episodic in nature, and it is not likely that there will be an entrainment event every day.  Occurence probability (occur_prob) is the probability of entraining fish of a species on any particular day. Stryke first simulates presence, and if fish are present Stryke simulates an entrainment rate.  This entrainment rate is then multiplied by the daily river discharge, and thus a simulated population is created.  Once there is a sample population, Stryke simulates fish lengths for each individual in the population.  The EPRI entrainment database also supplies information on fish lengths, which Stryke fits a log normal distribution to.    

| Field             | Data Type |                                             Comment                                           |
|-------------------|-----------|-----------------------------------------------------------------------------------------------|
|Common Name        |String     |(required)                                                                                     |
|Scientific Name    |String     |(required)                                                                                     |
|Season             |String     |(required) hydrologic season, must be related to a season on the Operating Scenarios tab       |
|Starting Population|Integer    |(not required) number of starting fish in the simulation (for anadromous mode)                 |
|(Ent. Event) shape |Float      |(not required) shape parameter describing daily entrainment event                              |
|(Ent. Event) location |Float      |(not required) location parameter describing daiy entrainment event                         |
|(Ent. Event) scale |Float      |(not required) scale parameter describing daily entrainment event.                             |
|dist               |String     |(not required) Distribution type describing daily entrainment event, must be one of (Log Normal, Weibull or Pareto) |
|max_ent_rate       |Float      |(not required) maximum entrainment event measured in fish per million cubic feet.              |
|occur_prob         |Float      |(not required) occurence probability                                                           |
|iterations         |Integer    |(required) number of simulation runs                                                           |
|Length_mean        |Float      |(not required) mean length (for anadromous mode)                                               |
|Length_sd          |Float      |(not required) standard deviation of length (for anadromous mode)                              |
|caudal_AR          |Float      |(not required) caudal fin aspect ratio, used in calculatio of swim speed.  See Sambalay 1990   |
|(Length) shape     |Float      |(not required) log normal shape parameter describing length of fish in population              |
|(Length) location  |Float      |(not required) log normal location parameter describing length of fish                         |
|(Length) shape     |Float      |(not required) log normal shape parameter describing length of fish                            |

![population](https://github.com/knebiolo/stryke/assets/61742537/ede729f8-4edf-4278-951b-b52aa4ad4238)
Note: The following columns have been hidden: Starting Popualtion, Length_mean, Length_sd, and caudal_AR.  The remaining columns depict a resident species set up. 

## Nodes and Edges

The next two tabs, **Nodes** and **Edges**, describe the migratory network simulated fish will move through.  Migratory networks are described mathematically with graphs.  Nodes are physical locations within the study area, and can include: river nodes, forebay, Units, tailrace, spill, etc.  Edges are logical migratory pathways that connect nodes together.  Stryke always models movements in a downstream direction, so the type of graph created is a directed acyclic graph.  It is directed in that fish are moving downstream while acyclic means travel is only one way.  The attributes for the **Nodes** tab are explained below. 

| Field             | Data Type |                                             Comment                                           |
|-------------------|-----------|-----------------------------------------------------------------------------------------------|
|Location           |String     |(required) physical location within the migratory network                                      |
|Surv_Fun           |String     |(required) the type of survival function applied at this node, must be 'a-priori' or a Unit ID |
|Survival           |Float      |(required) a-prior determined survival rate, if node is a Unit leave 0                         |

The following picture decpicts the correct set up for a simple 3-unit run-of-river impact assessment.

![nodesx](https://github.com/knebiolo/stryke/assets/61742537/0f1dec29-939e-4c05-8c91-1efbcbea8319)

Edges are logical pathways that connect two nodes.  Since Stryke simulates movement over a directed acyclic graph, edges are always in one direction (upstream to downstream).  Edges are always organized as From Node : To Node.  

| Field             | Data Type |                                             Comment                                           |
|-------------------|-----------|-----------------------------------------------------------------------------------------------|
|_from              |String     |(required) From Node, must match 1 Node from the Nodes tab                                     |
|_to                |String     |(required) To Node, must match 1 Node from the Nodes tab                                       |
|weight             |Float      |(required) leave as 1.                                                                         |

The following depicts the correct set up for the same, simple 3-unit run-of-river impact assessment.  Note: movement is always one way, always downstream.

![edges](https://github.com/knebiolo/stryke/assets/61742537/5b742a03-a083-46de-922e-9a1ca96b4e81)

## Unit Parameters

The unit parameters tab is another complex tab that contains measurable properties of the project's turbines and facilities such hydraulic head, runner diameter, number of blades, etc.  Note not all parameters are required for all turbine types.

| Field             | Data Type |                                             Comment                                           |
|-------------------|-----------|-----------------------------------------------------------------------------------------------|
|Unit               |String     |(required) unit identifier, must match identifiers used on **Nodes**, **Edges**, and **Operating Scenarios tab** |
|Runner Type        |String     |(required) type of runner, must be one of (Kaplan, Francis, or Propeller)                      |
|intake_vel         |Float      |(not required) If measured, intake velocity in ft/s                                            |
|op_order           |Integer    |(required) Preferred operating order of turbines                                               |
|H                  |Float      |(required) Hydraulic head (ft)                                                                 |
|RPM                |Float      |(required) runner revolutions per minute at maximum efficiency                                 |
|D                  |Float      |(required - Kaplan, Propeller) runner diameter (ft)                                            |
|$\eta$             |Float      |(required - Francis) turbine efficiency (%)                                                    |
|N                  |Integer    |(required) number of blades (Kaplan and Propeller) or buckets (Francis)                        |
|Qopt               |Float      |(required) most efficient discharge (cfs)                                                      |
|Qcap               |Float      |(required) hydraulic capacity of unit (cfs)                                                    |
|Qper               |Float      |(not required) percent of capacity at optimum discharge                                        |
|B                  |Float      |(required - Francis) runner inlet height (ft)                                                  |
|$\iota$            |Float      |(required - Francis) ratio of exit swirl to no exit swirl - leave at 1.1                       |
|D1                 |Float      |(required - Francis) diameter of runner at inlet (ft)                                          |
|D2                 |Float      |(required - Francis) diameter of runner at outlet (ft)                                         |
|$\lambda$          |Float      |(required) blade strike to mortality correlation factor, not all strikes result in death (USFWS recommends 0.2)|

With data entered into the spreadsheet interface, open up the project notebook and start simulating!

# Notebook Interface

## Fit Entrainment Rates

fish = stryke.epri(Genus = 'Alosa', Month = [1,2,3,4,5,6,7,8,9], HUC02= [2,3,4,5,7])

Families and genera of fishes for which there are data within the EPRI dataset.

| Family                | Genus       |                    |
|-----------------------|-------------|--------------------|
| Acipenseridae         | Acipenser   | Lepisosteus        |
| Amiidae               | Alosa       | Lepomis            |
| Anguillidae           | Ambloplites | Lethenteron        |
| Atherinopsidae        | Ameiurus    | Lota               |
| Catostomidae          | Amia        | Luxilus            |
| Centrarchidae         | Ammocrypta  | Margariscus        |
| Clupeidae             | Anguilla    | Micropterus        |
| Cottidae              | Aplodinotus | Minytrema          |
| Cyprinidae            | Campostoma  | Morone             |
| Esocidae              | Carassius   | Moxostoma          |
| Fundulidae            | Carpiodes   | Nacomis            |
| Gasterosteidae        | Catostomus  | Nocomis            |
| Ictaluridae           | Chrosomus   | Notemigonus        |
| Lepisosteidae         | Coregonus   | Notropis           |
| Lotidae               | Cottus      | Noturus            |
| Moronidae             | Couesius    | Oncorhynchus       |
| Osmeridae             | Culaea      | Opsopoeodus        |
| Percidae              | Cyprinella  | Osmerus            |
| Percopsidae           | Cyprinus    | Perca              |
| Petromyzontiformes    | Dorosoma    | Percina            |
| Salmonidae            | Erimyzon    | Percopsis          |
| Sciaenidae            | Esox        | Petromyzon         |
| Umbridae              | Etheostoma  | Pimephales         |
|                       | Exoglossum  | Pomoxis            |
|                       | Fundulus    | Pylodictis         |
|                       | Gasterosteus| Rhinichthys        |
|                       | Hybognathus | Salmo              |
|                       | Hypentelium | Salmonidae         |
|                       | Hypomesus   | Salvelinus         |
|                       | Ichthyomyzon| Sander             |
|                       | Ictalurus   | Semotilus          |
|                       | Labidesthes | Umbra              |
|                       | Lampetra    |                    |

