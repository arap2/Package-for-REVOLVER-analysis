# Package-for-REVOLVER-analysis
revolver_analysis v 0.1.
Detailed in revolver_analysis.py is a program for the automated mapping of evolutionary trajectories and the coupled 
generation and evaluations of decision trees from REVOLVER outputs. This program can also be used to stratify independent
single sample tumor cohorts. 

revolver_analysis.py has been tested using real-life mesothelioma patients, utilising datasets derived from REVOLVER 
analysis of the MEDUSA22 cohort. Current version only supports use in linux environments, this will be changed in the 
next update.


Datasets
revolver_analysis requires 3 datasets to carry out the full pipeline; 1 to map evolutionary trajectories, 1 to train the
decision tree classifier, and if desired an independent daataset to make predictions on.

Provided in the repository are the three datasets used to develop and test revolver_analysis. These can be downloaded and 
used to familarise users with running the program, there is a walkthrough on how to use these datasets in the User Guide.

MED_transitional_data.csv = REVOLVER output of the occurance of key transitions for all patients in the MEDUSA22 cohort.
MED_alteration_data.csv = REVOLVER output of the occurance of key alterations for all patients in the MEDUSA22 cohort.
TCGA_REVOLVER_driver_matrix.csv = Independent single tumor sample dataset stratified using a simple decision tree (see user
guide).

Requirements
A .txt document is provided containing all the dependencies of the program, this can be downloaded and used to batch install
all dependencies following the instructions in the user guide.


User Guide
Reference the user guide for more detailed information about downloading, using and implementing revolver_analysis for 
external use.

Literature
Development of an Automated Approach for the Classification of MPM Patients into Clinically Relevant Repeat Evolutionary Trajectories
Adam Pullen, University of Leicester, 30/12/2020

