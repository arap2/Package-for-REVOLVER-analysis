# importing all essential packages 
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import csv, pymysql, graphviz, os, sys, pprint
import tkinter as tk
from matplotlib import pyplot as plt # module used to plot the decision tree once trained.
from tkinter import *
from tkinter import filedialog
from sqlalchemy import create_engine
from sklearn.metrics import plot_confusion_matrix
from sklearn.tree import DecisionTreeClassifier # importinng the decision tree classification module. 
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix # these are metrics used to validate the accuracy of the decision tree classifier
from sklearn.model_selection import train_test_split
from pick import pick


# DEFINING FUNCTION
# //////////////////////////////////////////////////////////////////////////////////

def input_file():
	# Generating a GUI for users to input data from their current working directory

	root = tk.Tk()
	filename = filedialog.askopenfilename(initialdir = cwd, 
                                          title = "Select a File", 
                                          filetypes = (("all files", 
                                                        "*.*"),
                                                        ("CSV files",
                                                        	"*.csv*"), 
                                                       ("Text files", 
                                                        "*.txt*"))) 
	root.withdraw()
	return filename

def create_table(csv_file):
	# creating a connection to a MySQL database and parsing the input data into
	# a new MyySQL table.

	df = pd.read_csv(csv_file)
	data = pd.DataFrame(df)
	engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
							.format(user="arap2",
								pw="pa55wd",
								db="MEDUSA"))

	df.to_sql("hetmap_data2", con = engine, if_exists='replace')


def let_user_pick(options):
	# Let the user choose which sections of the pipeline they want to run.

	print("Please choose (or enter 0 to exit):")
	decision = False
	for idx, element in enumerate(options):
		print("{}) {}".format(idx+1,element))
	while decision is False:
		i = input("Enter number: ")
		try:
			if int(i) == 0:
				break
			if 0 < int(i) <= len(options):
				return int(i)
			else:
				print("number out of range")
		except:
			print("Please enter a corresponding number")
	return exit()

def name_output(directory):
	# checks the current working directory for any files named analysis_output and
	# creates an indexed version of the directory for each run of the program.

	name = "analysis_output"
	counter = 0
	file_exists = True

	while file_exists == True:
		if os.path.isdir(name) == True and counter >= 1:
			name = name.replace(f"{str(counter)}", f"{str(counter+1)}")
			counter +=1
		elif os.path.isdir(name) == True and counter < 1:
			counter +=1 
			name = f"{name}({str(counter)})"
		elif os.path.isdir(name) == False:
			file_exists = False

	return name

def sort_predictions(data_frame, predictions):
	# sorts the unlabelled array returned from sk-learn's predict() method and returns
	# a dictionary of predictions matched to the original sample IDs.

	predictions_dict = {}
	for index, row in data_frame.iterrows():
		entry = {row[0]:predictions[index]}
		predictions_dict.update(entry)
	return predictions_dict

def filter_alterations(data_frame, cutoff):

	recurrent_alterations = []
	for key, value in data_frame.iteritems():
		if sum(value) >= int(cutoff):
				recurrent_alterations.append(key)
	return recurrent_alterations

# //////////////////////////////////////////////////////////////////////////////////

# pipeline choice
options = ["Produce ETs only", "Train a decision tree", "Run the full pipeline"]
process = let_user_pick(options)
				# setting the pipeline for this run of the program.
cwd = os.getcwd()
name = name_output(cwd)
new_dir = name
path = os.path.join(cwd, new_dir)
try:
	os.mkdir(path)
except OSError as error:
	print(str(error) + "\n" + """files will be overwritten, do you still wish
	 to continue?""")
				# creating a new folder to store the output of the program in
				# this folder is nested in the current working directory.
# add option to cancel script (y/n) here in the future.

# PRODUCING EVOLUTIONARY TRAJECTORIES //////////////////////////////////////////////////////////////////

# user input
print("Chose a file to produce ETs")
filename = input_file()
create_table(filename) # parsing the csv file into a MySQL table.
				# user inputs a csv file used to map the evolutioanry trajectories.
				# choice of input data must be csv with column names delimited by a '__'
				# the csv file is converted to a table in a mysql database.


# parsing the data.
df = pd.read_csv(filename)
df_features = df.drop(columns=['sample', 'cluster'])
clusters = df.cluster.unique() # Extracting each unique cluster from the input data.
print("""Enter a cutoff value for recurrent alterations 
	(whole integers only):""")
ET_cutoff = int(input()) # user inputs the desired cutoff value.
recurrent_alterations = filter_alterations(df_features, ET_cutoff)
				# converts the csv file into a pandas dataframe and splits this into
				# two: 1) a dataframe of features. 2) a list of cluster labels.
				# The cutoff is set which is used to find all alterations that occur
				# >= to threshold.

# establishing a connection to the pymysql database.
con = pymysql.connect(
	host = 'localhost',
	user = 'arap2',
	password = 'pa55wd',
	database = 'MEDUSA'
	)
cur = con.cursor()


# sorting the transitional data into clusters.
labels = []		# labels is a list of list in which each nested list corresponds to
				# each cluster and each value corresponds to an alterations 
				# (order is conserved).
ETs = {}		# a dictionary is created where the clusters are the keys and the
				# values are a list of transitions that pass the cutoff.
with con.cursor() as cur: # iterating the MySQL table.
	for cluster in clusters:
		c_labels = []				# holds all the sum of occurance for each transition that passes the cutoff for the current cluster. 
									# used later as edge labels in ET mapping.
		value = [] 					# holds all the transitions that will be placed in the ET for the current cluster.
		for alteration in recurrent_alterations:
			sql_query = """select sum(%s) from hetmap_data2 where 
			cluster = '%s'"""		# sql query to return how many times a transition occurs
									# per cluster, if this sum is >= the cutoff, it is added to the ET
									# for that cluster.
			cur.execute(sql_query % (alteration, cluster)) # executing sql query 
			result = cur.fetchall()
			count = result[0][0]
			if count >= ET_cutoff: # assessing if the transition passess the threshold set by the user.
				value.append(alteration)
				c_labels.append(count) 
		ET_entry = {cluster:value} 
		ETs.update(ET_entry)
		labels.append(c_labels)
				# Within each nested list is the count for each alteration that passed the cutoff, e.g. labels[0][0]
				# contains the count for the first alteration in the cluster 1 ET.


# graphing the transitional data.
input_features = [] # a list containing the features (alterations) to be passed onto decision tree traiing.
colour_map = []	# contains the schema for colouring the nodes in the decision tree.
label_count = 0
for key, value in ETs.items(): # iterating the dictionary of clusters and transitions.
	g = nx.DiGraph() 				# initialising a new networkx graph for each cluster.
	for feature in value:
		fsplit = feature.split("__") # separating the transition into its two component parts. 
		g.add_edge(fsplit[0],fsplit[1], length=2) # adding an edge to the graph between 
												  # the two components of a transition.
	for node in g.nodes():			# defining the colour schema for the current cluster.
		if node == 'GL':
			colour_map.append('red')
		else:
			colour_map.append('blue')
			input_features.append(node) # appending to a list containing all the alterations 
										# excluding the transitional information.
	pos = nx.circular_layout(g)			# setting the layout of the figure.
	fig = plt.figure(figsize=(12,12))
	nx.draw_networkx_nodes(g, pos, node_size=4500, node_color=colour_map, alpha=0.5) # settings for plotting nodes
	nx.draw_networkx_edges(g, pos, width=2, alpha=0.5, edge_color='black')	# settings for plotting edges
	nx.draw_networkx_labels(g, pos, font_size=16, font_family='sans-serif') # settings for plotting node labels
					# setting parameters for the graphing of
					# the nodes, edges and labels.
	label_dictionary = {}
	for index, edge in enumerate(g.edges()):
		entry = {edge:labels[label_count][index]}
		label_dictionary.update(entry)
					# putting transition counts for each cluster
					# into a dictionary for use with networkx.
	nx.draw_networkx_edge_labels(g, pos, edge_labels=label_dictionary, 
		font_family='sans-serif', font_size=16) # adding the counts of each transition to appropriate transition edges for the current ET (cluster).
	
	file = f"ET{key}.png"					# naming figure 
	fig.savefig(os.path.join(path, file)) 	# save fig using this name
	plt.clf()
	colour_map = []							# resetting the coulour_map for the next cluster.	
	label_count +=1							# moving into the next nested list in the labels list.

if process == 1: 							# generate ETs only then exit the program - pipeline choice 1.
	exit()


# TRAINING THE DECISION TREE ///////////////////////////////////////////////////////////////

#user input
unique_features = set(input_features) 		# the recurrent alterations to feed into decision tree classification.
title = 'Choose features to train the algorithm with'
selection = pick(list(unique_features), title, multiselect=True, min_selection_count=4)
											# user selects from the recurrent alterations derived from the transitional data
											# to use as candidates in node selection for the decision tree classifier.
selection = [x[0] for x in selection] 		# unpacking the output from the user selection.
					

# parsing the data
feature_labels = []
filename_2 = input_file()
df2 = pd.read_csv(filename_2, delimiter='\t')
df2_features = pd.DataFrame(df2, columns=selection)
classes = pd.DataFrame(df2, columns=['CLUSTER'])
					# user inputs a csv files.
					# converts the csv file into a pandas dataframe and splits this into
					# two: 1) a dataframe of features. 2) a dataframe of cluster data.

# training the decision tree
clf = DecisionTreeClassifier(random_state=0, max_features=len(selection))
				# setting the parameters for the classification decision tree model we are going to use.
clf.fit(df2_features, classes) 
				# building the classification model (decision tree) on the training dataset (this is the features and classes)


# plotting the decision tree below
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300) # parameters for plotting decision tree.
file = "tree.dot"
					#generating DOT data for graphviz plot (method of plotting the decision tree)
tree.export_graphviz(clf, out_file=os.path.join(path, file), feature_names=selection, class_names=clusters,
	filled=True)	# outputting the figure to the output_directory generated at the start of the script and assigning figure lables.

os.system("dot -Tpng tree.dot -o full_script_tree_med.png")
					# terminal command to convert .dot file to .png file
output_summary = open("output_summary.txt", "w")
output_summary.close()
os.chdir(path)
os.system("dot -Tpng tree.dot -o full_script_tree_med.png")
					# terminal command to convert .dot file to .png file only supported on linux for now.
output_summary = open('output_summary.txt', 'a')
					# creating a new file to hold information on the run of the program.


# EVALUATING THE DECISION TREE /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
predictions = clf.predict(df2_features) 
sorted_predictions = sort_predictions(df2, predictions)
					# predictions on the training dataset used to validate the model below:
		 			# class predictions are sorted for easier display.

# printing parameter choices and predictions to outfile.
print(f"""Evolutionary trajectories mapped using: {filename} as input""", "\n", file=output_summary)
print(f"cutoff value: {ET_cutoff}", "\n", file=output_summary)
print(f"chosen feature set is {selection}", "\n", file=output_summary)
print(f"Decision tree trained on: {filename_2}", "\n", file=output_summary)
print("Predictions made during self validation: ", sorted_predictions, "\n", file=output_summary)

#performing statistical analyses of the decision tree classifier
c_matrix = confusion_matrix(classes, predictions, labels=clusters) 
classification_report = classification_report(classes, predictions)

# Printing the statistical analyses to outfile.
print("\n", "Model Evalutation", file=output_summary)
print( "\n", "classification matrix: ", "\n", c_matrix, file=output_summary)
print("\n", "classification_report: ", "\n", classification_report, 
	file=output_summary)
					# writing the predictions and their evalutations into output_summary.

if process == 2:	# generate decision tree, evalutate then exit the program.
	exit()

# BINNING INDEPENDET DATA //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# user input and data parsing.
print("Select a file for stratification")
filename = input_file()
assignment_data = pd.read_csv(filename)
assignment_data2 = pd.DataFrame(assignment_data, columns=selection)
assignment_clusters = pd.DataFrame(assignment_data, columns=['CLUSTER'])
					# user inputting independent data and parsing as previously done for training dataset.

# assignments of classes to the new dataset.
assignments = clf.predict(assignment_data2)
sorted_assignments = sort_predictions(assignment_data, assignments)
# evaluating assignments
matrix = confusion_matrix(assignment_clusters, assignments)

# writing the predictions and evaluations to into output_summary.
print("\n", f"""predictions made on independent dataset: {filename} """, file=output_summary)
print("\n", sorted_assignments, file=output_summary)
print("\n", "classification matrix: ", "\n", matrix,
	"\n", file=output_summary)
print("classification accuracy on independent dataset:", 
	clf.score(assignment_data2, assignment_clusters), file=output_summary)


# OVERFITTING ANALYSIS /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# testing pruning on the accuracy of test predictions
pruning_path = clf.cost_complexity_pruning_path(df2_features, classes) # determining effective pruning values (ccp_alphas) for the current training data.
ccp_alphas, impurities = pruning_path.ccp_alphas, pruning_path.impurities # extracting node impurities for each pruning value.
clfs=[] # holds each tree trained using a different ccp_alpha value.
for ccp_alpha in ccp_alphas: # training a new decision tree for each new value of ccp_alpha.
	p_clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
	p_clf.fit(df2_features, classes)
	clfs.append(p_clf)

clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]

train_scores = [clf.score(df2_features, classes) for clf in clfs]
test_scores = [clf.score(assignment_data2, assignment_clusters) for clf in clfs]
	# plotting the classification accuracy of each decision tree in clfs for both the training
	# and test datasets.

fig, ax = plt.subplots() # setting figure parameters and labels
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker='o', label="train",
        drawstyle="steps-post") # plotting all the classification accuracies of each tree with the training dataset.
ax.plot(ccp_alphas, test_scores, marker='o', label="test",
        drawstyle="steps-post")# plotting all the classification accuracies of each tree with the test dataset.
ax.legend()
fig_title = "accuracy vs alpha plot.png" # figure title
fig.savefig(os.path.join(path, fig_title)) # saving figure to the current output directory.

output_summary.close()
# /////////////////////////////////////////////////////////////////////////////////////