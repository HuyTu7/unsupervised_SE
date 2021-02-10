from __future__ import division, print_function

from datetime import timedelta
import pandas as pd
import numpy as np
import sklearn
import random
import pdb
from demos import cmd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.tree import DecisionTreeRegressor
from os import listdir

import collections
try:
   import cPickle as pickle
except:
   import pickle
from learners import Treatment, TM, SVM, RF, DT, NB, LR
from jitterbug import *
import warnings
warnings.filterwarnings('ignore')

BUDGET = 50
POOL_SIZE = 10000
INIT_POOL_SIZE = 10
np.random.seed(4789)


def load_csv(path="../new_data/original/"):
    data={}
    for file in listdir(path):
        if ".csv" in file:
            try:
                df = pd.read_csv(path+file)
                data[file.split(".csv")[0]] = df
            except:
                print("Ill-formated file", file)
    return data

def getHigherValueCutoffs(data, percentileCutoff, class_category):
	'''
	Parameters
	----------
	data : in pandas format
	percentileCutoff : in integer
	class_category : [TODO] not needed

	Returns
	-------
	'''
	abc = data.quantile(float(percentileCutoff) / 100)
	abc = np.array(abc.values)[:-1]
	return abc


def filter_row_by_value(row, cutoffsForHigherValuesOfAttribute):
	'''
	Shortcut to filter by rows in pandas
	sum all the attribute values that is higher than the cutoff
	----------
	row
	cutoffsForHigherValuesOfAttribute

	Returns
	-------
	'''
	rr = row[:-1]
	condition = np.greater(rr, cutoffsForHigherValuesOfAttribute)
	res = np.count_nonzero(condition)
	return res


def getInstancesByCLA(data, percentileCutOff, positiveLabel):
	'''
	- unsupervised clustering by median per attribute
	----------
	data
	percentileCutOff
	positiveLabel

	Returns
	-------

	'''
	# get cutoff per fixed percentile for all the attributes
	cutoffsForHigherValuesOfAttribute = getHigherValueCutoffs(data, percentileCutOff, "Label")
	# get K for all the rows
	K = data.apply(lambda row: filter_row_by_value(row, cutoffsForHigherValuesOfAttribute), axis = 1)
	# cutoff for the cluster to be partitioned into
	cutoffOfKForTopClusters = np.median(K)
	instances = [1 if x > cutoffOfKForTopClusters else 0 for x in K]
	data["CLA"] = instances
	data["K"] = K
	return data


def getInstancesByRemovingSpecificAttributes(data, attributeIndices, invertSelection):
	'''
	removing the attributes
	----------
	data
	attributeIndices
	invertSelection

	Returns
	-------
	'''
	if not invertSelection:
		data_res = data.drop(data.columns[attributeIndices], axis=1)
	else:
		# invertedIndices = np.in1d(range(len(attributeIndices)), attributeIndices)
		# data.drop(data.columns[invertedIndices], axis=1, inplace=True)
		data_res = data[attributeIndices]
		data_res['Label'] = data['Label'].values
	return data_res


def getInstancesByRemovingSpecificInstances(data, instanceIndices, invertSelection):
	'''
	removing instances
	----------
	data
	instanceIndices
	invertSelection

	Returns
	-------

	'''
	if not invertSelection:
		data.drop(instanceIndices, axis=0, inplace=True)
	else:
		invertedIndices = np.in1d(range(data.shape[0]), instanceIndices)
		data.drop(invertedIndices, axis=0, inplace=True)
	return data


def getSelectedInstances(data, cutoffsForHigherValuesOfAttribute, positiveLabel):
	'''
	select the instances that violate the assumption
	----------
	data
	cutoffsForHigherValuesOfAttribute
	positiveLabel

	Returns
	-------
	'''

	violations = data.apply(lambda r: getViolationScores(r,
														 data['Label'],
														 cutoffsForHigherValuesOfAttribute),
							axis=1)
	violations = violations.values
	# get indices of the violated instances
	selectedInstances = (violations > 0).nonzero()[0]
	# remove randomly 90% of the instances that violate the assumptions
	selectedInstances = np.random.choice(selectedInstances, int(selectedInstances.shape[0] * 0.9), replace=False)
	# for index in range(data.shape[0]):
	# 	if violations[index] > 0:
	# 		selectedInstances.append(index)
	return selectedInstances


def getCLAMIresults(seed=0, input="../new_data/csc/", output="../results/CSC_CLAMI_"):
	'''
	main method for most of the methods:
	- CLA
	- CLAMI
	- FLASH_CLAMI
	----------
	seed
	input
	output

	Returns
	-------
	'''
	treatments = ["CLA", "FLASH_CLA", "CLA+RF", "CLA+NB"]
	data = load_csv(path=input)
	columns = ["Treatment"]+list(data.keys())
	result = {}
	keys = list(data.keys())
	keys.sort()
	print(keys)
	for target in keys:
		print(target)
		result[target] = [CLA(data, target, None, 50)]
		print(result[target][-1])
		result[target] += [tune_CLAMI(data, target, None, 50)]
		print(result[target][-1])
		result[target] += CLAMI(data, target, None, 50)
	result["Treatment"] = treatments
	# Output results to tables
	metrics = result[columns[-1]][0].keys()
	for metric in metrics:
		df = {key: (result[key] if key == "Treatment" else [dict[metric] for dict in result[key]]) for key in result}
		pd.DataFrame(df, columns=columns).to_csv(output + "unsupervised_" + metric + ".csv",
												 line_terminator="\r\n", index=False)


def two_step_Jitterbug(data, target, model = "RF", est = False, T_rec = 0.90, inc=False, seed = 0):
	np.random.seed(seed)
	jitterbug = Jitterbug(data,target)
	jitterbug.find_patterns()
	jitterbug.test_patterns(include=inc)
	jitterbug.ML_hard(model = model, est = est, T_rec = T_rec)
	stats = jitterbug.eval()
	print(stats)
	return stats


def getJIT_CLAresults(seed=0, input="../new_data/corrected/", output="../results/SE_JITCLA_"):
	treatments = ["JITCLA"]
	data = load_csv(path=input)
	columns = ["Treatment"]+list(data.keys())
	result = {}
	keys = list(data.keys())
	keys.sort()
	print(keys)
	for target in keys:
		print(target)
		result[target] = [two_step_Jitterbug(data,target,est=True,inc=False,seed=seed)]
	result["Treatment"] = treatments
	# Output results to tables
	metrics = result[columns[-1]][0].keys()
	for metric in metrics:
		df = {key: (result[key] if key == "Treatment" else [dict[metric] for dict in result[key]]) for key in result}
		pd.DataFrame(df, columns=columns).to_csv(output + "unsupervised_" + metric + ".csv",
												 line_terminator="\r\n", index=False)
	data_type = "SE_JITCLA"
	with open("../dump/%s_result.pickle" % data_type, "wb") as f:
		pickle.dump(result, f)


def CLA(data, target, positiveLabel, percentileCutoff, suppress=0, experimental=0):
	treatment = Treatment(data, target)
	treatment.preprocess()
	testdata = treatment.full_test
	data = getInstancesByCLA(testdata, percentileCutoff, positiveLabel)
	treatment.y_label = ["yes" if y == 1 else "no" for y in data["Label"]]
	treatment.decisions = ["yes" if y == 1 else "no" for y in data["CLA"]]
	treatment.probs = data["K"]
	return treatment.eval()


def CLAMI(data, target, positiveLabel, percentileCutoff, suppress=0, experimental=0):
	'''
	CLAMI - Clustering, Labeling, Metric/Features Selection,
			Instance selection, and Supervised Learning
	----------

	Returns
	-------

	'''
	treatment = Treatment(data, target)
	treatment.preprocess()
	data = treatment.full_train
	testdata = treatment.full_test
	cutoffsForHigherValuesOfAttribute = getHigherValueCutoffs(data, percentileCutoff, "Label")
	print("get cutoffs")
	data = getInstancesByCLA(data, percentileCutoff, positiveLabel)
	print("get CLA instances")

	metricIdxWithTheSameViolationScores = getMetricIndicesWithTheViolationScores(data,
																				 cutoffsForHigherValuesOfAttribute,
																				 positiveLabel)
	print("get Features and the violation scores")
	# pdb.set_trace()
	keys = list(metricIdxWithTheSameViolationScores.keys())
	# start with the features that have the lowest violation scores
	keys.sort()
	for k in keys:
		selectedMetricIndices = metricIdxWithTheSameViolationScores[k]
		print(selectedMetricIndices)
		# pick those features for both train and test sets
		trainingInstancesByCLAMI = getInstancesByRemovingSpecificAttributes(data,
																			selectedMetricIndices, True)
		newTestInstances = getInstancesByRemovingSpecificAttributes(testdata,
																	selectedMetricIndices, True)
		# restart looking for the cutoffs in the train set
		cutoffsForHigherValuesOfAttribute = getHigherValueCutoffs(trainingInstancesByCLAMI,
																  percentileCutoff, "Label")
		# get instaces that violated the assumption in the train set
		instIndicesNeedToRemove = getSelectedInstances(trainingInstancesByCLAMI,
													   cutoffsForHigherValuesOfAttribute,
													   positiveLabel)
		# remove the violated instances
		trainingInstancesByCLAMI = getInstancesByRemovingSpecificInstances(trainingInstancesByCLAMI,
																		   instIndicesNeedToRemove, False)

		# make sure that there are both classes data in the training set
		zero_count = trainingInstancesByCLAMI[trainingInstancesByCLAMI["Label"] == 0].shape[0]
		one_count = trainingInstancesByCLAMI[trainingInstancesByCLAMI["Label"] == 1].shape[0]
		if zero_count > 0 and one_count > 0:
			break

	pdb.set_trace()
	results = []
	treaments = ["RF", "NB"]
	for mlAlg in treaments:
		results.append(training_CLAMI(trainingInstancesByCLAMI, newTestInstances, target, mlAlg))
	return results


def MI(data, tunedata, selectedMetricIndices, percentileCutoff, positiveLabel, target):
	print(selectedMetricIndices)
	trainingInstancesByCLAMI = getInstancesByRemovingSpecificAttributes(data,
																		selectedMetricIndices, True)
	newTuneInstances = getInstancesByRemovingSpecificAttributes(tunedata,
																selectedMetricIndices, True)
	cutoffsForHigherValuesOfAttribute = getHigherValueCutoffs(trainingInstancesByCLAMI,
															  percentileCutoff, "Label")
	instIndicesNeedToRemove = getSelectedInstances(trainingInstancesByCLAMI,
												   cutoffsForHigherValuesOfAttribute,
												   positiveLabel)
	trainingInstancesByCLAMI = getInstancesByRemovingSpecificInstances(trainingInstancesByCLAMI,
																	   instIndicesNeedToRemove, False)
	zero_count = trainingInstancesByCLAMI[trainingInstancesByCLAMI["Label"] == 0].shape[0]
	one_count = trainingInstancesByCLAMI[trainingInstancesByCLAMI["Label"] == 1].shape[0]
	if zero_count > 0 and one_count > 0:
		return selectedMetricIndices, training_CLAMI(trainingInstancesByCLAMI, newTuneInstances, target, "RF")
	else:
		return -1, -1


def transform_metric_indices(shape, indices):
	array = np.array([0] * shape)
	array[indices] = 1
	return array


def tune_CLAMI(data, target, positiveLabel, percentileCutoff, suppress=0, experimental=0, metric="APFD"):
	treatment = Treatment(data, target)
	treatment.preprocess()
	data = treatment.full_train
	sss = StratifiedShuffleSplit(n_splits=1, test_size=.25, random_state=47)
	testdata = treatment.full_test
	X, y = data[data.columns[:-1]], data[data.columns[-1]]
	for train_index, tune_index in sss.split(X, y):
		train_df = data.iloc[train_index]
		tune_df = data.iloc[tune_index]
		train_df.reset_index(drop=True, inplace=True)
		tune_df.reset_index(drop=True, inplace=True)
		cutoffsForHigherValuesOfAttribute = getHigherValueCutoffs(train_df, percentileCutoff, "Label")
		print("get cutoffs")
		train_df = getInstancesByCLA(train_df, percentileCutoff, positiveLabel)
		print("get CLA instances")

		metricIdxWithTheSameViolationScores = getMetricIndicesWithTheViolationScores(train_df,
																					 cutoffsForHigherValuesOfAttribute,
																					 positiveLabel)
		# pdb.set_trace()
		keys = list(metricIdxWithTheSameViolationScores.keys())
		# keys.sort()
		evaluated_configs = random.sample(keys, INIT_POOL_SIZE*2)
		evaluated_configs = [metricIdxWithTheSameViolationScores[k] for k in evaluated_configs]

		tmp_scores = []
		tmp_configs = []
		for selectedMetricIndices in evaluated_configs:
			selectedMetricIndices, res = MI(train_df, tune_df, selectedMetricIndices,
											percentileCutoff, positiveLabel, target)
			if isinstance(res, dict):
				tmp_configs.append(transform_metric_indices(data.shape[1], selectedMetricIndices))
				tmp_scores.append(res)

		ids = np.argsort([x[metric] for x in tmp_scores])[::-1][:1]
		best_res = tmp_scores[ids[0]]
		best_config = np.where(tmp_configs[ids[0]] == 1)[0]

		# number of eval
		this_budget = BUDGET
		eval = 0
		lives = 5
		print("Initial Population: %s" % len(tmp_scores))
		searchspace = [transform_metric_indices(data.shape[1], metricIdxWithTheSameViolationScores[k])
					   for k in keys]
		while this_budget > 0:
			cart_model = DecisionTreeRegressor()
			cart_model.fit(tmp_configs, [x[metric] for x in tmp_scores])

			cart_models = []
			cart_models.append(cart_model)
			next_config_id = acquisition_fn(searchspace, cart_models)
			next_config = metricIdxWithTheSameViolationScores[keys.pop(next_config_id)]
			searchspace.pop(next_config_id)
			next_config, next_res = MI(train_df, tune_df,
									   next_config, percentileCutoff,
									   positiveLabel, target)
			if not isinstance(next_res, dict):
				continue

			next_config_normal = transform_metric_indices(data.shape[1], next_config)
			tmp_scores.append(next_res)
			tmp_configs.append(next_config_normal)
			try:
				if abs(next_res[metric] - best_res[metric]) >= 0.03:
					lives = 5
				else:
					lives -= 1

				# pdb.set_trace()
				if isBetter(next_res, best_res, metric):
					best_config = next_config
					best_res = next_res

				if lives == 0:
					print("***" * 5)
					print("EARLY STOPPING!")
					print("***" * 5)
					break

				this_budget -= 1
				eval += 1
			except:
				pdb.set_trace()
	_, res = MI(train_df, testdata, best_config, percentileCutoff, positiveLabel, target)
	return res


def training_CLAMI(trainingInstancesByCLAMI, newTestInstances, target, model, all=True):
	pdb.set_trace()
	treatments = {"RF": RF, "SVM": SVM, "LR": LR, "NB": NB, "DT": DT, "TM": TM}
	treatment = treatments[model]
	clf = treatment(trainingInstancesByCLAMI, target)
	print(target, model)
	clf.test_data = newTestInstances[newTestInstances.columns.difference(['Label'])].values
	clf.y_label = np.array(["yes" if x == 1 else "no" for x in newTestInstances["Label"].values])

	try:
		clf.train_data = trainingInstancesByCLAMI.values[:, :-1]
		clf.x_label = np.array(["yes" if x == 1 else "no" for x in trainingInstancesByCLAMI['Label']])
		clf.train()
		results = clf.eval()
		if all:
			return results
		else:
			return results["APFD"] + results["f1"]
	except:
		pdb.set_trace()


def getViolationScores(data, labels, cutoffsForHigherValuesOfAttribute, key=-1):
	'''
	get violation scores
	----------
	data
	labels
	cutoffsForHigherValuesOfAttribute
	key

	Returns
	-------

	'''
	violation_score = 0
	if key not in ["Label", "K", "CLA"]:
		if key != -1:
			# violation score by columns
			categories = labels.values
			cutoff = cutoffsForHigherValuesOfAttribute[key]
			# violation: less than a median and class = 1 or vice-versa
			violation_score += np.count_nonzero(np.logical_and(categories == 0, np.greater(data.values, cutoff)))
			violation_score += np.count_nonzero(np.logical_and(categories == 1, np.less_equal(data.values, cutoff)))
		else:
			# violation score by rows
			row = data.values
			row_data, row_label = row[:-1], row[-1]
			# violation: less than a median and class = 1 or vice-versa
			row_label_0 = np.array(row_label == 0).tolist() * row_data.shape[0]
			violation_score += np.count_nonzero(np.logical_and(row_label_0,
															   np.greater(row_data, cutoffsForHigherValuesOfAttribute)))
			row_label_1 = np.array(row_label == 0).tolist() * row_data.shape[0]
			violation_score += np.count_nonzero(np.logical_and(row_label_1,
															   np.less_equal(row_data, cutoffsForHigherValuesOfAttribute)))

	# for attrIdx in range(data.shape[1] - 3):
	# 	# if attrIdx not in ["Label", "CLA"]:
	# 	attr_data = data[attrIdx].values
	# 	cutoff = cutoffsForHigherValuesOfAttribute[attrIdx]
	# 	violations.append(getViolationScoreByColumn(attr_data, data["Label"], cutoff))
	return violation_score


def acquisition_fn(search_space, cart_models):
    vals = []
    predicts = []
    ids = []
    ids_only = []
    for cart_model in cart_models:
        predicted = cart_model.predict(search_space)
        predicts.append(predicted)
        ids.append(np.argsort(predicted)[::1][:1])
    for id in ids:
        val = [pred[id[0]] for pred in predicts]
        vals.append(val)
        ids_only.append(id[0])

    return bazza(ids_only, vals)


def bazza(config_ids, vals, N=20):
    dim = len(vals)
    rand_vecs = [[np.random.uniform() for i in range(dim)] for j in range(N)]
    min_val = 9999
    min_id = 0
    for config_id, val in zip(config_ids, vals):
        projection_val = 0
        for vec in rand_vecs:
            projection_val += np.dot(vec, val)
        mean = projection_val/N
        if mean < min_val:
            min_val = mean
            min_id = config_id

    return min_id

def isBetter(new, old, metric):
    if metric == "d2h":
        return new[metric] < old[metric]
    else:
        return new[metric] > old[metric]


def getMetricIndicesWithTheViolationScores(data, cutoffsForHigherValuesOfAttribute, positiveLabel):
	'''
	get all the features that violated the assumption
	----------
	data
	cutoffsForHigherValuesOfAttribute
	positiveLabel

	Returns
	-------

	'''
	# cutoffs for all the columns/features
	cutoffsForHigherValuesOfAttribute = {i: x for i, x in enumerate(cutoffsForHigherValuesOfAttribute)}
	# use pandas apply per column to find the violation scores of all the features
	violations = data.apply(lambda col: getViolationScores(col, data['Label'], cutoffsForHigherValuesOfAttribute, key=col.name),
							axis = 0)
	violations = violations.values
	metricIndicesWithTheSameViolationScores = collections.defaultdict(list)

	# store the violated features that share the same violation scores together
	for attrIdx in range(data.shape[1] - 3):
		key = violations[attrIdx]
		metricIndicesWithTheSameViolationScores[key].append(attrIdx)
	return metricIndicesWithTheSameViolationScores


def plot_recall_cost(which = "overall"):
	'''
	draw the recall cost curve for all the methods 
	----------
	which

	Returns
	-------

	''''''
    path = "../dump/"+which+"_result.pickle"
    with open(path,"rb") as f:
        results = pickle.load(f)

    font = {'family': 'normal',
            'weight': 'bold',
            'size': 20}

    plt.rc('font', **font)
    paras = {'lines.linewidth': 5, 'legend.fontsize': 20, 'axes.labelsize': 30, 'legend.frameon': False,
             'figure.autolayout': True, 'figure.figsize': (16, 8)}

    plt.rcParams.update(paras)

    lines = ['-',':','--',(0,(4,2,1,2)),(0,(3,2)),(0,(2,1,1,1))]

    for project in results:
        fig = plt.figure()
        for i,treatment in enumerate(results[project]):
            plt.plot(results[project][treatment]["CostR"], results[project][treatment]["TPR"], linestyle = lines[i], label=treatment)
        plt.legend()
        plt.ylabel("Recall")
        plt.xlabel("Cost")
        plt.grid()
        plt.savefig("../figures_"+which+"/" + project + ".png")
        plt.close(fig)

if __name__ == "__main__":
	eval(cmd())


