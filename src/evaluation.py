from __future__ import division, print_function

from collections import Counter
import numpy as np
import pdb
from clami import *
from learners import *

# jitterbug.find_patterns()
# jitterbug.easy_code()
# jitterbug.test_patterns()

class Easy_CLAMI(object):
    def __init__(self, data, target, thres=0.05):
        self.data = data
        self.target = target
        self.thres = thres
        self.stats_test = {'tp': 0, 'p': 0}

    def preprocess(self):
        treatment = Treatment(self.data, self.target)
        treatment.preprocess()
        testdata = treatment.full_test
        traindata = treatment.full_train
        self.train_data = getInstancesByCLA(traindata, 50, None)
        self.x_label = np.array(["yes" if y == 1 else "no" for y in self.train_data["Label"]])
        self.test_data = getInstancesByCLA(testdata, 50, None)
        self.y_label = np.array(["yes" if y == 1 else "no" for y in self.test_data["Label"]])

    def find_patterns(self):
        self.thresholds = [np.percentile(self.train_data['K'], 50 + x*5) for x in range(1, 9, 1)]
        self.best_thres = -1
        best_prec = -1
        self.left_train = None
        for t in self.thresholds:
            left_train = range(self.train_data.shape[0])
            left_train, stats_train = self.remove(self.train_data, self.x_label, left_train, t)
            print(t, stats_train)
            prec = float(stats_train["tp"]) / (stats_train["p"] + 0.00001)
            if prec > best_prec:
                self.left_train = left_train
                best_prec = prec
                self.best_thres = t
        print("Best Threshold : %s,  Best Prec : %s" % (self.best_thres, best_prec))

    def remove(self, data, label, left, thres):
        K = data['K'].values
        p_arr = np.where(K > thres)[0]
        tp_arr = np.where(label == "yes")[0]
        tp_arr = np.intersect1d(p_arr, tp_arr)
        p = p_arr.shape[0]
        tp = tp_arr.shape[0]
        # for row in left:
        #     if data.iloc[row]['K'] > thres:
        #         to_remove.add(row)
        #         p += 1
        #         if label[row] == "yes":
        #             tp += 1

        #left = list(set(left) - to_remove)
        left = np.setdiff1d(left, p_arr)
        return left, {"p": p, "tp": tp}

    def test_patterns(self,output=False):
        left_test = range(self.test_data.shape[0])
        self.stats_test = {"tp": 0, "p": 0}
        self.left_test, stats_test = self.remove(self.test_data, self.y_label, left_test, self.best_thres)
        self.stats_test["tp"] += stats_test["tp"]
        self.stats_test["p"] += stats_test["p"]
        # save the "hard to find" data
        if output:
            self.rest = self.data[self.target].loc[left_test]
            self.rest.to_csv("../new_data/rest/csc/"+self.target+".csv", line_terminator="\r\n", index=False)
        return self.stats_test




class Eval():
    def __init__(self, y_label):
        self.y_label = y_label

    def confusion(self, decisions):
        tp, fp, fn, tn = 0, 0, 0, 0
        for i, d in enumerate(decisions):
            gt = self.y_label[i]
            if d == "yes" and gt == "yes":
                tp += 1
            elif d == "yes" and gt == "no":
                fp += 1
            elif d == "no" and gt == "yes":
                fn += 1
            elif d == "no" and gt == "no":
                tn += 1
        return tp, fp, fn, tn


    def retrieval_curves(self, labels):
        stat = Counter(labels)
        t = stat["yes"]
        n = stat["no"]
        tp = 0
        fp = 0
        tn = n
        fn = t
        cost = 0
        costs = [cost]
        tps = [tp]
        fps = [fp]
        tns = [tn]
        fns = [fn]
        for label in labels:
            cost += 1.0
            costs.append(cost)
            if label == "yes":
                tp += 1.0
                fn -= 1.0
            else:
                fp += 1.0
                tn -= 1.0
            fps.append(fp)
            tps.append(tp)
            tns.append(tn)
            fns.append(fn)
        costs = np.array(costs)
        tps = np.array(tps)
        fps = np.array(fps)
        tns = np.array(tns)
        fns = np.array(fns)

        tpr = tps / (tps + fns)
        fpr = fps / (fps + tns)
        costr = costs / (t + n)
        return {"TPR": tpr, "FPR": fpr, "CostR": costr}


    def AUC(self, ys, xs):
        assert len(ys) == len(xs), "Size must match."
        x_last = 0
        if xs[-1] < 1.0:
            xs.append(1.0)
            ys.append(ys[-1])
        auc = 0.0
        for i, x in enumerate(xs):
            y = ys[i]
            auc += y * (x - x_last)
            x_last = x
        return auc


    def eval(self):
        tp, fp, fn, tn = self.confusion(decisions)
        result = {}
        if tp == 0:
            result["precision"] = 0
            result["recall"] = 0
            result["f1"] = 0
        else:
            result["precision"] = float(tp) / (tp + fp)
            result["recall"] = float(tp) / (tp + fn)
            result["f1"] = 2 * result["precision"] * result["recall"] / (result["precision"] + result["recall"])
        if fp == 0:
            result["fall-out"] = 0
        else:
            result["fall-out"] = float(fp) / (fp + tn)

        order = np.argsort(self.probs)[::-1]
        labels = np.array(self.y_label)[order]
        rates = self.retrieval_curves(labels)
        for r in rates:
            result[r] = rates[r]
        result["AUC"] = self.AUC(rates["TPR"], rates["FPR"])
        result["APFD"] = self.AUC(rates["TPR"], rates["CostR"])
        result["p@10"] = Counter(labels[:10])["yes"] / float(len(labels[:10]))
        result["p@100"] = Counter(labels[:100])["yes"] / float(len(labels[:100]))
        result["g1"] = (2 * result["recall"] * (1 - result["fall-out"])) / (result["recall"] + 1 - result["fall-out"])
        return result