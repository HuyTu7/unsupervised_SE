from __future__ import division, print_function

from collections import Counter
import numpy as np

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