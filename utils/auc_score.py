# CODE FROM XIAORAN CHEN https://github.com/aubreychen9012/AutoEncoder_AnomalyDetection

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.utils import column_or_1d
from pdb import set_trace as bp


def compute_tpr_fpr(y_true, y_score, threshold):
    y_true = column_or_1d(y_true)
    y_score = column_or_1d(y_score)

    sort_idxs = np.argsort(y_score, kind = "mergesort")
    y_true = y_true[sort_idxs]
    y_score = y_score[sort_idxs]

    n_fp = [0]*len(threshold)
    n_tp = [0]*len(threshold)
    #tpfn = [0]*len(threshold)
    #fptn = [0]*len(threshold)

    for i in range(len(threshold)):
        stop = np.searchsorted(y_score, threshold[i], 'right')
        tp = np.sum(y_true[stop:]>0.5)
        fp = np.sum(y_score>threshold[i])-tp
        # y_score_binary = (y_score>=i)
        # fp = np.logical_and(y_true !=  y_score_binary,  y_score_binary != 0).sum()
        # fn = np.logical_and(y_true !=  y_score_binary,  y_score_binary == 0).sum()
        # tp = np.logical_and(y_true ==  y_score_binary, y_true != 0).sum()
        # tn = np.logical_and(y_true ==  y_score_binary, y_true == 0).sum()
        #tn, fp, fn, tp = confusion_matrix(y_true, y_score_binary).ravel()
        #tpr = tp/(tp+fn)
        #fpr = fp/(fp+tn)
        n_fp[i]=fp
        n_tp[i]=tp
        #tpfn[i]=tp+fn
        #fptn[i]=fp+tn
        # n_tp.append(tp)
        # n_fp.append(fp)
        # tpfn.append(tp+fn)
        # fptn.append(fp+tn)
    return np.stack((n_tp,n_fp))


def compute_area_threshold(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(roc_true, res_array)
    area = auc(fpr, tpr)
    dif_r = tpr-fpr
    threshold_max = thresholds[argmax(dif_r)]
    return area, threshold_max



def compute_auc_score(tpr, fpr):
    def _binary_roc_auc_score(y_true, y_score, sample_weight=None):
        if len(np.unique(y_true)) != 2:
            raise ValueError("Only one class present in y_true. ROC AUC score "
                             "is not defined in that case.")

        fpr, tpr, _ = roc_curve(y_true, y_score,
                                sample_weight=sample_weight)
        if max_fpr is None or max_fpr == 1:
            return auc(fpr, tpr)
        if max_fpr <= 0 or max_fpr > 1:
            raise ValueError("Expected max_frp in range ]0, 1], got: %r"
                             % max_fpr)

        # Add a single point at max_fpr by linear interpolation
        stop = np.searchsorted(fpr, max_fpr, 'right')
        x_interp = [fpr[stop - 1], fpr[stop]]
        y_interp = [tpr[stop - 1], tpr[stop]]
        tpr = np.append(tpr[:stop], np.interp(max_fpr, x_interp, y_interp))
        fpr = np.append(fpr[:stop], max_fpr)
        partial_auc = auc(fpr, tpr)

        # McClish correction: standardize result to be 0.5 if non-discriminant
        # and 1 if maximal
        min_area = 0.5 * max_fpr**2
        max_area = max_fpr
        return 0.5 * (1 + (partial_auc - min_area) / (max_area - min_area))

    y_type = type_of_target(y_true)
    if y_type == "binary":
        labels = np.unique(y_true)
        y_true = label_binarize(y_true, labels)[:, 0]

    return _average_binary_score(
        _binary_roc_auc_score, y_true, y_score, average,
sample_weight=sample_weight)


