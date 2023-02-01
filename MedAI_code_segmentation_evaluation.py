
import os
import cv2
import random
import warnings
import argparse
import glob

import numpy as np

from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import torch

random.seed(0)
np.random.seed(0)

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="")

parser.add_argument("-i", "--submission-dir", type=str)
parser.add_argument("-o", "--output-dir", type=str)
parser.add_argument("-t", "--truth-dir", type=str)

SUPPORTED_FILETYPES = [".jpg", ".jpeg", ".png"]
CSV_VAL_ORDER = ["accuracy", "jaccard", "dice", "f1", "recall", "precision"]

def read_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask.max() > 127:
        mask = mask / 255.0
    mask = mask > 0.5
    mask = mask.astype(np.uint8)
    mask = mask.reshape(-1)
    return mask

def filter_filtypes(path):
    _, fileext = os.path.splitext(path)
    return fileext in SUPPORTED_FILETYPES

def dice_score(y_true, y_pred):
    return np.sum(y_pred[y_true == 1] == 1) * 2.0 / (np.sum(y_pred[y_pred == 1] == 1) + np.sum(y_true[y_true == 1] == 1))

def TP_TN_FP_FN(true, pred):
    minus = true - pred
    FP = (minus==-1).float().sum(dim=1)
    FN = (minus==1).float().sum(dim=1)
    TP_plus_TN = (minus==0).float().sum(dim=1)

    multi = true * pred
    TP = multi.sum(dim=1)
    TN = TP_plus_TN - TP
    return TP, TN, FP, FN

def calculate_metrics_torch(true, pred, ROI='polyp',metrics=None, reduction=None,
                            cloned_detached=False):
    '''The input are tensors of shape (batch, C, H, W)'''

    batch_size = pred.shape[0]
    # (batch, C, H, W)->(batch,HW)
    if cloned_detached:
        true = true.argmax(dim=1).view(batch_size, -1).float()
        pred = pred.argmax(dim=1).view(batch_size, -1).float()
    else:
        true = true.clone().detach().argmax(dim=1).view(batch_size, -1).float()
        pred = pred.clone().detach().argmax(dim=1).view(batch_size, -1).float()

    #-------------------------------------------
    if metrics== None:
        metrics = 'accuracy', 'jaccard', 'dic', 'recall', 'precision'
    elif type(metrics)==str:
        metrics = [metrics]

    #-------------------------------------------
    if ROI=='polyp':
        pass
    elif ROI=='background':
        true = 1- true
        pred = 1- pred
    true.requires_grad = False
    pred.requires_grad = False
    #--------------------------------------------
    TP, TN, FP, FN =TP_TN_FP_FN(true,pred)
    #--------------------------------------------
    results = []
    for metric in metrics:
        result = 0
        if metric=='jaccard':
            iou = TP/(TP + FN + FP)
            result = iou
        elif metric=='accuracy':
            acc = (TP+TN)/(TP + FN + FP + TN)
            result = acc
        elif metric=='dic':
            dic = 2*TP/(2*TP + FP + FN)
            result = dic
        elif metric=='recall':
            recall = TP/(TP+FN)
            result = recall
        elif metric == 'precision':
            TP_FP = (TP + FP)
            TP_FP[TP_FP==0]+=1
            prec = TP / TP_FP
            result = prec
        else:
            continue

        if reduction=='mean':
            result = torch.mean(result).cpu().numpy().round(5)
        else:
            result = result.cpu().numpy().round(5)

        results.append(result)

    results = np.array(results)
    results_dic = {metric:results[index] for index, metric in enumerate(metrics)}
    if len(results)==1:
        results_dic = results[0]

    return results_dic

def calculate_metrics(y_true, y_pred):
    score_accuracy = accuracy_score(y_true, y_pred)
    score_jaccard = jaccard_score(y_true, y_pred, average="binary")
    #score_f1 = f1_score(y_true, y_pred, average="binary") #It is equal to Dice!!
    score_recall = recall_score(y_true, y_pred, average="binary")
    score_precision = precision_score(y_true, y_pred, average="binary", zero_division=0)
    score_dice = dice_score(y_true, y_pred)
    # tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    # score_specificity = tn / (tn + fp)
    return [score_accuracy, score_jaccard, score_dice, score_recall, score_precision]


def get_filename(filepath):
    return os.path.splitext(os.path.basename(filepath))[0]

def invert_mask(mask):
    _mask = mask.copy()
    mask[_mask == 1] = 0
    mask[_mask == 0] = 1
    return mask

def evaluate_submission(submission_dir, output_dir, ground_truth_dir):

    submission_attributes = os.path.basename(submission_dir).split("_")
    
    team_name = submission_attributes[1]
    run_id = "_".join(submission_attributes[2:-1])
    task_name = submission_attributes[-1]

    team_result_path = os.path.join(output_dir, team_name, task_name, run_id)

    if not os.path.exists(team_result_path):
        os.makedirs(team_result_path)
        
    true_masks = sorted(glob.glob(os.path.join(ground_truth_dir, "*")))

    pred_masks = sorted(glob.glob(os.path.join(submission_dir, "*")))
    pred_masks = list(filter(filter_filtypes, pred_masks))

    mean_score = []

    detailed_metrics_filename = "%s_%s_%s_detailed_metrics.csv" % (team_name, task_name, run_id)
    average_metrics_filename = "%s_%s_%s_average_metrics.csv" % (team_name, task_name, run_id)

    with open(os.path.join(team_result_path, detailed_metrics_filename), "w") as f:

        f.write("filename;%s\n" % ";".join(CSV_VAL_ORDER))

        assert len(true_masks) == len(pred_masks)

        for index, (y_true_path, y_pred_path) in enumerate(zip(true_masks, pred_masks)):

            print("Progress [%i / %i]" % (index + 1, len(true_masks)), end="\r")
            
            assert get_filename(y_true_path) == get_filename(y_pred_path)

            y_true = read_mask(y_true_path)
            y_pred = read_mask(y_pred_path)

            if y_true.max() == 0:
                y_true = invert_mask(y_true)
                y_pred = invert_mask(y_pred)

            metrics = calculate_metrics(y_true, y_pred)

            results_line = "%s;" % get_filename(y_true_path)
            results_line += ";".join(["%0.4f" % score for score in metrics])
            results_line += "\n"
            
            f.write(results_line)

            mean_score.append(metrics)

        print("\n")

    mean_score = np.mean(mean_score, axis=0)

    with open(os.path.join(team_result_path, average_metrics_filename), "w") as f:
        f.write("metric;value\n")
        f.write("\n".join(["%s;%0.4f" % (header, score) for header, score in zip(CSV_VAL_ORDER, mean_score)]))

    with open(os.path.join(output_dir, "%s_all_average_metrics.csv" % task_name), "a") as f:
        f.write("%s;%s;%s;" % (team_name, task_name, run_id))
        f.write(";".join(["%0.4f" % score for score in mean_score]))
        f.write("\n")

def IOU_class01(target, predicted,count=0):

    # if target.shape != predicted.shape:
    #     print("target has dimension", target.shape, ", predicted values have shape", predicted.shape)
    #     return

    # if target.dim() != 3:
    #     print("target has dim", target.dim(), ", Must be 4.")
    #     return
    target = target.clone().detach().cpu()
    predicted = predicted.clone().detach().cpu()  # simgoid is implecitly applied with calculating the loss

    # if count%80==0:
    #     show_image(target.numpy(),predicted.numpy())





    iou_list = []
    # for each image in the batch
    for i in range(target.shape[0]):
        target_arr = target[i,:, :, :].numpy().argmax(0)
        predicted_arr = predicted[i, :, :, :].numpy().argmax(0)
        iou_score_1 =  _iou(target_arr,predicted_arr)
        iou_score_0 = _iou(1-target_arr, 1-predicted_arr)

        iou_list.append((iou_score_0+iou_score_1)/2)


    # miou = iousum / target.shape[0]
    return np.mean(iou_list)

def _iou(target_arr,predicted_arr):
    intersection = np.logical_and(target_arr, predicted_arr).sum()
    union = np.logical_or(target_arr, predicted_arr).sum()
    if union == 0:
        if np.sum(target_arr) == 0: #if the image doesn't have a polyp at all, then, nothing to calculate here
            iou_score = 1
        else:
            iou_score = 0
    else:
        iou_score = intersection / union
    return iou_score
if __name__ == "__main__":

    args = parser.parse_args()

    submission_dir = args.submission_dir
    output_dir = args.output_dir
    truth_dir = args.truth_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, "%s_all_average_metrics.csv" % os.path.basename(submission_dir)), "w") as f:
        f.write("team-name;task-name;run-id;%s\n" % ";".join(CSV_VAL_ORDER)) 

    for submission_dir in glob.glob(os.path.join(submission_dir, "*")):
        print("Evaluating %s..." % submission_dir)
        evaluate_submission(submission_dir, output_dir, truth_dir)