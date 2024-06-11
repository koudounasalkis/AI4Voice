import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

import argparse

""" 
    This code computes the entropy-based ensemble of the two models
"""
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="vocal")
    parser.add_argument(
        "--cs_confidence",
        help="CS confidence scores",
        default="cs_confidence_scores.csv",
        type=str,
        required=False)
    parser.add_argument(
        "--sv_confidence",
        help="SV confidence scores",
        default="sv_confidence_scores.csv",
        type=str,
        required=False)
    parser.add_argument(
        "--df_train",
        help="Path to train file",
        default="data/df_train.csv",
        type=str,
        required=False)
    parser.add_argument(
        "--label",
        help="Label to predict; choose one from ['s/p', 'category', 'macro-category']",
        default="category",
        type=str,
        required=False)
    parser.add_argument(
        "--save_ensemble_confidence_scores",
        help="whether to save ensemble confidence scores or not",
        action="store_true",
        required=False)
    args = parser.parse_args()

    ## Load confidence scores
    results_cs = pd.read_csv(args.cs_confidence, index_col=None)
    results_sv = pd.read_csv(args.sv_confidence, index_col=None)
    print(results_cs.columns)

    ## Get label names
    df_train = pd.read_csv(args.df_train, index_col=None)
    categories = df_train[args.label].unique().tolist()
    label2id, id2label = dict(), dict()
    for i, label in enumerate(categories):
        label2id[label] = str(i)
        id2label[str(i)] = label
    labels = [int(label2id[c]) for c in results_cs['category']]
    
    ## Check predictions and metrics
    prediction_cs = results_cs[categories].values.argmax(axis=-1)
    print("\nCS predictions")
    print("F1 Macro: ", round(f1_score(labels, prediction_cs, average="macro"),4))
    print("Accuracy: ", round(accuracy_score(labels, prediction_cs),4))
    if args.label == "s/p":
        print("AUC: ", round(roc_auc_score(labels, prediction_cs),4))
    else:
        # print top-3 accuracy
        top3 = 0
        for i, (pred, label) in enumerate(zip(prediction_cs, labels)):
            top3 += int(label in np.argsort(results_cs[categories].values[i])[-3:])
        print("Top-3 Accuracy: ", round(top3/len(labels),4))

    print("----------------\n")

    prediction_sv = results_sv[categories].values.argmax(axis=1)
    labels_sv = [int(label2id[c]) for c in results_sv['category']]
    print("SV predictions")
    print("F1 Macro: ", round(f1_score(labels_sv, prediction_sv, average="macro"),4))
    print("Accuracy: ", round(accuracy_score(labels_sv, prediction_sv),4))
    if args.label == "s/p":
        print("AUC: ", round(roc_auc_score(labels_sv, prediction_sv),4))
    else:
        # print top-3 accuracy
        top3 = 0
        for i, (pred, label) in enumerate(zip(prediction_sv, labels)):
            top3 += int(label in np.argsort(results_sv[categories].values[i])[-3:])
        print("Top-3 Accuracy: ", round(top3/len(labels),4))
    print("----------------\n")

    ## Ensemble
    print("ENSEMBLE predictions: entropy")
    prediction_proba_cs = results_cs[categories].values
    prediction_proba_sv = results_sv[categories].values
    threshold_entropy = list(np.arange(0.1, 3.0, 0.1))
    best_accuracy = 0
    best_f1 = 0
    best_auc = 0
    best_top3_acc = 0
    best_te = 0
    best_predictions = []
    best_ensemble_confidence_scores = []
    for te in threshold_entropy:
        ensemble_confidence_scores = []
        final_predictions_entropy = []
        for proba_cs, proba_sv, pred_cs, pred_sv in zip(prediction_proba_cs, prediction_proba_sv, prediction_cs, prediction_sv):
            entropy_cs = np.sum(- proba_cs * np.log(proba_cs))
            entropy_sv = np.sum(- proba_sv * np.log(proba_sv))
            entropy = np.argmin(np.array([entropy_cs, entropy_sv]))
            if entropy_sv < te:
                final_predictions_entropy.append(pred_cs)
                ensemble_confidence_scores.append(proba_cs)
            else:
                final_predictions_entropy.append(pred_sv)
                ensemble_confidence_scores.append(proba_sv)
        
        accuracy = accuracy_score(labels, final_predictions_entropy)
        f1 = f1_score(labels, final_predictions_entropy, average="macro")
        if accuracy > best_accuracy or (accuracy == best_accuracy and  f1 > best_f1):
            best_accuracy = accuracy
            best_f1 = f1
            if args.label == "s/p":
                best_auc = roc_auc_score(labels, final_predictions_entropy)
            else:
                top3 = 0
                for i, (pred, label) in enumerate(zip(final_predictions_entropy, labels)):
                    top3 += int(label in np.argsort(results_cs[categories].values[i])[-3:])
                best_top3_acc = top3/len(labels)
            best_te = te
            best_predictions = final_predictions_entropy
            best_ensemble_confidence_scores = ensemble_confidence_scores

    print("----------------")
    print("Threshold ", best_te)
    print("F1 Macro: ", round(best_f1,4))
    print("Accuracy: ", round(best_accuracy,4))
    if args.label == "s/p":
        print("AUC: ", round(best_auc,4))
    else:
        print("Top-3 Accuracy: ", round(best_top3_acc,4))
    print("----------------\n")
    if args.save_ensemble_confidence_scores:
        print("Saving ensemble confidence scores...")
        ensemble = pd.DataFrame(best_ensemble_confidence_scores, columns=categories)
        ensemble['category'] = results_cs['category']
        ensemble.to_csv("ensemble_confidence_scores.csv", index=False)


    ## Ensemble with maximum confidence
    print("ENSEMBLE predictions: maximum")
    prediction_proba_cs = results_cs[categories].values
    prediction_proba_sv = results_sv[categories].values

    final_predictions_max = []
    for proba_cs, proba_sv, pred_cs, pred_sv in zip(prediction_proba_cs, prediction_proba_sv, prediction_cs, prediction_sv):
        max_cs = np.max(proba_cs)
        max_sv = np.max(proba_sv)
        if max_cs > max_sv:
            final_predictions_max.append(pred_cs)
        else:
            final_predictions_max.append(pred_sv)

    print("----------------")
    print("F1 Macro: ", round(f1_score(labels, final_predictions_max, average="macro"),4))
    print("Accuracy: ", round(accuracy_score(labels, final_predictions_max),4))
    if args.label == "s/p":
        print("AUC: ", round(roc_auc_score(labels, final_predictions_max),4))
    else:
        top3 = 0
        for i, (pred, label) in enumerate(zip(final_predictions_max, labels)):
            top3 += int(label in np.argsort(results_cs[categories].values[i])[-3:])
        print("Top-3 Accuracy: ", round(top3/len(labels),4))
    print("----------------\n")
    
    ## Ensemble with minimum confidence
    print("ENSEMBLE predictions: mimimum")
    prediction_proba_cs = results_cs[categories].values
    prediction_proba_sv = results_sv[categories].values

    final_predictions_min = []
    for proba_cs, proba_sv, pred_cs, pred_sv in zip(prediction_proba_cs, prediction_proba_sv, prediction_cs, prediction_sv):
        min_cs = np.min(proba_cs)
        min_sv = np.min(proba_sv)
        if min_cs > min_sv:
            final_predictions_min.append(pred_cs)
        else:
            final_predictions_min.append(pred_sv)

    print("----------------")
    print("F1 Macro: ", round(f1_score(labels, final_predictions_min, average="macro"),4))
    print("Accuracy: ", round(accuracy_score(labels, final_predictions_min),4))
    if args.label == "s/p":
        print("AUC: ", round(roc_auc_score(labels, final_predictions_min),4))
    else:
        top3 = 0
        for i, (pred, label) in enumerate(zip(final_predictions_min, labels)):
            top3 += int(label in np.argsort(results_cs[categories].values[i])[-3:])
        print("Top-3 Accuracy: ", round(top3/len(labels),4))
    print("----------------\n")
    
    ## Ensemble with average confidence
    print("ENSEMBLE predictions: average")
    prediction_proba_cs = results_cs[categories].values
    prediction_proba_sv = results_sv[categories].values

    final_predictions_avg = []
    for proba_cs, proba_sv, pred_cs, pred_sv in zip(prediction_proba_cs, prediction_proba_sv, prediction_cs, prediction_sv):
        avg_cs = np.mean(proba_cs)
        avg_sv = np.mean(proba_sv)
        if avg_cs > avg_sv:
            final_predictions_avg.append(pred_cs)
        else:
            final_predictions_avg.append(pred_sv)
    
    print("----------------")
    print("F1 Macro: ", round(f1_score(labels, final_predictions_avg, average="macro"),4))
    print("Accuracy: ", round(accuracy_score(labels, final_predictions_avg),4))
    if args.label == "s/p":
        print("AUC: ", round(roc_auc_score(labels, final_predictions_avg),4))
    else:
        top3 = 0
        for i, (pred, label) in enumerate(zip(final_predictions_avg, labels)):
            top3 += int(label in np.argsort(results_cs[categories].values[i])[-3:])
        print("Top-3 Accuracy: ", round(top3/len(labels),4))
    print("----------------\n")