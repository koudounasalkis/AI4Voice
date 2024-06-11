import torch
import pandas as pd
import argparse
import numpy as np
import os
import random
from sklearn.metrics import confusion_matrix

from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

from dataset import Dataset
from utils import WeightedTrainer, define_training_args, compute_class_weights
from utils import compute_metrics, compute_metrics_binary

import warnings
warnings.filterwarnings("ignore")

""" Define Command Line Parser """
def parse_cmd_line_params():
    parser = argparse.ArgumentParser(description="vocal")
    parser.add_argument(
        "--batch",
        help="batch size",
        default=8, 
        type=int,
        required=False)
    parser.add_argument(
        "--df_train",
        help="path to the train df",
        default="data/df_train.csv",
        type=str,
        required=False) 
    parser.add_argument(
        "--df_test",
        help="path to the test df",
        default="data/df_test.csv",
        type=str,
        required=False)
    parser.add_argument(
        "--feature_extractor",
        help="model to use for training",
        default="facebook/hubert-base-ls960",  
        type=str,                          
        required=False)   
    parser.add_argument(
        "--model",
        help="model to use for training",
        default="facebook/hubert-base-ls960",  
        type=str,                          
        required=False)                     
    parser.add_argument(
        "--output_dir",
        help="path to the output directory",
        default="results",
        type=str,
        required=False)
    parser.add_argument(
        "--label",
        help="Label to predict; choose one from ['s/p', 'category', 'macro-category']",
        default="category",
        type=str,
        required=False)
    parser.add_argument(
        "--save_confidence_scores",
        help="whether to save confidence scores or not",
        action="store_true",
        required=False)
    parser.add_argument(
        "--tts",
        help="whether to validate on tts samples or not",
        action="store_true",
        required=False)
    parser.add_argument(
        "--confusion_matrix",
        help="whether to save confusion matrix or not",
        action="store_true",
        required=False)
    parser.add_argument(
        "--max_duration",
        help="Maximum audio duration",
        default=10.0,
        type=float,
        required=False)
    args = parser.parse_args()
    return args



""" Read and Process Data"""
def read_data(df_train_path, df_test_path, label_name):
    df_train = pd.read_csv(df_train_path, index_col=None)
    df_test = pd.read_csv(df_test_path, index_col=None)
    
    ## Prepare Labels
    labels = df_train[label_name].unique()
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label
    num_labels = len(id2label)

    ## Train
    for index in range(0,len(df_train)):
        df_train.loc[index,'label'] = label2id[df_train.loc[index,label_name]]
    df_train['label'] = df_train['label'].astype(int)

    ## Validation
    for index in range(0,len(df_test)):
        df_test.loc[index,'label'] = label2id[df_test.loc[index,label_name]]
    df_test['label'] = df_test['label'].astype(int)

    return df_train, df_test, num_labels, label2id, id2label, labels
 

""" Main Program """
if __name__ == '__main__':

    ## Utils 
    args = parse_cmd_line_params()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    ## Set seed 
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    ## Train & Test df
    df_train, df_test, num_labels, label2id, id2label, labels = read_data(args.df_train, args.df_test, args.label)

    ## Model & Feature Extractor
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.feature_extractor)
    print("------------------------------------")
    print(f"Loading model from {args.model}")
    model = AutoModelForAudioClassification.from_pretrained(
        args.model, 
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
        local_files_only=True
        ).to(device)
    print("Model loaded successfully!")
    print("------------------------------------\n")

    ## Test Dataset and Class Weights
    print("----------------------------------")
    print("Loading Test dataset...")
    test_dataset = Dataset(
        examples=df_test, 
        feature_extractor=feature_extractor, 
        max_duration=args.max_duration
        )
    print("Dataset loaded successfully!")
    print("----------------------------------\n")

    ## Training Arguments
    training_arguments = define_training_args(args.output_dir, args.batch)
    class_weights = compute_class_weights(df_train)

    ## Trainer 
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_arguments,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics_binary if args.label=='s/p' else compute_metrics)

    ## Evaluate
    print("----------------------------------")
    print("Evaluating...")
    predictions = trainer.predict(test_dataset)

    ## Compute confidence scores
    if args.save_confidence_scores:
        print("Saving confidence scores...")
        sof = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=-1)
        df = pd.DataFrame(sof, columns=labels)
        df["category"] = [id2label[str(label_id)] for label_id in predictions.label_ids]
        output_name = f"{args.output_dir}/confidence_scores_tts" if args.tts \
            else f"{args.output_dir}/confidence_scores"
        df.to_csv(f"{output_name}.csv", index=False)


    ## Compute confusion matrix
    if args.confusion_matrix:
        print("Saving confusion matrix...")
        preds = np.argmax(predictions.predictions, axis=1)
        cm = confusion_matrix(predictions.label_ids, preds)
        output_name = f"{args.output_dir}/confusion_matrix_tts" if args.tts \
            else f"{args.output_dir}/confusion_matrix"
        print(cm)
        np.save(output_name, cm)

        # plot confusion matrix
        import matplotlib.pyplot as plt
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(10,10))
        sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues, cbar=False)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(f"{args.output_dir}/confusion_matrix.png")
        plt.close()