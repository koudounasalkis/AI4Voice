import torch
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import pandas as pd
import argparse
import random
import numpy as np
import os
import librosa

import warnings
warnings.filterwarnings("ignore")

from dataset import Dataset
from utils import WeightedTrainer, define_training_args, \
    compute_metrics, compute_class_weights, compute_metrics_binary
    

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
        "--steps",
        help="number of training steps",
        default=300, ## 2500 for category
        type=int,
        required=False)
    parser.add_argument(
        "--feature_extractor",
        help="feature extractor to use",
        default="facebook/hubert-base-ls960",  
        type=str,                          
        required=False) 
    parser.add_argument(
        "--model",
        help="model to use",
        default="facebook/hubert-base-ls960",  
        type=str,                          
        required=False)                     
    parser.add_argument(
        "--df_train",
        help="path to the train df",
        default="data/df_train.csv",
        type=str,
        required=False) 
    parser.add_argument(
        "--df_val",
        help="path to the val df",
        default="data/df_val.csv",
        type=str,
        required=False) 
    parser.add_argument(
        "--df_test",
        help="path to the test df",
        default="data/df_test.csv",
        type=str,
        required=False)  
    parser.add_argument(
        "--save_confidence_scores",
        help="whether to save confidence scores or not",
        action="store_true",
        required=False)
    parser.add_argument(
        "--output_dir",
        help="path to the output directory",
        default="results",
        type=str,
        required=False)
    parser.add_argument(
        "--warmup_steps",
        help="number of warmup steps",
        default=10,
        type=int,
        required=False)
    parser.add_argument(
        "--lr",
        help="learning rate",
        default=1e-4,
        type=float,
        required=False)
    parser.add_argument(
        "--max_duration",
        help="Maximum audio duration",
        default=10.0,
        type=float,
        required=False)
    parser.add_argument(
        "--label",
        help="Label to predict; choose one from ['s/p', 'category', 'macro-category']",
        default="category",
        type=str,
        required=False)
    parser.add_argument(
        "--augmentation",
        help="whether to augment or not the data",
        action="store_true",
        required=False)
    args = parser.parse_args()
    return args


""" Read and Process Data"""
def read_data(df_train_path, df_val_path, label_name):
    df_train = pd.read_csv(df_train_path, index_col=None)
    df_val = pd.read_csv(df_val_path, index_col=None)
    
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
    for index in range(0,len(df_val)):
        df_val.loc[index,'label'] = label2id[df_val.loc[index,label_name]]
    df_val['label'] = df_val['label'].astype(int)

    print("Label2Id: ", label2id)
    print("Id2Label: ", id2label)
    print("Num Labels: ", num_labels)

    return df_train, df_val, num_labels, label2id, id2label, labels


""" Define model and feature extractor """
def define_model(
    model_checkpoint, 
    feature_extractor_checkpoint, 
    num_labels, 
    label2id, 
    id2label, 
    device="cuda"
    ):
    feature_extractor = AutoFeatureExtractor.from_pretrained(feature_extractor_checkpoint)
    if model_checkpoint == "facebook/hubert-base-ls960" \
        or model_checkpoint == "facebook/wav2vec2-base" \
        or model_checkpoint == "microsoft/wavlm-base-plus" \
        or model_checkpoint == "ALM/hubert-base-audioset":
        print(f"Loading {model_checkpoint} model from HF Hub...")
        model = AutoModelForAudioClassification.from_pretrained(
            model_checkpoint, 
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label).to(device)
    else:
        print(f"Loading {model_checkpoint} model from local files...")
        model = AutoModelForAudioClassification.from_pretrained(
            model_checkpoint, 
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
            local_files_only=True,
            ignore_mismatched_sizes=True).to(device)
    return feature_extractor, model


""" Main Program """
if __name__ == '__main__':

    ## Utils 
    args = parse_cmd_line_params()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    print("------------------------------------")
    print("Running with the following parameters:")
    print("Batch size: ", args.batch)
    print("Number of steps: ", args.steps)
    print("Model: ", args.model)
    print("Warmup steps: ", args.warmup_steps)
    print("Learning rate: ", args.lr)
    print("Maximum audio duration: ", args.max_duration)
    print("Label to predict: ", args.label)
    print("------------------------------------\n")

    for seed in range(0,10):

        ## Set seed 
        print("Seed: ", seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        output_dir = os.path.join(args.output_dir, str(seed))
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir) 
        
        ## Train & Validation df
        df_train, df_val, num_labels, label2id, id2label, labels = read_data(
            args.df_train, 
            args.df_val, 
            args.label
            )
        print("Num labels: ", num_labels)

        ## Model & Feature Extractor
        model_checkpoint = args.model
        model_name = model_checkpoint.split("/")[-1]
        feature_extractor, model = define_model(
            model_checkpoint, 
            args.feature_extractor, 
            num_labels, 
            label2id, 
            id2label, 
            device
            )

        ## Train & Val Datasets 
        max_duration = args.max_duration
        train_dataset = Dataset(
            examples=df_train, 
            feature_extractor=feature_extractor, 
            max_duration=max_duration,
            augmentation=args.augmentation
            )
        val_dataset = Dataset(
            examples=df_val, 
            feature_extractor=feature_extractor, 
            max_duration=max_duration
            )

        ## Training Arguments and Class Weights
        training_arguments = define_training_args(
            output_dir=output_dir, 
            batch_size=args.batch, 
            num_steps=args.steps, 
            lr=args.lr, 
            gradient_accumulation_steps=1,
            warmup_steps=args.warmup_steps)
        class_weights = compute_class_weights(df_train)

        ## Trainer 
        trainer = WeightedTrainer(
            class_weights=class_weights,
            model=model,
            args=training_arguments,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics_binary if args.label=='s/p' else compute_metrics
            )

        ## Train and Evaluate
        trainer.train()
        trainer.evaluate()

        ## Evaluate
        print("----------------------------------")
        df_test = pd.read_csv(args.df_test, index_col=None)
        for index in range(0,len(df_test)):
            df_test.loc[index,'label'] = label2id[df_test.loc[index,args.label]]
        df_test['label'] = df_test['label'].astype(int)
        test_dataset = Dataset(
            examples=df_test, 
            feature_extractor=feature_extractor, 
            max_duration=max_duration
            )
        
        print("Evaluating...")
        predictions = trainer.predict(test_dataset)
        with open(f"{output_dir}/predictions.txt", "w") as f:
            f.write(f"Accuracy: {predictions.metrics['test_accuracy']}\n")
            f.write(f"F1: {predictions.metrics['test_f1_macro']}\n")
            if args.label == "s/p":
                f.write(f"AUC: {predictions.metrics['test_auc']}\n")
            else:
                f.write(f"Top-3 Accuracy: {predictions.metrics['test_top3_accuracy']}\n")

        ## Compute confidence scores
        if args.save_confidence_scores:
            print("Saving confidence scores...")
            sof = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=-1)
            df = pd.DataFrame(sof, columns=labels)
            df["category"] = [id2label[str(label_id)] for label_id in predictions.label_ids]
            df.to_csv(f"{output_dir}/confidence_scores.csv", index=False)