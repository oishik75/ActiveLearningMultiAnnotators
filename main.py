import argparse
import numpy as np
import pandas as pd
import os
import shutil

from active_learning import MultiAnnotatorActiveLearner

def main():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument("--data_path", default="data/medical.csv")
    parser.add_argument("--boot_size", type=float, default=0.1)
    parser.add_argument("--test_size", type=float, default=0.3)
    # Boot
    parser.add_argument("--boot_lr", type=float, default=1e-5)
    parser.add_argument("--boot_n_epochs", type=int, default=1000)
    parser.add_argument("--boot_log_epochs", type=int, default=10)
    parser.add_argument("--boot_strategy", default="random")
    parser.add_argument("--samples_per_cluster", type=int, default=1)
    # Active Learning
    parser.add_argument("--active_lr", type=float, default=1e-5)
    parser.add_argument("--active_n_epochs", type=int, default=400)
    parser.add_argument("--active_log_epochs", type=int, default=1000)
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--instance_strategy", default="random")
    parser.add_argument("--classifier_name", default="svm") # logistic_regression neural_net xgboost
    # Knowledgebase
    parser.add_argument("--use_knowledgebase", type=bool, default=True)
    parser.add_argument("--weight_threshold", type=float, default=0.7)
    parser.add_argument("--similarity_threshold", type=float, default=0.95)
    # Misc
    parser.add_argument("--log_dir", default="logs/medical/")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--labeling_type", default="max")
    parser.add_argument("--exp_name", default=None)

    args = parser.parse_args()

    if args.exp_name is None:
        args.exp_name = f"{args.classifier_name}_{args.instance_strategy}_labeling{args.labeling_type}_budget{args.budget}_bootsize{args.boot_size}_activeepochs{args.active_n_epochs}_kb{args.use_knowledgebase}_bootstrategy{args.boot_strategy}"

    suffix="" # to store suffix for exp_name passed by user if any
    while os.path.isdir(args.log_dir + args.exp_name + suffix):
        choice = input(f"Log dir {args.log_dir + args.exp_name + suffix} already exists. Delete existing (y/n)? ")
        if choice.lower() == "y":
            shutil.rmtree(args.log_dir + args.exp_name + suffix)
        elif choice.lower() == "n":
            suffix = input("Enter suffix to be appended to exp_name: ")

    args.exp_name = args.exp_name + suffix

    print(args)

    data = pd.read_csv(args.data_path)

    al = MultiAnnotatorActiveLearner(data, args)

    al.fully_supervised()

    al.boot_phase()

    al.active_learning_phase(verbose=False)

if __name__ == "__main__":
    main()