import sys
import pandas as pd
import argparse

def main(argv):
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir",
                        type=str,
                        default="../kasteren_house_a",
                        nargs="?",
                        help="Dataset dir")
    parser.add_argument("--dataset_file",
                        type=str,
                        default="kasterenA_groundtruth.csv",
                        nargs="?",
                        help="Dataset file")
    parser.add_argument("--train_percentage",
                        type=float,
                        default=0.7,
                        nargs="?",
                        help="Training percentage 0-100%")
    args = parser.parse_args()
    print('Loading dataset...')
    sys.stdout.flush()
    # dataset of actions and activities
    DATASET = args.dataset_dir + "/" + args.dataset_file
    df_dataset = pd.read_csv(DATASET, parse_dates=[[0, 1]], header=None, index_col=0, sep=' ')
    df_dataset.columns = ['sensor', 'action', 'event', 'activity']
    df_dataset.index.names = ["timestamp"]
    print("Len of original dataset: " + str(len(df_dataset)))
    # split into consecutive days
    df_dataset_day_list = []
    for group in df_dataset.groupby(df_dataset.index.day, sort=False):
        df_dataset_day_list.append(group[1])
    num_days = len(df_dataset_day_list)
    num_days_train = int(num_days * args.train_percentage)
    num_days_test = num_days - num_days_train
    df_dataset_day_train_list = df_dataset_day_list[0:num_days_train]
    df_dataset_day_test_list = df_dataset_day_list[num_days_train:len(df_dataset_day_list)]
    print("Num days train: " + str(num_days_train))
    print("Num days test: " + str(num_days_test))
    # concatenate into train and test
    df_dataset_train = pd.concat(df_dataset_day_train_list)
    df_dataset_test = pd.concat(df_dataset_day_test_list)
    print("Len of train dataset: " + str(len(df_dataset_train)))
    print("Len of test dataset: " + str(len(df_dataset_test)))
    # write to CSV
    df_dataset_train.to_csv("/activity_segmentation/results/" + args.dataset_file.replace('.csv', '') + "_train.csv", header=False, sep=" ")
    df_dataset_test.to_csv("/activity_segmentation/results/" + args.dataset_file.replace('.csv', '') + "_test.csv", header=False, sep=" ")

if __name__ == "__main__":
    main(sys.argv)