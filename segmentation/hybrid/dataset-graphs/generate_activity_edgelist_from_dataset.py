import json
import sys
import argparse
import pandas as pd

def main(argv):
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir",
                        type=str,
                        default="kasteren_house_a/reduced",
                        nargs="?",
                        help="Dataset dir")
    parser.add_argument("--dataset_file",
                        type=str,
                        default="base_kasteren_reduced_train.csv",
                        nargs="?",
                        help="Dataset CSV file")
    args = parser.parse_args()
    print('Loading dataset...')
    sys.stdout.flush()
    # dataset of actions and activities
    DATASET = '../../' + args.dataset_dir + "/" + args.dataset_file
    df_dataset = pd.read_csv(DATASET, parse_dates=[[0, 1]], header=None, index_col=0, sep=' ')
    df_dataset.columns = ['sensor', 'action', 'event', 'activity']
    df_dataset.index.names = ["timestamp"]
    # create action - activity dict
    action_activity_dict = {}
    for index, row in df_dataset.iterrows():
        if row['action'] in action_activity_dict:
            if row['activity'] != "None": # do not take into account None labels
                if row['activity'] != "InstallSensor" and row['activity'] != "SettingUpSensors":
                    values = action_activity_dict[row['action']]
                    values.add(row['activity'])
                    action_activity_dict[row['action']] = values
        else:
            action_activity_dict[row['action']] = set()
    # check dict struct
    print(action_activity_dict)
    # calculate edges of the graph
    edge_list = []
    for action, activities in action_activity_dict.items():
        for another_action, other_activities in action_activity_dict.items():
            if (action != another_action):
                # check activities correspondance
                if any(x in activities for x in other_activities):
                    edge_list.append([action, another_action])
    # write graph edges to file
    with open('/activity_segmentation/hybrid/retrofitting/lexicons/' + args.dataset_dir + '/actions_activities_from_data.edgelist', "w") as edgelist_file:
        for edge in edge_list:
            edgelist_file.write(str(edge[0]) + " " + str(edge[1]) + "\n")

if __name__ == "__main__":
    main(sys.argv)