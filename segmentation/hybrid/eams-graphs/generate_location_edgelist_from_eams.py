import json
import sys
import argparse

def check_location(locations_a, locations_b):
    for location_a in locations_a:
        for location_b in locations_b:
            if location_a == location_b:
                return True
    return False

def transform_activity_dict_to_action_dict(activity_dict):
    action_dict = {}
    for activity, knowledge in activity_dict.items():
        for action in knowledge['actions']:
            key = action
            if key in action_dict:
                values = action_dict[key]
                locations = values['locations']
                locations.extend(knowledge['locations'])
                action_dict[key] = values
            else:
                values = {}
                values['locations'] = knowledge['locations']
                action_dict[key] = values
    return action_dict

def main(argv):
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--eams_dir",
                        type=str,
                        default="kasteren_house_a/reduced",
                        nargs="?",
                        help="Eams dir")
    parser.add_argument("--eams_json",
                        type=str,
                        default="eams.json",
                        nargs="?",
                        help="Eams json file")
    args = parser.parse_args()
    # read EAMs from file
    DIR = args.eams_dir
    EAMS_FILE = '../../' + DIR + "/" + args.eams_json
    with open(EAMS_FILE) as json_file:
        eams = json.load(json_file)
    # check EAMs struct
    print(eams)
    # transform activity knowledge to action knowledge
    action_dict = transform_activity_dict_to_action_dict(eams)
    # check new struct
    print(action_dict)
    # calculate edges of the graph
    edge_list = []
    for action, knowledge in action_dict.items():
        for another_action, another_knowledge in action_dict.items():
            if (action != another_action):
                # check locations correspondance
                if check_location(knowledge['locations'], another_knowledge['locations']):
                    edge_list.append([action, another_action])
    # write graph edges to file
    with open('/activity_segmentation/hybrid/retrofitting/lexicons/' + DIR + '/actions_locations.edgelist', "w") as edgelist_file:
        for edge in edge_list:
            edgelist_file.write(str(edge[0]) + " " + str(edge[1]) + "\n")

if __name__ == "__main__":
    main(sys.argv)