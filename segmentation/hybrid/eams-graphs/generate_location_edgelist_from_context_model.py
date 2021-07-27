import json
import sys
import argparse

def main(argv):
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--context_dir",
                        type=str,
                        default="kasteren_house_a/reduced",
                        nargs="?",
                        help="Eams dir")
    parser.add_argument("--context_model_json",
                        type=str,
                        default="context_model.json",
                        nargs="?",
                        help="Eams json file")
    args = parser.parse_args()
    # read EAMs from file
    DIR = args.context_dir
    CONTEXT_MODEL_FILE = '/activity_segmentation/' + DIR + "/" + args.context_model_json
    with open(CONTEXT_MODEL_FILE) as json_file:
        context = json.load(json_file)
    # check EAMs struct
    print(context)
    # calculate edges of the graph
    context_objects = context['objects']
    edge_list = []
    for action, knowledge in context_objects.items():
        for another_action, another_knowledge in context_objects.items():
            if (action != another_action):
                # check locations correspondance
                if knowledge['location'] == another_knowledge['location']:
                    edge_list.append([action, another_action])
    # write graph edges to file
    with open('/activity_segmentation/segmentation/hybrid/retrofitting/lexicons/' + DIR + '/actions_locations_context.edgelist', "w") as edgelist_file:
        for edge in edge_list:
            edgelist_file.write(str(edge[0]) + " " + str(edge[1]) + "\n")

if __name__ == "__main__":
    main(sys.argv)