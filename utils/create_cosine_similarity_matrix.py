import sys
import argparse

from gensim.models import Word2Vec
from scipy import spatial

import csv

# Results DIR
DIR = '/results/'

def main(argv):
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",
                        type=str,
                        default='tests/5_execution.model',
                        nargs="?",
                        help="Model name")
    parser.add_argument("--vector_file_name",
                        type=str,
                        default='tests/5_execution_retrofitted_activities.vector',
                        nargs="?",
                        help="Vector file name")
    parser.add_argument("--use_vector_file",
                        type=bool,
                        default=False,
                        nargs="?",
                        help="Use vector file (True), do not use vector file (False)")
    parser.add_argument("--output_file",
                        type=str,
                        default='cosine_similarity_matrix.csv',
                        nargs="?",
                        help="Name for output file")
    args = parser.parse_args()
    # load word2vec model
    if args.use_vector_file:
        vectors_dict = dict()
        with open(args.vector_file_name) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                vectors_dict[row[0]] = [float(i) for i in row[1].split()]
        similarities = dict()
        for item in vectors_dict:
            print("Item " + item)
            for another_item in vectors_dict:
                print("Another item " + another_item)
                if item in similarities.keys():
                    similarities[item].append((another_item, 1 - spatial.distance.cosine(vectors_dict[item], vectors_dict[another_item])))
                else:
                    similarities[item] = [(another_item, 1 - spatial.distance.cosine(vectors_dict[item], vectors_dict[another_item]))]
            print("Finished CS calculation for item " + another_item)
    else:
        model = Word2Vec.load(args.model_name)
        similarities = dict()
        for item in model.wv.vocab:
            print("Item " + item)
            for another_item in model.wv.vocab:
                print("Another item " + another_item)
                if item in similarities.keys():
                    similarities[item].append((another_item, 1 - spatial.distance.cosine(model[item], model[another_item])))
                else:
                    similarities[item] = [(another_item, 1 - spatial.distance.cosine(model[item], model[another_item]))]
            print("Finished CS calculation for item " + another_item)
    print(similarities)
    # write cosine similarity matrix to vector file
    with open(DIR + args.output_file, 'w') as file:
        counter = 0
        for item in similarities:
            if counter == 0:
                file.write("Action, ")
                file.write(item)
            else:
                file.write(", " + item)
            counter += 1
        file.write("\n")
        for item in similarities:
            file.write(item + ", ")
            counter = 0
            for similarity in similarities[item]:
                if counter == 0:
                    file.write(str(similarity[1]))
                else:
                    file.write(", " + str(similarity[1]))
                counter += 1
            file.write("\n")
    print("CS matrix written to file")

if __name__ == "__main__":
    main(sys.argv)