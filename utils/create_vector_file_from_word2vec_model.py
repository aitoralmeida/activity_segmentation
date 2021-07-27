import sys
import argparse

from gensim.models import Word2Vec

# Results DIR
DIR = '/activity_segmentation/results/'

def main(argv):
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir",
                        type=str,
                        default='kasteren_house_a/reduced/word2vec_context/',
                        nargs="?",
                        help="Dir for word2vec models")
    parser.add_argument("--model_folder",
                        type=str,
                        default='context_window_1_window_1_iterations_5_embedding_size_50/',
                        nargs="?",
                        help="Folder for word2vec models")
    parser.add_argument("--exe",
                        type=int,
                        default=5,
                        nargs="?",
                        help="Number of executions")
    args = parser.parse_args()
    for e in range(args.exe):
        # load word2vec model
        model_name = str(e) + "_execution.model"
        model = Word2Vec.load(DIR + args.model_dir + args.model_folder + 'train/word2vec_models/' + model_name)
        # write model to vector file
        vector_file_name = str(e) + "_execution.vector"
        model.wv.save_word2vec_format(DIR + args.model_dir + args.model_folder + 'train/word2vec_models/' + vector_file_name, binary=False)

if __name__ == "__main__":
    main(sys.argv)