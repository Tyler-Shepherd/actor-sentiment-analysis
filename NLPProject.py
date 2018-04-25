import csv
import sys
from pprint import pprint
import NameTagging
import LexiconSentimentAnalysis
import NLPParsing
import ReadDNNResults

sys.path.insert(0, './DNN-Sentiment')

import getSentiment


if __name__ == "__main__":
    if len(sys.argv) <= 2:
        print("Improper number of arguments. Please run with python3 NLPProject.py <review filename> <algo choice>")
        print("-l for lexicon, -r for RNN, -c for CNN, -a for AT-LSTM")
        sys.exit()

    review_filename = sys.argv[1]
    algo_type = sys.argv[2]

    review_file = open(review_filename, "r")

    review = ""

    for line in review_file:
        review += line[:line.find('\n')] + " "

    review_file.close()

    # Name tagging
    # Gets all named entities of type PER in the review

    per_entities = NameTagging.name_tagging(review)

    print("per entities", per_entities, flush = True)

    named_entities = [i for i in per_entities if NameTagging.is_actor(i)]

    print("named entity actors", named_entities, flush = True)


    # Could use entity linking to determine whether each of these are an actor


    # Write actor sentences to file needed for RNN and CNN
    if algo_type == "-r" or algo_type == "-c":
        # write to file for DNN-Sentiment to read
        output_file = open("./DNN-Sentiment/data/actor_sentences.txt", "w")
        for entity in named_entities:
            actor_sentences = NLPParsing.get_actor_sentences(entity, review)

            for sen in actor_sentences:
                output_file.write(sen + ".")
        output_file.close()

        if algo_type == "-r":
            # Old: 1524375413
            # New: 1524617308

            all_predictions,all_scores = getSentiment.getSentimentRNN("actor_sentences",'1524617308')
            getSentiment.saveSentiment("actor_sentences",all_predictions,all_scores)
        else:
            all_predictions,all_scores = getSentiment.getSentimentCNN("actor_sentences",'1524270913')
            getSentiment.saveSentiment("actor_sentences",all_predictions,all_scores)


    for entity in named_entities:
        if algo_type == "-l":
            sentiment = LexiconSentimentAnalysis.GetSentiment(entity, review)
        elif algo_type == "-r" or algo_type == "-c":
            actor_sentences = NLPParsing.get_actor_sentences(entity, review)
            sentiment = ReadDNNResults.GetSentiment("./DNN-Sentiment/data/actor_sentences.csv", actor_sentences)

        print(entity, sentiment, flush=True)