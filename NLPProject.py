import csv
import sys
from pprint import pprint
import NameTagging
import LexiconSentimentAnalysis
import NLPParsing
import ReadDNNResults
import os
from subprocess import call

sys.path.insert(0, './lib/DNN-Sentiment')

import getSentiment


sys.path.insert(0, './lib/LSTM/reviews')
import actorReview

if __name__ == "__main__":
    if len(sys.argv) <= 2:
        print("Improper number of arguments. Please run with python3 NLPProject.py <review filename> <algo choice>")
        print("-l for lexicon, -r for RNN, -c for CNN, -a for AT-LSTM")
        sys.exit()

    review_filename = sys.argv[1]
    algo_type = sys.argv[2]

    review_file = open("./data/examples/"+review_filename, "r")

    review = ""

    for line in review_file:
        review += line[:line.find('\n')] + " "

    review_file.close()

    # Name tagging
    # Gets all named entities of type PER in the review

    per_entities = NameTagging.name_tagging(review)

    print("PER entities", per_entities, flush = True)

    named_entities = [i for i in per_entities if NameTagging.is_actor(i)]

    print("named entity actors", named_entities, flush = True)


    # Could use entity linking to determine whether each of these are an actor


    # Write actor sentences to file needed for RNN and CNN
    if algo_type == "-r" or algo_type == "-c":
        # write to file for DNN-Sentiment to read
        output_file = open("./lib/DNN-Sentiment/data/actor_sentences.txt", "w")
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
    elif algo_type == "-a":
        output_file = open("./lib/LSTM/data/actors/new_review.txt", "w")

    for entity in named_entities:
        if algo_type == "-l":
            sentiment = LexiconSentimentAnalysis.GetSentiment(entity, review)
            print(entity, ": ", NLPParsing.int_to_sentiment(sentiment), flush=True)
        elif algo_type == "-r" or algo_type == "-c":
            actor_sentences = NLPParsing.get_actor_sentences(entity, review)
            sentiment = ReadDNNResults.GetSentiment("./lib/DNN-Sentiment/data/actor_sentences.csv", actor_sentences)
            print(entity, ": ", NLPParsing.int_to_sentiment(sentiment), flush=True)
        elif algo_type == "-a":
            rev_line = review.replace(entity,"$T$")
            rev_line = rev_line.replace(entity.lower(), "$T$")

            # Replace all mentions of just one word in the actors name (i.e. their last name)
            for word in entity.split():
                if len(word) < 2:
                    # Ignore single letters (like an initial)
                    continue
                rev_line = rev_line.replace(word, "$T$")
                rev_line = rev_line.replace(word, "$T$")

            while "$T$ $T$" in rev_line:
                rev_line = rev_line.replace("$T$ $T$", "$T$")
            rev_line = rev_line.split(".")
            for sen in rev_line:
                if "$T$" in sen:
                    sen += '\n'
                    aspect_line = entity + "\n"
                    sentiment_line = "0"
                    output_file.write(sen + aspect_line + sentiment_line +"\n")
    #To ensure that the attention network LSTM model works, add three random reviews to end of the test dataset.
    if algo_type == "-a":
        positive = "$T$ did a phenomenal job in black panther which overall was a solid movie but a little bit overhyped..\n"
        positive += "chadwick boseman\n" + "1\n"
        output_file.write(positive)
        neutral = "$T$ did a mediocre job in black panther which overall was a solid movie but a little bit overhyped.\n"
        neutral += "mark jones\n" +  "0\n"
        output_file.write(neutral)
        terrible = "$T$ did a horrible job in black panther which overall was a solid movie but a little bit overhyped.\n"
        terrible += "johnny adams\n" + "-1\n"
        output_file.write(terrible)
        output_file.close()
        os.chdir("lib/LSTM/reviews")
        call(["py", "-3", "actorReview.py"])