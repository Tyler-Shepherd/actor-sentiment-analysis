import csv
from pprint import pprint
from operator import itemgetter
import math
import NameTagging

# Amartya's column numbers
# has_actor_col = -1
# review_col = 29
# actor1_col = 30
# actor1_sentiment_col = 31
# actors_start_col = 32
# actors_end_col = 0


# Tyler's column numbers
has_actor_col = 38
review_col = 27
actor1_col = 28
actor1_sentiment_col = 29
actors_start_col = 30
actors_end_col = 2



def sentiment_to_int(sentiment):
    if sentiment == "Positive":
        return 1
    elif sentiment == "Negative":
        return -1
    elif sentiment == "Neutral":
        return 0
    else:
        return 0 # Incorrectly formatted sentiment, assume neutral


def int_to_sentiment(sent_int):
    if sent_int == 1:
        return "Positive"
    elif sent_int == 0:
        return "Neutral"
    elif sent_int == -1:
        return "Negative"
    else:
        return "Neutral" # Incorrect number, assume neutral




def get_review_data():
    unknown_str = "Unknown"
    counter = 0
    review_data = []

    # review_data is list of dicts, each representing a review
    # Each review is a dict with keys "actor_info", "title" and "review"
    # "review" gives full review text
    # "title" gives title
    # "actor_info" gives a list of actors
    # Each actor represented by a dict with keys "name", "sentences", and "sentiment"
    # "name" gives actor name as reported by turk
    # "sentences" gives list of sentences that contain the actors name
    # "sentiment" gives -1/0/1 as reported by turk

    with open("Turk_Results.csv", 'r') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            counter += 1
            if counter == 1:
                continue
            if row[has_actor_col] == "No":
                continue
            current_dict = dict()

            movie_review = row[review_col]
            current_dict["review"] = movie_review
            info = row[actor1_col].split("|")
            movie_name = info[0]
            if len(movie_name.replace(" ", "")) == 0:
                movie_name = "Unknown" + str(counter)
            current_dict["title"] = movie_name

            first_actor_name = info[1]
            first_actor_sentiment = sentiment_to_int(row[actor1_sentiment_col])
            first_actor = {"name": first_actor_name, "sentiment": first_actor_sentiment, "sentences": []}
            current_dict["actor_info"] = []
            current_dict["actor_info"].append(first_actor)
            for b in range(actors_start_col, len(row) - 1 - actors_end_col, 2):
                if row[b] == "{}":
                    break
                next_actor = {"name": row[b], "sentiment": sentiment_to_int(row[b+1]), "sentences": []}
                current_dict["actor_info"].append(next_actor)

            review_data.append(current_dict)

    word_identifier = dict()  # For building word ids
    actor_identifier = dict()  # For building actor ids

    counter = 1
    actor_counter = 1
    for review in review_data:
        movie_review_terms = review["review"].split(" ")
        for term in movie_review_terms:
            if not term in word_identifier:
                word_identifier[term] = counter
                counter += 1

        # movie_review_sentences = review["review"].split(".")

        for actor in review["actor_info"]:
            actor["sentences"] = get_actor_sentences(actor["name"], review["review"])

            if not actor["name"] in actor_identifier:
                actor_identifier[actor["name"]] = actor_counter
                actor_counter += 1


        # for sentence in movie_review_sentences:
        #     for actor in review["actor_info"]:
        #         actor_name = actor["name"]

        #         if not actor_name in actor_identifier:
        #             actor_identifier[actor_name] = actor_counter
        #             actor_counter += 1

        #         if actor_name in sentence or actor_name.lower() in sentence:
        #             # TODO: update this to be better
        #             actor["sentences"].append(sentence)
        #             continue
        #         for word in actor_name.split(" "):
        #             if len(word) < 2:
        #                 continue
        #             elif word in sentence:
        #                 actor["sentences"].append(sentence)
        #                 continue

    return review_data, word_identifier, actor_identifier



# Formats and prints train or test data
def print_data_output(filename, data):
    output_file = open(filename, 'w')

    num_positive = 0
    num_negative = 0
    num_neutral = 0

    for i in range(len(data)):
        data_point = data[i]

        review = data_point[0]
        actor = data_point[1]

        actor_name = actor["name"]

        # Convert actor names to $T$
        rev_line = review.replace(actor_name,"$T$")
        rev_line = rev_line.replace(actor_name.lower(), "$T$")

        # Replace all mentions of just one word in the actors name (i.e. their last name)
        for word in actor_name.split():
            if len(word) < 2:
                # Ignore single letters (like an initial)
                continue
            rev_line = rev_line.replace(word, "$T$")
            rev_line = rev_line.replace(word, "$T$")

        while "$T$ $T$" in rev_line:
            rev_line = rev_line.replace("$T$ $T$", "$T$")

        rev_line += '\n'

        aspect_line = actor_name + "\n"
        sentiment_line = str(actor["sentiment"]) + "\n"

        if actor["sentiment"] == 1:
            num_positive += 1
        elif actor["sentiment"] == 0:
            num_neutral += 1
        elif actor["sentiment"] == -1:
            num_negative += 1

        output_file.write(rev_line + aspect_line + sentiment_line)
    output_file.close()

    return num_positive, num_neutral, num_negative




# Returns train_data, test_data
# train_data and test_data are both lists of tuples like (review, actor)
# where review is the full review text and actor is a dict with keys "name" "sentences" and "sentiment"
def split_train_and_test(review_data):
    full_data = []
    for review in review_data:
        review_text = review["review"]
        for actor in review["actor_info"]:
            full_data.append((review_text, actor))

    num_data = len(full_data)

    num_train = int(math.ceil(0.8 * num_data))

    train_data = full_data[:num_train]
    test_data = full_data[num_train:]

    return train_data, test_data



# Returns list of sentences containing actor_name from the review
def get_actor_sentences(actor_name, review):
    sentences = review.split('.')

    actor_sentences = []

    for sentence in sentences:
        if actor_name in sentence or actor_name.lower() in sentence:
            actor_sentences.append(sentence)
            continue
        for word in actor_name.split(" "):
            if len(word) <= 2:
                continue #ignore initials
            elif word in sentence:
                actor_sentences.append(sentence)
                break

    return actor_sentences




if __name__ == "__main__":
    # lis_movies, lis_actor_sentences, word_identifier, actor_identifier, all_reviews, actor_and_review = get_review_data()
    review_data, word_identifier, actor_identifier = get_review_data()

    g = sorted(word_identifier.items(), key=itemgetter(1))
    actor_g = sorted(actor_identifier.items(), key = itemgetter(1))

    # Print word ids
    output_file = open("word_id.txt", 'w')
    for each in g:
        str_line = each[0] + " " + str(each[1]) + "\n"
        output_file.write(str_line)
    output_file.close()

    # Print actor ids
    actor_output_file = open("actor_id.txt", 'w')
    for actor in actor_g:
        str_line = actor[0] + " " + str(actor[1]) + "\n"
        actor_output_file.write(str_line)
    actor_output_file.close()

    # Print all review text
    review_output_file = open("movie_reviews.txt", 'w')
    for review in review_data:
        rev_line = review["review"] +"\n"
        review_output_file.write(rev_line)
    review_output_file.close()

    # Print sentences each actor is in
    actor_sentences_output_file = open("actor_sentences.txt", 'w')
    for review in review_data:
        for actor in review["actor_info"]:
            if len(actor["sentences"]) == 0:
                continue
            sentence = ""
            for sen in actor["sentences"]:
                sentence += sen
            sentence += '.\n'

            actor_sentences_output_file.write(sentence)
    actor_sentences_output_file.close()

    # Get train and test data
    train_data, test_data = split_train_and_test(review_data)

    print("Num data:", len(train_data) + len(test_data))
    print("Num train", len(train_data))
    print("Num test", len(test_data))

    num_positive, num_neutral, num_negative = print_data_output("actor_train.txt", train_data)

    print("Train data (positive, neutral, negative)", num_positive, num_neutral, num_negative)

    num_positive, num_neutral, num_negative = print_data_output("actor_test.txt", test_data)

    print("Test data (positive, neutral, negative)", num_positive, num_neutral, num_negative)

    # Name tagging
    # Gets all named entities of type PER in the review
    # Currently unused
    # Could use entity linking to determine whether each of these are an actor

    # for review in all_reviews:
    #     named_entities = name_tagging.name_tagging(review)
    #     print(named_entities)
