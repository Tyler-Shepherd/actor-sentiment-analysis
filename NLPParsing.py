import csv
from pprint import pprint
from operator import itemgetter

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



# if __name__ == "__main__":
def get_review_data():
    counter = 0
    lis_movies = []
    unknown_str = "Unknown"
    lis_actor_sentences = []
    with open("Turk_Results.csv", 'rb') as csvfile:
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
            # print movie_name
            first_actor = info[1]
            first_actor_review = row[actor1_sentiment_col]
            current_dict["actor_info"] = dict()
            current_dict["actor_info"][first_actor] = first_actor_review
            for b in range(actors_start_col, len(row) - 1 - actors_end_col, 2):
                if row[b] == "{}":
                    break
                current_dict["actor_info"][row[b]] = row[b + 1]

            lis_movies.append(current_dict)

    word_identifier = dict()
    actor_identifier = dict()
    all_reviews = []

    counter = 1
    actor_counter = 1
    for review in lis_movies:
        actor_dict = dict()
        all_reviews.append(review["review"])
        movie_review_terms = review["review"].split(" ")
        for term in movie_review_terms:
            if not term in word_identifier:
                word_identifier[term] = counter
                counter += 1
        movie_review_sentences = review["review"].split(".")
        for sentence in movie_review_sentences:
            for actor_name in review["actor_info"]:
                if not actor_name in actor_identifier:
                    actor_identifier[actor_name] = actor_counter
                    actor_counter += 1
                if actor_name in sentence:
                    if not actor_name in actor_dict:
                        actor_dict[actor_name] = []
                    actor_dict[actor_name].append(sentence)
        lis_actor_sentences.append(actor_dict)

    return lis_movies, lis_actor_sentences, word_identifier, actor_identifier, all_reviews

if __name__ == "__main__":
    lis_movies, lis_actor_sentences, word_identifier, actor_identifier, all_reviews = get_review_data()
    g = sorted(word_identifier.items(), key=itemgetter(1))
    actor_g = sorted(actor_identifier.items(), key = itemgetter(1))
    output_file = open("word_id.txt", 'wb')
    for each in g:
        str_line = each[0] + " " + str(each[1]) + "\n"
        output_file.write(str_line)
    output_file.close()

    actor_output_file = open("actor_id.txt", 'wb')
    for actor in actor_g:
        str_line = actor[0] + " " + str(actor[1]) + "\n"
        actor_output_file.write(str_line)
    actor_output_file.close()

    review_output_file = open("movie_reviews.txt", 'wb')
    for review in all_reviews:
        rev_line = review +"\n"
        review_output_file.write(rev_line)
    review_output_file.close()

    # pprint(lis_actor_sentences)
