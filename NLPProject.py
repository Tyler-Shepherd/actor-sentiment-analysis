import csv
from pprint import pprint


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

    for review in lis_movies:
        actor_dict = dict()
        movie_review_sentences = review["review"].split(".")
        for sentence in movie_review_sentences:
            for actor_name in review["actor_info"]:
                if actor_name in sentence:
                    if not actor_name in actor_dict:
                        actor_dict[actor_name] = []
                    actor_dict[actor_name].append(sentence)
        lis_actor_sentences.append(actor_dict)

    return lis_movies, lis_actor_sentences
    # pprint(lis_actor_sentences)
