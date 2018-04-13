import csv

if __name__ == "__main__":
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
            if row[-1] == "No":
                continue
            current_dict = dict()
            movie_review = row[29]
            current_dict["review"] = movie_review
            info = row[30].split("|")
            movie_name = info[0]
            if len(movie_name.replace(" ", "")) == 0:
                movie_name = "Unknown" + str(counter)
            current_dict["title"] = movie_name
            print movie_name
            first_actor = info[1]
            first_actor_review = row[31]
            current_dict["actor_info"] = dict()
            current_dict["actor_info"][first_actor] = first_actor_review
            for b in range(32, len(row) - 1, 2):
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

    print lis_actor_sentences
