#!/usr/bin/env hy

; Run this cell to import packages.
(import pickle)
(import numpy :as np)
(import pandas :as pd)
(import matplotlib.pyplot :as plt)

(import utils [get_vectors])

(setv data (pd.read_csv "capitals.txt" :delimiter " "))
(setv data.columns ["city1" "country1" "city2" "country2"])

; print first five elements in the DataFrame
;; (print (data.head 5))


(setv word_embeddings (pickle.load (open "word_embeddings_subset.p" "rb")))
;; (print (len word_embeddings))  

(defn cosine_similarity [A B]
    "
    Input:
        A: a numpy array which corresponds to a word vector
        B: A numpy array which corresponds to a word vector
    Output:
        cos: numerical number representing the cosine similarity between A and B.
    "

    (setv dot (np.dot A B))
    (setv norma (np.linalg.norm A))
    (setv normb (np.linalg.norm B))
    (setv cos (/ dot (* norma normb)))

    cos)

(defn test_cosine_similarity []
    (setv king (get word_embeddings "king"))
    (setv queen (get word_embeddings "queen"))

    (print (cosine_similarity king queen)))
;; (test-cosine-similarity)


(defn euclidean [A B]
    "
    Input:
        A: a numpy array which corresponds to a word vector
        B: A numpy array which corresponds to a word vector
    Output:
        d: numerical number representing the Euclidean distance between A and B.
    "

    ; euclidean distance

    (np.linalg.norm (- A B)))

(defn test_euclidean []
    (setv king (get word_embeddings "king"))
    (setv queen (get word_embeddings "queen"))

    (print (euclidean king queen)))
;; (test-euclidean)

(defn get_country [city1 country1 city2 embeddings]:
    "
    Input:
        city1: a string (the capital city of country1)
        country1: a string (the country of capital1)
        city2: a string (the capital city of country2)
        embeddings: a dictionary where the keys are words and values are their embeddings
    Output:
        countries: a dictionary with the most likely country and its similarity score
    "
    ; store the city1, country 1, and city 2 in a set called group
    (setv group #{city1 country1 city2})

    ; get embeddings of city 1
    (setv city1_emb (get embeddings city1))
    ; get embedding of country 1
    (setv country1_emb (get embeddings country1))
    ; get embedding of city 2
    (setv city2_emb (get embeddings city2))

    ; get embedding of country 2 (it's a combination of the embeddings of country 1, city 1 and city 2)
    ; Remember: King - Man + Woman = Queen
    (setv vec (+ city2_emb (- country1_emb city1_emb)))

    ; Initialize the similarity to -1 (it will be replaced by a similarities that are closer to +1)
    (setv similarity -1)

    ; initialize country to an empty string
    (setv country "")

    ; loop through all words in the embeddings dictionary
    (for [word (embeddings.keys)]
        ; first check that the word is not already in the 'group'
        (when (not (in word group))
            ; get the word embedding
            (setv word_emb (get embeddings word))

            ; calculate cosine similarity between embedding of country 2 and the word in the embeddings dictionary
            (setv cur_similarity (cosine_similarity word_emb vec))

            ; if the cosine similarity is more similar than the previously best similarity...
            (when (> cur_similarity similarity)
                ; update the similarity to the new, better similarity
                (setv similarity cur_similarity)

                ; store the country as a tuple, which contains the word and the similarity
                (setv country #(word cur_similarity)))))

    country)

(print (get_country "Athens" "Greece" "Cairo" word_embeddings))

(defn get_accuracy [word_embeddings data]
    "
    Input:
        word_embeddings: a dictionary where the key is a word and the value is its embedding
        data: a pandas dataframe containing all the country and capital city pairs
    
    Output:
        accuracy: the accuracy of the model
    "

    ; initialize num correct to zero
    (setv num_correct 0)

    ; loop through the rows of the dataframe
    (for [[i row] (data.iterrows)]

        ; get city1
        (setv city1 (get row "city1"))

        ; get country1
        (setv country1 (get row "country1"))

        ; get city2
        (setv city2 (get row "city2"))

        ; get country2
        (setv country2 (get row "country2"))

        ; use get_country to find the predicted country2
        (setv #(predicted_country2 _) (get_country city1 country1 city2 word_embeddings))

        ; if the predicted country2 is the same as the actual country2...
        (when (= predicted_country2 country2)
            ; increment the number of correct by 1
            (setv num_correct (+ num_correct 1))))

    ; get the number of rows in the data dataframe (length of dataframe)
    (setv m (len data))

    ; calculate the accuracy by dividing the number correct by m
    (/ num_correct m))

(setv accuracy (get_accuracy word_embeddings data))
(print f"Accuracy is {accuracy :.2f}")