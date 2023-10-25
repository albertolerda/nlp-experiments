#!/usr/bin/env hy

(import pdb)
(import pickle)
(import string)

(import time)

(import gensim)
(import matplotlib.pyplot :as plt)
(import nltk)
(import numpy :as np)
(import scipy)
(import sklearn)
(import gensim.models [KeyedVectors])
(import nltk.corpus [stopwords twitter_samples])
(import nltk.tokenize [TweetTokenizer])

(import utils [cosine_similarity get_dict process_tweet])
(import os [getcwd])

; add folder, tmp2, from our local workspace containing pre-downloaded corpora files to nltk's data path
(setv filePath f"{(getcwd)}/../tmp2/")
(nltk.data.path.append filePath)

(nltk.download "stopwords")
(nltk.download "twitter_samples")
;; 1. The word embeddings data for English and French words
;; Exercise 01: Translating English dictionary to French by using embeddings

(setv en_embeddings_subset (pickle.load (open "en_embeddings.p" "rb")))
(setv fr_embeddings_subset (pickle.load (open "fr_embeddings.p" "rb")))

(setv en_fr_train (get_dict "en-fr.train.txt"))
(print "The length of the English to French training dictionary is" (len en_fr_train))
(setv en_fr_test (get_dict "en-fr.test.txt"))
(print "The length of the English to French test dictionary is" (len en_fr_train))

(defn get_matrices [en_fr french_vecs english_vecs]
    "
    Input:
        en_fr: English to French dictionary
        french_vecs: French words to their corresponding word embeddings.
        english_vecs: English words to their corresponding word embeddings.
    Output: 
        X: a matrix where the columns are the English embeddings.
        Y: a matrix where the columns correspong to the French embeddings.
        R: the projection matrix that minimizes the F norm ||X R -Y||^2.
    "

    ; X_l and Y_l are lists of the english and french word embeddings
    (setv X_l [])
    (setv Y_l [])

    ; get the english words (the keys in the dictionary) and store in a set()
    (setv english_set (set (english_vecs.keys)))

    ; get the french words (keys in the dictionary) and store in a set()
    (setv french_set (set (french_vecs.keys)))

    ; store the french words that are part of the english-french dictionary (these are the values of the dictionary)
    (setv french_words (set (en_fr.values)))

    ; loop through all english, french word pairs in the english french dictionary
    (for [[en_word fr_word] (en_fr.items)]
        ; check that the french word has an embedding and that the english word has an embedding
        (when (and (in fr_word french_set) (in en_word english_set))
            ; get the english embedding
            (setv en_vec (get english_vecs en_word))

            ; get the french embedding
            (setv fr_vec (get french_vecs fr_word))

            ; add the english embedding to the list
            (X_l.append en_vec)

            ; add the french embedding to the list
            (Y_l.append fr_vec)))

    ; stack the vectors of X_l into a matrix X
    (setv X (np.vstack X_l))

    ; stack the vectors of Y_l into a matrix Y
    (setv Y (np.vstack Y_l))
    #(X Y))


(setv #(X_train Y_train) (get_matrices en_fr_train fr_embeddings_subset en_embeddings_subset))

; Exercise 02: Implementing translation mechanism described in this section.
(defn compute_loss [X Y R]
    "
    Inputs: 
        X: a matrix of dimension (m,n) where the columns are the English embeddings.
        Y: a matrix of dimension (m,n) where the columns correspong to the French embeddings.
        R: a matrix of dimension (n,n) - transformation matrix from English to French vector space embeddings.
    Outputs:
        L: a matrix of dimension (m,n) - the value of the loss function for given X, Y and R.
    "
    ; m is the number of rows in X
    (setv m (get X.shape 0))
    
    ; diff is XR - Y
    (setv diff (- (np.dot X R) Y))

    ; diff_squared is the element-wise square of the difference
    (setv diff_squared (* diff diff))

    ; sum_diff_squared is the sum of the squared elements
    (setv sum_diff_squared (np.sum diff_squared))

    ; loss i the sum_diff_squard divided by the number of examples (m)
    (/ sum_diff_squared m))

; Exercise 03
; Step 2: Computing the gradient of loss in respect to transform matrix R

(defn compute_gradient [X Y R]
    "
    Inputs: 
        X: a matrix of dimension (m,n) where the columns are the English embeddings.
        Y: a matrix of dimension (m,n) where the columns correspong to the French embeddings.
        R: a matrix of dimension (n,n) - transformation matrix from English to French vector space embeddings.
    Outputs:
        g: a matrix of dimension (n,n) - gradient of the loss function L for given X, Y and R.
    "
    ; m is the number of rows in X
    (setv m (get X.shape 0))

    ; gradient is X^T(XR - Y) * 2/m
    (* 2 (/ (np.dot (np.transpose X) (- (np.dot X R) Y)) m)))


(defn align_embeddings [X Y [train_steps 100] [learning_rate 0.0003]]
    "
    Inputs:
        X: a matrix of dimension (m,n) where the columns are the English embeddings.
        Y: a matrix of dimension (m,n) where the columns correspong to the French embeddings.
        train_steps: positive int - describes how many steps will gradient descent algorithm do.
        learning_rate: positive float - describes how big steps will  gradient descent algorithm do.
    Outputs:
        R: a matrix of dimension (n,n) - the projection matrix that minimizes the F norm ||X R -Y||^2
    "
    (np.random.seed 129)

    ; the number of columns in X is the number of dimensions for a word vector (e.g. 300)
    ; R is a square matrix with length equal to the number of dimensions in th  word embedding
    (setv R (np.random.rand (get X.shape 1) (get X.shape 1)))

    (for [i (range train_steps)]
        (when (= (% i 25) 0)
            (print f"loss at iteration {i} is: {(compute_loss X Y R) :.4f}"))
        ; use the function that you defined to compute the gradient
        (setv gradient (compute_gradient X Y R))

        ; update R by subtracting the learning rate times gradient
        (setv R (- R (* learning_rate gradient))))
    R)


(defn test_align_embeddings []
    (np.random.seed 129)
    (setv m 10)
    (setv n 5)
    (setv X (np.random.rand m n))
    (setv Y (* (np.random.rand m n) .1))
    (setv R (align_embeddings X Y)))

;; (test_align_embeddings)

;; (setv R_train (align_embeddings X_train Y_train :train_steps 400 :learning_rate 0.8))


; 2.2 Testing the translation

(defn nearest_neighbor [v candidates [k 1]]
    "
    Input:
      - v, the vector you are going find the nearest neighbor for
      - candidates: a set of vectors where we will find the neighbors
      - k: top k nearest neighbors to find
    Output:
      - k_idx: the indices of the top k closest vectors in sorted form
    "
    (setv similarity_l [])

    ; for each candidate vector...
    (for [row candidates]
        ; get the cosine similarity
        (setv cos_similarity (cosine_similarity v row))

        ; append the similarity to the list
        (.append similarity_l cos_similarity))
        
    ; sort the similarity list and get the indices of the sorted list
    (setv sorted_ids (np.argsort similarity_l))

    ; get the indices of the k most similar candidate vectors
    (setv k_idx (cut sorted_ids (- k) None))

    k_idx)
(defn test_nearest_neighbot []
    (setv v (np.array [1 0 1]))
    (setv candidates (np.array [[1 0 5] [-2 5 3] [2 0 1] [6 -9 5] [9 9 9]]))
    (print (get candidates (nearest_neighbor v candidates 3))))
(test_nearest_neighbot)


; Test your translation and compute its accuracy

(defn test_vocabulary [X Y R]
    "
    Input:
        X: a matrix where the columns are the English embeddings.
        Y: a matrix where the columns correspong to the French embeddings.
        R: the transform matrix which translates word embeddings from
        English to French word vector space.
    Output:
        accuracy: for the English to French capitals
    "

    ; The prediction is X times R
    (setv pred (np.dot X R))

    ; initialize the number correct to zero
    (setv num_correct 0)

    ; loop through each row in pred (each transformed embedding)
    (for [i (range (len pred))]
        ; get the index of the nearest neighbor of pred at row 'i'; also pass in the candidates in Y
        (setv pred_idx (get (nearest_neighbor (get pred i) Y) 0))

        ; if the index of the nearest neighbor equals the row of i... \
        (when (= pred_idx i)
            ; increment the number correct by 1.
            (setv num_correct (+ num_correct 1))))

    ; accuracy is the number correct divided by the number of rows in 'pred' (also number of rows in X)
    (setv accuracy (/ num_correct (len pred)))

    accuracy)

;; Commented to speed up later execises
;; (setv #(X_val Y_val) (get_matrices en_fr_test fr_embeddings_subset en_embeddings_subset))

;; (setv acc (test_vocabulary X_val Y_val R_train))  ; this might take a minute or two
;; (print f"accuracy on test set is {acc :.3f}")


; 3. LSH and document search

(setv all_positive_tweets (twitter_samples.strings "positive_tweets.json"))
(setv all_negative_tweets (twitter_samples.strings "negative_tweets.json"))
(setv all_tweets (+ all_positive_tweets all_negative_tweets))


(defn get_document_embedding [tweet en_embeddings]
    "
    Input:
        - tweet: a string
        - en_embeddings: a dictionary of word embeddings
    Output:
        - doc_embedding: sum of all word embeddings in the tweet
    "
    (setv doc_embedding (np.zeros 300))

    ; process the document into a list of words (process the tweet)
    (setv processed_doc (process_tweet tweet))
    (for [word processed_doc]
        ; add the word embedding to the running total for the document embedding
        (setv doc_embedding (+ doc_embedding 
            (try (get en_embeddings word)
                (except [KeyError] 0)))))
    doc_embedding)

(defn test_document_embedding []
    (setv custom_tweet "RT @Twitter @chapagain Hello There! Have a great day. :) #good #morning http://chapagain.com.np")
    (setv tweet_embedding (get_document_embedding custom_tweet en_embeddings_subset))
    (print (cut tweet_embedding -5 None))
)

(test_document_embedding)


(defn get_document_vecs [all_docs en_embeddings]
    "
    Input:
        - all_docs: list of strings - all tweets in our dataset.
        - en_embeddings: dictionary with words as the keys and their embeddings as the values.
    Output:
        - document_vec_matrix: matrix of tweet embeddings.
        - ind2Doc_dict: dictionary with indices of tweets in vecs as keys and their embeddings as the values.
    "

    ; the dictionary's key is an index (integer) that identifies a specific tweet
    ; the value is the document embedding for that document
    (setv ind2Doc_dict {})

    ; this is list that will store the document vectors
    (setv document_vec_l [])

    (for [[i doc] (enumerate all_docs)]
        ; get the document embedding of the tweet
        (setv doc_embedding (get_document_embedding doc en_embeddings))

        ; save the document embedding into the ind2Tweet dictionary at index i
        (setv (get ind2Doc_dict i) doc_embedding)

        ; append the document embedding to the list of document vectors
        (.append document_vec_l doc_embedding))

    ; convert the list of document vectors into a 2D array (each row is a document vector)
    (setv document_vec_matrix (np.vstack document_vec_l))

    #(document_vec_matrix ind2Doc_dict))

(setv #(document_vecs ind2Tweet) (get_document_vecs all_tweets en_embeddings_subset))

(print f"length of dictionary {(len ind2Tweet)}")
(print f"shape of document_vecs {document_vecs.shape}")


(setv my_tweet "i am sad")
(process_tweet my_tweet)
(setv tweet_embedding (get_document_embedding my_tweet en_embeddings_subset))
(setv idx (np.argmax (cosine_similarity document_vecs tweet_embedding)))
(print (get all_tweets idx))

; 3.3 Finding the most similar tweets with LSH


(setv N_VECS (len all_tweets)) ; This many vectors.
(setv N_DIMS (len (get ind2Tweet 1))) ; Vector dimensionality.
(print f"Number of vectors is {N_VECS} and each has {N_DIMS} dimensions.")

; The number of planes. We use log2(625) to have ~16 vectors/bucket.
(setv N_PLANES 10)
; Number of times to repeat the hashing to improve the search.
(setv N_UNIVERSES 25)

;3.4 Getting the hash number for a vector


; Exercise 09: Implementing hash buckets


(np.random.seed 0)
(setv planes_l (lfor _ (range N_UNIVERSES) (np.random.normal :size #(N_DIMS N_PLANES))))


(defn hash_value_of_vector [v planes]
    "Create a hash for a vector; hash_id says which random hash to use.
    Input:
        - v:  vector of tweet. It's dimension is (1, N_DIMS)
        - planes: matrix of dimension (N_DIMS, N_PLANES) - the set of planes that divide up the region
    Output:
        - res: a number which is used as a hash for your vector
    "
    ; for the set of planes,
    ; calculate the dot product between the vector and the matrix containing the planes
    ; remember that planes has shape (300, 10)
    ; The dot product will have the shape (1,10)
    (setv dot_product (np.dot v planes))

    ; get the sign of the dot product (1,10) shaped vector
    (setv sign_of_dot_product (>= dot_product 0))

    ; set h to be false (eqivalent to 0 when used in operations) if the sign is negative,
    ; and true (equivalent to 1) if the sign is positive (1,10) shaped vector
    (setv h sign_of_dot_product)

    ; remove extra un-used dimensions (convert this from a 2D to a 1D array)
    (setv h (np.squeeze h))

    ; initialize the hash value to 0
    (setv hash_value 0)

    (setv n_planes (get planes.shape 1))
    (for [i (range n_planes)]
        ; increment the hash value by 2^i * h_i
        (setv hash_value (+ hash_value (* (get h i) (** 2 i)))))

    ; cast hash_value as an integer
    (setv hash_value (int hash_value))

    hash_value)


(defn test_hash_value_of_vector []
    (np.random.seed 0)
    (setv idx 0)
    (setv planes (get planes_l idx))  ; get one 'universe' of planes to test the function
    (setv vec (np.random.rand 1 300))
    (print (+ f" The hash value for this vector,"
              f"and the set of planes at index {idx},"
              f"is {(hash_value_of_vector vec planes)}")))
(test_hash_value_of_vector)

(defn make_hash_table [vecs planes]
    "
    Input:
        - vecs: list of vectors to be hashed.
        - planes: the matrix of planes in a single \"universe\", with shape (embedding dimensions, number of planes).
    Output:
        - hash_table: dictionary - keys are hashes, values are lists of vectors (hash buckets)
        - id_table: dictionary - keys are hashes, values are list of vectors id's
                            (it's used to know which tweet corresponds to the hashed vector)
    "

    ; number of planes is the number of columns in the planes matrix
    (setv num_of_planes (get planes.shape 1))

    ; number of buckets is 2^(number of planes)
    (setv num_buckets (** 2 num_of_planes))

    ; create the hash table as a dictionary.
    ; Keys are integers (0,1,2.. number of buckets)
    ; Values are empty lists
    (setv hash_table (dfor i (range num_buckets) i []))

    ; create the id table as a dictionary.
    ; Keys are integers (0,1,2... number of buckets)
    ; Values are empty lists
    (setv id_table (dfor i (range num_buckets) i []))

    ; for each vector in 'vecs'
    (for [[i v] (enumerate vecs)]
        ; calculate the hash value for the vector
        (setv h (hash_value_of_vector v planes))

        ; store the vector into hash_table at key h,
        ; by appending the vector v to the list at key h
        (.append (get hash_table h) v)

        ; store the vector's index 'i' (each document is given a unique integer 0,1,2...)
        ; the key is the h, and the 'i' is appended to the list at key h
        (.append (get id_table h) i))

    #(hash_table id_table))

(defn test_make_hash_table []
    (np.random.seed 0)
    (setv planes (get planes_l 0)) ; get one 'universe' of planes to test the function
    (setv vec (np.random.rand 1 300))
    (setv [tmp_hash_table tmp_id_table] (make_hash_table document_vecs planes))

    (print f"The hash table at key 0 has {(len (get tmp_hash_table 0))} document vectors")
    (print f"The id table at key 0 has {(len (get tmp_id_table 0))}")
    (print f"The first 5 document indices stored at key 0 of are {(get tmp_id_table 0 (slice 0 5))}"))
(test_make_hash_table)


(setv hash_tables [])
(setv id_tables [])
(for [universe_id (range N_UNIVERSES)]  ; there are 25 hashes
    (print "working on hash universe #:" universe_id)
    (setv planes (get planes_l universe_id))
    (setv [hash_table id_table] (make_hash_table document_vecs planes))
    (hash_tables.append hash_table)
    (id_tables.append id_table))

(defn approximate_knn [doc_id v planes_l [k 1] [num_universes_to_use N_UNIVERSES]]
    "Search for k-NN using hashes."
    (assert (<= num_universes_to_use N_UNIVERSES))

    ; Vectors that will be checked as possible nearest neighbor
    (setv vecs_to_consider_l [])

    ; list of document IDs
    (setv ids_to_consider_l [])

    ; create a set for ids to consider, for faster checking if a document ID already exists in the set
    (setv ids_to_consider_set (set))

    ; loop through the universes of planes
    (for [universe_id (range num_universes_to_use)]

        ; get the set of planes from the planes_l list, for this particular universe_id
        (setv planes (get planes_l universe_id))

        ; get the hash value of the vector for this set of planes
        (setv hash_value (hash_value_of_vector v planes))

        ; get the hash table for this particular universe_id
        (setv hash_table (get hash_tables universe_id))

        ; get the list of document vectors for this hash table, where the key is the hash_value
        (setv document_vectors_l (get hash_table hash_value))

        ; get the id_table for this particular universe_id
        (setv id_table (get id_tables universe_id))

        ; get the subset of documents to consider as nearest neighbors from this id_table dictionary
        (setv new_ids_to_consider (get id_table hash_value))

        ; remove the id of the document that we're searching
        (when (in doc_id new_ids_to_consider)
            (.remove new_ids_to_consider doc_id)
            (print f"removed doc_id {doc_id} of input vector from new_ids_to_search"))

        ; loop through the subset of document vectors to consider
        (for [[i new_id] (enumerate new_ids_to_consider)]
            ; if the document ID is not yet in the set ids_to_consider...
            (when (not (in new_id ids_to_consider_set))
                ; access document_vectors_l list at index i to get the embedding
                ; then append it to the list of vectors to consider as possible nearest neighbors
                (setv document_vector_at_i (get document_vectors_l i))

                ; append the new_id (the index for the document) to the list of ids to consider
                (.append vecs_to_consider_l document_vector_at_i)
                (.append ids_to_consider_l new_id)

                ; also add the new_id to the set of ids to consider
                ; (use this to check if new_id is not already in the IDs to consider)
                (.add ids_to_consider_set new_id))))

    ; Now run k-NN on the smaller set of vecs-to-consider.
    (print (% "Fast considering %d vecs" (len vecs_to_consider_l)))

    ; convert the vecs to consider set to a list, then to a numpy array
    (setv vecs_to_consider_arr (np.array vecs_to_consider_l))

    ; call nearest neighbors on the reduced list of candidate vectors
    (setv nearest_neighbor_idx_l (nearest_neighbor v vecs_to_consider_arr :k k))

    ; Use the nearest neighbor index list as indices into the ids to consider
    ; create a list of nearest neighbors by the document ids
    (setv nearest_neighbor_ids (lfor idx nearest_neighbor_idx_l (get ids_to_consider_l idx)))

    nearest_neighbor_ids)

(setv doc_id 0)
(setv doc_to_search (get all_tweets doc_id))
(setv vec_to_search (get document_vecs doc_id))


(setv nearest_neighbor_ids (approximate_knn doc_id vec_to_search planes_l :k 3 :num_universes_to_use 5))


(print f"Nearest neighbors for document {doc_id}")
(print f"Document contents: {doc_to_search}")
(print "")

(for [neighbor_id nearest_neighbor_ids]
    (print f"Nearest neighbor at document id {neighbor_id}")
    (print f"document contents: {(get all_tweets neighbor_id)}"))
