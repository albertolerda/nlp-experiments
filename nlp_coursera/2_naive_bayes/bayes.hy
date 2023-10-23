#!/usr/bin/env hy

;; Import functions and data
(import utils [process_tweet lookup])
(import nltk.corpus [stopwords twitter_samples])
(import numpy :as np)
(import pandas :as pd)
(import nltk)
(import os [getcwd])
(import string)

(setv file-path f"{(getcwd)}/../tmp2/")
(print file-path)
(nltk.data.path.append file-path)

(nltk.download "twitter_samples")
(nltk.download "stopwords")

;; split the data into two pieces, one for training and one for testing (validation set)
(setv all-positive-tweets (.strings twitter_samples "positive_tweets.json"))
(setv all-negative-tweets (twitter_samples.strings "negative_tweets.json"))

(setv test-pos (cut all-positive-tweets 4000 None))
(setv train-pos (cut all-positive-tweets 0 4000))
(setv test-neg (cut all-negative-tweets 4000 None))
(setv train-neg (cut all-negative-tweets 0 4000))

(setv train_x (+ train_pos train_neg))
(setv test_x (+ test_pos test_neg))

(setv train_y (np.append (np.ones #((len train-pos) 1)) (np.zeros #((len train-neg) 1)) :axis 0))
(setv test_y (np.append (np.ones #((len test-pos) 1)) (np.zeros #((len test-neg) 1)) :axis 0))


(print (+ "train_y.shape = " (str train_y.shape)))
(print (+ "test_y.shape = " (str test_y.shape)))

;;; Part 1.1 Implementing your helper functions
(defn count_tweets [result tweets ys]
    "
    Input:
        result: a dictionary that will be used to map each pair to its frequency
        tweets: a list of tweets
        ys: a list corresponding to the sentiment of each tweet (either 0 or 1)
    Output:
        result: a dictionary mapping each pair to its frequency
    "

    (for [[y tweet] (zip (np.squeeze ys) tweets)]
        (for [word (process_tweet tweet)]
            ; define the key, which is the word and label tuple
            (setv pair #(word y))

            (if (in pair result)
                (setv (get result pair) (+ (get result pair) 1))
            ; else, if the key is new, add it to the dictionary and set the count to 1
                (setv (get result pair) 1))))

    result)

(defn test_count_tweets []
    (setv result {})
    (setv tweets ["i am happy" "i am tricked" "i am sad" "i am tired" "i am tired"])
    (setv ys [1 0 0 0 0])
    (count_tweets result tweets ys))

(print (test_count_tweets))


;;;; Part 2: Train your model using Naive Bayes
; Build the freqs dictionary for later uses
(setv freqs (count_tweets {} train_x train_y))


(defn train_naive_bayes [freqs train_x train_y]
    "
    Input:
        freqs: dictionary from (word, label) to how often the word appears
        train_x: a list of tweets
        train_y: a list of labels correponding to the tweets (0,1)
    Output:
        logprior: the log prior. (equation 3 above)
        loglikelihood: the log likelihood of you Naive bayes equation. (equation 6 above)
    "
    (setv loglikelihood {})
    (setv logprior 0)

    ; calculate V, the number of unique words in the vocabulary
    (setv vocab (set (lfor pair (freqs.keys) (get pair 0))))
    (setv V (len vocab))

    ; calculate N_pos and N_neg
    (setv N_pos 0)
    (setv N_neg 0)
    (for [pair (freqs.keys)]
        ; if the label is positive (greater than zero)
        (if (> (get pair 1) 9)
            ; Increment the number of positive words by the count for this (word, label) pair
            (setv N_pos (+ N_pos (get freqs pair)))
            ; else, the label is negative
            ; increment the number of negative words by the count for this (word,label) pair
            (setv N_neg (+ N_pos (get freqs pair)))))

    ; Calculate D, the number of documents
    (setv D (len train_y))

    ; Calculate D_pos, the number of positive documents (*hint: use sum(<np_array>))
    (setv D_pos (np.sum train_y))

    ; Calculate D_neg, the number of negative documents (*hint: compute using D and D_pos)
    (setv D_neg (- D D_pos))

    ; Calculate logprior
    (setv logprior (- (np.log D_pos) (np.log D_neg)))

    ; For each word in the vocabulary...
    (for [word vocab]
        ; get the positive and negative frequency of the word
        (setv freq_pos (try (get freqs #(word 1)) (except [KeyError] 0)))
        (setv freq_neg (try (get freqs #(word 0)) (except [KeyError] 0)))

        ; calculate the probability that each word is positive, and negative
        (setv p_w_pos (/ (+ freq_pos 1) (+ N_pos V)))
        (setv p_w_neg (/ (+ freq_neg 1) (+ N_neg V)))

        ; calculate the log likelihood of the word
        (setv (get loglikelihood word) (- (np.log p_w_pos) (np.log p_w_neg))))

    #(logprior loglikelihood))

(setv #(logprior loglikelihood) (train_naive_bayes freqs train_x train_y))
(defn test_train_naive_bayes []
    (print logprior)
    (print (len loglikelihood)))
(test_train_naive_bayes)


;;;; Part 3: Test your naive bayes
(defn naive_bayes_predict [tweet logprior loglikelihood]
    "
    Input:
        tweet: a string
        logprior: a number
        loglikelihood: a dictionary of words mapping to numbers
    Output:
        p: the sum of all the logliklihoods of each word in the tweet (if found in the dictionary) + logprior (a number)

    "
    ; process the tweet to get a list of words
    (setv word_l (process_tweet tweet))

    ; initialize probability to logprior
    (setv p logprior)

    (for [word word_l]

        ; check if the word exists in the loglikelihood dictionary
        (when (in word loglikelihood)
            ; add the log likelihood of that word to the probability
            (setv p (+ p (get loglikelihood word)))))

    p)

(defn test_naive_bayes_predict []
    (setv #(logprior loglikelihood) (train_naive_bayes freqs train_x train_y))
    (setv my_tweet "She smiled.")
    (setv p (naive_bayes_predict my_tweet logprior loglikelihood))
    (print "The expected output is" p))
(test-naive-bayes-predict)


(defn test_naive_bayes [test_x test_y logprior loglikelihood]
    "
    Input:
        test_x: A list of tweets
        test_y: the corresponding labels for the list of tweets
        logprior: the logprior
        loglikelihood: a dictionary with the loglikelihoods for each word
    Output:
        accuracy: (# of tweets classified correctly)/(total # of tweets)
    "
    (setv accuracy 0)  ; return this properly

    (setv y_hats [])
    (for [tweet test_x]
        ; if the prediction is > 0
        (if (> (naive_bayes_predict tweet logprior loglikelihood) 0)
            ; the predicted class is 1
            (setv y_hat_i 1)
            ; otherwise the predicted class is 0
            (setv y_hat_i 0))

        ; append the predicted class to the list y_hats
        (.append y_hats y_hat_i))

    ; error is the average of the absolute values of the differences between y_hats and test_y
    (setv error (np.mean (np.absolute (- y_hats (np.squeeze(np.asarray test_y))))))

    ; Accuracy is 1 minus the error
    (setv accuracy (- 1 error))

    accuracy)

(print (% "Naive Bayes accuracy = %0.4f" 
      (test_naive_bayes test_x test_y logprior loglikelihood)))
