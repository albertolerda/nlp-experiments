;; Import functions and data
(import nltk)
(import os [getcwd])
(import numpy :as np)
(import pandas :as pd)
(import nltk.corpus [twitter_samples])
(import utils [process_tweet build_freqs])

(setv file-path f"{(getcwd)}/../tmp2/")
(print file-path)
(nltk.data.path.append file-path)

;; Just first time download data
; (nltk.download "twitter_samples")
; (nltk.download "stopwords")

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

;; create frequency dictionary
(setv freqs (build_freqs train_x train_y))

;; check the output
(print (+ "type(freqs) = " (str (type freqs))))
(print (+ "len(freqs) = " (str (len (freqs.keys)))))


;; Logistic regression
(defn sigmoid [z]
    (/ 1.0 (+ 1.0 (np.exp (- z)))))

(defn gradientDescent [x y theta alpha num_iters]
    "
    Input:
        x: matrix of features which is (m,n+1)
        y: corresponding labels of the input matrix x, dimensions (m,1)
        theta: weight vector of dimension (n+1,1)
        alpha: learning rate
        num_iters: number of iterations you want to train your model for
    Output:
        J: the final cost
        theta: your final weight vector
    "
    ; get 'm', the number of rows in matrix x
    (setv m (get x.shape 0))
    
    (for [i (range 0 num_iters)]
        ; get z, the dot product of x and theta
        (setv z (np.dot x theta))
        
        ; get the sigmoid of z
        (setv h (sigmoid z))
        
        ; calculate the cost function
        (setv J (/ (+ (np.dot (np.transpose y) (np.log h)) (np.dot (- 1.0 (np.transpose y)) (np.log (- 1.0 h)))) (- m)))

        ; update the weights theta
        (setv theta (- theta (* (/ alpha m) (np.dot (np.transpose x) (- h y)))))

        (print J))
        
    (setv J (float J))
    #(J theta))

;; Test for gradient descent
;; (np.random.seed 1)
;; (setv tmp_X (np.append (np.ones #(10 1)) (* (np.random.rand 10 2) 2000) :axis 1))
;; (setv tmp_Y (.astype (> (np.random.rand 10 1) 0.35) float))
;; 
;; (setv #(tmp_J tmp_theta) (gradientDescent tmp_X tmp_Y (np.zeros #(3 1)) 1e-8 700))
;; (print f"The cost after training is {tmp_J :.8f}.")
;; (print f"The resulting vector of weights is {(lfor t (np.squeeze tmp_theta) (round t 8))}")

;;;; Part 2: Extracting the features

(defn extract_features [tweet freqs]
    "
    Input: 
        tweet: a list of words for one tweet
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
    Output: 
        x: a feature vector of dimension (1,3)
    "
    ; process_tweet tokenizes, stems, and removes stopwords
    (setv word_l (process_tweet tweet))
    
    ; 3 elements in the form of a 1 x 3 vector
    (setv x (np.zeros #(1 3)))
    
    ; bias term is set to 1
    (setv (get x 0 0) 1)
    
    ; loop through each word in the list of words
    (for [word word_l]
        
        ; increment the word count for the positive label 1
        (setv (get x 0 1) (+ (get x 0 1) 
            (try
                (get freqs #(word 1))
                (except [LookupError] 0))))
        ; increment the word count for the negative label 0
        (setv (get x 0 2) (+ (get x 0 2) 
            (try
                (get freqs #(word 0))
                (except [LookupError] 0)))))
        
    (assert (= x.shape #(1 3)))
    x)

(setv tmp1 (extract_features (get train_x 0) freqs))
(print tmp1)


;;;; Part 3: Training Your Model
; collect the features 'x' and stack them into a matrix 'X'
(setv X (np.zeros #((len train_x) 3)))
(for [i (range (len train_x))]
    (setv (cut X i None) (extract_features (get train_x i) freqs)))

; training labels corresponding to X
(setv Y train_y)

; Apply gradient descent
(setv #(J theta) (gradientDescent X Y (np.zeros #(3, 1)) 1e-9 1500))
(print f"The cost after training is {J :.8f}.")
(print f"The resulting vector of weights is {(lfor t (np.squeeze theta) (round t 8))}")

;;;; Part 4: Test your logistic regression
(defn predict_tweet [tweet freqs theta]
    "
    Input: 
        tweet: a string
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
        theta: (3,1) vector of weights
    Output: 
        y_pred: the probability of a tweet being positive or negative
    "
    
    ; extract the features of the tweet and store it into x
    (setv x (extract-features tweet freqs))
    
    ; make the prediction using x and theta
    (sigmoid (np.dot x theta)))

(for [tweet ["I am happy" "I am bad" "this movie should have been great." "great" "great great" "great great great" "great great great great"]]
    (print (% "%s -> %f" #(tweet (predict_tweet tweet freqs theta)))))

(print (predict_tweet "I am learning :)" freqs theta))

(print (predict_tweet "Surfing is difficult, but I won't stop trying to improve" freqs theta))

;; Check performance using the test set

(defn test_logistic_regression [test_x test_y freqs theta]
    "
    Input: 
        test_x: a list of tweets
        test_y: (m, 1) vector with the corresponding labels for the list of tweets
        freqs: a dictionary with the frequency of each pair (or tuple)
        theta: weight vector of dimension (3, 1)
    Output: 
        accuracy: (# of tweets classified correctly) / (total # of tweets)
    "
    
    ; the list for storing predictions
    (setv y_hat [])
    
    (for [tweet test_x]
        ; get the label prediction for the tweet
        (setv y_pred (predict_tweet tweet freqs theta))
        
        (.append y-hat
            (if (> y_pred 0.5) 1.0 0.0)))

    ; With the above implementation, y_hat is a list, but test_y is (m,1) array
    ; convert both to one-dimensional arrays in order to compare them using the '==' operator
    (setv accuracy (/ (np.sum (= (np.asarray y-hat) (np.squeeze test_y))) (len y-hat)))

    accuracy)

(setv tmp_accuracy (test_logistic_regression test_x test_y freqs theta))
(print f"Logistic regression model's accuracy = {tmp_accuracy :.4f}")


(print "Label Predicted Tweet")
(for [[x y] (zip test_x test_y)]
    (setv y_hat (predict_tweet x freqs theta))

    (when (> (np.abs (- y (> y_hat 0.5))) 0)
        (do
            (print "THE TWEET IS:" x)
            (print "THE PROCESSED TWEET IS:" (process_tweet x))
            (print (% "%d\t%0.8f\t%s" #(y y_hat (.encode (.join " " (process_tweet x)) "ascii" "ignore")))))))