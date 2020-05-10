import math
from collections import defaultdict
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class NaiveBayesMovieReviewClassification:   
        
    def __init__(self, k = 0.5):
        self.k = k
        self.word_probabilities = defaultdict(lambda: [0, 0])
        self.log_total_positive_probability = self.log_total_negative_probability = 0
        self.total_positive_count = self.total_negative_count = 0
        self.lemmatizer = WordNetLemmatizer()
        self.confusion_matrix = [[0,0],[0,0]]
        
    def preprocess(self, text):
        text = text.lower()
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(text) # tokenize text
        
        stopword_set = set(stopwords.words('english'))
        # Remove all stopwords
        filtered_words = [self.lemmatizer.lemmatize(word) for word in tokens if not word in stopword_set]
        return filtered_words

    def train(self, x, y):    
        y = y.values.tolist()
        x = x.values.tolist()
        
        word_sentiment_count = defaultdict(lambda: [0, 0])
        for i, row in enumerate(x):
            words = self.preprocess(row[0])
            for word in words :
                if(y[i][0] == 1):
                    self.total_positive_count += 1
                else:
                    self.total_negative_count += 1
                word_sentiment_count[word][y[i][0]] += 1
        total_word_count = self.total_positive_count + self.total_negative_count
        self.log_total_positive_probability = math.log(self.total_positive_count / total_word_count)
        self.log_total_negative_probability = math.log(self.total_negative_count / total_word_count)
        
        for word, sentiment_counts in word_sentiment_count.items():
            self.word_probabilities[word][1] = math.log(sentiment_counts[1] + self.k) - math.log(self.total_positive_count + (2 * self.k)) + self.log_total_positive_probability
            self.word_probabilities[word][0] = math.log(sentiment_counts[0] + self.k) - math.log(self.total_negative_count + (2 * self.k)) + self.log_total_negative_probability
        
        print('Finished training!')
    
    def test(self, x, y):
        y = y.values.tolist()
        x = x.values.tolist()
        # print('Total Positive Probability : ' +str(total_positive_probability))
        # print('Total Negative Probability : ' +str(total_negative_probability))
        for i, row in enumerate(x):
            predicted_class = self.classify(row[0])
            self.confusion_matrix[y[i][0]][predicted_class] += 1 
        
        true_positive = self.confusion_matrix[1][1]
        false_positive = self.confusion_matrix[0][1]
        actual_positive = self.confusion_matrix[1][0] + self.confusion_matrix[1][1]
        
        true_negative = self.confusion_matrix[0][0]
        false_negative = self.confusion_matrix[1][0]
        actual_negative = self.confusion_matrix[0][0] + self.confusion_matrix[0][1]
        
        predicted_positive = true_positive + false_positive
        predicted_negative = true_negative + false_negative
        
        
        
        print('         | Predicted Negative | Predicted Positive |')
        print('----------------------------------------------------------')
        print('Negative |      {0}                    {1}           {2}'.format(true_negative, false_positive, actual_negative))
        print('----------------------------------------------------------')
        print('Positive |      {0}                    {1}           {2}'.format(false_negative, true_positive, actual_positive))
        print('----------------------------------------------------------')
        print('         |      {0}                    {1}           {2}'.format(predicted_negative, predicted_positive, predicted_negative + predicted_positive))

    def classify(self, review):
        words = self.preprocess(review)
        positive_probability = negative_probability = 0
        for word in words:
            if(self.word_probabilities.get(word) is not None):
                positive_probability += self.word_probabilities.get(word)[1]
                negative_probability += self.word_probabilities.get(word)[0]                
            else:
                positive_probability += math.log(self.k) - math.log(self.total_positive_count + (2 * self.k)) + self.log_total_positive_probability
                negative_probability += math.log(self.k) - math.log(self.total_negative_count + (2 * self.k)) + self.log_total_negative_probability

        return 1 if positive_probability > negative_probability else 0

nb = NaiveBayesMovieReviewClassification(0.5)

import pandas as pd

# working on training set
df = pd.read_csv('./Train.csv')
df.describe()
x_train = df.iloc[:, :-1]
y_train = df.iloc[:, 1:2]
nb.train(x_train, y_train)

# working on test set
df = pd.read_csv('./Test.csv')
df.describe()
x_test = df.iloc[:, :-1]
y_test = df.iloc[:, 1:2]
nb.test(x_test, y_test)