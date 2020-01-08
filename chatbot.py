'''
** PACKAGES TO INSTALL:
 - pip install nltk : Natural Language Toolkit package which is a popular package for NLP
 - pip install newspaper3k: python package for extracting and parsing newspaper articles
'''

# STEP 1: Importing Libraries & Package
from newspaper import Article # We will use the 'newspaper library to extract the text from the website by using the 'Article' class
import random # We will use the 'random' library to generate a random number for our greeting response
import string # We will use the 'string' library to process the standard Python string.
from sklearn.feature_extraction.text import TfidfVectorizer # Used to get the Tf-Idf Vectorized class to vectorize and evaluate how important a word to a document
from sklearn.metrics.pairwise import cosine_similarity # Used to see how similar the text is to the user queries
import nltk
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# STEP 2: Download packages from NLTK
nltk.download('punkt', quiet = True)
nltk.download('wordnet', quiet = True)

# STEP 3: Get the article URL
article = Article('https://www.mayoclinic.org/diseases-conditions/chronic-kidney-disease/symptoms-causes/syc-20354521')
article.download()
article.parse()
article.nlp()
corpus = article.text

print(corpus)

# STEP 4: Tokenize data
text = corpus
sent_tokens = nltk.sent_tokenize(text)

print(sent_tokens)


# STEP 5: Greeting function
GREETING_INPUT = ['hello', 'hi', 'hi there']
GREETING_RESPONSE= ['hello', 'hi', 'hi there']

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUT:
            return random.choice(GREETING_RESPONSE)

# STEP 6: Lemmatization function

# create a dictionary to remove the punctuation
remove_punc_dict = dict((ord(punct), None) for punct in string.punctuation)

# function to return a list of lemmatized lower case words after removing punctuation
def lemmonized(text):
    return nltk.word_tokenize(text.lower().translate(remove_punc_dict))

# STEP 7: Generating response
def response(user_response):
    robo_response = '' # empty response for the bot
    sent_tokens.append(user_response)
    tfidfVec = TfidfVectorizer(tokenizer = lemmonized, stop_words = 'english')
    tfidf = tfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    score = flat[-2]
    if(score == 0):
        robo_response =  robo_response + " Apologies, I dont understand"
    else:
        robo_response = robo_response + sent_tokens[idx]

    sent_tokens.remove(user_response)

    return robo_response

# STEP 8: 
flag = True
print("DOCBot: Hello. I'm your assistant. Anything I can help? If you want to exit, type \'Bye\'")
bye_phrase = ['bye', 'ok bye', 'bye bye']
while(flag):
    user_response = input()
    user_response = user_response.lower()
    if(user_response != 'bye'):
        if(user_response == 'thanks' or user_response == 'thank you'):
            flag = False
            print("DOCBot: You're welcome")
        else:
            if(greeting(user_response) != None):
                print("DOCBot: " + greeting(user_response))
            else:
                print("DOCBot: " + response(user_response))
    else:
        flag = False
        print("DOCBot: my pleasure to talk with you")

