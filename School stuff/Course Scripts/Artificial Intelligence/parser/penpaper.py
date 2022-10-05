import os
import requests
import io #codecs
import nltk
from nltk import word_tokenize, sent_tokenize 

# Text version of https://kilgarriff.co.uk/Publications/2005-K-lineer.pdf
if os.path.isfile('text.txt'):
    with io.open('text.txt', encoding='utf8') as fin:
        text = fin.read()
from nltk.corpus import stopwords
import re
text = text.lower()

m = re.sub(r'\b[A-Z]+\b', '', text)
        
text_tokens = word_tokenize(m)
s=set(stopwords.words('english'))
a= ['a', 'an', 'and', 'as', 'at', 'for', 'from', 'in', 'into', 'of', 'on', 'or', 'the', 'to']

words = [word for word in text_tokens if word.isalpha()]

last = [w for w in words if w not in a]




