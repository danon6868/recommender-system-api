import re
import string
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer


wnl = WordNetLemmatizer()


def preprocessing(line, token=wnl):
    line = line.lower()
    line = re.sub(r"[{}]".format(string.punctuation), " ", line)
    line = line.replace("\n\n", " ").replace("\n", " ")
    line = " ".join([token.lemmatize(x) for x in line.split(" ")])
    return line


tfidf = TfidfVectorizer(stop_words="english", preprocessor=preprocessing)
