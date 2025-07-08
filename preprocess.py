import string
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def preprocess(text):
    
    text = text.lower()

    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])

    # Tokenize and remove stopwords
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]

    # Return cleaned text
    return ' '.join(tokens)


