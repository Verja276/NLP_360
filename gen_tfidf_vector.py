import pickle
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_text(text):
    # remove special chars and numbers
    text = re.sub("[^A-Za-z]+", " ", text)
    
    # remove stopwords
    tokens = nltk.word_tokenize(text)
    tokens = [w for w in tokens if not w.lower() in stopwords.words("english")]
    text = " ".join(tokens)
    text = text.lower().strip()
    
    return text



"""df = pd.read_csv('StructuredQA.csv')

'''sample = df['Question'][1]
print(sample)
print(preprocess_text(sample))'''

df['preprocessedQuestion'] = df['Question'].apply(lambda text: preprocess_text(text))
print(df.head())

vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=5, max_df=0.95)

vectors = vectorizer.fit_transform(df['preprocessedQuestion']).toarray()
vectors = pd.DataFrame(vectors)

with open('vectorizer.pkl','wb') as f:
    pickle.dump(vectorizer,f)
    
print(vectors.head())
vectors.to_csv('TFIDFVectors.csv', index=False)"""