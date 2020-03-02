import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF
from debate_cleaning import *
from nltk.stem.wordnet import WordNetLemmatizer
from matplotlib import pyplot as plt

def build_text_vectorizer(contents, use_tfidf=True, use_stemmer=False, max_features=None):

    Vectorizer = TfidfVectorizer if use_tfidf else CountVectorizer
    tokenizer = RegexpTokenizer(r"[\w']+")
    stem = PorterStemmer().stem if use_stemmer else (lambda x: x)
    # stem = WordNetLemmatizer().lemmatize
    stop_set = set(stopwords.words('english'))
    stop_set = stop_set.union({'go','way','say','back','sure','would','think','see','first','second','one','said','go','put',
    'get','going','think','get','done','having','has','need','want','us','got','well','crosstalk'})

    # Closure over the tokenizer et al.
    def tokenize(text):
        tokens = tokenizer.tokenize(text)
        stems = [stem(token) for token in tokens if token not in stop_set]
        return stems

    vectorizer_model = Vectorizer(tokenizer=tokenize, max_features=max_features)
    vectorizer_model.fit(contents)
    vocabulary = np.array(vectorizer_model.get_feature_names())

    # Closure over the vectorizer_model's transform method.
    def vectorizer(X):
        return vectorizer_model.transform(X).toarray()

    return vectorizer, vocabulary

def hand_label_topics(H, vocabulary):

    hand_labels = []
    for i, row in enumerate(H):
        top_five = np.argsort(row)[::-1][:20]
        print('topic', i)
        print('-->', ' '.join(vocabulary[top_five]))
        label = input('please label this topic: ')
        hand_labels.append(label)
        print()
    return hand_labels

def analyze_article(article_index, contents, W, hand_labels, candidate_dictionary):


    speaker = df.iloc[article_index]['speaker']
    if speaker not in candidate_dictionary.keys():
        candidate_dictionary[speaker] = {}

    probs = softmax(W[article_index], temperature=0.01)
    for prob, label in zip(probs, hand_labels):
        if label not in candidate_dictionary[speaker].keys():
            candidate_dictionary[speaker][label] = prob
        else:
            candidate_dictionary[speaker][label] += prob

    return candidate_dictionary

def softmax(v, temperature=1.0):
    '''
    A heuristic to convert arbitrary positive values into probabilities.
    See: https://en.wikipedia.org/wiki/Softmax_function
    '''
    expv = np.exp(v / temperature)
    s = np.sum(expv)
    return expv / s

if __name__ == '__main__':
    df = pd.read_csv("/Users/bechis/dsi/repo/Capstone3/data/debate_transcripts.csv", encoding= 'unicode_escape')
    df1 = pd.read_csv("/Users/bechis/DSI/repo/capstone3/data/debate_transcripts_nevada.csv", encoding= 'unicode_escape')
    df2 = pd.read_csv("/Users/bechis/dsi/repo/capstone3/data/South_Carolina_Debate.csv", encoding= 'unicode_escape')
    df = pd.concat([df, df1, df2])
    df = unusable_rows(df)
    df = get_candidates(df)

    plt.style.use('default')

    candidates_list = ['Joe Biden', 'Bernie Sanders', 'Amy Klobuchar', 'Elizabeth Warren', 'Pete Buttigieg', 'Tom Steyer']

    contents = df['speech']
    vectorizer, vocabulary = build_text_vectorizer(contents,
                                use_tfidf=True,
                                use_stemmer=False,
                                max_features=5000)
    X = vectorizer(contents)

    nmf = NMF(n_components=13, max_iter=100, alpha=0.0)

    W = nmf.fit_transform(X)
    H = nmf.components_

    rand_articles = np.random.choice(list(range(len(W))), 15)
    hand_labels = hand_label_topics(H, vocabulary)

    candidate_dictionary = {}

    for i in range(len(W)):
         candidate_dictionary = analyze_article(i, contents, W, hand_labels, candidate_dictionary)

    joe_count = len(df[df['speaker']=='Amy Klobuchar'])
    fig = plt.figure(figsize=(8,8))
    count = 1
    vals = [0,5,10,15,20,25]

    for x in candidate_dictionary.keys():
        joe_count = len(df[df['speaker']==x])
        labels = []
        values = []
        print(x)
        for y in candidate_dictionary[x]:
            print(str(y) + ' ' + str(round(((candidate_dictionary[x][y] / joe_count)*100),2)))
            labels.append(y)
            values.append(round(((candidate_dictionary[x][y] / joe_count)*100),2))

        if x in candidates_list:
            ax = fig.add_subplot(3,2,count)
            length_labels = np.arange(len(labels))
            ax.barh(length_labels, values)
            # ax.set_ylabel('Debate Topics')
            ax.set_xlim(right = 25)
            ax.set_xticks(vals)
            ax.set_xticklabels(['{}%'.format(x) for x in vals])
            ax.set_yticks(length_labels)
            ax.set_yticklabels(labels)
            ax.set_title(x)
            plt.tight_layout()
            count += 1
        print('\n')

    plt.show()

    # plt.savefig('candidates_loading.png', transparent = True)
