import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import re
import math
from sklearn.model_selection import train_test_split
from collections import Counter, defaultdict
from bs4 import BeautifulSoup
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
# from wordcloud import WordCloud

MNB = None

class Tokenizer:
  
  def clean(self, text):
      no_html = BeautifulSoup(text).get_text()
      clean = re.sub("[^a-z\s]+", " ", no_html, flags=re.IGNORECASE)
      return re.sub("(\s+)", " ", clean)


  def tokenize(self, text):
      clean = self.clean(text).lower()
      stopwords_en = stopwords.words("english")
      return [w for w in re.split("\W+", clean) if not w in stopwords_en]



class MultinomialNaiveBayes:
  
    def __init__(self, classes, tokenizer):
      self.tokenizer = tokenizer
      self.classes = classes
      
    def group_by_class(self, X, y):
      data = dict()
      for c in self.classes:
        data[c] = X[np.where(y == c)]
      return data
          
    def fit(self, X, y):
        self.n_class_items = {}
        self.log_class_priors = {}
        self.word_counts = {}
        self.vocab = set()

        n = len(X)
        
        grouped_data = self.group_by_class(X, y)
        
        for c, data in grouped_data.items():
          self.n_class_items[c] = len(data)
          self.log_class_priors[c] = math.log(self.n_class_items[c] / n)
          self.word_counts[c] = defaultdict(lambda: 0)
          
          for text in data:
            counts = Counter(self.tokenizer.tokenize(text))
            for word, count in counts.items():
                if word not in self.vocab:
                    self.vocab.add(word)

                self.word_counts[c][word] += count
                
        return self
      
    def laplace_smoothing(self, word, text_class):
      num = self.word_counts[text_class][word] + 1
      denom = self.n_class_items[text_class] + len(self.vocab)
      return math.log(num / denom)
      
    def predict(self, X):
        result = []
        for text in X:
          
          class_scores = {c: self.log_class_priors[c] for c in self.classes}

          words = set(self.tokenizer.tokenize(text))
          for word in words:
              if word not in self.vocab: continue

              for c in self.classes:
                
                log_w_given_c = self.laplace_smoothing(word, c)
                class_scores[c] += log_w_given_c
                
          result.append(max(class_scores, key=class_scores.get))

        return result

def initial_training():
  global MNB
  print('imported')
  # get_ipython().run_line_magic('matplotlib', 'inline')

  sns.set(style='whitegrid', palette='muted', font_scale=1.5)

  rcParams['figure.figsize'] = 14, 8

  RANDOM_SEED = 42

  np.random.seed(RANDOM_SEED)
  nltk.download('stopwords')

  train = dataset=pd.read_csv('IMDB Dataset.csv.zip',sep=',')

  print(train.head())

  # f = sns.countplot(x='sentiment', data=train)
  # f.set_title("Sentiment distribution")
  # f.set_xticklabels(['Negative', 'Positive'])
  # plt.xlabel("");

  # text = " ".join(review for review in train.review)


  # wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white", stopwords=stopwords.words("english")).generate(text)
  # plt.figure()
  # plt.imshow(wordcloud, interpolation="bilinear")
  # plt.axis("off")
  # plt.show();

  print('splitting')

  X = train['review'].values
  y = train['sentiment'].values
    
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

  print('training')

  MNB = MultinomialNaiveBayes(
      classes=np.unique(y), 
      tokenizer=Tokenizer()
  ).fit(X_train, y_train)


  print('predicting')
  y_hat = MNB.predict(X_test)

  print("accuracy:", accuracy_score(y_test, y_hat))

  # return MNB

def pred_imdb(s):
    # s = input('enter the imdb review page link: ')
    # s = 'https://www.imdb.com/title/tt8178634/reviews/?ref_=tt_ql_urv'
    s1 = s.split('/')
    s1 = s1[4]

    url = "https://www.imdb.com/title/"+s1+"/reviews/_ajax?ref_=undefined&paginationKey={}"
    key = ""
    data = {"title": [], "review": []}

    while True:
        response = requests.get(url.format(key))
        soup = BeautifulSoup(response.content, "html.parser")
        # Find the pagination key
        pagination_key = soup.find("div", class_="load-more-data")
        if not pagination_key:
            break

        # Update the `key` variable in-order to scrape more reviews
        key = pagination_key["data-key"]
        for title, review in zip(soup.find_all(class_="title"), soup.find_all(class_="text show-more__control")):
            data["title"].append(title.get_text(strip=True))
            data["review"].append(review.get_text())

    kgf2 = pd.DataFrame(data)

    print('prediction...')

    y = MNB.predict(kgf2['review'])

    kgf2['sentiment'] = pd.Series(y)
    
    p = kgf2[kgf2['sentiment'] == 'positive']['title'].count()
    n = kgf2[kgf2['sentiment'] == 'negative']['title'].count()


    print('output')
    return [int(p),int(n)]
    # Data to plot
    labels = 'Positive', 'negative'
    sizes = [p,n]
    colors = ['green', 'red']
    #explode = (0.1, 0, 0, 0)  # explode 1st slice

    # Plot
    # plt.pie(sizes, labels=labels, colors=colors,
    # autopct='%1.1f%%', shadow=False, startangle=140)

    # plt.axis('equal')
    # plt.show()