import pandas as pd
from matplotlib import pyplot as plt

import pymorphy2
morph = pymorphy2.MorphAnalyzer()

import nltk
import sys # чтобы получить класс по его имени в get_fragments

import math
import numpy



def tokenize(text):
  tokens = nltk.word_tokenize(text, language = 'russian')
  tokens = ([token.lower() for token in tokens]) # Lowercase the tokens
  tokens = ([token for token in tokens if any(c.isalpha() for c in token)]) # Filter out punctuation
  return tokens


class Author:
  class Valid:
    def __init__(self):
      self.word_length = []
      self.sentence_length = []
      self.service_freq = []
      self.verb_freq = []
      self.noun_freq = []
      self.adj_freq = []
      self.dist_to_chehov = []
      self.dist_to_bunin = []

  def __init__(self):
    self.train_set = self.Valid()
    self.test_set = self.Valid()
    self.num_of_words = 0
    self.num_of_tokens = 0
  
  
  def add_params(self, text, valid):
    
    # === Признаки по частям речи и количеству слов/предложений ===

    count_sentences = text.count('.')
    symb = text.replace('.', '').replace(' ', '')
    words = text.split(' ')
    words = list(filter(None, words)) #удалить пустые строки из-за пробелов в начале предложений

    self.num_of_words += len(words)
    # def part(word):
    #     return morph.parse(word)[0].tag.POS

    # parts = [("CONJ", 0), ("PREP", 0), ("PRCL", 0),  ("VERB", 0), ("NOUN", 0), ("ADJF", 0), ("ADJS", 0),]

    # for word in words:
    #   for i in range(len(parts)):
    #     if parts[i][0] == str(part(word)):
    #       parts[i] = (parts[i][0], parts[i][1] + 1)
    #       break

    # serv_part = (parts[0][1] + parts[1][1] + parts[2][1]) / len(words) * 100
    # verb_part = parts[3][1] / len(words) * 100
    # noun_part = parts[4][1] / len(words) * 100
    # adj_part = (parts[5][1] + parts[6][1]) / len(words) * 100



    # === Дельта Бёрроуза (между частотным словарём обоих текстов и данным отрывком) ===

    fragm_tokens = tokenize(text)
    overall = len(fragm_tokens)
    feature_freqs[text] = {}

    self.num_of_tokens += overall

    # Calculate each feature's presence in the fragment
    for feature in features:
        presence = fragm_tokens.count(feature)
        feature_freqs[text][feature] = presence / overall

    dist_to_chehov = 0
    for feature in features:
       dist_to_chehov += math.fabs(feature_freqs['chehov'][feature] - feature_freqs[text][feature])
    dist_to_chehov /= len(features)

    dist_to_bunin = 0
    for feature in features:
       dist_to_bunin += math.fabs(feature_freqs['bunin'][feature] - feature_freqs[text][feature])
    dist_to_bunin /= len(features)


    

    # === Добавление готовых свойств в класс ===

    # if valid == 'train':
    #   self.train_set.word_length.append(len(symb) / len(words))
    #   self.train_set.sentence_length.append(len(words) / count_sentences)
    # else:
    #   self.test_set.word_length.append(len(symb) / len(words))
    #   self.test_set.sentence_length.append(len(words) / count_sentences)

    # if valid == 'train':
    #   self.train_set.service_freq.append(serv_part)
    #   self.train_set.verb_freq.append(verb_part)
    # else:
    #   self.test_set.service_freq.append(serv_part)
    #   self.test_set.verb_freq.append(verb_part)

    # self.train_set.noun_freq.append(noun_part)
    # self.train_set.adj_freq.append(adj_part)

    if valid == 'train':
      self.train_set.dist_to_chehov.append(dist_to_chehov)
      self.train_set.dist_to_bunin.append(dist_to_bunin)
    else:
      self.test_set.dist_to_chehov.append(dist_to_chehov)
      self.test_set.dist_to_bunin.append(dist_to_bunin)



def get_text(author):
  with open(f'{author}/text.txt') as f:
    full_text = f.read()

  return full_text.replace('?', '.').replace('!', '.').replace('...', '.').replace('\n' , ' ').replace(',', '').replace('\"', '').replace('-- ' , '').replace('(', '').replace(')', '')




def share_fragments(author, num_of_sentences, cross):
  full_text = get_text(author)
  start = 0
  finish = full_text.find('.', start + 1)
  num_of_fragments = 0
  while True:
    for i in range(num_of_sentences):
      finish = full_text.find('.', finish+1)
      if finish == -1:
        return
    finish += 1 # включая точку
    fragment = full_text[start:finish]
    start = finish
    num_of_fragments += 1
    if num_of_fragments >= cross * 5 and num_of_fragments < cross * 5 + 20:
      getattr(sys.modules[__name__], author).add_params(fragment, 'test') # вызвать класс по его имени
    else:
      getattr(sys.modules[__name__], author).add_params(fragment, 'train')


chehov = Author()
bunin = Author()

def create_corpus_dict(cross):
  global whole_corpus

  authors = ('chehov', 'bunin')

  
  def get_train_corpus(author, num_of_sentences, cross):
    corpus = []
    full_text = get_text(author)
    start = 0
    finish = full_text.find('.', start + 1)
    num_of_fragments = 0
    while True:
      for i in range(num_of_sentences):
        finish = full_text.find('.', finish+1)
        if finish == -1:
          return corpus
      finish += 1 # включая точку
      fragment = full_text[start:finish]
      start = finish
      num_of_fragments += 1
      if num_of_fragments < cross * 5 or num_of_fragments >= cross * 5 + 20:
        corpus += tokenize(fragment)

  author_tokens = {}
  author_tokens['chehov'] = get_train_corpus('chehov', 49, cross)
  author_tokens['bunin'] = get_train_corpus('bunin', 45, cross)

  whole_corpus = author_tokens['chehov'] + author_tokens['bunin']

  # Get a frequency distribution
  whole_corpus_freq_dist = list(nltk.FreqDist(whole_corpus).most_common(100))
  global features
  features = [word for word,freq in whole_corpus_freq_dist]
  global feature_freqs
  feature_freqs = {}

  # Calculating features for each subcorpus
  for author in authors:
    feature_freqs[author] = {} # A dictionary for each candidate's features

    overall = len(author_tokens[author]) # the number of tokens in the author's subcorpus

    # Calculate each feature's presence in the subcorpus
    for feature in features:
        presence = author_tokens[author].count(feature)
        feature_freqs[author][feature] = presence / overall


  corpus_features = {} # The data structure into which we will be storing the "corpus standard" statistics

  for feature,freq in whole_corpus_freq_dist:
    # Create a sub-dictionary that will contain the feature's standard deviation
    corpus_features[feature] = {}

    # Calculate the standard deviation using the basic formula for a sample
    feature_stdev = 0
    for author in authors:
        diff = feature_freqs[author][feature] - freq
        feature_stdev += diff*diff
    feature_stdev = math.sqrt(feature_stdev)
    corpus_features[feature]["StdDev"] = feature_stdev



# ========    кросс-валидация   ===========

accuracy = [0, 0, 0, 0, 0]

for i in range(5):
  create_corpus_dict(i)
  share_fragments('chehov', 49, i)
  share_fragments('bunin', 45, i)


  # =========   Обучение   ==========
  # plt.tight_layout()  #что делает этот метод?

  # x_blue = chehov.train_set.word_length
  # y_blue = chehov.train_set.sentence_length
  # x_red = bunin.train_set.word_length
  # y_red = bunin.train_set.sentence_length
  # plt.xlabel('Длина слов (в символах)')
  # plt.ylabel('Длина предложений (в словах)')


  # x_blue = chehov.train_set.service_freq
  # y_blue = chehov.train_set.verb_freq
  # x_red = bunin.train_set.service_freq
  # y_red = bunin.train_set.verb_freq
  # plt.xlabel('Индекс аналитичности')
  # plt.ylabel('Индекс глагольности')

  # x_blue = chehov.noun_freq
  # y_blue = chehov.adj_freq
  # x_red = bunin.noun_freq
  # y_red = bunin.adj_freq
  # plt.xlabel('Индекс субстантивности')
  # plt.ylabel('Индекс адъективности')


  x_blue = chehov.train_set.dist_to_chehov
  y_blue = chehov.train_set.dist_to_bunin
  x_red = bunin.train_set.dist_to_chehov
  y_red = bunin.train_set.dist_to_bunin
  plt.xlabel('Дельта Бёрроуза до корпуса Бунина')
  plt.ylabel('Дельта Бёрроуза до корпуса Чехова')


  # plt.scatter(x_blue, y_blue, c = 'blue')
  # plt.scatter(x_red, y_red, c = 'red')


  #Центры масс
  x1 = sum(x_blue) / len(x_blue)
  y1 = sum(y_blue) / len(y_blue)
  x2 = sum(x_red) / len(x_red)
  y2 = sum(y_red) / len(y_red)


  # plt.scatter(x1, y1, c = 'green')
  # plt.scatter(x2, y2, c = 'green')
  # plt.scatter((x1 + x2) / 2, (y1 + y2) / 2, c = 'green')

  # Разделяющая прямая между ними
  a = 0
  k = (x2*x2 - x1*x1) / (y1 - y2)
  b = (x1*x1 + y1*y1 - x2*x2 - y2*y2) / (2 * (y1 - y2))

  # x1_divide = (3*x2 - x1) / 2
  # x2_divide = (3*x1 - x2) / 2
  # x1_divide = x1
  # x2_divide = x2
  # y1_divide = k*x1_divide + b
  # y2_divide = k*x2_divide + b


  a1 = x1 + x2
  b1 = (x1 + x2) / 3
  a2 = y1 + y2 - b
  b2 = (y1 + y2 + b) / 3

  # plt.plot((4.74, 5.04), (25.75, 5.30))
  # plt.plot((35, 20), (8, 16.7))
  plt.plot((a1, b1), (a2, b2))
  


  # =======  Тест  ========

  # x_blue = chehov.test_set.word_length
  # y_blue = chehov.test_set.sentence_length
  # x_red = bunin.test_set.word_length
  # y_red = bunin.test_set.sentence_length

  # x_blue = chehov.test_set.service_freq
  # y_blue = chehov.test_set.verb_freq
  # x_red = bunin.test_set.service_freq
  # y_red = bunin.test_set.verb_freq
  
  x_blue = chehov.test_set.dist_to_chehov
  y_blue = chehov.test_set.dist_to_bunin
  x_red = bunin.test_set.dist_to_chehov
  y_red = bunin.test_set.dist_to_bunin


  plt.scatter(x_blue, y_blue, c = 'blue')
  plt.scatter(x_red, y_red, c = 'red')

  # plt.show()

  errs = 0
  for x,y in zip(x_blue,y_blue):
    if (x - a1)*(b2 - a2) - (y - a2)*(b1 - a1) < 0:
      errs += 1

  for x,y in zip(x_red,y_red):
    if (x - a1)*(b2 - a2) - (y - a2)*(b1 - a1) > 0:
      errs += 1

  accuracy[i] = 100 * (1 - (errs / (len(x_blue) + len(x_red))))

  print(i+1, ") Accuracy: ", round(accuracy[i], 2), "%", sep = '')


avg_accuracy = sum(accuracy) / 5

print("Average accuracy:", round(avg_accuracy, 2), "%")

print('chehov words: ', chehov.num_of_words/5)
print('chehov tokens: ', chehov.num_of_tokens/5)
print('bunin words: ', bunin.num_of_words/5)
print('bunin tokens: ', bunin.num_of_tokens/5)

