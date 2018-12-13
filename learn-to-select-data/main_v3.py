# -*- coding:utf-8 -*-
 
from scipy.stats import entropy
from scipy.spatial import distance
from gensim.models import LdaMulticore, LdaModel
from gensim.matutils import Sparse2Corpus
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from robo.fmin import bayesian_optimization
from collections import Counter
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import logging
import itertools
import pickle
import scipy
import time
import yaml
import os

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=196)
 
logging.basicConfig(level=logging.INFO)
 
class Metric():
   
    @classmethod
    def jensen_shannon(cls, repr1, repr2):
        avg_repr = 0.5 * (repr1 + repr2)
        sim = 1.0 - 0.5 * (entropy(repr1, avg_repr) + entropy(repr2, avg_repr))
        return 0.0 if np.isinf(sim) else sim
  
    @classmethod
    def renyi(cls, repr1, repr2, alpha=0.99):
        log_sum = np.sum([np.power(p, alpha) / np.power(q, alpha-1) for (p, q) in zip(repr1, repr2)])
        sim = 1.0 / (alpha - 1) * np.log(log_sum)
        return 0.0 if np.isinf(sim) else sim
  
    @classmethod
    def kullback_leibler(cls, repr1, repr2):
        return entropy(repr1, repr2)
   
    @classmethod
    def cosine(cls, repr1, repr2):
        if repr1 is None or repr2 is None: return 0
        sim = 1.0 - distance.cosine(repr1, repr2)
        return 0.0 if np.isinf(sim) else sim
   
    @classmethod
    def euclidean(cls, repr1, repr2):
        sim = np.sqrt(np.sum([np.power(p-q, 2) for (p, q) in zip(repr1, repr2)]))
        return sim
  
    @classmethod
    def variational(cls, repr1, repr2):
        sim = np.sum([np.abs(p-q) for (p, q) in zip(repr1, repr2)])
        return sim
  
    @classmethod
    def bhattacharyya(cls, repr1, repr2):
        sim = - np.log(np.sum([np.sqrt(p*q) for (p, q) in zip(repr1, repr2)]))
        return 0.0 if np.isinf(sim) else sim
   
    @classmethod
    def num_word_types(cls, words):
        return len(set(words))
  
    @classmethod
    def type_token_ratio(cls, words):
        return cls.num_word_types(words) / len(words)
  
    @classmethod
    def entropy(cls, p_words):
        elist = np.asarray([p_word * np.log(p_word) for p_word in p_words if p_words != 0.0])
        elist = np.nan_to_num(elist)
        return -np.sum(elist)
  
    @classmethod
    def simpsons_index(cls, p_words):
        return np.sum([np.power(p_word, 2) for p_word in p_words])
  
    @classmethod
    def quadratic_entropy(cls, p_word_vector_pairs):
        summed, length = 0, len(p_word_vector_pairs)
        for idx_1 in range(length):
            for idx_2 in range(idx_1, length):
                p_1, vec_word_1 = p_word_vector_pairs[idx_1]
                p_2, vec_word_2 = p_word_vector_pairs[idx_2]
                value = cls.cosine(vec_word_1, vec_word_2) * p_1 * p_2
                if idx_1 == idx_2: summed += value
                else: summed += 2 * value
        return summed
      
    @classmethod
    def renyi_entropy(cls, p_words, alpha = 0.99):
        summed = np.sum([np.power(p_word, alpha) for p_word in p_words])
        summed = 0.0001 if summed == 0 else summed
        return 1.0 / (1 - alpha) * np.log(summed)

    
class AmazonReviewDataset():
   
    def __init__(self, dirpath, word2vector_path, max_vocab_size=10000, num_topics=50, num_topic_iterations=2000, num_topic_passes=10, reproc=False):
        self.dirpath = dirpath
        self.word2vector_path = word2vector_path
        self.max_vocab_size = max_vocab_size
        self.num_topics = num_topics
        self.num_topic_iterations = num_topic_iterations
        self.num_topic_passes = num_topic_passes
        self.reproc = reproc
        self.domains = os.listdir(self.dirpath)
        if not os.path.exists("./preproc_data"):
            os.makedirs("./preproc_data")
            print("Initialize the Pre-processed data")
        elif self.reproc:
            os.rmdir("./preproc_data")
            os.makedirs("./preproc_data")
            print("Re-construct Pre-processed data")
        else:
            print("Re-use History Pre-processed data")
   
    "domain2data"
    ##################################################
    def load_domain2data(self):
      
        def file_parser(domain, split):
          
            def line_parser(line):
                features, review = line.split(' ')[:-1], []
                for feature in features:
                    ngram, count = feature.split(':')
                    for _ in range(int(count)):
                        review.append(ngram)
                return review
          
            file_path = os.path.join(self.dirpath, domain, '{}.review'.format(split))
            with open(file_path, "r") as f:
                reviews = [line_parser(line) for line in f]
            return reviews
      
        if os.path.exists("./preproc_data/domain2data.pkl"):
            with open("./preproc_data/domain2data.pkl", "rb") as filer:
                self.domain2data = pickle.load(filer)
        else:
            self.domain2data = {domain: {"labeled":[], "label":[], "unlabeled": None} for domain in self.domains}
            for domain in self.domains:
                for split in ['positive', 'negative', 'unlabeled']:
                    reviews = file_parser(domain, split)
                    if split == 'unlabeled':
                        self.domain2data[domain]['unlabeled'] = reviews
                    else:
                        self.domain2data[domain]['labeled'] += reviews
                        self.domain2data[domain]['label'] += [1 if split == "positive" else 0] * len(reviews)
                self.domain2data[domain]["label"] = np.array(self.domain2data[domain]["label"])
            with open("./preproc_data/domain2data.pkl", "wb") as filew:
                pickle.dump(self.domain2data, filew)
        print("Load domain2data has done.")
       
    ##################################################
    def load_global_vocab(self):
        if os.path.exists("./preproc_data/vocab.txt"):
            self.word2id = {}
            with open("./preproc_data/vocab.txt", 'r') as f:
                for i, line in enumerate(f):
                    if i >= self.max_vocab_size: break
                    word, idx = line.split('\t')
                    self.word2id[word] = int(idx.strip())
            self.vocab_size = len(self.word2id)
            self.id2word = {index: word for word, index in self.word2id.items()}
        else:
            texts = []
            if not hasattr(self, "domain2data"):
                self.load_domain2data()
            for domain in self.domain2data:
                texts.extend(self.domain2data[domain]["labeled"])
                texts.extend(self.domain2data[domain]["unlabeled"])
            word_counts = Counter(itertools.chain(*texts))
            most_common = word_counts.most_common(n=self.max_vocab_size)
            self.word2id = {word: index for index, (word, _) in enumerate(most_common)}
            self.id2word = {index: word for word, index in self.word2id.items()}
            with open("./preproc_data/vocab.txt", 'w') as f:
                for word, index in sorted(self.word2id.items(), key=lambda d:d[1]):
                    f.write('%s\t%d\n' % (word, index))
            self.vocab_size = len(self.word2id)
        print("Load vocab has done.")
       
    "word2vector"
    ##################################################
    def load_word2vector(self):
        if os.path.exists("./preproc_data/word2vector.pkl"):
            with open("./preproc_data/word2vector.pkl", "rb") as filer:
                self.word2vector = pickle.load(filer)
        else:
            self.word2vector = {}
            if not hasattr(self, "word2id"):
                self.load_global_vocab()
            with open(self.word2vector_path, 'r') as f:
                for i, line in enumerate(f):
                    if i == 0: continue
                    word = line.split(' ')[0]
                    if word not in self.word2id: continue
                    line = ' '.join(line.split(' ')[1:]).strip()
                    vector = np.fromstring(line, dtype=float, sep=' ')
                    self.word2vector[word] = vector
            with open("./preproc_data/word2vector.pkl", "wb") as filer:
                pickle.dump(self.word2vector, filer)
        print("Load word2vector has done.")
   
    "topic_model"
    ##################################################
    def load_topic_model(self):
        if not hasattr(self, "word2id"):
            self.load_globel_vocab()
        self.vectorizer = CountVectorizer(vocabulary=self.word2id, tokenizer=lambda x: x, preprocessor=lambda x: x)
        file_path = "./preproc_data/topic_model.pkl"
        if os.path.exists(file_path):
            self.topic_model = LdaModel.load(file_path)
        else:
            texts = []
            if not hasattr(self, "domain2data"):
                self.load_domain2data()
            for domain in self.domain2data:
                texts.extend(self.domain2data[domain]["labeled"])
                texts.extend(self.domain2data[domain]["unlabeled"])
            corpus = self.vectorizer.fit_transform(texts)
            corpus = Sparse2Corpus(corpus, documents_columns=False)
            self.topic_model = LdaMulticore(
                corpus=corpus,
                num_topics=self.num_topics,
                id2word=self.id2word,
                iterations=self.num_topic_iterations,
                passes=self.num_topic_passes
            )
            self.topic_model.save(file_path)
           
    "tfidf"
    ##################################################
    def load_domain2tfidf(self):
        if os.path.exists("./preproc_data/domain2tfidf.pkl"):
            with open("./preproc_data/domain2tfidf.pkl", "rb") as filer:
                self.domain2tfidf = pickle.load(filer)
        else:
            if not hasattr(self, "domain2data"):
                self.load_domain2data()
            if not hasattr(self, "word2id"):
                self.load_global_vocab()
            self.domain2tfidf = {domain: {"labeled":[], "label":[], "unlabeled": []} for domain in self.domains}
            for domain in self.domain2data:
                vectorizer = TfidfVectorizer(vocabulary=self.word2id, tokenizer=lambda x: x, preprocessor=lambda x: x)
                vectorizer.fit(self.domain2data[domain]["labeled"] + self.domain2data[domain]["unlabeled"])
                for key in self.domain2tfidf[domain]:
                    self.domain2tfidf[domain][key] = self.domain2data[domain][key] if key == "label" \
                    else vectorizer.transform(self.domain2data[domain][key])
            with open("./preproc_data/domain2tfidf.pkl", "wb") as filew:
                pickle.dump(self.domain2tfidf, filew)
        print("Load domain2tfidf has done.")
   
    "texts"
    ##################################################
    def get_texts(self, domains, unlabeled=True):
        texts = []
        if not hasattr(self, "domain2data"):
            self.load_domain2data()
        for domain in domains:
            texts.extend(self.domain2data[domain]["labeled"])
            if unlabeled:
                texts.extend(self.domain2data[domain]["unlabeled"])
        return texts
   
    "distribution"
    ##################################################
    def get_texts_term_distribution(self, texts):
        if not hasattr(self, "word2id"):
            self.load_global_vocab()
        term_distribution = np.zeros(len(self.word2id))
        for text in texts:
            for word in text:
                if word in self.word2id:
                    term_distribution[self.word2id[word]] += 1
        term_distribution /= np.sum(term_distribution)
        if np.isnan(np.sum(term_distribution)):
            term_distribution = np.zeros(self.vocab_size)
        return term_distribution
   
    "topic"
    ##################################################
    def get_texts_topic_distribution(self, texts):
        if not hasattr(self, "vectorizer"):
            self.load_topic_model()
        vectorized_corpus = self.vectorizer.transform(texts)
        gensim_corpus = Sparse2Corpus(vectorized_corpus, documents_columns=False)
        topic_representations = []
        for doc in gensim_corpus:
            topic_representations.append(
                [topic_prob for (_, topic_prob) in self.topic_model.get_document_topics(doc, minimum_probability=0.)]
            )
        return np.array(topic_representations)
   
    "word2vec"
    ##################################################
    def get_texts_word2vec_distribution(self, texts):
        if not hasattr(self, "word2vector"):
            self.load_word2vector()
        word_embeds, t = [], 10e-5
        texts_term_distribution_weights = self.get_texts_term_distribution(texts)
        for text in texts:
            word_count, doc_vector = 0, np.zeros(len(list(self.word2vector.values())[0]))
            for word in text:
                if word not in self.word2vector: continue
                doc_vector += np.sqrt(t / (texts_term_distribution_weights[self.word2id[word]])) * self.word2vector[word]
                word_count += 1
            doc_vector = doc_vector if word_count == 0 else doc_vector/word_count
            word_embeds.append(doc_vector)
        return np.array(word_embeds)
   
    "model feature"
    ##################################################
    def get_model_feature(self, domains):
        if not hasattr(self, "domain2tfidf"):
            self.load_domain2tfidf()
        X, Y, D = [], [], []
        for domain in domains:
            X.extend(self.domain2tfidf[domain]["labeled"])
            Y.extend(self.domain2tfidf[domain]["label"])
            D.extend([domain] * self.domain2tfidf[domain]["labeled"].shape[0])
        X = scipy.sparse.vstack(X).toarray()
        Y = np.asarray(Y)
        return X, Y, D
   
    "metric feature"
    ###################################################
    def get_metric_feature(self, target_domain, metric_dict):
        metric_names = [(metric_type, metric_name) for metric_type in metric_dict for metric_name in metric_dict[metric_type]]
        feature = []
        if "term" in metric_dict:
            term_feature = self.get_term_feature(target_domain, metric_names)
            feature.append(term_feature)
        if "topic" in metric_dict:
            topic_feature = self.get_topic_feature(target_domain, metric_names)
            feature.append(topic_feature)
        if "word2vec" in metric_dict:
            word2vec_feature = self.get_word2vec_feature(target_domain, metric_names)
            feature.append(word2vec_feature)
        if "diversity" in metric_dict:
            diversity_feature = self.get_diversity_feature(target_domain, metric_names)
            feature.append(diversity_feature)
        feature = np.concatenate(feature, axis=1)
        return feature
   
    def get_term_feature(self, target_domain, metric_names):
        filepath = "./preproc_data/term_feature_{}.pkl".format(target_domain)
        if os.path.exists(filepath):
            with open(filepath, "rb") as filer:
                term_feature = pickle.load(filer)
        else:
            source_texts = self.get_texts([domain for domain in self.domains if domain != target_domain], unlabeled=False)
            target_texts = self.get_texts([target_domain], unlabeled=False)
            texts_distribution = [self.get_texts_term_distribution([text]) for text in source_texts]
            domain_distribution = self.get_texts_term_distribution(target_texts)
            rvalues = []
            for text_distribution in texts_distribution:
                values = []
                for metric_name in metric_names:
                    metric_type, metric_func = metric_name
                    if metric_type != "term": continue
                    if metric_func in ['jensen_shannon', 'renyi', 'cosine', 'euclidean', 'variational', 'bhattacharyya']:
                        values.append(getattr(Metric, metric_func)(text_distribution, domain_distribution))
                rvalues.append(values)
            term_feature = np.asarray(rvalues)
            with open(filepath, "wb") as filew:
                pickle.dump(term_feature, filew)
        return term_feature
   
    def get_topic_feature(self, target_domain, metric_names):
        filepath = "./preproc_data/topic_feature_{}.pkl".format(target_domain)
        if os.path.exists(filepath):
            with open(filepath, "rb") as filer:
                topic_feature = pickle.load(filer)
        else:
            source_texts = self.get_texts([domain for domain in self.domains if domain != target_domain], unlabeled=False)
            target_texts = self.get_texts([target_domain], unlabeled=False)
            texts_distribution = self.get_texts_topic_distribution(source_texts)
            domain_distribution = np.mean(self.get_texts_topic_distribution(target_texts), axis=0)
            rvalues = []
            for text_distribution in texts_distribution:
                values = []
                for metric_name in metric_names:
                    metric_type, metric_func = metric_name
                    if metric_type != "topic": continue
                    if metric_func in ['jensen_shannon', 'renyi', 'cosine', 'euclidean', 'variational', 'bhattacharyya']:
                        values.append(getattr(Metric, metric_func)(text_distribution, domain_distribution))
                rvalues.append(values)
            topic_feature = np.asarray(rvalues)
            with open(filepath, "wb") as filew:
                pickle.dump(topic_feature, filew)
        return topic_feature
   
    def get_word2vec_feature(self, target_domain, metric_names):
        filepath = "./preproc_data/word2vec_feature_{}.pkl".format(target_domain)
        if os.path.exists(filepath):
            with open(filepath, "rb") as filer:
                word2vec_feature = pickle.load(filer)
        else:
            source_texts = self.get_texts([domain for domain in self.domains if domain != target_domain], unlabeled=False)
            target_texts = self.get_texts([target_domain], unlabeled=False)
            texts_distribution = self.get_texts_word2vec_distribution(source_texts)
            domain_distribution = np.mean(self.get_texts_word2vec_distribution(target_texts), axis=0)
            rvalues = []
            for text_distribution in texts_distribution:
                values = []
                for metric_name in metric_names:
                    metric_type, metric_func = metric_name
                    if metric_type != "word2vec": continue
                    if metric_func in ['cosine', 'euclidean', 'variational']:
                        values.append(getattr(Metric, metric_func)(text_distribution, domain_distribution))
                rvalues.append(values)
            word2vec_feature = np.asarray(rvalues)
            with open(filepath, "wb") as filew:
                pickle.dump(word2vec_feature, filew)
        return word2vec_feature
   
    def get_diversity_feature(self, target_domain, metric_names):
        filepath = "./preproc_data/diversity_feature_{}.pkl".format(target_domain)
        if os.path.exists(filepath):
            with open(filepath, "rb") as filer:
                diversity_feature = pickle.load(filer)
        else:
            if not hasattr(self, "word2vector"):
                self.load_word2vector()
            source_texts = self.get_texts([domain for domain in self.domains if domain != target_domain], unlabeled=False)
            term_distribution = self.get_texts_term_distribution(source_texts)
            rvalues = []
            for source_text in source_texts:
                p_words, p_word_vector_pairs = [], []
                for word in set(source_text):
                    if word in self.word2id:
                        p_words.append(term_distribution[self.word2id[word]])
                        if word in self.word2vector:
                            p_word_vector_pairs.append((term_distribution[self.word2id[word]], self.word2vector[word]))
                    else:
                        p_words.append(0.0)
                values = []
                for metric_name in metric_names:
                    metric_type, metric_func = metric_name
                    if metric_type != "diversity": continue
                    if metric_func in ['num_word_types', 'type_token_ratio']:
                        values.append(getattr(Metric, metric_func)(source_text))
                    elif metric_func in ['entropy', 'simpsons_index', 'renyi_entropy']:
                        values.append(getattr(Metric, metric_func)(p_words))
                    elif metric_func in ['quadratic_entropy']:
                        values.append(getattr(Metric, metric_func)(p_word_vector_pairs))
                    else:
                        raise AttributeError()
                rvalues.append(values)
            diversity_feature = np.asarray(rvalues)
            with open(filepath, "wb") as filew:
                pickle.dump(diversity_feature, filew)
        return diversity_feature

class DataSelector():
    
    def __init__(self, dataset, select_num=1600, feature=None, weights=None, isbalance=True):
        self.dataset = dataset
        self.select_num = select_num
        self.weights = weights
        self.feature = feature
        self.isbalance = isbalance
    
    def random(self, target_domain):
        X, Y, _ = self.dataset.get_model_feature([domain for domain in self.dataset.domains if domain != target_domain])
        select_X, _, select_Y, _ = train_test_split(X, Y, train_size=self.select_num, stratify=Y)
        return select_X, select_Y
    
    def all_source_data(self, target_domain):
        select_X, select_Y, _ = self.dataset.get_model_feature([domain for domain in self.dataset.domains if domain != target_domain])
        return select_X, select_Y
    
    def most_similar_domain(self, target_domain):
        max_sim, most_sim_domain = 0, None
        target_texts = self.dataset.get_texts([target_domain], unlabeled=False)
        trg_term_dist = self.dataset.get_texts_term_distribution(target_texts)
        for domain in self.dataset.domains:
            if domain == target_domain:continue
            texts = self.dataset.get_texts([domain], unlabeled=False)
            src_term_dist = self.dataset.get_texts_term_distribution(texts)
            sim = Metric.jensen_shannon(src_term_dist, trg_term_dist)
            if sim > max_sim:
                max_sim, most_sim_domain = sim, domain
        select_X, select_Y, _ = self.dataset.get_model_feature([most_sim_domain])
        return select_X, select_Y
    
    def most_similar_examples(self, target_domain):
        assert (self.feature is not None) and (self.weights is not None), "Weights must be assigned!"
        source_X, source_Y, _ = self.dataset.get_model_feature([domain for domain in self.dataset.domains if domain != target_domain])
        scores = self.feature.dot(np.transpose(self.weights))
        sorted_idx, _ = zip(*sorted(zip(range(len(scores)), scores), key=lambda x:x[1], reverse=True))
        selected_idx = []
        if self.isbalance:
            if isinstance(source_Y, (list, tuple)):
                label_sets = list(set(source_Y))
            elif isinstance(source_Y, np.ndarray):
                label_sets = list(set(source_Y.tolist()))
            else:
                raise NotImplementedError()
            for label in label_sets:
                selected_idx.extend([idx for idx in sorted_idx if source_Y[idx] == label][:int(self.select_num/len(label_sets))])      
        else:
            selected_idx = sorted_idx[:self.select_num]
        select_X, select_Y = source_X[selected_idx], source_Y[selected_idx]
        return select_X, select_Y
        

class BayesOptimizer():
   
    def __init__(self, model, lower, upper, num_iterations, select_num, isbalance=True):
        self.model = model
        self.lower = lower
        self.upper = upper
        self.num_iterations = num_iterations
        self.select_num = select_num
        self.isbalance = isbalance
   
    def _select(self, feature, weights, labels=None):
        scores = feature.dot(np.transpose(weights))
        sorted_idx, _ = zip(*sorted(zip(range(len(scores)), scores), key=lambda x:x[1], reverse=True))
        selected_idx = []
        if self.isbalance:
            assert labels is not None, "When use balance mode, labels is required!"
            if isinstance(labels, (list, tuple)):
                label_sets = list(set(labels))
            elif isinstance(labels, np.ndarray):
                label_sets = list(set(labels.tolist()))
            else:
                raise NotImplementedError()
            for label in label_sets:
                selected_idx.extend([idx for idx in sorted_idx if labels[idx] == label][:int(self.select_num/len(label_sets))])      
        else:
            selected_idx = sorted_idx[:self.select_num]
        return selected_idx
    
    def get_weights(self, feature, X, Y, val_X, val_Y):
       
        def objective_function(weights):
            selected_idx = self._select(feature, weights, Y)
            select_X, select_Y = X[selected_idx], Y[selected_idx]
            self.model.train(select_X, select_Y)
            error = self.model.loss(val_X, val_Y)
            return error
       
        return bayesian_optimization(
            objective_function=objective_function,
            lower=self.lower,
            upper=self.upper,
            num_iterations=self.num_iterations
        )['x_opt']

class Model():
   
    def __init__(self, name="svm.SVC"):
        if name == "svm.SVC":
            self.clf = svm.SVC()
        else:
            raise NotImplementedError()
       
    def train(self, X, Y):
        self.clf.fit(X, Y)
       
    def infer(self, X, Y):
        pY = self.clf.predict(X)
        acc = accuracy_score(Y, pY)
        return pY, acc
   
    def loss(self, X, Y):
        _, acc = self.infer(X, Y)
        error = 1 - float(acc)
        return error
    

class Experiment():
    
    def __init__(self, dataset, selector, valid_size=100):
        self.dataset = dataset
        self.selector = selector
        self.valid_size = valid_size
        
    def resample_val_test(self, target_domain):
        target_X, target_Y, _ = self.dataset.get_model_feature([target_domain])
        test_target_X, val_target_X, test_target_Y, val_target_Y = train_test_split(
            target_X, target_Y, 
            test_size=self.valid_size, 
            stratify=target_Y
        )
        self.test_data = (test_target_X, test_target_Y)
        self.val_data = (val_target_X, val_target_Y)
    
    def __call__(self, target_domain, baseline, resample=False):
        #######################################################
        if not hasattr(self, "test_data") or resample:
            target_X, target_Y, _ = self.dataset.get_model_feature([target_domain])
            test_target_X, val_target_X, test_target_Y, val_target_Y = train_test_split(
                target_X, target_Y, 
                test_size=self.valid_size, 
                stratify=target_Y
            )
            self.test_data = (test_target_X, test_target_Y)
            self.val_data = (val_target_X, val_target_Y)
        else:
            pass
        #######################################################
        train_data = getattr(self.selector, baseline)(target_domain)
        train_num_samples = train_data[0].shape[0]
        model = Model()
        start_time = time.time()
        model.train(*train_data)
        cost_time = time.time() - start_time
        print("Train random! cost: {} on {} samples".format(cost_time, train_num_samples))
        val_num_samples = self.val_data[0].shape[0]
        _, val_acc = model.infer(*self.val_data)
        val_loss = model.loss(*self.val_data)
        print("Valid random! loss: {}, acc: {} on {} samples".format(val_loss, val_acc, val_num_samples))
        test_num_samples = self.test_data[0].shape[0]
        _, test_acc = model.infer(*self.test_data)
        test_loss = model.loss(*self.test_data)
        print("Test random! loss: {}, acc: {} on {} samples".format(test_loss, test_acc, test_num_samples))
        
if __name__ == "__main__":
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument('--dirpath', type=str, default="/home/yangming/Datasets/amazon-reviews/processed_acl/")
    parser.add_argument('--word2vector_path', type=str, default="/home/yangming/Datasets/Glove/glove.42B.300d.txt")
    parser.add_argument('--metric_dict_path', type=str, default="./metric_dict.yaml")
    parser.add_argument('--reproc', type=ast.literal_eval, default=False)
    parser.add_argument('--isbalance', type=ast.literal_eval, default=True)
    parser.add_argument('--target_domain', type=str, default="kitchen")
    parser.add_argument('--max_vocab_size', type=int, default=10000)
    parser.add_argument('--num_topics', type=int, default=50)
    parser.add_argument('--num_topic_iterations', type=int, default=2000)
    parser.add_argument('--num_topic_passes', type=int, default=10)
    parser.add_argument('--select_num', type=int, default=1600)
    parser.add_argument('--valid_size', type=int, default=100)
    parser.add_argument('--num_iterations', type=int, default=100)
    parser.add_argument('--cv_fold', type=int, default=10)
    parser.add_argument('--select_methods', nargs='+', type=str, default="most_similar_examples random all_source_data most_similar_domain")
    args = parser.parse_args()
    #####################################################################
    dataset = AmazonReviewDataset(
        dirpath=args.dirpath, 
        word2vector_path=args.word2vector_path, 
        max_vocab_size=args.max_vocab_size, 
        num_topics=args.num_topics, 
        num_topic_iterations=args.num_topic_iterations, 
        num_topic_passes=args.num_topic_passes, 
        reproc=args.reproc
    )
    #####################################################################
    feature, weights = None, None
    if "most_similar_examples" in args.select_methods:
        proxy_model = Model()
        with open(args.metric_dict_path, "r") as filer:
            metric_dict = yaml.load(filer)
        start_time = time.time()
        feature = dataset.get_metric_feature(args.target_domain, metric_dict)
        print("build metric feature for target domain: {} cost: {}".format(args.target_domain, time.time() - start_time))
        n_dim = feature.shape[-1]
        lower = np.array(n_dim * [-1])
        upper = np.array(n_dim * [1])
        opt = BayesOptimizer(
            model=proxy_model, 
            lower=lower, 
            upper=upper, 
            num_iterations=args.num_iterations,
            select_num=args.select_num,
            isbalance=args.isbalance
        )
        X, Y, _ = dataset.get_model_feature([domain for domain in dataset.domains if domain !=args.target_domain])
        target_X, target_Y, _ = dataset.get_model_feature([args.target_domain])
        _, val_X, _, val_Y = train_test_split(target_X, target_Y, test_size=args.valid_size, stratify=target_Y)
        start_time = time.time()
        weights = opt.get_weights(feature, X, Y, val_X, val_Y)
    #####################################################################
    selector = DataSelector(dataset, select_num=args.select_num, feature=feature, weights=weights)
    #####################################################################
    exp = Experiment(dataset, selector, valid_size=args.valid_size)
    for i in range(args.cv_fold):
        print("############################")
        print("cv-{}".format(i))
        exp.resample_val_test(args.target_domain)
        for baseline in args.select_methods.split():
            print("------------------------")
            print(baseline)
            exp(args.target_domain, baseline)
            print("------------------------")
        print("############################")