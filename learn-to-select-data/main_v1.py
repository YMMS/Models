# -*- coding:utf-8 -*-

import os
import logging
import pickle
import itertools
import operator
import scipy
import gensim
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from robo.fmin import bayesian_optimization
from collections import Counter

####################################################################
def read_data(dir_path):
    domains = os.listdir(dir_path)
    domain2data = {domain: [[], [], None] for domain in domains}
    for domain in domains:
        for split in ['positive', 'negative', 'unlabeled']:
            file_path = os.path.join(dir_path, domain, '%s.review' % split)
            reviews = []
            with open(file_path, "r") as f:
                for line in f:
                    features = line.split(' ')[:-1]
                    review = []
                    for feature in features:
                        ngram, count = feature.split(':')
                        for _ in range(int(count)):
                            review.append(ngram)
                    reviews.append(review)
            if split == 'unlabeled':
                domain2data[domain][2] = reviews
            else:
                domain2data[domain][0] += reviews
                domain2data[domain][1] += [1 if split == "positive" else 0] * len(reviews)
        domain2data[domain][1] = np.array(domain2data[domain][1])
    return domain2data

####################################################################
def train_and_evaluate(train_data, train_labels, val_data, val_labels, test_data=None, test_labels=None):
    clf = svm.SVC()
    clf.fit(train_data, train_labels)
    val_predictions = clf.predict(val_data)
    val_accuracy = accuracy_score(val_labels, val_predictions)
    test_accuracy = None
    if test_data is not None and test_labels is not None:
        test_predictions = clf.predict(test_data)
        test_accuracy = accuracy_score(test_labels, test_predictions)
    return val_accuracy, test_accuracy

"For baseline: most_similar_domain"
####################################################################
def get_most_similar_domain(trg_domain, domain2term_dists, similarity_name='jensen-shannon'):
    highest_sim_score, most_similar_domain = 0, None
    trg_term_dist = domain2term_dists[trg_domain]
    for domain, src_term_dist in domain2term_dists.items():
        if domain == trg_domain:
            continue
        sim_score = similarity_name2value(similarity_name, src_term_dist, trg_term_dist)
        if sim_score > highest_sim_score:
            highest_sim_score, most_similar_domain = sim_score, domain
    return most_similar_domain

"loss"
####################################################################
def objective_function(feature_weights):
    train_subset, train_labels_subset = get_data_subsets(feature_values, feature_weights, X_train, y_train, hparam.num_train_examples)
    val_accuracy, _ = train_and_evaluate(train_subset, train_labels_subset, X_val, y_val, X_test, y_test)
    error = 1 - float(val_accuracy)
    return error

"similarity feature names"
####################################################################
def get_feature_names(feature_set_names):
    features = []
    if 'similarity' in feature_set_names:
        features += ['jensen-shannon', 'renyi', 'cosine', 'euclidean', 'variational', 'bhattacharyya']
    if 'topic_similarity' in feature_set_names:
        features += ['topic_%s' % s for s in ['jensen-shannon', 'renyi', 'cosine', 'euclidean', 'variational', 'bhattacharyya']]
    if 'word_embedding_similarity' in feature_set_names:
        features += ['word_embedding_%s' % s for s in ['cosine', 'euclidean', 'variational']]
    if 'diversity' in feature_set_names:
        features += ['num_word_types', 'type_token_ratio', 'entropy', 'simpsons_index', 'quadratic_entropy', 'renyi_entropy']
    return features

"vocab"
####################################################################
class Vocab:

    def __init__(self, max_vocab_size, vocab_path):
        self.max_vocab_size = max_vocab_size
        self.vocab_path = vocab_path
        self.size = 0
        self.word2id = {}
        self.id2word = {}

    def load(self):
        with open(self.vocab_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= self.max_vocab_size: break
                word, idx = line.split('\t')
                self.word2id[word] = int(idx.strip())
        self.size = len(self.word2id)
        self.id2word = {index: word for word, index in self.word2id.items()}

    def create(self, texts, lowercase=True):
        if lowercase:
            texts = [[word.lower() for word in text] for text in texts]

        word_counts = Counter(itertools.chain(*texts))
        most_common = word_counts.most_common(n=self.max_vocab_size)
        self.word2id = {word: index for index, (word, _) in enumerate(most_common)}
        self.id2word = {index: word for word, index in self.word2id.items()}
        with open(self.vocab_path, 'w') as f:
            for word, index in sorted(self.word2id.items(), key=operator.itemgetter(1)):
                f.write('%s\t%d\n' % (word, index))
        self.size = len(self.word2id)

""
####################################################################
def get_all_docs(domain_data_pairs, unlabeled=True):
    docs, labels, domains = [], [], []
    for domain, (labeled_docs, doc_labels, unlabeled_docs) in domain_data_pairs:
        length_of_docs = 0
        if not scipy.sparse.issparse(labeled_docs):
            docs += labeled_docs
            length_of_docs += len(labeled_docs)
            if unlabeled:
                docs += unlabeled_docs
                length_of_docs += len(labeled_docs)
        else:
            docs.append(labeled_docs)
            length_of_docs += labeled_docs.shape[0]
            if unlabeled and unlabeled_docs is not None:
                docs.append(unlabeled_docs)
                length_of_docs += unlabeled_docs.shape[0]
        labels.append(doc_labels)
        domains += [domain] * length_of_docs
    if scipy.sparse.issparse(labeled_docs):
        docs = scipy.sparse.vstack(docs)
    return docs, np.hstack(labels), domains

"word embedding"
####################################################################
def load_word_vectors(file, vocab_word_vec_file, word2id):
    word2vector = {}
    if os.path.exists(vocab_word_vec_file):
        with open(vocab_word_vec_file, 'r') as f:
            for line in f:
                word = line.split(' ')[0]
                line = ' '.join(line.split(' ')[1:]).strip()
                vector = np.fromstring(line, dtype=float, sep=' ')
                word2vector[word] = vector
        return word2vector
    with open(file, 'r') as f:
        for i, line in enumerate(f):
            if i == 0: continue
            word = line.split(' ')[0]
            if word not in word2id: continue
            line = ' '.join(line.split(' ')[1:]).strip()
            vector = np.fromstring(line, dtype=float, sep=' ')
            word2vector[word] = vector

    with open(vocab_word_vec_file, 'w') as f:
        for word, vector in word2vector.items():
            f.write('%s %s\n' % (word, ' '.join([str(c) for c in vector])))
    return word2vector

"global vocab; local domain2data"
####################################################################
def get_tfidf_data(domain2data, vocab):
    domain2tfidf_data = {}
    for domain, (labeled_examples, labels, unlabeled_examples) in domain2data.items():
        vectorizer = TfidfVectorizer(vocabulary=vocab.word2id, tokenizer=lambda x: x, preprocessor=lambda x: x)
        vectorizer.fit(labeled_examples + unlabeled_examples)
        tfidf_labeled_examples = vectorizer.transform(labeled_examples)
        unlabeled_examples = unlabeled_examples[:100000]
        tfidf_unlabeled_examples = vectorizer.transform(unlabeled_examples) if len(unlabeled_examples) != 0 else None
        domain2tfidf_data[domain] = [tfidf_labeled_examples, labels, tfidf_unlabeled_examples]
    return domain2tfidf_data

""
####################################################################
def get_term_dist(docs, vocab, lowercase=True):
    term_dist = np.zeros(vocab.size)
    for doc in docs:
        for word in doc:
            if lowercase:
                word = word.lower()
            if word in vocab.word2id:
                term_dist[vocab.word2id[word]] += 1
    term_dist /= np.sum(term_dist)
    if np.isnan(np.sum(term_dist)):
        term_dist = np.zeros(vocab.size)
    return term_dist

""
####################################################################
def get_domain_term_dists(term_dist_path, domain2data, vocab, lowercase=True):
    domain2term_dist = {}
    if os.path.exists(term_dist_path):
        with open(term_dist_path, 'r') as f:
            for line in f:
                domain, term_dist = line.strip().split('\t')
                term_dist = np.fromstring(term_dist, count=vocab.size, sep=' ')
                domain2term_dist[domain] = term_dist
        return domain2term_dist
    for domain, (examples, _, unlabeled_examples) in domain2data.items():
        domain2term_dist[domain] = get_term_dist(examples + unlabeled_examples, vocab, lowercase)
    with open(term_dist_path, 'w') as f:
        for domain, term_dist in domain2term_dist.items():
            f.write('%s\t%s\n' % (domain, ' '.join([str(c) for c in term_dist])))
    return domain2term_dist

""
####################################################################
def get_topic_distributions(examples, vectorizer, lda_model):
    vectorized_corpus = vectorizer.transform(examples)
    gensim_corpus = gensim.matutils.Sparse2Corpus(vectorized_corpus, documents_columns=False)
    topic_representations = []
    for doc in gensim_corpus:
        topic_representations.append([topic_prob for (_, topic_prob) in lda_model.get_document_topics(doc, minimum_probability=0.)])
    return np.array(topic_representations)

""
####################################################################
def weighted_sum_of_embeddings(docs, word2id, word2vector, term_dist):
    t = 10e-5
    word_embed_representations = []
    for doc in docs:
        doc_vector = np.zeros(len(list(word2vector.values())[0]))
        word_vector_count = 0
        for word in doc:
            if word in word2vector:
                vector = word2vector[word]
                doc_vector += np.sqrt(t / (term_dist[word2id[word]])) * vector
                word_vector_count += 1
        if word_vector_count == 0:
            word_vector_count = 1
        doc_vector /= word_vector_count
        word_embed_representations.append(doc_vector)
    return np.array(word_embed_representations)

""
####################################################################
def train_topic_model(examples, vocab, num_topics=50, num_iterations=2000, num_passes=10):
    vectorizer = CountVectorizer(vocabulary=vocab.word2id, tokenizer=lambda x: x, preprocessor=lambda x: x)
    lda_corpus = vectorizer.fit_transform(examples)
    lda_corpus = gensim.matutils.Sparse2Corpus(lda_corpus, documents_columns=False)
    lda_model = gensim.models.LdaMulticore(lda_corpus, num_topics=num_topics, id2word=vocab.id2word, iterations=num_iterations, passes=num_passes)
    return vectorizer, lda_model
####################################################################
def get_data_subsets(feature_vals, feature_weights, train_data, train_labels, num_train_examples):
    scores = feature_vals.dot(np.transpose(feature_weights))
    sorted_index_score_pairs = sorted(zip(range(len(scores)), scores), key=operator.itemgetter(1), reverse=True)
    top_indices, _ = zip(*sorted_index_score_pairs)
    top_pos_indices = [idx for idx in top_indices if train_labels[idx] == 1][:int(num_train_examples/2)]
    top_neg_indices = [idx for idx in top_indices if train_labels[idx] == 0][:int(num_train_examples/2)]
    top_indices = top_pos_indices + top_neg_indices
    if isinstance(train_data, list):
        return [train_data[idx] for idx in top_indices], train_labels[top_indices]
    return train_data[top_indices], train_labels[top_indices]
####################################################################
def log_to_file(log_file, run_dict, trg_domain, args):
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))
    with open(log_file, 'a') as f:
        for method, scores in run_dict.items():
            best_feature_weights = ''
            if len(scores) == 0: continue
            if method.startswith('bayes-opt'):
                val_accuracies, test_accuracies, best_feature_weights = zip(*scores)
            else:
                val_accuracies, test_accuracies = zip(*scores)
            mean_val, std_val = np.mean(val_accuracies), np.std(val_accuracies)
            mean_test, std_test = np.mean(test_accuracies), np.std(test_accuracies)
            f.write('%s\t%s\t%s\t%.4f (+-%.4f)\t%.4f (+-%.4f)\t[%s]\t[%s]\t%s\t'
                    '%s\n'
                    % (trg_domain, method, ' '.join(args.feature_sets),
                       mean_val, std_val, mean_test, std_test,
                       ', '.join(['%.4f' % v for v in val_accuracies]),
                       ', '.join(['%.4f' % t for t in test_accuracies]),
                       str(list(best_feature_weights)),
                       ' '.join(['%s=%s' % (arg, str(getattr(args, arg)))
                                 for arg in vars(args)])))
####################################################################
def jensen_shannon_divergence(repr1, repr2):
    avg_repr = 0.5 * (repr1 + repr2)
    sim = 1 - 0.5 * (scipy.stats.entropy(repr1, avg_repr) + scipy.stats.entropy(repr2, avg_repr))
    if np.isinf(sim): return 0
    return sim
####################################################################
def renyi_divergence(repr1, repr2, alpha=0.99):
    log_sum = np.sum([np.power(p, alpha) / np.power(q, alpha-1) for (p, q) in zip(repr1, repr2)])
    sim = 1 / (alpha - 1) * np.log(log_sum)
    if np.isinf(sim): return 0
    return sim
####################################################################
def cosine_similarity(repr1, repr2):
    if repr1 is None or repr2 is None: return 0
    sim = 1 - scipy.spatial.distance.cosine(repr1, repr2)
    if np.isnan(sim): return 0
    return sim
####################################################################
def euclidean_distance(repr1, repr2):
    sim = np.sqrt(np.sum([np.power(p-q, 2) for (p, q) in zip(repr1, repr2)]))
    return sim
####################################################################
def variational_distance(repr1, repr2):
    sim = np.sum([np.abs(p-q) for (p, q) in zip(repr1, repr2)])
    return sim
####################################################################
def kl_divergence(repr1, repr2):
    sim = scipy.stats.entropy(repr1, repr2)
    return sim
####################################################################
def bhattacharyya_distance(repr1, repr2):
    sim = - np.log(np.sum([np.sqrt(p*q) for (p, q) in zip(repr1, repr2)]))
    if np.isinf(sim): return 0
    return sim
####################################################################
def similarity_name2value(s_name, repr1, repr2):
    if s_name == 'jensen-shannon':
        return jensen_shannon_divergence(repr1, repr2)
    if s_name == 'renyi':
        return renyi_divergence(repr1, repr2)
    if s_name == 'cos' or s_name == 'cosine':
        return cosine_similarity(repr1, repr2)
    if s_name == 'euclidean':
        return euclidean_distance(repr1, repr2)
    if s_name == 'variational':
        return variational_distance(repr1, repr2)
    if s_name == 'kl':
        return kl_divergence(repr1, repr2)
    if s_name == 'bhattacharyya':
        return bhattacharyya_distance(repr1, repr2)
    raise ValueError('%s is not a valid feature name.' % s_name)
####################################################################
def get_feature_representations(feature_names, examples, trg_examples, vocab, word2vec=None, topic_vectorizer=None, lda_model=None, lowercase=True):
    features = np.zeros((len(examples), len(feature_names)))
    if lowercase:
        examples = [[word.lower() for word in example] for example in examples]
        trg_examples = [[word.lower() for word in trg_example] for trg_example in trg_examples]
    train_term_dist = get_term_dist(examples, vocab)
    term_dists = np.array([get_term_dist([example], vocab) for example in examples])
    trg_term_dist = get_term_dist(trg_examples, vocab)
    topic_dists, trg_topic_dist = None, None
    if any(f_name.startswith('topic') for f_name in feature_names):
        topic_dists = get_topic_distributions(examples, topic_vectorizer, lda_model)
        trg_topic_dist = np.mean(get_topic_distributions(trg_examples, topic_vectorizer, lda_model), axis=0)
    word_reprs, trg_word_repr = None, None
    if any(f_name.startswith('word_embedding') for f_name in feature_names):
        word_reprs = weighted_sum_of_embeddings(examples, vocab.word2id, word2vec, train_term_dist)
        trg_word_repr = np.mean(weighted_sum_of_embeddings(trg_examples, vocab.word2id, word2vec, trg_term_dist), axis=0)
    for i in range(len(examples)):
        for j, f_name in enumerate(feature_names):
            if f_name.startswith('topic'):
                f = similarity_name2value(f_name.split('_')[1], topic_dists[i], trg_topic_dist)
            elif f_name.startswith('word_embedding'):
                f = similarity_name2value(f_name.split('_')[2], word_reprs[i], trg_word_repr)
            elif f_name in ['jensen-shannon', 'renyi', 'cosine', 'euclidean', 'variational', 'bhattacharyya']:
                f = similarity_name2value(f_name, term_dists[i], trg_term_dist)
            elif f_name in ['num_word_types', 'type_token_ratio', 'entropy', 'simpsons_index', 'quadratic_entropy', 'renyi_entropy']:
                f = diversity_feature_name2value(f_name, examples[i], train_term_dist, vocab.word2id, word2vec)
            else:
                raise ValueError('%s is not a valid feature name.' % f_name)
            features[i, j] = f
    return features
####################################################################
def number_of_word_types(example):
    return len(set(example))


def type_token_ratio(example):
    return number_of_word_types(example) / len(example)

def entropy(example, train_term_dist, word2id):
    summed = 0
    for word in set(example):
        if word in word2id:
            p_word = train_term_dist[word2id[word]]
            summed += p_word * np.log(p_word)
    return - summed


def simpsons_index(example, train_term_dist, word2id):
    score = np.sum([np.power(train_term_dist[word2id[word]], 2) if word in word2id else 0 for word in set(example)])
    return score


def quadratic_entropy(example, train_term_dist, word2id, word2vec):
    """Calculates Quadratic Entropy."""
    assert word2vec is not None, ('Error: Word vector representations have to be available for quadratic entropy.')
    summed = 0
    for word_1 in set(example):
        if word_1 not in word2id or word_1 not in word2vec:
            continue  # continue as the product will be 0
        for word_2 in set(example):
            if word_2 not in word2id or word_2 not in word2vec:
                continue  # continue as the product will be 0
            p_1 = train_term_dist[word2id[word_1]]
            p_2 = train_term_dist[word2id[word_2]]
            vec_1 = word2vec[word_1]
            vec_2 = word2vec[word_2]
            sim = cosine_similarity(vec_1, vec_2)
            summed += sim * p_1 * p_2
    return summed


def renyi_entropy(example, domain_term_dist, word2id):
    alpha = 0.99
    summed = np.sum([np.power(domain_term_dist[word2id[word]], alpha) if word in word2id else 0 for word in set(example)])
    if summed == 0:
        summed = 0.0001
    score = 1 / (1 - alpha) * np.log(summed)
    return score
####################################################################
def diversity_feature_name2value(f_name, example, train_term_dist, word2id, word2vec):
    if f_name == 'num_word_types':
        return number_of_word_types(example)
    if f_name == 'type_token_ratio':
        return type_token_ratio(example)
    if f_name == 'entropy':
        return entropy(example, train_term_dist, word2id)
    if f_name == 'simpsons_index':
        return simpsons_index(example, train_term_dist, word2id)
    if f_name == 'quadratic_entropy':
        return quadratic_entropy(example, train_term_dist, word2id, word2vec)
    if f_name == 'renyi_entropy':
        return renyi_entropy(example, train_term_dist, word2id)
    raise ValueError('%s is not a valid feature name.' % f_name)
####################################################################
def read_feature_weights_file(feature_weights_path):
    print('Reading feature weights from %s...' % feature_weights_path)
    with open(feature_weights_path, 'r') as f:
        for line in f:
            feature_weights_domain, feature_set, feature_weights = line.split('~')
            feature_weights = feature_weights.strip('[]\n')
            feature_weights = feature_weights.split(', ')
            feature_weights = [float(f) for f in feature_weights]
            print('Feature weights domain: %s. Feature set: %s. Feature weights: %s' % (feature_weights_domain, feature_set, str(feature_weights)))
            yield feature_weights_domain, feature_set, feature_weights
####################################################################
def train_pretrained_weights(feature_values, X_train, y_train, train_domains, num_train_examples, X_val, y_val, X_test, y_test, trg_domain, args, feature_names):
    for feat_weights_domain, feat_weights_feats, feature_weights in read_feature_weights_file(args.feature_weights_file):
        assert len(feature_weights) == len(feature_names)
        assert set(args.feature_sets) == set(feat_weights_feats.split(' '))

        if trg_domain != feat_weights_domain: continue
        train_domain_subset, _ = get_data_subsets(feature_values, feature_weights, train_domains, y_train, args.task, num_train_examples)
        for subset_domain in set(train_domain_subset):
            print('# of %s in train data for trg domain %s: %d' % (subset_domain, trg_domain, train_domain_subset.count(subset_domain)))
            continue
        train_subset, labels_subset = get_data_subsets(feature_values, feature_weights, X_train, y_train, args.task, num_train_examples)
        val_accuracy, test_accuracy = train_and_evaluate(train_subset, labels_subset, X_val, y_val, X_test, y_test)
        dict_key = ('%s-X-domain-%s-%s' % ("bayes_opt", feat_weights_domain, feat_weights_feats))
        log_to_file(args.log_file, {dict_key: [(val_accuracy, test_accuracy, feature_weights)]}, trg_domain, args)
####################################################################
class HParam():
    
    def __init__(self):
        self.experiment_id = "20181120"
        self.data_path = "./amazon-reviews/processed_acl/"
        self.word2vec_path = "./glove.42B.300d.txt"
        self.feature_weights_file = None
        self.model_dir = "./models/{}".format(self.experiment_id)
        self.log_file = "./logs/{}".format(self.experiment_id)
        self.max_vocab_size = 10000
        self.vector_size = 300
        self.num_iterations = 100
        self.num_train_examples = 1600
        self.num_runs = 1
        self.target_domains = ['books', 'dvd', 'electronics', 'kitchen']
        self.baselines = []
        self.feature_sets = ['similarity', 'topic_similarity', 'word_embedding_similarity', 'diversity']
        self.logging = True

hparam = HParam()

if hparam.logging:
    logging.basicConfig(level=logging.INFO)

if not os.path.exists(hparam.model_dir):
    os.makedirs(hparam.model_dir)

"target domains which like to run"
task_trg_domains = hparam.target_domains
"features used as similarity metric"
feature_names = get_feature_names(hparam.feature_sets)
"preprocessed data"
preproc_data_path = os.path.join(hparam.model_dir, 'preproc_data.pkl')

"preproc_data: domain2data"
#############################################
if not os.path.exists(preproc_data_path):
    domain2data = read_data(hparam.data_path)
    with open(preproc_data_path, 'wb') as f:
        pickle.dump(domain2data, f)
else:
    with open(preproc_data_path, 'rb') as f:
        domain2data = pickle.load(f)

"vocab: all domains"
#############################################
vocab_path = os.path.join(hparam.model_dir, 'vocab.txt')
vocab = Vocab(hparam.max_vocab_size, vocab_path)
if not os.path.exists(vocab_path):
    tokenised_sentences = get_all_docs(domain2data.items(), unlabeled=True)[0]
    vocab.create(tokenised_sentences)
    del tokenised_sentences
else:
    vocab.load()

"embedding of third part"
#############################################
vocab_word2vec_file = os.path.join(hparam.model_dir, 'vocab_word2vec.txt')
word2vec = load_word_vectors(hparam.word2vec_path, vocab_word2vec_file, vocab.word2id)

"tf-idf: "
#############################################
domain2train_data = get_tfidf_data(domain2data, vocab)
term_dist_path = os.path.join(hparam.model_dir, 'term_dist.txt')
domain2term_dist = get_domain_term_dists(term_dist_path, domain2data, vocab)
for trg_domain in task_trg_domains:
    "Used for training task model: source domain; unlabeled=False"
    X_train, y_train, train_domains = get_all_docs([(k, v) for (k, v) in sorted(domain2train_data.items()) if k != trg_domain], unlabeled=False)
    "Used for metric: source domain; unlabeled=False"
    examples, y_train_check, train_domains_check = get_all_docs([(k, v) for (k, v) in sorted(domain2data.items()) if k != trg_domain], unlabeled=False)
    "Used for metric: topic model; all domain; unlabeled=True; texts"
    topic_vectorizer, lda_model = train_topic_model(get_all_docs(domain2data.items(), unlabeled=True)[0], vocab)
    "Used for metric: examples=source domain texts; domain2data[trg_domain][0]=target domain texts"
    feature_values = get_feature_representations(feature_names, examples, domain2data[trg_domain][0], vocab, word2vec, topic_vectorizer, lda_model)
    "z-normo"
    feature_values = stats.zscore(feature_values, axis=0)
    "free memory"
    del examples, y_train_check, train_domains_check
    "log data structure"
    run_dict = {
        "random": [],
        "most_similar_domain": [],
        "most_similar_examples": [], 
        "all_source_data": [], 
        "bayes_opt": []
    }
    ""
    for _ in range(hparam.num_runs):
        "target domain dataset"
        X_test, y_test, _ = domain2train_data[trg_domain]
        "split target domain dataset into `test`, `val`"
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=100, stratify=y_test)
        if hparam.feature_weights_file:
            train_pretrained_weights(feature_values, X_train, y_train, train_domains, hparam.num_train_examples, X_val, y_val, X_test, y_test, trg_domain, feature_names)
            continue
        "baselines"
        ################################################
        for baseline in hparam.baselines:
            "train_subset, labels_subset"
            ############################################
            if baseline == "random":
                train_subset, _, labels_subset, _ = train_test_split(X_train, y_train, train_size=hparam.num_train_examples, stratify=y_train)
            elif baseline == 'all_source_data':
                train_subset, labels_subset = X_train, y_train
            elif baseline == "most_similar_domain":
                most_similar_domain = get_most_similar_domain(trg_domain, domain2term_dist)
                train_subset, labels_subset, _ = domain2train_data[most_similar_domain]
                train_subset, _, labels_subset, _ = train_test_split(train_subset, labels_subset, train_size=hparam.num_train_examples, stratify=labels_subset)
            elif baseline == "most_similar_examples":
                one_all_weights = np.ones(len(feature_names))
                one_all_weights[1:] = 0
                train_subset, labels_subset = get_data_subsets(feature_values, one_all_weights, X_train, y_train, hparam.num_train_examples)
            else:
                raise ValueError('%s is not a baseline.' % baseline)
            "val_accuracy, test_accuracy"
            #############################################
            val_accuracy, test_accuracy = train_and_evaluate(train_subset, labels_subset, X_val, y_val, X_test, y_test)
            run_dict[baseline].append((val_accuracy, test_accuracy))
        "Paper Method"
        "Weights"
        ################################################
        lower = np.array(len(feature_names) * [-1])
        upper = np.array(len(feature_names) * [1])
        res = bayesian_optimization(objective_function, lower=lower, upper=upper, num_iterations=hparam.num_iterations)
        best_feature_weights = res['x_opt']
        "data"
        ################################################
        train_subset, labels_subset = get_data_subsets(feature_values, best_feature_weights, X_train, y_train, hparam.num_train_examples)
        "train & evaluate"
        ################################################
        val_accuracy, test_accuracy = train_and_evaluate(train_subset, labels_subset, X_val, y_val, X_test, y_test)
        "log"
        run_dict["bayes_opt"].append((val_accuracy, test_accuracy, best_feature_weights))
    "run_dict"
    log_to_file(hparam.log_file, run_dict, trg_domain, hparam)

