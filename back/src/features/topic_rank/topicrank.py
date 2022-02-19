
"""TopicRank keyphrase extraction model.

Graph-based ranking approach to keyphrase extraction described in:
https://aclanthology.org/I13-1062.pdf

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import string
from itertools import combinations

import networkx as nx
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

from .base import LoadFile


FRENCH_STOP_WORDS = {
    "alors",
    "au",
    "aucuns",
    "aussi",
    "autre",
    "avant",
    "avec",
    "avoir",
    "bon",
    "car",
    "ce",
    "cela",
    "ces",
    "ceux",
    "chaque",
    "ci",
    "comme",
    "comment",
    "dans",
    "des",
    "du",
    "dedans",
    "dehors",
    "depuis",
    "devrait",
    "doit",
    "donc",
    "dos",
    "début",
    "elle",
    "elles",
    "en",
    "encore",
    "essai",
    "est",
    "et",
    "eu",
    "fait",
    "faites",
    "fois",
    "font",
    "hors",
    "ici",
    "il",
    "ils",
    "je",
    "juste",
    "la",
    "le",
    "les",
    "leur",
    "là",
    "ma",
    "maintenant",
    "mais",
    "mes",
    "mien",
    "moins",
    "mon",
    "mot",
    "même",
    "ni",
    "nommés",
    "notre",
    "nous",
    "ou",
    "où",
    "par",
    "parce",
    "pas",
    "peut",
    "peu",
    "plupart",
    "pour",
    "pourquoi",
    "quand",
    "que",
    "quel",
    "quelle",
    "quelles",
    "quels",
    "qui",
    "sa",
    "sans",
    "ses",
    "seulement",
    "si",
    "sien",
    "son",
    "sont",
    "sous",
    "soyez",
    "sujet",
    "sur",
    "ta",
    "tandis",
    "tellement",
    "tels",
    "tes",
    "ton",
    "tous",
    "tout",
    "toute",
    "trop",
    "très",
    "tu",
    "voient",
    "vont",
    "votre",
    "vous",
    "vu",
    "ça",
    "étaient",
    "état",
    "étions",
    "été",
    "être",
    "plus",
    "celui",
    "entre",
    "vers",
    "dont",
    "divers",
    "pendant",
    "non",
    "certain",
    "chose",
}


class TopicRank(LoadFile):

    """TopicRank keyphrase extraction model.

    Parameterized example::

        import pke
        import string
        from nltk.corpus import stopwords

        # 1. create a TopicRank extractor.
        extractor = pke.unsupervised.TopicRank()

        # 2. load the content of the document.
       extractor.load_document(input='path/to/input.xml')

        # 3. select the longest sequences of nouns and adjectives, that do
        #    not contain punctuation marks or stopwords as candidates.
        pos = {'NOUN', 'PROPN', 'ADJ'}
        stoplist = list(string.punctuation)
        stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
        stoplist += stopwords.words('english')
        extractor.candidate_selection(pos=pos, stoplist=stoplist)

        # 4. build topics by grouping candidates with HAC (average linkage,
        #    threshold of 1/4 of shared stems). Weight the topics using random
        #    walk, and select the first occuring candidate from each topic.
        extractor.candidate_weighting(threshold=0.74, method='average')

        # 5. get the 10-highest scored candidates as keyphrases
        keyphrases = extractor.get_n_best(n=10)

    """

    def __init__(self):
        """Redefining initializer for TopicRank.
        """

        super(TopicRank, self).__init__()

        self.graph = nx.Graph()
        """ The topic graph. """

        self.topics = []
        """ The topic container. """

    def candidate_selection(self, pos=None):
        """Selects longest sequences of nouns and adjectives as keyphrase
        candidates.

        Args:
            pos (set): the set of valid POS tags, defaults to ('NOUN',
                'PROPN', 'ADJ').

        """

        # define default pos tags set
        if pos is None:
            pos = {'NOUN', 'PROPN', 'ADJ'}

        # select sequence of adjectives and nouns
        # print(self.sentences[0].stems)
        self.longest_pos_sequence_selection(valid_pos=pos)


        # initialize stoplist list if not provided

        stoplist = set(self.stoplist).union(FRENCH_STOP_WORDS)

        # filter candidates containing stopwords or punctuation marks
        self.candidate_filtering(stoplist=set(string.punctuation).union({'-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-'}.union(stoplist)))


    def vectorize_candidates(self):
        """Vectorize the keyphrase candidates.

        Returns:
            C (list): the list of candidates.
            X (matrix): vectorized representation of the candidates.

        """

        # build the vocabulary, i.e. setting the vector dimensions
        dim = set([])
        #print("candiate items:",self.candidates.items())
        # for k, v in self.candidates.iteritems():
        # iterate Python 2/3 compatible
        for (k, v) in self.candidates.items():
            for w in v.lexical_form:
                dim.add(w)
        dim = list(dim)

        # vectorize the candidates Python 2/3 + sort for random issues
        C = list(self.candidates)  # .keys()
        C.sort()

        X = np.zeros((len(C), len(dim)))
        for i, k in enumerate(C):
            for w in self.candidates[k].lexical_form:
                X[i, dim.index(w)] += 1

        #print(C,X)
        return C, X

    def topic_clustering(self, threshold=0.74, method='single'):
        """Clustering candidates into topics.

        Args:
            threshold (float): the minimum similarity for clustering, defaults
                to 0.74, i.e. more than 1/4 of stem overlap similarity.
            method (str): the linkage method, defaults to average.

        """

        # handle document with only one candidate
        if len(self.candidates) == 1:
            self.topics.append([list(self.candidates)[0]])
            return

        if len(self.candidates) == 0:
            self.topics = []
            return

        # vectorize the candidates
        candidates, X = self.vectorize_candidates()

        # compute the distance matrix
        Y = pdist(X)

        # compute the clusters
        Z = linkage(Y, method=method)

        # form flat clusters
        clusters = fcluster(Z, t=threshold, criterion='distance')

        # for each topic identifier
        for cluster_id in range(1, max(clusters) + 1):
            self.topics.append([candidates[j] for j in range(len(clusters))
                                if clusters[j] == cluster_id])

    def build_topic_graph(self):
        """Build topic graph."""

        # adding the nodes to the graph
        # print(self.topics)
        self.graph.add_nodes_from(range(len(self.topics)))

        # loop through the topics to connect the nodes
        for i, j in combinations(range(len(self.topics)), 2):
            self.graph.add_edge(i, j, weight=0.0)
            for c_i in self.topics[i]:
                for c_j in self.topics[j]:
                    for p_i in self.candidates[c_i].offsets:
                        for p_j in self.candidates[c_j].offsets:
                            gap = abs(p_i - p_j)
                            if p_i < p_j:
                                gap -= len(self.candidates[c_i].lexical_form) - 1
                            if p_j < p_i:
                                gap -= len(self.candidates[c_j].lexical_form) - 1
                            self.graph[i][j]['weight'] += 1.0 / gap

        mapping = {i: self.topics[i][0] for i in range(len(self.topics))}
        self.graph = nx.relabel_nodes(self.graph, mapping)

    def candidate_weighting(self,
                            threshold=0.74,
                            method='average',
                            heuristic=None):
        """Candidate ranking using random walk.

        Args:
            threshold (float): the minimum similarity for clustering, defaults
                to 0.74.
            method (str): the linkage method, defaults to average.
            heuristic (str): the heuristic for selecting the best candidate for
                each topic, defaults to first occurring candidate. Other options
                are 'frequent' (most frequent candidate, position is used for
                ties).

        """

        # cluster the candidates
        self.topic_clustering(threshold=threshold, method=method)

        # build the topic graph
        self.build_topic_graph()


        # compute the word scores using random walk
        w = nx.pagerank_scipy(self.graph, alpha=0.85, weight='weight')

        # loop through the topics
        for i, topic in enumerate(self.topics):

            # get the offsets of the topic candidates
            offsets = [self.candidates[t].offsets[0] for t in topic]

            # get first candidate from topic
            if heuristic == 'frequent':

                # get frequencies for each candidate within the topic
                freq = [len(self.candidates[t].surface_forms) for t in topic]

                # get the indexes of the most frequent candidates
                indexes = [j for j, f in enumerate(freq) if f == max(freq)]

                # offsets of the indexes
                indexes_offsets = [offsets[j] for j in indexes]
                most_frequent = indexes_offsets.index(min(indexes_offsets))
                self.weights[topic[most_frequent]] = w[self.topics[i][0]]

            else:
                first = offsets.index(min(offsets))
                # print(w)
                self.weights[topic[first]] = w[self.topics[i][0]]

