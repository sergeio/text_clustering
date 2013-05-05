from collections import defaultdict
import re

from similarity import similarity
from k_means import KMeans


def word_frequencies(word_vector):
    """What percent of the time does each word in the vector appear?

    Returns a dictionary mapping each word to its frequency.

    """
    num_words = len(word_vector)
    frequencies = defaultdict(float)
    for word in word_vector:
        frequencies[word] += 1.0 / num_words

    return dict(frequencies)


def compare_vectors(word_vector1, word_vector2):
    """Numerical similarity between lists of words. Higher is better.

    Uses cosine similarity.
    Result range: 0 (bad) - 1 (uses all the same words in the same proportions)

    """
    all_words = list(set(word_vector1).union(set(word_vector2)))
    frequency_dict1 = word_frequencies(word_vector1)
    frequency_dict2 = word_frequencies(word_vector2)

    frequency_vector1 = [frequency_dict1.get(word, 0) for word in all_words]
    frequency_vector2 = [frequency_dict2.get(word, 0) for word in all_words]

    return similarity(frequency_vector1, frequency_vector2)


def vectorize_text(text):
    """Takes in text, processes it, and vectorizes it."""

    def remove_punctuation(text):
        """Removes special characters from text."""
        return re.sub('[,.?";:\-!@#$%^&*()]', '', text)

    def remove_common_words(text_vector):
        """Removes 50 most common words in the uk english.

        source: http://www.bckelk.ukfsn.org/words/uk1000n.html

        """
        common_words = set(['the', 'and', 'to', 'of', 'a', 'I', 'in',
            'was', 'he', 'that', 'it', 'his', 'her', 'you', 'as',
            'had', 'with', 'for', 'she', 'not', 'at', 'but', 'be',
            'my', 'on', 'have', 'him', 'is', 'said', 'me', 'which',
            'by', 'so', 'this', 'all', 'from', 'they', 'no', 'were',
            'if', 'would', 'or', 'when', 'what', 'there', 'been',
            'one', 'could', 'very', 'an', 'who'])
        return [word for word in text_vector if word not in common_words]

    text = text.lower()
    text = remove_punctuation(text)
    words_list = text.split()
    words_list = remove_common_words(words_list)

    return words_list


def compare_texts(text1, text2):
    """How similar are the two input paragraphs?"""
    return compare_vectors(vectorize_text(text1), vectorize_text(text2))


################################


def make_word_lists(paragraphs):
    return map(vectorize_text, paragraphs)

def make_word_set(word_lists):
    """ """
    return set(word for words in word_lists for word in words)

def make_word_vectors(word_set, word_lists):

    def vectorize(frequency_dict):
        return [frequency_dict.get(word, 0) for word in word_set]

    frequencies = map(word_frequencies, word_lists)

    return map(vectorize, frequencies)

def translator(clusters, paragraph_map):
    """Translate vectors back into paragraphs, to make them human-readable."""
    def item_translator(vector):
        return paragraph_map.get(str(vector))

    def cluster_translator(cluster):
        return map(item_translator, cluster)

    return map(cluster_translator, clusters)

def cluster_paragraphs(paragraphs, num_clusters=2):
    word_lists = make_word_lists(paragraphs)
    word_set = make_word_set(word_lists)
    word_vectors = make_word_vectors(word_set, word_lists)

    paragraph_map = dict(zip(map(str, word_vectors), paragraphs))

    k_means = KMeans(num_clusters, word_vectors)
    k_means.main_loop()
    return translator(k_means.clusters, paragraph_map)



# the `vectorize_text` function is not actually vectorizing, it's just
# splitting/stripping.  The vectorization happens in the `compare_texts`
# function, where the word-lists are replaced by the frequency of their
# occurence. I should rename functions to rectify.
