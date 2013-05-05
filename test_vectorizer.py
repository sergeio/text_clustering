from string import split
from unittest import TestCase, main

from vectorizer import (
    cluster_paragraphs,
    make_word_set,
    make_word_vectors,
)


class WhenCreatingAVectorSpace(TestCase):

    def setUp(self):
        text1 = 'this is some text'
        text2 = 'unique text'
        text3 = 'and even way more text'

        self.word_vectors = map(split, [text1, text2, text3])

        self.returned = make_word_set(self.word_vectors)

    def test_foo(self):
        for vector in self.word_vectors:
            for word in vector:
                self.assertTrue(word in self.returned)

class WhenCalculatingWordVectors(TestCase):

    def setUp(self):
        text1 = 'this is some text'
        text2 = 'unique text'
        text3 = 'and even way more text'

        self.word_vectors = [
            ['some', 'text'],
            ['unique', 'text'],
            ['even', 'way', 'more', 'text'],
        ]
        self.vector_space = make_word_set(self.word_vectors)

        self.returned = make_word_vectors(self.vector_space, self.word_vectors)

    def test_return_value(self):
        expected_return = [
            [0, 0.5, 0.5, 0, 0, 0],
            [0, 0.5, 0, 0, 0.5, 0],
            [0.25, 0.25, 0, 0.25, 0, 0.25],
        ]
        self.assertEqual(self.returned, expected_return)

class WhenClusteringParagraphs(TestCase):

    def setUp(self):
        self.text1 = 'A study on the effectiveness of milk and micronutrients.'
        self.text2 = 'A study on the effectiveness of milk.'
        self.text3 = 'Something completely unrelated'

        self.returned = cluster_paragraphs(
            [self.text1, self.text2, self.text3])

    def test_cluster_correctness(self):
        self.assertTrue([self.text1, self.text2] in self.returned)
        self.assertTrue([self.text3] in self.returned)


if __name__ == '__main__':
    main()
