import numpy as np
import pandas as pd 
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import correlation, hamming, jaccard, minkowski, cosine
from zenml.vector import Vector
from zenml.text import Text
from zenml.utils import values_equal


class Similarity:
    def __init__(self, a:Vector, b:Vector) -> None:
        self.a = a
        self.b = b


    def euclidean(self) -> float:
        """
        Euclidean distance between two vectors
        """
        return pairwise_distances([self.a.values], [self.b.values], metric='euclidean')[0][0]


    def hamming(self) -> float:
        """
        Hamming distance between two vectors
        """
        return pairwise_distances([self.a.values], [self.b.values], metric=hamming)[0][0]


    def jaccard(self) -> float:
        """
        Jaccard distance between two vectors
        """
        return pairwise_distances([self.a.values], [self.b.values], metric=jaccard)[0][0]


    def minkowski(self) -> float:
        """
        Minkowski distance between two vectors
        """
        return pairwise_distances([self.a.values], [self.b.values], metric=minkowski)[0][0]


    def correlation(self) -> float:
        """
        Correlation distance between two vectors. Note that perfect
        correlation will have a distance of 0.
        """
        return pairwise_distances([self.a.values], [self.b.values], metric=correlation)[0][0]


    def cosine(self) -> float:
        """
        Cosine angle between two vectors
        """
        return pairwise_distances([self.a.values], [self.b.values], metric=cosine)[0][0]


    def variable_match(self) -> Vector:
        """
        Returns a binary vector representing the equality of
        pairwise elements in two series
        """
        if self.a.dtype in ('int64', 'bool'):
            return (self.a == self.b) * 1
        elif self.a.dtype == 'float64':
            return pd.Series(np.isclose(self.a, self.b, rtol=1e-05, atol=1e-08, equal_nan=False) * 1)
        elif self.a.dtype == 'object':
            return pd.Series(np.array([True if values_equal(model_pair[0], model_pair[1]) else False
                            for model_pair in list(zip(self.a.values, self.b.values))]) * 1)
        else:
            raise Exception('datatime64 not supported yet')


    def cosine_text(self, regex_exp:str) -> Vector:
        if self.a.dtype != 'object' or self.b.dtype != 'object':
            raise Exception('Only pandas series of type object are supported')
        else:
            text_vector_a = Text(self.a)
            text_vector_b = Text(self.a)
            word_counts_a = text_vector_a.word_counts(regex_exp)
            word_counts_b = text_vector_b.word_counts(regex_exp)
            # TODO: finish implementation
            return None


