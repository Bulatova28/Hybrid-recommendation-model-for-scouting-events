import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin


def convert_to_int(x):
    return x.astype(int)


class SentenceTransformerEmbedder(BaseEstimator, TransformerMixin):
    def __init__(self, model_name='google/embeddinggemma-300m', output_dim=128):
        self.model_name = model_name
        self.output_dim = output_dim
        self.model = SentenceTransformer(model_name, device='cpu')
        self.cache = {}

    def fit(self,x,y=None):
        return self

    def transform(self, x):
        if not hasattr(self, 'cache'):
            self.cache = {}

        if isinstance(x, pd.DataFrame):
            text_data = x.astype(str).agg(' '.join, axis=1).tolist()
        else:
            text_data = [str(i) for i in x]
        
        to_encode = [text for text in text_data if text not in self.cache]
        
        if to_encode:
            new_embeds = self.model.encode(to_encode, convert_to_numpy=True)
            for text, embed in zip(to_encode, new_embeds):
                self.cache[text] = embed
        
        results = np.array([self.cache[text] for text in text_data])
        
        if self.output_dim:
            results = results[:, :self.output_dim]
            
        return results
    

class CyclicEncoder(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        x_array = np.array(x)
        self.n_categories_ = []
        for i in range(x_array.shape[1]):
            self.n_categories_.append(len(np.unique(x_array[:, i])))
        return self

    def transform(self, x):
        x_array = np.array(x)
        results = []
        for i in range(x_array.shape[1]):
            n = self.n_categories_[i]
            val = x_array[:, i]
            
            sin = np.sin(2 * np.pi * val / n)
            cos = np.cos(2 * np.pi * val / n)
            results.append(np.c_[sin, cos])
            
        return np.hstack(results)