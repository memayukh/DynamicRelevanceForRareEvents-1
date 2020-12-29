
import numpy as np
from sklearn import preprocessing
import pandas as pd


def RelevanceOverSampling(D, target, size, relevance, categorical_col = []):

    if(int(sum(relevance)) == 0):
        return D

    relevance = [float(i)/sum(relevance) for i in relevance]

    new_target_values = np.random.choice(D[target], size , p=relevance)

    new_indices = []
    for y_instance in new_target_values:
        index = D.index[D[target] == y_instance].tolist()
        new_indices.append(index[-1])


    extra_data = D.loc[new_indices]

    new_D = D.append(extra_data, ignore_index = True)

    return new_D









