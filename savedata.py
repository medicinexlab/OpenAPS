"""
saveloadarray.py
"""

import pickle

def save_data(data_item, name):
    """
    This function saves the data as a pickle file.
    """

    with open('./savedata/' + name + '.pickle', 'wb') as handle:
        pickle.dump(data_item, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_array(name):
    """
    This function loads the data from the pickle file.
    """
    with open('./savedata/' + name + '.pickle', 'rb') as handle:
        data_item = pickle.load(handle)

    return data_item
