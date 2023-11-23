import os
import pickle
from abc import ABC, abstractclassmethod

import numpy as np
from loguru import logger

from crs.download import build

class BaseDataset(ABC):
    def __init__(self, opt, dpath, resource, restore=False, save=False):
        self.opt = opt
        self.dpath = dpath
        dfile = resource['file']
        build(dpath, dfile, version=resource['version'])
        
        if not restore:
            train_data, valid_data, test_data, self.vocab = self._load_data()
            logger.info('[Finish data load]')
            self.train_data, self.valid_data, self.test_data, self.side_data = self._data_preprocess(train_data, valid_data, test_data)

            embedding = opt.get('embedding', None)
            if embedding:
                self.side_data["embedding"] = np.load(os.path.join(self.dpath, embedding))
                logger.debug(f'[Load pretrained embedding from {embedding}]')
            logger.info('[Finish data preprocess]')
        else:
            self.train_data, self.valid_data, self.test_data, self.side_data, self.vocab = self._load_from_restore()

        if save:
            data = (self.train_data, self.valid_data, self.test_data, self.side_data, self.vocab)
            self._save_to_restore(data)
    

    @abstractclassmethod
    def _load_data(self):
        pass

    @abstractclassmethod
    def _data_preprocess(self, train_data, valid_data, test_data):
        pass

    @abstractclassmethod
    def _load_from_restore(self, file_name="all_data.pkl"):
        if not os.path.exists(os.path.join(self.dpath, file_name)):
            raise ValueError(f'Saved dataset [{file_name}] does not exist')
        with open(os.path.join(self.dpath, file_name), 'rb') as f:
            dataset = pickle.load(f)
        logger.info(f'Restore dataset from [{file_name}]')
        return dataset

    @abstractclassmethod
    def _save_to_restore(self, data, file_name="all_data.pkl"):
        if not os.path.exists(self.dpath):
            os.makedirs(self.dpath)
        save_path = os.path.join(self.dpath, file_name)
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f'[Save dataset to {file_name}]')