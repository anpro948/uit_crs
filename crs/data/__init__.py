from crs.data.dataloader import *
from crs.data.dataset import *

dataset_register_table = {
    'ReDial': ReDialDataset,
}

dataset_language_map = {
    'ReDial': 'en',
}

dataloader_register_table = {
    # 'TGReDial': TGReDialDataLoader,
    # 'TGRec': TGReDialDataLoader,
    # 'TGConv': TGReDialDataLoader,
    # 'TGRec_TGConv': TGReDialDataLoader,

    'ReDialRec': ReDialDataLoader,
    'ReDialConv': ReDialDataLoader,
    'ReDialRec_ReDialConv': ReDialDataLoader,

    # 'BERT': TGReDialDataLoader,
    # 'GPT2': TGReDialDataLoader,
}


def get_dataset(opt, tokenize, restore, save) -> BaseDataset:
    dataset = opt['dataset'] #example -> "Redial"
    if dataset in dataset_register_table:
        return dataset_register_table[dataset](opt, tokenize, restore, save)
    else:
        raise NotImplementedError(f'The dataset [{dataset}] has not been implemented')


def get_dataloader(opt, dataset, vocab) -> BaseDataLoader:
    model_name = opt['model_name']
    if model_name in dataloader_register_table:
        return dataloader_register_table[model_name](opt, dataset, vocab)
    else:
        raise NotImplementedError(f'The dataloader [{model_name}] has not been implemented')