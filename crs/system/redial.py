import torch
from loguru import logger

from crs.data import dataset_language_map
from crs.evaluator.metrics.base import AverageMetric
from crs.evaluator.metrics.gen import PPLMetric
from crs.system.base import BaseSystem
from crs.system.utils.functions import ind2txt
from math import floor

from crs.config import PRETRAIN_PATH
from crs.data import get_dataloader, dataset_language_map

class ReDialSystem(BaseSystem):
    """This is the system for Redial model"""

    def __init__(self, opt, train_dataloader, valid_dataloader, test_dataloader, vocab, side_data, restore_system=False,
                 interact=False, debug=False, tensorboard=False):
        """

        Args:
            opt (dict): Indicating the hyper parameters.
            train_dataloader (BaseDataLoader): Indicating the train dataloader of corresponding dataset.
            valid_dataloader (BaseDataLoader): Indicating the valid dataloader of corresponding dataset.
            test_dataloader (BaseDataLoader): Indicating the test dataloader of corresponding dataset.
            vocab (dict): Indicating the vocabulary.
            side_data (dict): Indicating the side data.
            restore_system (bool, optional): Indicating if we store system after training. Defaults to False.
            interact (bool, optional): Indicating if we interact with system. Defaults to False.
            debug (bool, optional): Indicating if we train in debug mode. Defaults to False.
            tensorboard (bool, optional) Indicating if we monitor the training performance in tensorboard. Defaults to False. 

        """
        super(ReDialSystem, self).__init__(opt, train_dataloader, valid_dataloader, test_dataloader, vocab, side_data,
                                           restore_system, interact, debug, tensorboard)
        self.ind2tok = vocab['conv']['ind2tok']
        self.end_token_idx = vocab['conv']['end']
        self.item_ids = side_data['rec']['item_entity_ids']
        self.id2entity = vocab['rec']['id2entity']

        self.rec_optim_opt = opt['rec']
        self.conv_optim_opt = opt['conv']
        self.rec_epoch = self.rec_optim_opt['epoch']
        self.conv_epoch = self.conv_optim_opt['epoch']
        self.rec_batch_size = self.rec_optim_opt['batch_size']
        self.conv_batch_size = self.conv_optim_opt['batch_size']

        self.language = dataset_language_map[self.opt['dataset']]

    def rec_evaluate(self, rec_predict, item_label):
        rec_predict = rec_predict.cpu()
        rec_predict = rec_predict[:, self.item_ids]
        _, rec_ranks = torch.topk(rec_predict, 50, dim=-1)
        rec_ranks = rec_ranks.tolist()
        item_label = item_label.tolist()
        for rec_rank, item in zip(rec_ranks, item_label):
            item = self.item_ids.index(item)
            self.evaluator.rec_evaluate(rec_rank, item)

    def conv_evaluate(self, prediction, response):
        prediction = prediction.tolist()
        response = response.tolist()
        for p, r in zip(prediction, response):
            p_str = ind2txt(p, self.ind2tok, self.end_token_idx)
            r_str = ind2txt(r, self.ind2tok, self.end_token_idx)
            self.evaluator.gen_evaluate(p_str, [r_str])

    def step(self, batch, stage, mode):
        assert stage in ('rec', 'conv')
        assert mode in ('train', 'valid', 'test')

        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device)

        if stage == 'rec':
            rec_loss, rec_scores = self.rec_model.forward(batch, mode=mode)
            rec_loss = rec_loss.sum()
            if mode == 'train':
                self.backward(rec_loss)
            else:
                self.rec_evaluate(rec_scores, batch['item'])
            rec_loss = rec_loss.item()
            self.evaluator.optim_metrics.add("rec_loss", AverageMetric(rec_loss))
        else:
            gen_loss, preds = self.conv_model.forward(batch, mode=mode)
            gen_loss = gen_loss.sum()
            if mode == 'train':
                self.backward(gen_loss)
            else:
                self.conv_evaluate(preds, batch['response'])
            gen_loss = gen_loss.item()
            self.evaluator.optim_metrics.add('gen_loss', AverageMetric(gen_loss))
            self.evaluator.gen_metrics.add('ppl', PPLMetric(gen_loss))

    def train_recommender(self):
        self.init_optim(self.rec_optim_opt, self.rec_model.parameters())

        for epoch in range(self.rec_epoch):
            self.evaluator.reset_metrics()
            logger.info(f'[Recommendation epoch {str(epoch)}]')
            logger.info('[Train]')
            for batch in self.train_dataloader['rec'].get_rec_data(batch_size=self.rec_batch_size):
                self.step(batch, stage='rec', mode='train')
            self.evaluator.report(epoch=epoch, mode='train')  # report train loss
            # val
            logger.info('[Valid]')
            with torch.no_grad():
                self.evaluator.reset_metrics()
                for batch in self.valid_dataloader['rec'].get_rec_data(batch_size=self.rec_batch_size, shuffle=False):
                    self.step(batch, stage='rec', mode='valid')
                self.evaluator.report(epoch=epoch, mode='valid')  # report valid loss
                # early stop
                metric = self.evaluator.optim_metrics['rec_loss']
                if self.early_stop(metric):
                    break
        # test
        logger.info('[Test]')
        with torch.no_grad():
            self.evaluator.reset_metrics()
            for batch in self.test_dataloader['rec'].get_rec_data(batch_size=self.rec_batch_size, shuffle=False):
                self.step(batch, stage='rec', mode='test')
            self.evaluator.report(mode='test')

    def train_conversation(self):
        self.init_optim(self.conv_optim_opt, self.conv_model.parameters())

        for epoch in range(self.conv_epoch):
            self.evaluator.reset_metrics()
            logger.info(f'[Conversation epoch {str(epoch)}]')
            logger.info('[Train]')
            for batch in self.train_dataloader['conv'].get_conv_data(batch_size=self.conv_batch_size):
                self.step(batch, stage='conv', mode='train')
            self.evaluator.report(epoch=epoch, mode='train')
            # val
            logger.info('[Valid]')
            with torch.no_grad():
                self.evaluator.reset_metrics()
                for batch in self.valid_dataloader['conv'].get_conv_data(batch_size=self.conv_batch_size,
                                                                         shuffle=False):
                    self.step(batch, stage='conv', mode='valid')
                self.evaluator.report(epoch=epoch, mode='valid')
                metric = self.evaluator.optim_metrics['gen_loss']
                if self.early_stop(metric):
                    break
        # test
        logger.info('[Test]')
        with torch.no_grad():
            self.evaluator.reset_metrics()
            for batch in self.test_dataloader['conv'].get_conv_data(batch_size=self.conv_batch_size, shuffle=False):
                self.step(batch, stage='conv', mode='test')
            self.evaluator.report(mode='test')

    def fit(self):
        self.train_recommender()
        self.train_conversation()

    def interact(self):
        self.init_interact()
        input_text = self.get_input(self.language)
        while not self.finished:
            # rec
            if hasattr(self, 'rec_model'):
                rec_input = self.process_input(input_text, 'rec')
                scores = self.rec_model.forward(rec_input, 'infer')

                scores = scores.cpu()[0]
                scores = scores[self.item_ids]
                _, rank = torch.topk(scores, 10, dim=-1)
                item_ids = []
                for r in rank.tolist():
                    item_ids.append(self.item_ids[r])
                first_item_id = item_ids[:1]
                self.update_context('rec', entity_ids=first_item_id, item_ids=first_item_id)

                print(f"[Recommend]:")
                for item_id in item_ids:
                    if item_id in self.id2entity:
                        print(self.id2entity[item_id])
            # conv
            if hasattr(self, 'conv_model'):
                conv_input = self.process_input(input_text, 'conv')
                preds = self.conv_model.forward(conv_input, 'infer').tolist()[0]
                p_str = ind2txt(preds, self.ind2tok, self.end_token_idx)

                token_ids, entity_ids, movie_ids, word_ids = self.convert_to_id(p_str, 'conv')
                self.update_context('conv', token_ids, entity_ids, movie_ids, word_ids)

                print(f"[Response]:\n{p_str}")
            # input
            input_text = self.get_input(self.language)

    def process_input(self, input_text, stage):
        token_ids, entity_ids, movie_ids, word_ids = self.convert_to_id(input_text, stage)
        self.update_context(stage, token_ids, entity_ids, movie_ids, word_ids)

        data = {'role': 'Seeker', 
                'context_tokens': self.context[stage]['context_tokens'],
                'context_entities': self.context[stage]['context_entities'],
                'context_words': self.context[stage]['context_words'],
                'context_items': self.context[stage]['context_items'],
                'items': self.context[stage]['entity_set'],
                }
        dataloader = get_dataloader(self.opt, data, self.vocab[stage])
        # if stage == 'rec':
        #     data = dataloader.rec_interact(data)
        # elif stage == 'conv':
        #     data = dataloader.conv_interact(data)

        data = [ele.to(self.device) if isinstance(ele, torch.Tensor) else ele for ele in data]
        return data

    def convert_to_id(self, text, stage):
        if self.language == 'zh':
            tokens = self.tokenize(text, 'pkuseg')
        elif self.language == 'en':
            tokens = self.tokenize(text, 'nltk')
        else:
            raise

        entities = self.link(tokens, self.side_data[stage]['entity_kg']['entity'])
        words = self.link(tokens, self.side_data[stage]['word_kg']['entity'])

        if self.opt['tokenize'][stage] in ('gpt2', 'bert'):
            language = dataset_language_map[self.opt['dataset']]
            path = os.path.join(PRETRAIN_PATH, self.opt['tokenize'][stage], language)
            tokens = self.tokenize(text, 'bert', path)

        token_ids = [self.vocab[stage]['tok2ind'].get(token, self.vocab[stage]['unk']) for token in tokens]
        entity_ids = [self.vocab[stage]['entity2id'][entity] for entity in entities if
                      entity in self.vocab[stage]['entity2id']]
        movie_ids = [entity_id for entity_id in entity_ids if entity_id in self.item_ids]
        word_ids = [self.vocab[stage]['word2id'][word] for word in words if word in self.vocab[stage]['word2id']]

        return token_ids, entity_ids, movie_ids, word_ids
        