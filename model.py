import torch
import torch.nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

import numpy as np

from attention_module import BAttention, SelfAttention, MultiHeadedAttention


class MultiInferRNNModel(torch.nn.Module):
    def __init__(self, gen_emb, domain_emb, args):
        '''double embedding + lstm encoder + dot self attention'''
        super(MultiInferRNNModel, self).__init__()

        self.args = args
        self.gen_embedding = torch.nn.Embedding(gen_emb.shape[0], gen_emb.shape[1])
        self.gen_embedding.weight.data.copy_(gen_emb)
        self.gen_embedding.weight.requires_grad = False

        self.domain_embedding = torch.nn.Embedding(domain_emb.shape[0], domain_emb.shape[1])
        self.domain_embedding.weight.data.copy_(domain_emb)
        self.domain_embedding.weight.requires_grad = False

        self.dropout1 = torch.nn.Dropout(0.5)
        self.dropout2 = torch.nn.Dropout(0)

        self.bilstm = torch.nn.LSTM(300+100, args.lstm_dim,
                                    num_layers=1, batch_first=True, bidirectional=True)
        self.attention_layer = SelfAttention(args)

        self.feature_linear = torch.nn.Linear(args.lstm_dim*2 + args.class_num*2, args.lstm_dim*2)
        self.cls_linear = torch.nn.Linear(args.lstm_dim*6, args.class_num)

        self.Gvec = torch.nn.parameter.Parameter(torch.empty((4, 200)))
        torch.nn.init.uniform_(self.Gvec)

        self.batt = BAttention(args)
        self.matt = MultiHeadedAttention(1, args.class_num)

        self.vec_fc = torch.nn.Linear(args.lstm_dim*4, args.lstm_dim*2)

    def _get_embedding(self, sentence_tokens, mask):
        gen_embed = self.gen_embedding(sentence_tokens)
        domain_embed = self.domain_embedding(sentence_tokens)
        embedding = torch.cat([gen_embed, domain_embed], dim=2)
        embedding = self.dropout1(embedding)
        embedding = embedding * mask.unsqueeze(2).float().expand_as(embedding)
        return embedding

    def _lstm_feature(self, embedding, lengths):
        embedding = pack_padded_sequence(embedding, lengths, batch_first=True)
        context, _ = self.bilstm(embedding)
        context, _ = pad_packed_sequence(context, batch_first=True)
        return context

    def _cls_logits(self, features):
        # features = self.dropout2(features)
        tags = self.cls_linear(features)
        return tags

    def feature_deal(self, feature, edim):
        feature = feature.unsqueeze(2).expand([-1,-1, edim, -1])
        feature_T = feature.transpose(1, 2)
        features = torch.cat([feature, feature_T], dim=3)

        fusion_feature = self.vec_fc(features)
        features = torch.cat([feature, feature_T, fusion_feature], dim=3)
        #print(features.size())
        #exit(0)
        return features

    def logits_choose(self, logits):
        logits_a = torch.max(logits, dim=1)[0]
        logits_b = torch.max(logits, dim=2)[0]
        logits = torch.cat([logits_a.unsqueeze(3), logits_b.unsqueeze(3)], dim=3)
        logits = torch.max(logits, dim=3)[0]
        return logits

    def logits_deal(self, logits, edim):
        logits = logits.unsqueeze(2).expand([-1, -1, edim, -1])
        logits_T = logits.transpose(1, 2)
        logits = torch.cat([logits, logits_T], dim=3)
        return logits

    def multi_hops(self, feature, lengths, mask, k):
        '''generate mask'''
        max_length = feature.shape[1]
        mask = mask[:, :max_length]
        mask_a = mask.unsqueeze(1).expand([-1, max_length, -1])
        mask_b = mask.unsqueeze(2).expand([-1, -1, max_length])
        mask = mask_a * mask_b
        mask = torch.triu(mask).unsqueeze(3).expand([-1, -1, -1, self.args.class_num])

        '''save all logits'''
        logits_list = []
        features = self.feature_deal(feature, max_length)
        Rvec = self._cls_logits(features)
        logits_list.append(Rvec)
        
        Hvec = feature
        for i in range(k):
            pre_Rvec = Rvec
            Rvec = pre_Rvec * mask

            Rvec_max = self.logits_choose(Rvec)
            # Rvec = self.logits_deal(Rvec_max)
            pre_Rvec_max = self.logits_choose(pre_Rvec)

            Rvec_max = self.matt(Rvec_max, Rvec_max, Rvec_max)

            #print(Hvec.size(), Rvec_max.size(), pre_Rvec_max.size()) 
            new_features = torch.cat([Hvec, Rvec_max, pre_Rvec_max], dim=2)
            Hvec = self.feature_linear(new_features)
            Hvec = self.batt(Hvec)
            features = self.feature_deal(Hvec, max_length)
            
            # for pair attention
            #features = self.batt(features.view((len(features), -1, self.args.lstm_dim*4))).view((len(features), max_length, max_length, self.args.lstm_dim*4))
            #print(features.size())
            #exit(0)
            Rvec = self._cls_logits(features)
            logits_list.append(Rvec)

        return logits_list

    def forward(self, sentence_tokens, lengths, mask):
        embedding = self._get_embedding(sentence_tokens, mask)
        lstm_feature = self._lstm_feature(embedding, lengths.cpu())

        # self attention
        lstm_feature_attention = self.attention_layer(lstm_feature, lstm_feature, mask[:,:lengths[0]])
        #lstm_feature_attention = self.attention_layer.forward_perceptron(lstm_feature, lstm_feature, mask[:, :lengths[0]])
        lstm_feature = lstm_feature + lstm_feature_attention

        logits = self.multi_hops(lstm_feature, lengths, mask, self.args.nhops)
        return [logits[-1]]

