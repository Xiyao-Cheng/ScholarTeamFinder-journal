#!/usr/bin/python3
from __future__ import absolute_import, division, print_function

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import collections


def Identity(x):
    return x


# ──────────────────────────────────────────────────────────────────────
# Box intersection modules
# ──────────────────────────────────────────────────────────────────────

class BoxOffsetIntersection(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layer1 = nn.Linear(dim, dim)
        self.layer2 = nn.Linear(dim, dim)
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, embeddings):
        layer1_act = F.relu(self.layer1(embeddings))
        gate = torch.sigmoid(self.layer2(torch.mean(layer1_act, dim=0)))
        offset, _ = torch.min(embeddings, dim=0)
        return offset * gate


class CenterIntersectionWithLSTM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lstm  = nn.LSTM(dim, dim // 2, bidirectional=True, batch_first=False)
        self.layer1 = nn.Linear(dim, dim)
        self.layer2 = nn.Linear(dim, dim)
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, embeddings):
        embeddings, _ = self.lstm(embeddings)
        layer1_act = F.relu(self.layer1(embeddings))
        attention  = F.log_softmax(self.layer2(layer1_act), dim=0)
        return torch.sum(attention * embeddings, dim=0)


class Regularizer:
    def __init__(self, base_add, min_val, max_val):
        self.base_add = base_add
        self.min_val  = min_val
        self.max_val  = max_val

    def __call__(self, x):
        return torch.clamp(x + self.base_add, self.min_val, self.max_val)


# ──────────────────────────────────────────────────────────────────────
# Text Query Encoder
# ──────────────────────────────────────────────────────────────────────

class TextQueryEncoder(nn.Module):
    def __init__(self, hidden_dim, model_name='allenai/scibert_scivocab_uncased',
                 freeze_bert=True, gamma=12.0, epsilon=2.0):
        super().__init__()
        self.bert       = AutoModel.from_pretrained(model_name)
        self.proj       = nn.Linear(self.bert.config.hidden_size, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        init_scale = (gamma + epsilon) / hidden_dim   
        self.scale = nn.Parameter(torch.tensor(init_scale))

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls     = outputs.last_hidden_state[:, 0, :]   # [CLS] token
        projected = self.layer_norm(self.proj(cls))
        return torch.tanh(projected) * self.scale.abs()

    def unfreeze(self):
        for param in self.bert.parameters():
            param.requires_grad = True


# ──────────────────────────────────────────────────────────────────────
# Main Model
# ──────────────────────────────────────────────────────────────────────

class KGReasoning(nn.Module):
    def __init__(self, nentity, nrelation, hidden_dim, gamma,
                 geo='box', test_batch_size=1, box_mode=None,
                 use_cuda=True, query_name_dict=None,
                 freeze_bert=True,
                 scibert_model='allenai/scibert_scivocab_uncased'):
        super().__init__()

        self.nentity        = nentity
        self.nrelation      = nrelation
        self.hidden_dim     = hidden_dim
        self.epsilon        = 2.0
        self.geo            = geo
        self.use_cuda       = use_cuda
        self.device         = torch.device('cuda' if use_cuda else 'cpu')
        self.query_name_dict = query_name_dict

        self.gamma = nn.Parameter(torch.Tensor([gamma]), requires_grad=False)
        self.embedding_range = nn.Parameter(
            torch.Tensor([(gamma + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        # ── Scholar (entity) embeddings ──────────────────────────────
        self.entity_dim   = hidden_dim
        self.relation_dim = hidden_dim

        self.entity_embedding = nn.Parameter(
            torch.zeros(nentity, self.entity_dim)
        )
        nn.init.uniform_(self.entity_embedding,
                         -self.embedding_range.item(),
                          self.embedding_range.item())

        self.relation_embedding = nn.Parameter(
            torch.zeros(nrelation, self.relation_dim)
        )
        nn.init.uniform_(self.relation_embedding,
                         -self.embedding_range.item(),
                          self.embedding_range.item())

        # ── Box geometry ─────────────────────────────────────────────
        if geo == 'box':
            activation = 'none'
            cen        = 0.02
            if isinstance(box_mode, (tuple, list)) and len(box_mode) == 2:
                activation, cen = box_mode
            self.cen = cen
            self.func = {'none': Identity, 'relu': F.relu,
                         'softplus': F.softplus}.get(activation, Identity)

            self.offset_embedding = nn.Parameter(
                torch.zeros(nrelation, self.entity_dim)
            )
            nn.init.uniform_(self.offset_embedding, 0.,
                             self.embedding_range.item())
            self.center_net = CenterIntersectionWithLSTM(self.entity_dim)
            self.offset_net = BoxOffsetIntersection(self.entity_dim)

        # ── ★ Text query encoder ──────────────────────────────────────
        self.text_encoder = TextQueryEncoder(
            hidden_dim  = hidden_dim,
            model_name  = scibert_model,
            freeze_bert = freeze_bert,
            gamma       = gamma,          
            epsilon     = self.epsilon,   # epsilon = 2.0
        )

        t = torch.arange(nentity).float().repeat(test_batch_size, 1)
        self.batch_entity_range = t.cuda() if use_cuda else t

    # ── Text encoding helper ──────────────────────────────────────────

    def encode_query_text(self, input_ids, attention_mask):
        return self.text_encoder(input_ids, attention_mask)

    def unfreeze_bert(self):
        self.text_encoder.unfreeze()

    # ── Box distance ──────────────────────────────────────────────────

    def cal_logit_box(self, entity_emb, query_center, query_offset):
        delta        = (entity_emb - query_center).abs()
        distance_out = F.relu(delta - query_offset)
        distance_in  = torch.min(delta, query_offset)
        return (self.gamma
                - torch.norm(distance_out, p=1, dim=-1)
                - self.cen * torch.norm(distance_in, p=1, dim=-1))

    # ── Embed query box given TEXT center ────────────────────────────

    def embed_query_box_with_text(self, query_text_emb, queries,
                                   query_structure, idx):
        
        all_relation_flag = all(e in ['r', 'n']
                                for e in query_structure[-1])
        if all_relation_flag:
            if query_structure[0] == 'e':
                embedding        = query_text_emb
                offset_embedding = torch.zeros_like(embedding)
                idx += 1   
            else:
                embedding, offset_embedding, idx = \
                    self.embed_query_box_with_text(
                        query_text_emb, queries, query_structure[0], idx)

            for rel_type in query_structure[-1]:
                if rel_type == 'n':
                    assert False, "box cannot handle negation"
                
                rel_idx   = queries[:, idx]
                r_emb     = torch.index_select(self.relation_embedding, 0, rel_idx)
                r_off_emb = torch.index_select(self.offset_embedding,   0, rel_idx)
                embedding       += r_emb
                offset_embedding += self.func(r_off_emb)
                idx += 1
        else:
            emb_list, off_list = [], []
            for sub_struct in query_structure:
                emb, off, idx = self.embed_query_box_with_text(
                    query_text_emb, queries, sub_struct, idx)
                emb_list.append(emb)
                off_list.append(off)
            embedding        = self.center_net(torch.stack(emb_list))
            offset_embedding = self.offset_net(torch.stack(off_list))

        return embedding, offset_embedding, idx


    # ── Forward ───────────────────────────────────────────────────────

    def forward(self, positive_sample, negative_sample, subsampling_weight,
                batch_queries_dict, batch_idxs_dict,
                input_ids=None, attention_mask=None):
        assert self.geo == 'box', "Only box geometry supported"
        return self.forward_box(
            positive_sample, negative_sample, subsampling_weight,
            batch_queries_dict, batch_idxs_dict,
            input_ids, attention_mask
        )

    def forward_box(self, positive_sample, negative_sample,
                    subsampling_weight, batch_queries_dict, batch_idxs_dict,
                    input_ids=None, attention_mask=None):

        # ── Encode query text once for the whole batch ────────────────
        if input_ids is not None:
            query_text_emb = self.encode_query_text(input_ids, attention_mask)
        else:
            # fallback：zero vector
            bs = sum(len(v) for v in batch_idxs_dict.values())
            query_text_emb = torch.zeros(bs, self.hidden_dim,
                                         device=self.device)

        all_center_embs, all_offset_embs, all_idxs = [], [], []
        all_union_center_embs, all_union_offset_embs, all_union_idxs = [], [], []

        for query_structure in batch_queries_dict:
            idxs    = batch_idxs_dict[query_structure]
            queries = batch_queries_dict[query_structure]
            text_emb_slice = query_text_emb[idxs]

            if 'u' in self.query_name_dict.get(query_structure, ''):
                center_emb, offset_emb, _ = self.embed_query_box_with_text(
                    text_emb_slice,
                    self.transform_union_query(queries, query_structure),
                    self.transform_union_structure(query_structure), 0)
                all_union_center_embs.append(center_emb)
                all_union_offset_embs.append(offset_emb)
                all_union_idxs.extend(idxs)
            else:
                center_emb, offset_emb, _ = self.embed_query_box_with_text(
                    text_emb_slice, queries, query_structure, 0)
                all_center_embs.append(center_emb)
                all_offset_embs.append(offset_emb)
                all_idxs.extend(idxs)

        # ── Concat embeddings ─────────────────────────────────────────
        if all_center_embs:
            all_center_embs = torch.cat(all_center_embs, 0).unsqueeze(1)
            all_offset_embs = torch.cat(all_offset_embs, 0).unsqueeze(1)
        if all_union_center_embs:
            all_union_center_embs = torch.cat(all_union_center_embs, 0).unsqueeze(1)
            all_union_offset_embs = torch.cat(all_union_offset_embs, 0).unsqueeze(1)
            n = all_union_center_embs.shape[0] // 2
            all_union_center_embs = all_union_center_embs.view(n, 2, 1, -1)
            all_union_offset_embs = all_union_offset_embs.view(n, 2, 1, -1)

        if subsampling_weight is not None:
            subsampling_weight = subsampling_weight[all_idxs + all_union_idxs]

        # ── Positive logit ────────────────────────────────────────────
        if positive_sample is not None:
            if all_center_embs is not None and len(all_center_embs):
                pos_emb = torch.index_select(
                    self.entity_embedding, 0,
                    positive_sample[all_idxs]).unsqueeze(1)
                positive_logit = self.cal_logit_box(
                    pos_emb, all_center_embs, all_offset_embs)
            else:
                positive_logit = torch.Tensor([]).to(self.device)

            if len(all_union_idxs):
                pos_emb_u = torch.index_select(
                    self.entity_embedding, 0,
                    positive_sample[all_union_idxs]).unsqueeze(1).unsqueeze(1)
                pos_union_logit = self.cal_logit_box(
                    pos_emb_u, all_union_center_embs, all_union_offset_embs)
                pos_union_logit = torch.max(pos_union_logit, dim=1)[0]
            else:
                pos_union_logit = torch.Tensor([]).to(self.device)

            positive_logit = torch.cat([positive_logit, pos_union_logit], 0)
        else:
            positive_logit = None

        # ── Negative logit ────────────────────────────────────────────
        if negative_sample is not None:
            if all_center_embs is not None and len(all_center_embs):
                neg_reg = negative_sample[all_idxs]
                bs, ns  = neg_reg.shape
                neg_emb = torch.index_select(
                    self.entity_embedding, 0,
                    neg_reg.view(-1).long()).view(bs, ns, -1)
                negative_logit = self.cal_logit_box(
                    neg_emb, all_center_embs, all_offset_embs)
            else:
                negative_logit = torch.Tensor([]).to(self.device)

            if len(all_union_idxs):
                neg_u  = negative_sample[all_union_idxs]
                bs, ns = neg_u.shape
                neg_emb_u = torch.index_select(
                    self.entity_embedding, 0,
                    neg_u.view(-1).long()).view(bs, 1, ns, -1)
                neg_union_logit = self.cal_logit_box(
                    neg_emb_u, all_union_center_embs, all_union_offset_embs)
                neg_union_logit = torch.max(neg_union_logit, dim=1)[0]
            else:
                neg_union_logit = torch.Tensor([]).to(self.device)

            negative_logit = torch.cat([negative_logit, neg_union_logit], 0)
        else:
            negative_logit = None

        return positive_logit, negative_logit, subsampling_weight, \
               all_idxs + all_union_idxs

    def transform_union_query(self, queries, query_structure):
        name = self.query_name_dict.get(query_structure, '')
        if name == '2u-DNF':
            queries = queries[:, :-1]
        elif name == 'up-DNF':
            queries = torch.cat([
                torch.cat([queries[:, :2], queries[:, 5:6]], 1),
                torch.cat([queries[:, 2:4], queries[:, 5:6]], 1)
            ], 0)
        return queries.reshape(queries.shape[0] * 2, -1)

    def transform_union_structure(self, query_structure):
        name = self.query_name_dict.get(query_structure, '')
        if name == '2u-DNF':
            return ('e', ('r',))
        elif name == 'up-DNF':
            return ('e', ('r', 'r'))
