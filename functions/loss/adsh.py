# Angular deep supervised hashing for image retrieval
import logging

import torch
import torch.nn.functional as F

from functions.loss.base_cls import BaseClassificationLoss


class ADSHLoss(BaseClassificationLoss):
    def __init__(self, alpha=1, beta=1, s=10, m=0.2, multiclass=False, method='cosface', **kwargs):
        super().__init__()
        logging.info("Loss parameters are: ", locals())
        self.alpha = alpha
        self.beta = beta
        self.s = s
        self.m = m

        self.method = method
        self.multiclass = multiclass
        self.weight = None


        #nips2020
        self.criterion = torch.nn.CrossEntropyLoss()
        self.criterion_regre = torch.nn.MSELoss()


    def get_hamming_distance(self):
        assert self.weight is not None, 'please set weights before calling this function'

        b = torch.div(self.weight, self.weight.norm(p=2, dim=-1, keepdim=True) + 1e-7)
        b = torch.tanh(b)

        nbit = b.size(1)

        hd = 0.5 * (nbit - b @ b.t())
        return hd

    # def get_margin_logits(self, logits, labels):
    #     y_onehot = torch.ones_like(logits)
    #     y_onehot.scatter_(1, torch.unsqueeze(labels, dim=-1), self.m)
    #
    #     norm = logits.norm(p=2, dim=-1, keepdim=True)
    #     theta = torch.div(logits, norm + 1e-7).acos().clamp(-0.999999, 0.999999)
    #
    #     theta += ()
    #
    #     margin_logits = y_onehot * logits
    #
    #     return margin_logits

    def get_margin_logits(self, logits, labels):
        y_onehot = torch.zeros_like(logits)
        y_onehot.scatter_(1, torch.unsqueeze(labels, dim=-1), self.m)

        if self.method == 'arcface':
            arc_logits = torch.acos(logits.clamp(-0.99999, 0.99999))
            logits = torch.cos(arc_logits + y_onehot)
            margin_logits = self.s * logits
        else:
            y_onehot = torch.zeros_like(logits)
            y_onehot.scatter_(1, torch.unsqueeze(labels, dim=-1), self.m)
            margin_logits = self.s * (logits - y_onehot)

        return margin_logits

    def get_dynamic_logits(self, logits, code_logits, labels):
        """
        return dynamic generated labels
        :param logits:
        :param code_logits:
        :param labels:
        :return:
        """
        assert self.weight is not None
        new_weight = []
        new_labels = F.one_hot(torch.tensor([labels.size(1)] * labels.size(0)), labels.size(1) + 1).to(labels.device)
        for l in range(labels.size(0)):
            w = self.weight[labels[l].bool()].mean(dim=0, keepdim=True)  # (L,D)
            new_weight.append(w)
        new_weight = torch.cat(new_weight, dim=0)  # (BS, D)

        code_logits_norm = torch.div(code_logits, code_logits.norm(p=2, dim=1, keepdim=True) + 1e-7)
        new_weight_norm = torch.div(new_weight, new_weight.norm(p=2, dim=1, keepdim=True) + 1e-7)
        new_logits = (code_logits_norm * new_weight_norm).sum(dim=1, keepdim=True)  # (BS, D) * (BS, D) -> (BS, 1)
        new_logits = torch.cat([logits, new_logits], dim=1)
        return new_logits, new_labels

    def nips2020attrloss(self,pre_attri, pre_class,label_a,label_v,attention,middle_graph):
        #xe 1 --attri 1e-4   regular 0.0005
        #l_xe 0 --l_attri 1e-2 --l_regular 0.5e-6
        #cpt 2e-9
        xe = 1
        attri = 1e-4
        cpt = 2e-9
        loss = 0
        if xe > 0:
            loss_xe = xe * self.criterion(pre_class, label_v)
            loss = loss_xe

        if attri > 0:
            loss_attri = attri * self.criterion_regre(pre_attri, label_a)
            loss += loss_attri

        if cpt > 0:
            batch_size, attri_dim, map_dim, map_dim = attention.size()
            peak_id = torch.argmax(attention.view(batch_size * attri_dim, -1), dim=1)
            peak_mask = middle_graph[peak_id, :, :].view(batch_size, attri_dim, map_dim, map_dim)
            cpt_loss = cpt * torch.sum(torch.sigmoid(attention) * peak_mask)
            loss += cpt_loss

        return loss

    def forward(self, logits, code_logits, labels, 
        pre_attri, pre_class,label_a,label_v,attention,middle_graph,
        onehot=True):
        if self.multiclass:
            logits, labels = self.get_dynamic_logits(logits, code_logits, labels)
            labels = labels.argmax(1)
        else:
            if onehot:
                labels = labels.argmax(1)


        ### nips2020
        attr_w = 1#2for cub  0.5for awa2  2for sun
        self.attrloss = self.nips2020attrloss(pre_attri, pre_class,label_a,label_v,attention,middle_graph)


        margin_logits = self.get_margin_logits(logits, labels)
        ce = F.cross_entropy(margin_logits, labels)

        hd = self.get_hamming_distance()
        triu = torch.ones_like(hd).bool()
        triu = torch.triu(triu, 1)

        hd_triu = hd[triu]

        meanhd = torch.mean(-hd_triu)
        varhd = torch.var(hd_triu)

        self.losses['ce'] = ce
        self.losses['meanhd'] = meanhd
        self.losses['varhd'] = varhd

        loss = ce + self.alpha * meanhd + self.beta * varhd +attr_w*self.attrloss
        return loss
