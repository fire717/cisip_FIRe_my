import torch.nn as nn
import torch.nn.functional as F
from models import register_network, BaseArch
from models.architectures.helper import get_hash_fc_with_normalizations, get_backbone
from models.layers.cossim import CosSim


@register_network('orthohash')
class ArchOrthoHash(BaseArch):
    """Arch OrthoHash"""

    def __init__(self, config, n_attr=0, **kwargs):
        super(ArchOrthoHash, self).__init__(config, **kwargs)

        hash_layer = config['loss_param'].get('hash_layer', 'identity')
        hash_kwargs = config['loss_param']
        cossim_ce = config['loss_param'].get('cossim_ce', True)
        learn_cent = config['loss_param'].get('learn_cent', True)

        self.backbone = get_backbone(backbone=self.backbone_name,
                                     nbit=self.nbit,
                                     nclass=self.nclass,
                                     pretrained=self.pretrained,
                                     freeze_weight=self.freeze_weight, **kwargs)

        if cossim_ce:
            self.ce_fc = CosSim(self.nbit, self.nclass, learn_cent)
        else:
            self.ce_fc = nn.Linear(self.nbit, self.nclass)

        self.hash_fc = get_hash_fc_with_normalizations(in_features=self.backbone.in_features,
                                                       nbit=self.nbit,
                                                       bias=self.bias,
                                                       kwargs=hash_kwargs)
        #nips2020
        self.softmax = nn.Softmax(dim=1)
        self.attr_conv = nn.Conv2d(2048, n_attr, 1, 1) #2048 for resnet101

    def get_features_params(self):
        return self.backbone.get_features_params()

    def get_hash_params(self):
        return list(self.ce_fc.parameters()) + list(self.hash_fc.parameters())

    def forward(self, x, attribute):
        x,features = self.backbone(x)

        ### add 2020nips attr branch
        attention = self.attr_conv(features) #64,85,7,7
        pre_attri = F.avg_pool2d(attention, kernel_size=7).view(features.shape[0], -1).double() #64,85
        pre_class = self.softmax(pre_attri.mm(attribute))#64x50

        v = self.hash_fc(x)
        u = self.ce_fc(v)
        return u, v, pre_attri, pre_class,attention
