import torch.nn as nn

class EmbedNet(nn.Module):
    def __init__(self, backbone, model):
        super(EmbedNet, self).__init__()
        self.backbone = backbone
        self.model = model

    def forward(self, x):
        x = self.backbone(x)
        embedded_x = self.model(x)
        return embedded_x

class TripletNet(nn.Module):
    def __init__(self, embed_net):
        super(TripletNet, self).__init__()
        self.embed_net = embed_net

    def forward(self, a, p, n):
        embedded_a = self.embed_net(a)
        embedded_p = self.embed_net(p)
        embedded_n = self.embed_net(n)
        return embedded_a, embedded_p, embedded_n

    def feature_extract(self, x):
        return self.embed_net(x)

class QuadrupletNet(nn.Module):
    def __init__(self, embed_net):
        super(QuadrupletNet, self).__init__()
        self.embed_net = embed_net

    def forward(self, a, p, n1, n2):
        embedded_a = self.embed_net(a)
        embedded_p = self.embed_net(p)
        embedded_n1 = self.embed_net(n1)
        embedded_n2 = self.embed_net(n2)
        return embedded_a, embedded_p, embedded_n1, embedded_n2

    def feature_extract(self, x):
        return self.embed_net(x)