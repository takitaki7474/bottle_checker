import chainer
import chainer.links as L
import chainer.functions as F

class MyNet_8(chainer.Chain):

    def __init__(self, n_out=11):
        super(MyNet_8, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 32, 3, 3, 1)
            self.conv2 = L.Convolution2D(32, 32, 3, 3, 1)
            self.conv3 = L.Convolution2D(32, 32, 3, 3, 1)
            self.conv4 = L.Convolution2D(32, 32, 3, 3, 1)
            self.conv5 = L.Convolution2D(32, 32, 3, 3, 1)
            self.conv6 = L.Convolution2D(32, 32, 3, 3, 1)
            self.conv7 = L.Convolution2D(32, 32, 3, 3, 1)
            self.conv8 = L.Convolution2D(32, 32, 3, 3, 1)
            self.fc4 = L.Linear(None, 512)
            self.fc5 = L.Linear(512, n_out)

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 2)
        h = F.relu(self.conv3(h))
        h = F.max_pooling_2d(F.relu(self.conv4(h)), 2)
        h = F.relu(self.conv5(h))
        h = F.max_pooling_2d(F.relu(self.conv6(h)), 2)
        h = F.relu(self.conv7(h))
        h = F.max_pooling_2d(F.relu(self.conv8(h)), 2)
        h = F.dropout(F.relu(self.fc4(h)))
        h = self.fc5(h)
        return h

class MyNet_6(chainer.Chain):

    def __init__(self, n_out):
        super(MyNet_6, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 32, 3, 3, 1)
            self.conv2 = L.Convolution2D(32, 32, 3, 3, 1)
            self.conv3 = L.Convolution2D(32, 32, 3, 3, 1)
            self.conv4 = L.Convolution2D(32, 32, 3, 3, 1)
            self.conv5 = L.Convolution2D(32, 32, 3, 3, 1)
            self.conv6 = L.Convolution2D(32, 32, 3, 3, 1)
            self.fc4 = L.Linear(None, 512)
            self.fc5 = L.Linear(512, n_out)

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 2)
        h = F.relu(self.conv3(h))
        h = F.max_pooling_2d(F.relu(self.conv4(h)), 2)
        h = F.relu(self.conv5(h))
        h = F.max_pooling_2d(F.relu(self.conv6(h)), 2)
        h = F.dropout(F.relu(self.fc4(h)))
        h = self.fc5(h)
        return h

class ConvBlock(chainer.Chain):

    def __init__(self, n_ch, pool_drop=False):
        w = chainer.initializers.HeNormal()
        super(ConvBlock, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(None, n_ch, 3, 1, 1, nobias=True, initialW=w)
            self.bn = L.BatchNormalization(n_ch)
        self.pool_drop = pool_drop

    def __call__(self, x):
        h = F.relu(self.bn(self.conv(x)))
        if self.pool_drop:
            h = F.max_pooling_2d(h, 2, 2)
            h = F.dropout(h, ratio=0.25)
        return h

class LinearBlock(chainer.Chain):

    def __init__(self, drop=False):
        w = chainer.initializers.HeNormal()
        super(LinearBlock, self).__init__()
        with self.init_scope():
            self.fc = L.Linear(None, 1024, initialW=w)
        self.drop = drop

    def __call__(self, x):
        h = F.relu(self.fc(x))
        if self.drop:
            h = F.dropout(h)
        return h

class DeepCNN(chainer.ChainList):

    def __init__(self, n_output):
        super(DeepCNN, self).__init__(
            ConvBlock(64),
            ConvBlock(64, True),
            ConvBlock(128),
            ConvBlock(128, True),
            ConvBlock(256),
            ConvBlock(256),
            ConvBlock(256),
            ConvBlock(256, True),
            LinearBlock(),
            LinearBlock(),
            L.Linear(None, n_output)
        )

    def __call__(self, x):
        for f in self:
            x = f(x)
        return x
