import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    ReLU,
    UpSampling2D,
    AveragePooling2D,
    GlobalAveragePooling2D,
    Concatenate,
    Activation,
    Reshape
)


def conv3x3(out_planes, stride=1):
    """3x3 convolution with padding"""
    return Conv2D(filters=out_planes, kernel_size=3, strides=stride,
                     padding="same", use_bias=True)


class BasicBlock:
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False):
        self.conv1 = conv3x3(planes, stride)
        self.bn1 = BatchNormalization()
        self.conv2 = conv3x3(planes)
        self.bn2 = BatchNormalization()
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def build_block(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = ReLU()(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            for layer in self.downsample:
                x = layer(x)
            residual = x

        out += residual

        if self.no_relu:
            return out
        else:
            return ReLU()(out)


class Bottleneck:
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=True):
        self.conv1 = Conv2D(filters=planes, kernel_size=1, use_bias=True)
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(filters=planes, kernel_size=3, strides=stride,
                               padding="same", use_bias=True)
        self.bn2 = BatchNormalization()
        self.conv3 = Conv2D(filters=planes * self.expansion, kernel_size=1,
                               use_bias=True)
        self.bn3 = BatchNormalization()
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def build_bottleneck(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = ReLU()(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = ReLU()(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            for layer in self.downsample:
                x = layer(x)
            residual = x

        out += residual
        if self.no_relu:
            return out
        else:
            return ReLU()(out)


class DAPPM:
    def __init__(self, inplanes, branch_planes, outplanes):
        self.scale1 = [
            AveragePooling2D(pool_size=(5, 5), strides=2, padding="same"),
            BatchNormalization(),
            ReLU(),
            Conv2D(filters=branch_planes, kernel_size=1, use_bias=False),
        ]
        self.scale2 = [
            AveragePooling2D(pool_size=(9, 9), strides=4, padding="same"),
            BatchNormalization(),
            ReLU(),
            Conv2D(filters=branch_planes, kernel_size=1, use_bias=False),
        ]
        self.scale3 = [
            AveragePooling2D(pool_size=(17, 17), strides=8, padding="same"),
            BatchNormalization(),
            ReLU(),
            Conv2D(filters=branch_planes, kernel_size=1, use_bias=False),
        ]
        self.scale4 = [
            AveragePooling2D(pool_size=(16, 16), strides=1),
            BatchNormalization(),
            ReLU(),
            Conv2D(filters=branch_planes, kernel_size=1, use_bias=False),     
        ]
        self.scale0 = [
            BatchNormalization(),
            ReLU(),
            Conv2D(filters=branch_planes, kernel_size=1, use_bias=False),     
        ]
        self.process1 = [
            BatchNormalization(),
            ReLU(),
            Conv2D(filters=branch_planes, kernel_size=3, padding="same", use_bias=False),     
        ]
        self.process2 = [
            BatchNormalization(),
            ReLU(),
            Conv2D(filters=branch_planes, kernel_size=3, padding="same", use_bias=False),     
        ]
        self.process3 = [
            BatchNormalization(),
            ReLU(),
            Conv2D(filters=branch_planes, kernel_size=3, padding="same", use_bias=False),                 
        ]
        self.process4 = [
            BatchNormalization(),
            ReLU(),
            Conv2D(filters=branch_planes, kernel_size=3, padding="same", use_bias=False),                             
        ]
        self.compression = [
            BatchNormalization(),
            ReLU(),
            Conv2D(filters=outplanes, kernel_size=1, use_bias=False),
        ]
        self.shortcut = [
            BatchNormalization(),
            ReLU(),
            Conv2D(filters=outplanes, kernel_size=1, use_bias=False),        
        ]

    def build_dappm(self, x):
        x_list = []

        _x = x
        for layer in self.scale0:
            _x = layer(_x)
        x_list.append(_x)

        _x = x
        for layer in self.scale1:
            _x = layer(_x)
        _x = UpSampling2D(size=(2, 2), interpolation="bilinear")(_x)
        for layer in self.process1:
            _x = layer(_x)
        x_list.append(_x + x_list[0])

        _x = x
        for layer in self.scale2:
            _x = layer(_x)
        _x = UpSampling2D(size=(4, 4), interpolation="bilinear")(_x)
        for layer in self.process2:
            _x = layer(_x)
        x_list.append(_x + x_list[1])

        _x = x
        for layer in self.scale3:
            _x = layer(_x)
        _x = UpSampling2D(size=(8, 8), interpolation="bilinear")(_x)
        for layer in self.process3:
            _x = layer(_x)
        x_list.append(_x + x_list[2])

        _x = x
        for layer in self.scale4:
            _x = layer(_x)
        _x = UpSampling2D(size=(16, 16), interpolation="bilinear")(_x)
        for layer in self.process4:
            _x = layer(_x)
        x_list.append(_x + x_list[3])

        cat = Concatenate()(x_list)
        for layer in self.compression:
            cat = layer(cat)
        cut = x
        for layer in self.shortcut:
            cut = layer(cut)
        out = cat + cut

        return out


class segmenthead:
    def __init__(self, inplanes, interplanes, outplanes, scale_factor=8):
        self.bn1 = BatchNormalization()
        self.conv1 = Conv2D(filters=interplanes, kernel_size=3, padding="same", use_bias=False)
        self.bn2 = BatchNormalization()
        self.relu = ReLU()
        self.conv2 = Conv2D(filters=outplanes, kernel_size=1, use_bias=True)
        self.scale_factor = scale_factor

    def build_segmenthead(self, x):
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(x))

        if self.scale_factor is not None:
            height = self.scale_factor
            width = self.scale_factor
            out = UpSampling2D(
                        size=(height, width),
                        interpolation='bilinear')(out)

        return out


class DualResNet:
    def __init__(self, block, layers=[2, 2, 2, 2], num_classes=1, planes=64, spp_planes=128, head_planes=64, input_shape=(1024, 1024, 1), augment=False, padding="same"):
        self.highres_planes = planes * 2
        self.augment = augment

        self.conv1 = [
            Conv2D(filters=planes, kernel_size=3, strides=2, padding=padding, input_shape=input_shape),
            BatchNormalization(),
            ReLU(),
            Conv2D(filters=planes, kernel_size=3, strides=2, padding=padding),
            BatchNormalization(),
            ReLU(),
        ]

        self.relu_layer2 = ReLU()
        self.relu_layer3 = ReLU()
        self.relu_layer4 = ReLU()
        self.relu_layer5 = ReLU()        
        self.relu_compression3 = ReLU()
        self.relu_compression4 = ReLU()                
        self.relu_down3 = ReLU()
        self.relu_down4 = ReLU()                
        self.relu_layer3_ = ReLU()
        self.relu_layer4_ = ReLU()
        self.relu_layer5_ = ReLU()        
        self.layer1 = self._make_layer(block, planes, planes, layers[0])
        self.layer2 = self._make_layer(block, planes, planes*2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, planes*2, planes*4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, planes*4, planes*8, layers[3], stride=2)
        self.layer5 = self._make_layer(Bottleneck, planes*8, planes*8, 1, stride=2)        
        self.compression3 = [
            Conv2D(filters=self.highres_planes,
                   kernel_size=1,
                   use_bias=True),
            BatchNormalization()
        ]
        self.compression4 = [
            Conv2D(filters=self.highres_planes,
                   kernel_size=1,
                   use_bias=True),
            BatchNormalization()
        ]
        self.down3 = [
            Conv2D(filters=planes*4,
                   kernel_size=3,
                   strides=2,
                   padding=padding,
                   use_bias=True),
            BatchNormalization()
        ]
        self.down4 = [
            Conv2D(filters=planes*4,
                   kernel_size=3,
                   strides=2,
                   padding=padding,
                   use_bias=True),
            BatchNormalization(),
            ReLU(),
            Conv2D(filters=planes*8,
                   kernel_size=3,
                   strides=2,
                   padding=padding,
                   use_bias=True),
            BatchNormalization(),
        ]
        self.layer3_ = self._make_layer(block, planes*2, self.highres_planes, 2)
        self.layer4_ = self._make_layer(block, self.highres_planes, self.highres_planes, 2)
        self.layer5_ = self._make_layer(Bottleneck, self.highres_planes, self.highres_planes, 1)
        self.spp = DAPPM(planes*16, spp_planes, planes*4)

        self.final_layer = segmenthead(planes*4, head_planes, num_classes)

    def build_model(self, x):
        layers = []
        # x.shape == (None, 256, 256, 32)
        for layer in self.conv1:
            x = layer(x)

        # x.shape == (None, 256, 256, 32)            
        for layer in self.layer1:
            x = layer.build_block(x)
        layers.append(x)

        # x.shape == (None, 128, 128, 64)
        x = self.relu_layer2(x)
        for layer in self.layer2:
            x = layer.build_block(x)
        layers.append(x)

        # x.shape == (None, 64, 64, 128)            
        x = self.relu_layer3(x)
        for layer in self.layer3:
            x = layer.build_block(x)
        layers.append(x)

        # x_.shape == (None, 128, 128, 64)                    
        x_ = self.relu_layer3_(layers[1])
        for layer in self.layer3_:
            x_ = layer.build_block(x_)

        # x_.shape == (None, 64, 64, 128)
        t = self.relu_down3(x_)
        for layer in self.down3:
            t = layer(t)
        x = x + t

        # x_.shape == (None, 128, 128, 64)
        t = self.relu_compression3(layers[2])
        for layer in self.compression3:
            t = layer(t)
        x_ = x_ + UpSampling2D(
            size=(2, 2),
            interpolation="bilinear")(t)
        
        if self.augment:
            temp = x_

        # x.shape == (None, 32, 32, 256)            
        x = self.relu_layer4(x)
        for layer in self.layer4:
            x = layer.build_block(x)
        layers.append(x)

        # x.shape == (None, 128, 128, 64)                    
        x_ = self.relu_layer4_(x_)
        for layer in self.layer4_:
            x_ = layer.build_block(x_)

        # x.shape == (None, 32, 32, 256)                                
        t = self.relu_down4(x_)
        for layer in self.down4:
            t = layer(t)
        x = x + t

        # x_.shape == (None, 128, 128, 64)
        t = self.relu_compression4(layers[3])
        for layer in self.compression4:
            t = layer(t)
        x_ = x_ + UpSampling2D(
            size=(4, 4),
            interpolation="bilinear")(t)

        # x_.shape == (None, 128, 128, 128)        
        x_ = self.relu_layer5_(x_)
        for layer in self.layer5_:
            x_ = layer.build_bottleneck(x_)

        # x.shape == (None, 16, 16, 512)                    
        x = self.relu_layer5(x)
        for layer in self.layer5:
            x = layer.build_bottleneck(x)

        # x.shape == (None, 128, 128, 128)                                
        t = self.spp.build_dappm(x)
        x = UpSampling2D(
            size=(8, 8),
            interpolation="bilinear")(t)

        # x.shape == (None, 1024, 1024, 1)
        out = self.final_layer.build_segmenthead(x + x_)
        out = Activation("sigmoid")(out)

        return out

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = [
                Conv2D(
                    filters=planes * block.expansion,
                    kernel_size=1,
                    strides=stride,
                    use_bias=True
                ),
                BatchNormalization(),                
            ]
            
        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == (blocks-1):
                layers.append(block(inplanes, planes, stride=1, no_relu=True))
            else:
                layers.append(block(inplanes, planes, stride=1, no_relu=False))

        return layers



def _get_stub_dataset():
    x = np.random.randn(3*1024*1024*1)
    x = x.reshape((3, 1024, 1024, 1))
    return x, x
    

def train():
    input_shape = (1024, 1024, 1)
    model = DualResNet(BasicBlock, [2, 2, 2, 2], num_classes=1, planes=32, spp_planes=128, head_planes=64, input_shape=input_shape)
    input_x = Input(shape=input_shape)
    out = model.build_model(input_x)
    model = tf.keras.models.Model(inputs=input_x, outputs=out)
    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics="accuracy"
    )
    model.summary()

    x, y = _get_stub_dataset()
    history = model.fit(
        x=x,
        y=y,
        epochs=1,
        batch_size=1,
        steps_per_epoch=len(x),
        verbose=1,
    )

    save_path = "trained_model"
    tf.saved_model.save(model, save_path)


if __name__ == "__main__":
    train()

    
        
    
