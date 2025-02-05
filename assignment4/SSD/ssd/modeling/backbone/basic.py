import torch
from torch import nn



class BasicModel(torch.nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
     where "output_channels" is the same as cfg.BACKBONE.OUT_CHANNELS
    """
    def __init__(self, cfg):
        super().__init__()
        output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_channels = output_channels
        image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
        self.output_feature_shape = cfg.MODEL.PRIORS.FEATURE_MAPS

        self.layer_1 = nn.Sequential(
                                    nn.Conv2d(in_channels=image_channels,stride=1, padding = 1,out_channels=32,kernel_size=3),
                                    nn.MaxPool2d(stride=2,kernel_size=2),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=32,stride=1, padding = 1, out_channels=64,kernel_size=3),
                                    nn.MaxPool2d(stride=2,kernel_size=2),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=64,stride=1, padding = 1, out_channels=64,kernel_size=3),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels = 64, stride = 2, padding = 1, out_channels = self.output_channels[0], kernel_size = 3)
        )
        self.layer_2 = nn.Sequential(
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels = self.output_channels[0], stride = 1, padding = 1, out_channels = 256, kernel_size = 3),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels = 256, stride = 2, padding = 1, out_channels = self.output_channels[1], kernel_size = 3)
        )
            
        self.layer_3 = nn.Sequential(
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels = self.output_channels[1], stride = 1, padding = 1, out_channels = 256, kernel_size = 3),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels = 256, stride = 2, padding = 1, out_channels = self.output_channels[2], kernel_size = 3)
        )
        self.layer_4 = nn.Sequential(
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels = self.output_channels[2], stride = 1, padding = 1, out_channels = 128, kernel_size = 3),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels = 128, stride = 2, padding = 1, out_channels = self.output_channels[3], kernel_size = 3)
        )
        self.layer_5 = nn.Sequential(
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels = self.output_channels[3], padding = 1, stride = 1, out_channels = 128, kernel_size = 3),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels = 128, stride = 2, padding = 1, out_channels = self.output_channels[4], kernel_size = 3)
        )
        self.layer_6 = nn.Sequential(
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels = self.output_channels[4], stride = 1, padding = 1, out_channels = 128, kernel_size = 3),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels = 128, stride = 2, padding = 0, out_channels = self.output_channels[5], kernel_size = 3)
        
        )
        

    def forward(self, x):
        """
        The forward function should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """
        out_features = []
        out = x

        out  = self.layer_1(out)
        out_features.append(out)

        out = self.layer_2(out)
        out_features.append(out)
        
        out = self.layer_3(out)
        out_features.append(out)

        out = self.layer_4(out)
        out_features.append(out)

        out = self.layer_5(out)
        out_features.append(out)

        out = self.layer_6(out)
        out_features.append(out)
        

        for idx, feature in enumerate(out_features):
            w, h = self.output_feature_shape[idx]
            out_channel = self.output_channels[idx]
            expected_shape = (out_channel, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        return tuple(out_features)



