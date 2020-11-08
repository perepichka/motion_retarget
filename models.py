"""Module defining Transmomo models. """

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class ConvEncoder(nn.Module):

    @classmethod
    def build_from_config(cls, config):
        conv_pool = None if config.conv_pool is None else getattr(nn, config.conv_pool)
        global_pool = None if config.global_pool is None else getattr(nn, config.global_pool)
        encoder = cls(config.channels, config.padding, config.kernel_size, config.conv_stride, conv_pool, global_pool)
        return encoder

    def __init__(self, channels, padding=3, kernel_size=8, conv_stride=2, conv_pool=None, global_pool=None):
        super(ConvEncoder, self).__init__()

        self.in_channels = channels[0]
        self.global_pool = getattr(F, config.body_encoder.global_pool) if global_pool is not None else None

        model = []
        acti = nn.LeakyReLU(0.2)

        nr_layer = len(channels) - 1


        for i in range(nr_layer):
            if conv_pool is None:
                model.append(nn.ReflectionPad1d(padding))
                model.append(nn.Conv1d(channels[i], channels[i+1], kernel_size=kernel_size, stride=conv_stride))
                model.append(acti)
            else:
                model.append(nn.ReflectionPad1d(padding))
                model.append(nn.Conv1d(channels[i], channels[i+1], kernel_size=kernel_size, stride=conv_stride))
                model.append(acti)
                model.append(conv_pool(kernel_size=2, stride=2))

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = x[:, :self.in_channels, :]
        x = self.model(x)

        # Final global pooling if required
        if self.global_pool is not None:
            kernel_size = body_code_seq.size(-1)
            x = self.global_pool(x, kernel_size)

        return x


class ConvDecoder(nn.Module):

    @classmethod
    def build_from_config(cls, config):
        decoder = cls(config.channels, config.kernel_size)
        return decoder

    def __init__(self, channels, kernel_size=7):
        super(ConvDecoder, self).__init__()

        model = []
        pad = (kernel_size - 1) // 2
        acti = nn.LeakyReLU(0.2)

        for i in range(len(channels) - 1):
            model.append(nn.Upsample(scale_factor=2, mode='nearest'))
            model.append(nn.ReflectionPad1d(pad))
            model.append(nn.Conv1d(channels[i], channels[i + 1],
                                            kernel_size=kernel_size, stride=1))
            if i == 0 or i == 1:
                model.append(nn.Dropout(p=0.2))
            if not i == len(channels) - 2:
                model.append(acti)

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):

    def __init__(self, *args, **kwargs):
        pass

    def forward(self, x):
        pass


class Autoencoder3f(nn.Module):


    def __init__(self, config):
        super(Autoencoder3f, self).__init__()

        assert config.motion_encoder.channels[-1] + config.body_encoder.channels[-1] + \
               config.view_encoder.channels[-1] == config.decoder.channels[0]

        self.n_joints = config.decoder.channels[-1] // 3
        self.body_reference = config.body_reference

        motion_cls = getattr(thismodule, config.motion_encoder.cls)
        body_cls = getattr(thismodule, config.body_encoder.cls)
        view_cls = getattr(thismodule, config.view_encoder.cls)

        self.encode_motion = motion_cls.build_from_config(config.motion_encoder)
        self.encode_body = body_cls.build_from_config(config.body_encoder)
        self.encode_view = view_cls.build_from_config(config.view_encoder)

        self.decoder = ConvDecoder.build_from_config(config.decoder)


    def forward(self, seqs):
        motion_code = self.encode_motion(x)
        body_code, _ = self.encode_body(x)
        view_code, _ = self.encode_view(x)
        out = self.decode(motion_code, body_code, view_code)
        out = rotate_and_maybe_project(out, body_reference=self.body_reference, project_2d=True)
        return self.reconstruct(seqs)


    def decode(self, motion_code, body_code, view_code):
        if body_code.size(-1) == 1:
            body_code = body_code.repeat(1, 1, motion_code.shape[-1])
        if view_code.size(-1) == 1:
            view_code = view_code.repeat(1, 1, motion_code.shape[-1])
        complete_code = torch.cat([motion_code, body_code, view_code], dim=1)
        out = self.decoder(complete_code)
        return out


if __name__ == '__main__':
    pass
