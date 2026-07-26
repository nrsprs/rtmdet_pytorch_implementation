import pytest
import torch

from rtmdet.layers.channel_attention import ChannelAttention
from rtmdet.layers.conv_module import ConvModule, DWConvModule
from rtmdet.layers.csp_layer import CSPLayer
from rtmdet.layers.csp_next_block import CSPNextBlock
from rtmdet.layers.sppf import SPFFBottleneck


class TestConvModule:
    def test_basic(self):
        m = ConvModule(64, 128, kernel_size=3, stride=1)
        x = torch.randn(2, 64, 32, 32)
        out = m(x)
        assert out.shape == (2, 128, 32, 32)

    def test_stride2(self):
        m = ConvModule(64, 128, kernel_size=3, stride=2)
        x = torch.randn(2, 64, 32, 32)
        out = m(x)
        assert out.shape == (2, 128, 16, 16)

    def test_kernel1(self):
        m = ConvModule(64, 128, kernel_size=1)
        x = torch.randn(2, 64, 32, 32)
        out = m(x)
        assert out.shape == (2, 128, 32, 32)

    def test_auto_padding(self):
        m = ConvModule(64, 128, kernel_size=3, stride=1, padding=None)
        x = torch.randn(2, 64, 32, 32)
        out = m(x)
        assert out.shape == (2, 128, 32, 32)

    def test_explicit_padding(self):
        m = ConvModule(64, 128, kernel_size=3, stride=1, padding=0)
        x = torch.randn(2, 64, 32, 32)
        out = m(x)
        assert out.shape == (2, 128, 30, 30)

    def test_channel_change(self):
        m = ConvModule(3, 32, kernel_size=3, stride=1)
        x = torch.randn(2, 3, 64, 64)
        out = m(x)
        assert out.shape == (2, 32, 64, 64)


class TestDWConvModule:
    def test_basic(self):
        m = DWConvModule(64, 64)
        x = torch.randn(2, 64, 32, 32)
        out = m(x)
        assert out.shape == (2, 64, 32, 32)

    def test_output_channels(self):
        m = DWConvModule(64, 64)
        x = torch.randn(2, 64, 32, 32)
        out = m(x)
        assert out.shape == (2, 64, 32, 32)


class TestCSPNextBlock:
    def test_with_residual(self):
        m = CSPNextBlock(64, add=True)
        x = torch.randn(2, 64, 32, 32)
        out = m(x)
        assert out.shape == (2, 64, 32, 32)
        assert not torch.allclose(out, x)

    def test_without_residual(self):
        m = CSPNextBlock(64, add=False)
        x = torch.randn(2, 64, 32, 32)
        out = m(x)
        assert out.shape == (2, 64, 32, 32)

    def test_preserves_shape(self):
        m = CSPNextBlock(128, add=True)
        x = torch.randn(4, 128, 16, 16)
        out = m(x)
        assert out.shape == x.shape


class TestCSPLayer:
    def test_equal_channels(self):
        m = CSPLayer(128, 128, n=3, add=True)
        x = torch.randn(2, 128, 32, 32)
        out = m(x)
        assert out.shape == (2, 128, 32, 32)

    def test_different_channels(self):
        m = CSPLayer(64, 128, n=3, add=False)
        x = torch.randn(2, 64, 32, 32)
        out = m(x)
        assert out.shape == (2, 128, 32, 32)

    def test_n_blocks_param_count(self):
        m1 = CSPLayer(64, 64, n=1, add=True)
        m3 = CSPLayer(64, 64, n=3, add=True)
        params1 = sum(p.numel() for p in m1.parameters())
        params3 = sum(p.numel() for p in m3.parameters())
        assert params3 > params1

    def test_with_attention(self):
        m = CSPLayer(64, 64, n=2, add=True, use_attention=True)
        x = torch.randn(2, 64, 16, 16)
        out = m(x)
        assert out.shape == (2, 64, 16, 16)

    def test_without_attention(self):
        m = CSPLayer(64, 64, n=2, add=True, use_attention=False)
        x = torch.randn(2, 64, 16, 16)
        out = m(x)
        assert out.shape == (2, 64, 16, 16)

    def test_c_out_must_be_even(self):
        with pytest.raises(AssertionError):
            CSPLayer(64, 127, n=2, add=True)


class TestChannelAttention:
    def test_shape(self):
        m = ChannelAttention(64)
        x = torch.randn(2, 64, 16, 16)
        out = m(x)
        assert out.shape == (2, 64, 16, 16)

    @pytest.mark.parametrize("channels", [32, 128, 256])
    def test_different_channels(self, channels):
        m = ChannelAttention(channels)
        x = torch.randn(2, channels, 8, 8)
        out = m(x)
        assert out.shape == (2, channels, 8, 8)


class TestSPFFBottleneck:
    def test_shape(self):
        m = SPFFBottleneck(256)
        x = torch.randn(2, 256, 16, 16)
        out = m(x)
        assert out.shape == (2, 256, 16, 16)

    def test_channel_math(self):
        m = SPFFBottleneck(256)
        # conv1: 256->128, concat 4x=512, conv2: 512->256
        assert m.conv1.conv.out_channels == 128
        assert m.conv2.conv.in_channels == 512
        assert m.conv2.conv.out_channels == 256