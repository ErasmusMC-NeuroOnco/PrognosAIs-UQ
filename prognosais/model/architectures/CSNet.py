import torch
import torch.nn as nn
import torch.nn.functional as F


class CSNet(nn.Module):
    """
    Three-dimensional CSNet architecture for joint segmentation and
    classification.

    The network uses a shared encoder, a segmentation decoder, and a
    classification head that predicts one or more configured classification tasks from the
    shared latent representation. It supports a classification-only mode that
    skips the decoder and uses encoder features only.

    Attributes:
        precision: Floating-point precision used for convolution layers.
        dropout_rate: Dropout rate applied in the convolutional and dense
            blocks.
        classification_tasks: Mapping of classification task names to the number
            of classes predicted for each task.
        in_channels: Number of input imaging channels.
        classification_only: Whether to skip the segmentation decoder and use
            only encoder features for classification.
    """

    def __init__(
        self,
        dropout_rate: float,
        classification_tasks: dict[str, int],
        in_channels: int = 4,
        precision: torch.dtype | None = None,
        classification_only: bool = False,
    ):
        """
        Initialize the CSNet model.

        Args:
            dropout_rate: Dropout probability used throughout the network.
            classification_tasks: Mapping of classification task names to their number of output classes.
            in_channels: Number of input image channels.
            precision: Torch dtype used for convolutional and linear layers.
            classification_only: If True, skip the segmentation decoder during the
                forward pass and use encoder features only for classification.
        """
        super().__init__()

        self.precision = precision
        self.dropout_rate = dropout_rate
        self.in_channels = in_channels
        self.classification_only = classification_only
        self.classification_tasks = classification_tasks

        # Encoder
        self.enc1 = self.encoder_block(self.in_channels, 32, first=True)
        self.enc1_2 = self.maxpool_batchnorm(32, first=True)
        self.enc2 = self.encoder_block(32, 64)
        self.enc2_2 = self.maxpool_batchnorm(64, kernel_size=2, stride=2, padding=1)
        self.enc3 = self.encoder_block(64, 128)
        self.enc3_2 = self.maxpool_batchnorm(128, kernel_size=3, stride=2, padding=1)
        self.enc4 = self.encoder_block(128, 256)
        self.enc4_2 = self.maxpool_batchnorm(256, kernel_size=3, stride=2, padding=1)

        # Bottleneck
        self.bottleneck = self.bottleneck_block(256, 512)
        self.bottleneck_bn = nn.BatchNorm3d(512)

        # Decoder
        self.upconv4 = self.upconv_block(512, 256)
        self.dec4 = self.decoder_block(512, 256)
        self.bn4 = nn.BatchNorm3d(256)

        self.upconv3 = self.upconv_block(256, 128)
        self.dec3 = self.decoder_block(256, 128)
        self.bn3 = nn.BatchNorm3d(128)

        self.upconv2 = self.upconv_block(128, 64)
        self.dec2 = self.decoder_block(128, 64)
        self.bn2 = nn.BatchNorm3d(64)

        self.upconv1 = self.deconv_block(64, 32)
        self.dec1 = self.decoder_block(64, 32)
        self.bn1 = nn.BatchNorm3d(32)

        # Final segmentation layer
        self.final_conv = nn.Conv3d(32, 2, kernel_size=1, dtype=self.precision)
        # self.softmax = nn.Softmax(dim=1)

        # Classification part
        self.global_pool = nn.AdaptiveMaxPool3d(1)
        self.dropout_layer = nn.Dropout(self.dropout_rate)

        if not self.classification_only:
            self.class_dense = nn.Sequential(
                nn.Linear(1472, 512), nn.ReLU()
            )
        else:
            self.class_dense = nn.Sequential(
                nn.Linear(992, 512), nn.ReLU()
            )

        self.classifiers = nn.ModuleDict(
            {
                task_name: nn.Linear(512, num_classes)
                for task_name, num_classes
                in self.classification_tasks.items()
            }
        )

    def encoder_block(
        self, in_channels: int, out_channels: int, first: bool = False
    ) -> nn.Sequential:
        """
        Build one encoder block for the contracting path.

        The first encoder block uses a single convolution because it operates
        directly on the input image tensor. Later encoder blocks use two
        convolution layers before dropout.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            first: If True, build the first encoder block with a single
                convolution. Otherwise build the standard two-convolution
                block.

        Returns:
            nn.Sequential: Encoder block layers.
        """

        # Decoder block with slight modifications if it is the first one.
        if first:
            encoder = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    dtype=self.precision,
                ),
                nn.ReLU(),
                nn.Dropout3d(self.dropout_rate),
            )
        # Typical decoder block.
        else:
            encoder = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    dtype=self.precision,
                ),
                nn.ReLU(),
                nn.Conv3d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    dtype=self.precision,
                ),
                nn.ReLU(),
                nn.Dropout3d(self.dropout_rate),
            )

        return encoder

    def maxpool_batchnorm(
        self, in_channels, first=False, kernel_size=None, stride=None, padding=None
    ):
        """
        Build the downsampling block used between encoder stages.

        The first downsampling block is implemented as a strided convolution
        followed by ReLU and batch normalization. Later blocks use max pooling
        followed by batch normalization.

        Args:
            in_channels: Number of input channels.
            first: If True, build the initial strided-convolution version.
            kernel_size: Max-pooling kernel size for the standard version.
            stride: Max-pooling stride for the standard version.
            padding: Max-pooling padding for the standard version.

        Returns:
            nn.Sequential: Downsampling block.

        """

        # Block with slight modifications if it is the first in the path.
        if first:

            block = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    in_channels,
                    kernel_size=9,
                    stride=3,
                    padding=4,
                    dtype=self.precision,
                ),
                nn.ReLU(),
                nn.BatchNorm3d(in_channels),
            )

        # Typical max pooling and batch normalization block.
        else:

            assert (
                kernel_size is not None and stride is not None and padding is not None
            ), "Please insert a valid kernel size, stride and padding."

            block = nn.Sequential(
                nn.MaxPool3d(kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm3d(in_channels),
            )

        return block

    def bottleneck_block(self, in_channels, out_channels):
        """
        Build the bottleneck block at the bottom of the U-shaped network.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.

        Returns:
            nn.Sequential: Bottleneck block with two convolutions and dropout.

        """

        bottleneck = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                dtype=self.precision,
            ),
            nn.ReLU(),
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                dtype=self.precision,
            ),
            nn.ReLU(),
            nn.Dropout3d(self.dropout_rate),
        )
        return bottleneck

    def upconv_block(self, in_channels, out_channels):
        """
        Build an upconvolution block for decoder upsampling.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.

        Returns:
            nn.Sequential: Upconvolution block.

        """
        return nn.Sequential(
            nn.ConvTranspose3d(
                in_channels, out_channels, kernel_size=2, stride=2, dtype=self.precision
            ),
            nn.ReLU(),
        )

    def deconv_block(self, in_channels, out_channels):
        """
        Build a larger-step deconvolution block for the final decoder stage.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.

        Returns:
            nn.Sequential: Deconvolution block.

        """
        return nn.Sequential(
            nn.ConvTranspose3d(
                in_channels, out_channels, kernel_size=9, stride=3, dtype=self.precision
            ),
            nn.ReLU(),
        )

    def decoder_block(self, in_channels, out_channels):
        """
        Build one decoder block for the expansive path.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.

        Returns:
            nn.Sequential: Decoder block.

        """
        decoder = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                dtype=self.precision,
            ),
            nn.ReLU(),
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                dtype=self.precision,
            ),
            nn.ReLU(),
            nn.Dropout3d(self.dropout_rate),
        )
        return decoder

    def upsample(self, tensor: torch.Tensor, target_size):
        """
        Resize a tensor to match a target spatial size.

        This helper is used before concatenating encoder and decoder features
        so that tensors from different resolutions align spatially.

        Args:
            tensor: Tensor to upsample.
            target_size: Spatial size to interpolate to.

        Returns:
            torch.Tensor: Interpolated tensor.

        """
        return F.interpolate(tensor, size=target_size, mode="trilinear")

    def forward(self, x: torch.Tensor):
        """
        Run a forward pass through the model.

        The method returns segmentation logits when the decoder is enabled and
        classification logits for every configured classification task.

        Args:
            x: Input tensor with shape [batch, channels, depth, height, width].

        Returns:
            tuple:
                seg_out: Segmentation logits, or None in classification-only
                    mode.
                classification_outputs: Logits for all configured tasks.

        """
        enc1 = self.enc1(x)
        pooled_enc1 = self.global_pool(enc1).view(enc1.size(0), -1)
        enc1_2 = self.enc1_2(enc1)

        enc2 = self.enc2(enc1_2)
        pooled_enc2 = self.global_pool(enc2).view(enc2.size(0), -1)
        enc2_2 = self.enc2_2(enc2)

        enc3 = self.enc3(enc2_2)
        pooled_enc3 = self.global_pool(enc3).view(enc3.size(0), -1)
        enc3_2 = self.enc3_2(enc3)

        enc4 = self.enc4(enc3_2)
        pooled_enc4 = self.global_pool(enc4).view(enc4.size(0), -1)
        enc4_2 = self.enc4_2(enc4)

        # Bottleneck
        bottleneck = self.bottleneck(enc4_2)
        pooled_bottleneck = self.global_pool(bottleneck).view(bottleneck.size(0), -1)
        bottleneck_2 = self.bottleneck_bn(bottleneck)

        if not self.classification_only:
            # Decoder
            upconv4 = self.upconv4(bottleneck_2)
            dec4 = self.dec4(
                torch.cat([self.upsample(upconv4, enc4.shape[2:]), enc4], dim=1)
            )
            pooled_dec4 = self.global_pool(dec4).view(dec4.size(0), -1)
            dec4 = self.bn4(dec4)

            upconv3 = self.upconv3(dec4)
            dec3 = self.dec3(
                torch.cat([self.upsample(upconv3, enc3.shape[2:]), enc3], dim=1)
            )
            pooled_dec3 = self.global_pool(dec3).view(dec3.size(0), -1)
            dec3 = self.bn3(dec3)

            upconv2 = self.upconv2(dec3)
            dec2 = self.dec2(
                torch.cat([self.upsample(upconv2, enc2.shape[2:]), enc2], dim=1)
            )
            pooled_dec2 = self.global_pool(dec2).view(dec2.size(0), -1)
            dec2 = self.bn2(dec2)

            upconv1 = self.upconv1(dec2)
            dec1 = self.dec1(
                torch.cat([self.upsample(upconv1, enc1.shape[2:]), enc1], dim=1)
            )
            pooled_dec1 = self.global_pool(dec1).view(dec1.size(0), -1)
            dec1 = self.bn1(dec1)

            # Final Convolution for Segmentation
            seg_out = self.final_conv(dec1)
            # seg_out = self.softmax(final_conv)

        if self.classification_only:
            # Classification
            seg_out = None
            class_features = torch.cat(
                [
                    pooled_enc1,
                    pooled_enc2,
                    pooled_enc3,
                    pooled_enc4,
                    pooled_bottleneck,
                ],
                dim=1,
            )
        else:
            # Classification
            class_features = torch.cat(
                [
                    pooled_enc1,
                    pooled_enc2,
                    pooled_enc3,
                    pooled_enc4,
                    pooled_bottleneck,
                    pooled_dec4,
                    pooled_dec3,
                    pooled_dec2,
                    pooled_dec1,
                ],
                dim=1,
            )

        class_features = self.dropout_layer(class_features)
        class_features = self.class_dense(class_features)

        classification_outputs = {
            task_name: classifier(class_features)
            for task_name, classifier in self.classifiers.items()
        }

        return seg_out, classification_outputs


if __name__ == "__main__":

    # Example usage
    classification_tasks = {
        "idh": 2,
        "onep19q": 2,
        "grade": 3,
    }
    model = CSNet(dropout_rate=0.2, classification_tasks=classification_tasks, classification_only=False)

    # Input tensor with the expected image size
    input_tensor = torch.rand(1, 4, 145, 182, 152, dtype=torch.float32)
    seg_output, classification_outputs = model(input_tensor)

    print(seg_output.shape)

    for task, prediction in classification_outputs.items():
        print(task, prediction.shape)
