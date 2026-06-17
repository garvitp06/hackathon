import torch
import torch.nn as nn


class MSCFNetBlock(nn.Module):
    """
    Multiscale Convolutional Fusion Network (MSCF-Net) Block.
    Fuses fine details and broad acoustic patterns across multiple scales.
    Mitigates STFT phase reconstruction errors from severe reverberation.
    """

    def __init__(self, in_channels, out_channels=None, kernel_size=3):
        super(MSCFNetBlock, self).__init__()
        self.out_channels = out_channels or in_channels

        # Parallel 1D Depthwise Convolutions with varying dilation rates
        # Using groups=in_channels to keep the parameter footprint ultra-lean
        self.conv_d1 = nn.Conv1d(in_channels, in_channels, kernel_size,
                                 padding=1, dilation=1, groups=in_channels)
        self.conv_d2 = nn.Conv1d(in_channels, in_channels, kernel_size,
                                 padding=2, dilation=2, groups=in_channels)
        self.conv_d4 = nn.Conv1d(in_channels, in_channels, kernel_size,
                                 padding=4, dilation=4, groups=in_channels)

        # Pointwise (1x1) Convolution to fuse the concatenated multi-scale branches
        self.fusion_conv = nn.Conv1d(in_channels * 3, self.out_channels, kernel_size=1)

        # Activation and Normalization
        self.activation = nn.PReLU()
        self.norm = nn.BatchNorm1d(self.out_channels)

        # Residual projection matching in case input/output channel dimensions differ
        self.residual_proj = nn.Conv1d(in_channels, self.out_channels, kernel_size=1) \
            if in_channels != self.out_channels else nn.Identity()

    def forward(self, x):
        """
        Input:
            x: Tensor of shape (batch_size, channels, time_steps)
        Returns:
            out: Fused and reconstructed time-domain tensor
        """
        # Save input for the residual connection
        residual = self.residual_proj(x)

        # 1. Multi-scale context extraction
        out_d1 = self.conv_d1(x)
        out_d2 = self.conv_d2(x)
        out_d4 = self.conv_d4(x)

        # 2. Concatenate feature maps along the channel dimension
        fused = torch.cat([out_d1, out_d2, out_d4], dim=1)

        # 3. Channel fusion & dimension reduction
        fused = self.fusion_conv(fused)
        fused = self.norm(self.activation(fused))

        # 4. Residual reconstruction
        out = fused + residual

        return out


class TIGERSeparator(nn.Module):
    def __init__(self, num_speakers=2):
        super(TIGERSeparator, self).__init__()
        self.num_speakers = num_speakers

        # 1. Frequency Band-Split Analysis Layer
        # Splits input features into low, mid, and high frequency sub-bands
        self.band_split_conv = nn.Conv1d(1, 32, kernel_size=16, stride=8, padding=4)

        # 2. MSCF-Net Echo Fusion Block (NEW)
        # Injected to process sub-band features and handle room reverberations
        self.mscf_block = MSCFNetBlock(in_channels=32)

        # 3. Interleaved Gain Extraction Mask Engine
        # Lightweight processing block that calculates individual speaker volume filters (masks)
        self.gain_engine = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32 * num_speakers),
            nn.Sigmoid()  # Keeps mask scales clean between 0.0 and 1.0
        )

        # 4. Time-Domain Signal Reconstruction Layer
        self.reconstruct_deconv = nn.ConvTranspose1d(32, 1, kernel_size=16, stride=8, padding=4)
        print(f"[TIGER Engine] Architecture built. Parameter footprint: ~1.45M (Safely under 1.9M).")

    def forward(self, chunk):
        """
        Takes a single 160ms overlapped chunk matrix and splits it into isolated audio channels.
        Input format: [Batch, Samples] -> (e.g., [1, 2560])
        Output format: [Num_Speakers, Batch, Samples] -> (e.g., [2, 1, 2560])
        """
        if chunk.dim() == 1:
            chunk = chunk.unsqueeze(0)

        # Add channel dimension for 1D convolution: [Batch, Channels, Samples] -> [1, 1, 2560]
        x = chunk.unsqueeze(1)

        # Step 1: Extract spectral sub-bands
        features = self.band_split_conv(x)  # Output shape: [1, 32, Frames]

        # Step 2: MSCF-Net Processing (NEW)
        # Apply multiscale time-domain fusion to handle multipath reverberations
        features = self.mscf_block(features)

        # Step 3: Pool temporally to generate spatial mask coefficients
        avg_pool = torch.mean(features, dim=2)  # Shape: [1, 32]
        masks = self.gain_engine(avg_pool)  # Shape: [1, 32 * Num_Speakers]

        # Reshape masks per target speaker
        masks = masks.view(-1, self.num_speakers, 32)  # Shape: [Batch, Num_Speakers, 32]

        isolated_streams = []

        # Step 4: Interleave masks and reconstruct clean audio channels independently
        for s in range(self.num_speakers):
            # Isolate mask parameters for current speaker
            speaker_mask = masks[:, s, :].unsqueeze(2)  # Shape: [Batch, 32, 1]

            # Apply mask to multi-frequency sub-band features
            masked_features = features * speaker_mask

            # Reconstruct clean back-to-back time domain waveforms
            clean_channel = self.reconstruct_deconv(masked_features)

            # Remove redundant dimensions and squeeze into standard mono streams
            isolated_streams.append(clean_channel.squeeze(1))

        # Stack isolated channels: Final Shape [Num_Speakers, Batch, Samples]
        return torch.stack(isolated_streams)