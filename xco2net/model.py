"""OCAP-Net model definition.

OCAP-Net (Observation-Conditioned Absorption Pattern Network) estimates XCO2
residuals from OCO-2 spectra. The model uses wavelength-aligned SNR weighting,
band-wise spectral encoding, and state-conditioned band-level gates.
"""

import torch
import torch.nn as nn


DTYPE = torch.float32


class SimpleMLP(nn.Module):
    """Small MLP used by the spectral-only prediction head."""

    def __init__(self, layer_sizes):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class MLPSpectral(nn.Module):
    """Shared band-wise spectral encoder.

    Input shape: [B, 3, 1016]
    Output shape: [B, 3, D]
    """

    def __init__(self, in_features=1016, hidden_features=256, out_features=256, dropout_rate=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class StateConditionedBandGate(nn.Module):
    """Generate three state-conditioned scalar gates for the O2, weak CO2, and strong CO2 bands."""

    def __init__(self, state_embed_dim, gate_hidden_dim=64, temperature=1.0, rescale=True):
        super().__init__()
        if temperature <= 0:
            raise ValueError("gate_temperature must be positive.")
        self.temperature = temperature
        self.rescale = rescale
        self.gate_mlp = nn.Sequential(
            nn.Linear(state_embed_dim, gate_hidden_dim),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, 3),
        )

    def forward(self, state_embedding):
        logits = self.gate_mlp(state_embedding)
        alpha = torch.softmax(logits / self.temperature, dim=-1)
        gate = 3.0 * alpha if self.rescale else alpha
        return gate.unsqueeze(-1), logits


class OCAPGateNet(nn.Module):
    """OCAP-Net with state-conditioned band gating and a spectral-only head.

    Inputs:
        features: [B, M] auxiliary physical states.
        radiances: [B, 3048, 2], ordered as three 1016-point bands with
            radiance/SNR aligned on the last dimension.

    Output:
        Predicted standardized XCO2 residual.
    """

    def __init__(
        self,
        patch_size,
        stride,
        feature_dim,
        d_model,
        final_mlp_layers,
        use_state_band_gate=True,
        gate_hidden_dim=64,
        gate_temperature=1.0,
        gate_rescale=True,
        **kwargs,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.d_model = d_model
        self.use_state_band_gate = bool(use_state_band_gate)
        self.d_state = 32

        self.physical_proj = nn.Sequential(nn.Linear(feature_dim, self.d_state))
        self.snr_layer = nn.Sequential(nn.Linear(patch_size, patch_size), nn.Sigmoid())
        self.feature_extractor = MLPSpectral(
            in_features=1016,
            hidden_features=d_model,
            out_features=d_model,
            dropout_rate=0.2,
        )

        self.band_gate = None
        if self.use_state_band_gate:
            self.band_gate = StateConditionedBandGate(
                state_embed_dim=self.d_state,
                gate_hidden_dim=gate_hidden_dim,
                temperature=gate_temperature,
                rescale=gate_rescale,
            )

        head_dim = 3 * d_model
        self.final_nn = SimpleMLP([head_dim] + final_mlp_layers)

    def _apply_band_gate(self, band_embeddings, z_state, aux=None):
        if not self.use_state_band_gate:
            return band_embeddings
        gate, gate_logits = self.band_gate(z_state)
        band_embeddings = band_embeddings * gate
        if aux is not None:
            aux["band_gates"] = gate.squeeze(-1)
            aux["band_gate_logits"] = gate_logits
        return band_embeddings

    def forward(self, features: torch.Tensor, radiances: torch.Tensor, return_aux: bool = False):
        batch_size = features.size(0)
        aux = {}

        z_state = self.physical_proj(features)

        x = radiances.unfold(1, self.patch_size, self.stride)
        rad, snr = x[:, :, 0, :], x[:, :, 1, :]
        snr_pos = self.snr_layer(snr)
        cap = torch.log1p(snr_pos)
        cap_max = cap.amax(dim=-1, keepdim=True).clamp_min(1e-6)
        rad_weighted = rad * (cap / cap_max)

        e = self.feature_extractor(rad_weighted)
        e = self._apply_band_gate(e, z_state, aux)
        e_flat = e.view(batch_size, -1)
        out = self.final_nn(e_flat)

        if return_aux:
            return out, aux
        return out


class OCAPNetGate(OCAPGateNet):
    """Alias for compatibility with older experiment names."""

    pass
