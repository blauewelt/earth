"""PixelMAE — a masked autoencoder over one pixel's channels, with a
queryable decoder that predicts the SAME pixel's masked channels and its
NEIGHBOURS in space and time from the bottleneck alone.

Shape of the idea (proposal §4–5, scaled to a pilot):

  channels of pixel (lat, lon, t)      ┌─ mask some observed channels
        │  one token per channel  ◄────┘
        ▼
  transformer encoder  (missing channels enter as explicit "missing" tokens —
        │               absence is information, not padding)
        ▼
  z = bottleneck(CLS)   d_z-dimensional; THE embedding
        ▼
  decoder(z, channel-id, Δlat, Δlon, Δt) → value
        Δ = (0,0,0)  reconstruct this pixel (masked channels score)
        Δ = space    predict the 4-neighbours' channels, same month
        Δ = time     predict this pixel's channels next/previous month

Why neighbour heads: a plain autoencoder can ace reconstruction by memorising
the seasonal cycle (proposal §5). Forcing z to answer for pixels it never saw
makes it carry STATE — the currency the AMOC probe then reads.

The decoder is a neural-field-style MLP conditioned on (z, query): it can be
asked for any offset at inference, which is what "use the embedding to predict
nearby pixels" means operationally.
"""
import torch
import torch.nn as nn


class PixelMAE(nn.Module):
    def __init__(self, n_chan, d_model=128, n_heads=4, n_layers=4,
                 d_z=32, d_dec=256, max_abs_offset=3):
        super().__init__()
        self.n_chan = n_chan
        self.d_z = d_z
        self.max_off = max_abs_offset

        # --- encoder tokens -------------------------------------------------
        self.val_proj = nn.Linear(1, d_model)
        self.chan_emb = nn.Embedding(n_chan, d_model)
        self.mask_tok = nn.Parameter(torch.zeros(d_model))     # channel masked by US
        self.miss_tok = nn.Parameter(torch.zeros(d_model))     # channel unobserved in the DATA
        self.cls_tok = nn.Parameter(torch.zeros(d_model))
        # coords/season enter as one non-maskable context token:
        # [sin m, cos m, lat/90, lon/180]
        self.ctx_proj = nn.Linear(4, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=4 * d_model,
            batch_first=True, norm_first=True, dropout=0.0)
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
        self.to_z = nn.Linear(d_model, d_z)

        # --- queryable decoder ---------------------------------------------
        # query = channel id + integer offset (Δlon, Δlat, Δmonth), each
        # embedded; decoder never sees the input values, only z.
        self.q_chan = nn.Embedding(n_chan, 64)
        self.q_off = nn.Embedding(2 * max_abs_offset + 1, 16)  # shared per axis
        self.decoder = nn.Sequential(
            nn.Linear(d_z + 64 + 3 * 16, d_dec), nn.GELU(),
            nn.Linear(d_dec, d_dec), nn.GELU(),
            nn.Linear(d_dec, 1),
        )
        for p in (self.mask_tok, self.miss_tok, self.cls_tok):
            nn.init.normal_(p, std=0.02)

    def encode(self, x, obs, mask, ctx):
        """x [B,C] values (0 where not observed) · obs [B,C] bool observed ·
        mask [B,C] bool masked-out-by-training · ctx [B,4] → z [B,d_z]."""
        B, C = x.shape
        ce = self.chan_emb.weight[None, :, :].expand(B, -1, -1)
        vt = self.val_proj(x.unsqueeze(-1)) + ce
        vt = torch.where((obs & ~mask).unsqueeze(-1), vt, torch.zeros_like(vt))
        vt = vt + torch.where((obs & mask).unsqueeze(-1),
                              self.mask_tok.expand(B, C, -1) + ce, torch.zeros_like(vt))
        vt = vt + torch.where((~obs).unsqueeze(-1),
                              self.miss_tok.expand(B, C, -1) + ce, torch.zeros_like(vt))
        toks = torch.cat([self.cls_tok.expand(B, 1, -1),
                          self.ctx_proj(ctx).unsqueeze(1), vt], dim=1)
        h = self.encoder(toks)
        return self.to_z(h[:, 0])

    def query(self, z, chan_idx, off):
        """z [B,d_z] · chan_idx [B,Q] · off [B,Q,3] ints in [-max,max] → [B,Q]."""
        B, Q = chan_idx.shape
        qc = self.q_chan(chan_idx)
        qo = self.q_off(off + self.max_off).reshape(B, Q, -1)
        zq = z.unsqueeze(1).expand(-1, Q, -1)
        return self.decoder(torch.cat([zq, qc, qo], dim=-1)).squeeze(-1)
