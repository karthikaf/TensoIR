"""
models/ipe.py

Integrated Positional Encoding (IPE) for directional cone inputs.

Replaces the standard single-direction positional encoding used in TensoIR's
environment map queries with Mip-NeRF's cone-aware IPE (NeP Step 3.3).

Reference:
  Mip-NeRF: "Representing Scenes as Neural Radiance Fields of Radiance" (Barron et al., 2021)
  NeP: Equation used in Step 3.3 of the NeP-TensoIR integration plan.

Convention: uses plain sin(k·x) / cos(k·x) (no extra π factor) to match
the original NeP snippet.  Attenuation is exp(-0.5 · k² · σ²) consistently.

Usage:
    from models.ipe import integrated_dir_enc
    ipe_features = integrated_dir_enc(reflection_dirs, theta_m, num_freqs=8)
"""

import torch


def integrated_dir_enc(directions: torch.Tensor,
                        theta_m: torch.Tensor,
                        num_freqs: int = 8) -> torch.Tensor:
    """
    Evaluates the expected (integrated) positional encoding over a cone of
    directions, as defined by Mip-NeRF / NeP.

    Instead of encoding a single direction vector, this function encodes the
    expected value of sin/cos features over all directions within a cone of
    half-angle `theta_m` centred on `directions`.  The attenuation factor
    exp(-0.5 * sigma^2 * k^2) analytically marginalises out the cone spread,
    producing a smoothly blurred encoding for large cones (rough surfaces) and
    a sharp encoding for small cones (smooth surfaces).

    Args:
        directions: (N, 3)  Central direction of each cone (e.g. reflection vector).
                             Need not be unit-normalised; normalisation is the
                             caller's responsibility.
        theta_m:    (N, 1)  Half-angle of the specular lobe cone in radians,
                             computed via NeP Eq. 10  (see relight_utils.py).
        num_freqs:  int     Number of frequency octaves.  Output dimension is
                             2 * 3 * num_freqs  (sin + cos, three spatial dims,
                             num_freqs octaves).

    Returns:
        (N, 6 * num_freqs)  Integrated positional encoding tensor.

    Mathematical derivation
    -----------------------
    For a Gaussian-approximated cone, using plain sin/cos(k·x):
        E[sin(k · x)] = sin(k · mu) * exp(-0.5 · k^2 · sigma^2)
        E[cos(k · x)] = cos(k · mu) * exp(-0.5 · k^2 · sigma^2)

    where mu = directions  and  sigma^2 = (theta_m^2) / 4  (cone variance approximation).
    k = 2^i for i in [0, num_freqs).
    """
    # -----------------------------------------------------------------
    # 1. Frequency bands: [2^0, 2^1, ..., 2^(num_freqs-1)]
    # -----------------------------------------------------------------
    freq_bands = 2.0 ** torch.linspace(
        0.0, num_freqs - 1, num_freqs,
        device=directions.device, dtype=directions.dtype
    )  # [num_freqs]

    # -----------------------------------------------------------------
    # 2. Project each direction onto the frequency bands
    #    directions: (N, 3)  ->  pts: (N, 3 * num_freqs)
    # -----------------------------------------------------------------
    # directions[..., None] : (N, 3, 1) * freq_bands : (num_freqs,) -> (N, 3, num_freqs)
    pts = (directions[..., None] * freq_bands).view(
        directions.shape[:-1] + (directions.shape[-1] * num_freqs,)
    )  # (N, 3 * num_freqs)

    # -----------------------------------------------------------------
    # 3. Cone variance  (scalar per ray, then broadcast over channels)
    #    Approximation: variance ~ theta_m^2 / 4
    #    theta_m: (N, 1)
    # -----------------------------------------------------------------
    variance = (theta_m ** 2) / 4.0  # (N, 1)

    # Expand variance to match per-channel frequency-squared weights
    # variance: (N, 1, 1) * freq_bands^2: (num_freqs,) -> (N, 1, num_freqs)
    # then view to (N, num_freqs) and repeat for 3 spatial dims -> (N, 3*num_freqs)
    var_bands = (variance.unsqueeze(-1) * (freq_bands ** 2))  # (N, 1, num_freqs)
    var_bands = var_bands.expand(
        directions.shape[0], 3, num_freqs
    ).reshape(directions.shape[0], 3 * num_freqs)  # (N, 3 * num_freqs)

    # -----------------------------------------------------------------
    # 4. Attenuation factor: exp(-0.5 * k^2 * sigma^2)
    # -----------------------------------------------------------------
    attenuation = torch.exp(-0.5 * var_bands)  # (N, 3 * num_freqs)

    # -----------------------------------------------------------------
    # 5. Integrated sin/cos features
    # -----------------------------------------------------------------
    ipe_sin = torch.sin(pts) * attenuation  # (N, 3 * num_freqs)
    ipe_cos = torch.cos(pts) * attenuation  # (N, 3 * num_freqs)

    return torch.cat([ipe_sin, ipe_cos], dim=-1)  # (N, 6 * num_freqs)
