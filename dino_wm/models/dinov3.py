import os

import torch
import torch.nn as nn
from torchvision.transforms import v2

# Patch-token width per backbone, mirroring tsd/models/dinov3_encoder_model.py.
# DINOv3 exposes no single attribute that covers both the ViT and ConvNeXt families,
# so this table is the authority — keep it in sync with that file.
_DINOV3_EMB_DIMS = {
    "dinov3_vits16": 384,
    "dinov3_vits16plus": 384,
    "dinov3_vitb16": 768,
    "dinov3_vitl16": 1024,
    "dinov3_vitl16plus": 1024,
    "dinov3_vith16plus": 1280,
    "dinov3_vit7b16": 4096,
    "dinov3_convnext_tiny": 768,
    "dinov3_convnext_small": 768,
    "dinov3_convnext_base": 1024,
    "dinov3_convnext_large": 1536,
}

# The normalization DINOv3 was pretrained under, and the one
# tools/preprocess_data_mmap.py applied when it wrote this dataset's
# obs/sensor_data/<camera>/dino_patch_features.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class DinoV3Encoder(nn.Module):
    """
    DINOv3 patch-token encoder, matched to the backbone and preprocessing that
    tools/preprocess_data_mmap.py used to write the dataset's `dino_patch_features`.

    Why the match has to be exact: VWorldModel.encode_obs short-circuits to the
    dataset's precomputed features whenever they are present, which is every batch of
    training — so the predictor and decoder only ever see features from *that* pipeline
    and this module is never called. It runs only at inference, where observations come
    back from the env carrying no precomputed features. Any disagreement between the two
    encoders is therefore invisible in training curves and silently corrupts every
    planning rollout, which is exactly what a dinov2_vits14 encoder here did: it happens
    to produce the same 196x384 shape as dinov3_vits16, so nothing raises and the
    predictor is simply fed a foreign feature space.

    Args:
        name: DINOv3 backbone id, e.g. "dinov3_vits16". Must contain "dino" — VWorldModel
            keys its encoder_transform off that substring, and the resulting resize is
            what makes the patch count line up with the decoder's grid.
        feature_key: which forward_features output to take ("x_norm_patchtokens" for the
            patch grid the decoder consumes, "x_norm_clstoken" for a single pooled token).
        repo_dir: local checkout of facebookresearch/dinov3. Loaded with source="local"
            because DINOv3 weights are not served through torch.hub the way DINOv2's are.
        weights: path to the pretrained .pth. This must be the same file the dataset was
            encoded with, not merely the same architecture.
    """

    def __init__(self, name, feature_key, repo_dir, weights):
        super().__init__()
        self.name = name
        self.feature_key = feature_key

        if name not in _DINOV3_EMB_DIMS:
            raise ValueError(
                f"Unknown DINOv3 backbone {name!r}. Known: {sorted(_DINOV3_EMB_DIMS)}"
            )
        if not os.path.isdir(repo_dir):
            raise FileNotFoundError(
                f"DINOv3 repo not found at {repo_dir!r}. Clone facebookresearch/dinov3 "
                "there, or point encoder.repo_dir at an existing checkout."
            )
        if not os.path.isfile(weights):
            raise FileNotFoundError(
                f"DINOv3 weights not found at {weights!r}. This must be the same "
                "checkpoint tools/preprocess_data_mmap.py encoded the dataset with."
            )

        self.base_model = torch.hub.load(
            repo_dir, name, source="local", weights=weights
        )
        self.emb_dim = _DINOV3_EMB_DIMS[name]

        if feature_key == "x_norm_patchtokens":
            self.latent_ndim = 2
        elif feature_key == "x_norm_clstoken":
            self.latent_ndim = 1
        else:
            raise ValueError(f"Invalid feature key: {feature_key}")

        # VWorldModel derives its resize from patch_size, so a wrong value here changes
        # the token count and silently mismatches the decoder's patch grid.
        self.patch_size = getattr(self.base_model, "patch_size", 16)

        self.normalize = v2.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)

    def forward(self, x):
        # x arrives in [-1, 1]: datasets/img_transforms.default_transform ends with
        # Normalize([0.5]*3, [0.5]*3), and VWorldModel.encoder_transform only resizes.
        # The preprocessing tool instead fed uint8 through ToDtype(scale=True) to reach
        # [0, 1] before normalizing, so undo the [-1, 1] convention to land in the same
        # place. (Its Resize ran on uint8 rather than after the cast, which differs in
        # principle but is a no-op here: both pipelines resize 224 -> 224 for a /16
        # backbone at image_size 224.)
        x = (x + 1.0) / 2.0
        x = self.normalize(x)

        emb = self.base_model.forward_features(x)[self.feature_key]
        if self.latent_ndim == 1:
            emb = emb.unsqueeze(1)  # dummy patch dim
        return emb
