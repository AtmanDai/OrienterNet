# Copyright (c) Meta Platforms, Inc. and affiliates.

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.functional import normalize

from . import get_model
from .base import BaseModel
from .bev_net import BEVNet
from .bev_projection import CartesianProjection, PolarProjectionDepth
from .map_encoder import MapEncoder
from .metrics import AngleError, AngleRecall, Location2DError, Location2DRecall
from .voting import (
    TemplateSampler,
    argmax_xyr,
    conv2d_fft_batchwise,
    expectation_xyr,
    log_softmax_spatial,
    mask_yaw_prior,
    nll_loss_xyr,
    nll_loss_xyr_smoothed,
)


class OrienterNet(BaseModel):
    default_conf = {
        "image_encoder": "???",
        "map_encoder": "???",
        "bev_net": "???",
        "latent_dim": "???",
        "matching_dim": "???",
        "scale_range": [0, 9],
        "num_scale_bins": "???",
        "z_min": None,
        "z_max": "???",
        "x_max": "???",
        "pixel_per_meter": "???",
        "num_rotations": "???",
        "add_temperature": False,
        "normalize_features": False,
        "padding_matching": "replicate",
        "apply_map_prior": True,
        "do_label_smoothing": False,
        "sigma_xy": 1,
        "sigma_r": 2,
        # depcreated
        "depth_parameterization": "scale",
        "norm_depth_scores": False,
        "normalize_scores_by_dim": False,
        "normalize_scores_by_num_valid": True,
        "prior_renorm": True,
        "retrieval_dim": None,
    }

    def _init(self, conf):
        assert not self.conf.norm_depth_scores
        assert self.conf.depth_parameterization == "scale"
        assert not self.conf.normalize_scores_by_dim
        assert self.conf.normalize_scores_by_num_valid
        assert self.conf.prior_renorm

        Encoder = get_model(conf.image_encoder.get("name", "feature_extractor_v2"))
        self.image_encoder = Encoder(conf.image_encoder.backbone)
        self.map_encoder = MapEncoder(conf.map_encoder)
        self.bev_net = None if conf.bev_net is None else BEVNet(conf.bev_net)

        ppm = conf.pixel_per_meter
        self.projection_polar = PolarProjectionDepth(
            conf.z_max,
            ppm,
            conf.scale_range,
            conf.z_min,
        )
        self.projection_bev = CartesianProjection(
            conf.z_max, conf.x_max, ppm, conf.z_min
        )
        self.template_sampler = TemplateSampler(
            self.projection_bev.grid_xz, ppm, conf.num_rotations
        )

        self.scale_classifier = torch.nn.Linear(conf.latent_dim, conf.num_scale_bins)
        if conf.bev_net is None:
            self.feature_projection = torch.nn.Linear(
                conf.latent_dim, conf.matching_dim
            )
        if conf.add_temperature:
            temperature = torch.nn.Parameter(torch.tensor(0.0))
            self.register_parameter("temperature", temperature)

    def _is_panorama_batch(self, data):
        """
        Check if the batch contains panorama views (groups of 3).

        Args:
            data: Input data dict containing 'image' tensor

        Returns:
            bool: True if batch size is multiple of 3 and >= 3, False otherwise
        """
        batch_size = data["image"].shape[0]
        return batch_size % 3 == 0 and batch_size >= 3

    def _create_equilateral_triangle_mask(self, h, w, device):
        """
        Create an equilateral triangle mask for concatenating BEV features,
        divided into three 120-degree isosceles triangles.

        The division lines connect the triangle's center to its vertices.
        - View1 (0°): Top region, pointing up.
        - View2 (120°): Bottom-left region.
        - View3 (240°): Bottom-right region.

        Args:
            h: Height of the feature map.
            w: Width of the feature map.
            device: PyTorch device for tensor creation.

        Returns:
            dict: Dictionary with 'triangle', 'view1', 'view2', 'view3' masks.
        """
        center_y, center_x = h / 2.0, w / 2.0
        side_length = h  # As per original design
        triangle_height = side_length * np.sqrt(3) / 2.0

        # Create coordinate grids
        y, x = torch.meshgrid(
            torch.arange(h, device=device, dtype=torch.float32),
            torch.arange(w, device=device, dtype=torch.float32),
            indexing="ij",
        )

        # Vertices of the equilateral triangle (centered at image center)
        top_v = (center_x, center_y - triangle_height * 2.0 / 3.0)
        bl_v = (center_x - side_length / 2.0, center_y + triangle_height / 3.0)
        br_v = (center_x + side_length / 2.0, center_y + triangle_height / 3.0)

        # Use barycentric coordinates to determine if a point is inside the triangle
        def point_in_triangle(px, py, v1, v2, v3):
            denom = (v2[1] - v3[1]) * (v1[0] - v3[0]) + (v3[0] - v2[0]) * (
                v1[1] - v3[1]
            )
            a = ((v2[1] - v3[1]) * (px - v3[0]) + (v3[0] - v2[0]) * (py - v3[1])) / denom
            b = ((v3[1] - v1[1]) * (px - v3[0]) + (v1[0] - v3[0]) * (py - v3[1])) / denom
            c = 1 - a - b
            return (a >= 0) & (b >= 0) & (c >= 0)

        triangle_mask = point_in_triangle(x, y, top_v, bl_v, br_v)

        # --- Correct view segmentation using center-to-vertex lines ---
        # Center of the triangle is (center_x, center_y)
        center_v = (center_x, center_y)

        # Define a function to check which side of a line a point is on
        # This uses the 2D cross-product concept.
        def get_side(p, line_start, line_end):
            return (line_end[0] - line_start[0]) * (p[1] - line_start[1]) - (
                line_end[1] - line_start[1]
            ) * (p[0] - line_start[0])

        # Define the three dividing lines from the center to the vertices
        side_of_line_to_top = get_side((x, y), center_v, top_v)
        side_of_line_to_bl = get_side((x, y), center_v, bl_v)
        side_of_line_to_br = get_side((x, y), center_v, br_v)

        # View 1 (top sector): Between the lines to BL and BR vertices.
        # This region is "above" the lines to BL and BR (in a rotational sense).
        view1_mask = triangle_mask & (side_of_line_to_bl >= 0) & (side_of_line_to_br < 0)

        # View 2 (bottom-left sector): Between the lines to BR and Top vertices.
        view2_mask = triangle_mask & (side_of_line_to_br >= 0) & (side_of_line_to_top < 0)

        # View 3 (bottom-right sector): Between the lines to Top and BL vertices.
        view3_mask = triangle_mask & (side_of_line_to_top >= 0) & (side_of_line_to_bl < 0)


        return {
            "triangle": triangle_mask,
            "view1": view1_mask,
            "view2": view2_mask,
            "view3": view3_mask,
        }

    def _concatenate_panorama_bev_features(self, f_bev, valid_bev):
        """
        Concatenate BEV features from 3 panorama views into equilateral triangle
        arrangement.

        Each panorama contains 3 views at 120-degree intervals:
        - view1 (0°): placed in top triangle region
        - view2 (120°): placed in bottom-left triangle region
        - view3 (240°): placed in bottom-right triangle region

        Args:
            f_bev: [B, C, H, W] where B is multiple of 3 - BEV features for all
                views
            valid_bev: [B, H, W] - Valid masks for each view

        Returns:
            f_bev_concat: [B//3, C, H, W] - Concatenated features in triangular
                arrangement
            valid_concat: [B//3, H, W] - Valid masks for concatenated features
        """
        B, C, H, W = f_bev.shape
        device = f_bev.device
        assert B % 3 == 0, "Batch size must be multiple of 3 for panorama processing"

        num_panoramas = B // 3

        # Create triangle masks once
        masks = self._create_equilateral_triangle_mask(H, W, device)

        # Initialize output tensors
        f_bev_concat = torch.zeros(
            (num_panoramas, C, H, W), device=device, dtype=f_bev.dtype
        )
        valid_concat = torch.zeros(
            (num_panoramas, H, W), device=device, dtype=torch.bool
        )

        # Process each panorama
        for pano_idx in range(num_panoramas):
            # Get the 3 views for this panorama
            start_idx = pano_idx * 3
            view_indices = [start_idx, start_idx + 1, start_idx + 2]
            view_names = ["view1", "view2", "view3"]

            for i, view_name in enumerate(view_names):
                view_idx = view_indices[i]
                mask = masks[view_name]

                # Apply both triangle mask and original valid mask
                combined_mask = mask & valid_bev[view_idx]

                # Copy features where mask is true
                f_bev_concat[pano_idx, :, combined_mask] = f_bev[
                    view_idx, :, combined_mask
                ]
                valid_concat[pano_idx, combined_mask] = True

        return f_bev_concat, valid_concat

    def _rotate_features_to_map(self, f_bev, valid_bev, yaw_view1):
        """
        Rotate the concatenated BEV features to align with map orientation.

        The rotation aligns the triangular panorama features with the encoded map tile
        based on the yaw angle of view1 (which determines the "forward" direction).

        Args:
            f_bev: [B, C, H, W] - Concatenated BEV features in triangular arrangement
            valid_bev: [B, H, W] - Valid masks for the features
            yaw_view1: [B] - Yaw angles of view1 for each panorama (in radians)

        Returns:
            f_bev_rotated: [B, C, H, W] - Rotated features aligned with map
            valid_rotated: [B, H, W] - Rotated valid masks
        """
        B, C, H, W = f_bev.shape
        device = f_bev.device

        # Convert yaw to rotation matrices for grid sampling
        # PyTorch grid_sample expects rotation in radians, counter-clockwise
        cos_yaw = torch.cos(
            -yaw_view1
        )  # Negative because we want to rotate features to match map
        sin_yaw = torch.sin(-yaw_view1)

        # Create rotation matrices [B, 2, 3] for affine transformation
        rotation_matrices = torch.zeros((B, 2, 3), device=device, dtype=f_bev.dtype)
        rotation_matrices[:, 0, 0] = cos_yaw
        rotation_matrices[:, 0, 1] = -sin_yaw
        rotation_matrices[:, 1, 0] = sin_yaw
        rotation_matrices[:, 1, 1] = cos_yaw
        # Translation is 0 (rotation around center)

        # Create sampling grid
        grid = F.affine_grid(rotation_matrices, (B, C, H, W), align_corners=False)

        # Rotate features
        f_bev_rotated = F.grid_sample(
            f_bev, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )

        # Rotate valid masks (need to add channel dimension for grid_sample)
        valid_expanded = valid_bev.unsqueeze(1).float()  # [B, 1, H, W]
        valid_rotated_expanded = F.grid_sample(
            valid_expanded,
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=False,
        )
        valid_rotated = (valid_rotated_expanded.squeeze(1) > 0.5).bool()  # [B, H, W]

        return f_bev_rotated, valid_rotated

    def exhaustive_voting(self, f_bev, f_map, valid_bev, confidence_bev=None):
        if self.conf.normalize_features:
            f_bev = normalize(f_bev, dim=1)
            f_map = normalize(f_map, dim=1)

        # Build the templates and exhaustively match against the map.
        if confidence_bev is not None:
            f_bev = f_bev * confidence_bev.unsqueeze(1)
        f_bev = f_bev.masked_fill(~valid_bev.unsqueeze(1), 0.0)
        templates = self.template_sampler(f_bev)
        with torch.autocast("cuda", enabled=False):
            scores = conv2d_fft_batchwise(
                f_map.float(),
                templates.float(),
                padding_mode=self.conf.padding_matching,
            )
        if self.conf.add_temperature:
            scores = scores * torch.exp(self.temperature)

        # Reweight the different rotations based on the number of valid pixels in each
        # template. Axis-aligned rotation have the maximum number of valid pixels.
        valid_templates = self.template_sampler(valid_bev.float()[None]) > (1 - 1e-4)
        num_valid = valid_templates.float().sum((-3, -2, -1))
        scores = scores / num_valid[..., None, None]
        return scores

    def _forward(self, data):
        pred = {}
        pred_map = pred["map"] = self.map_encoder(data)
        f_map = pred_map["map_features"][0]

        # Extract image features.
        level = 0
        f_image = self.image_encoder(data)["feature_maps"][level]
        camera = data["camera"].scale(1 / self.image_encoder.scales[level])
        camera = camera.to(data["image"].device, non_blocking=True)

        # Estimate the monocular priors.
        pred["pixel_scales"] = scales = self.scale_classifier(f_image.moveaxis(1, -1))
        f_polar = self.projection_polar(f_image, scales, camera)

        # Map to the BEV.
        with torch.autocast("cuda", enabled=False):
            f_bev, valid_bev, _ = self.projection_bev(
                f_polar.float(), None, camera.float()
            )
        pred_bev = {}
        if self.conf.bev_net is None:
            # channel last -> classifier -> channel first
            f_bev = self.feature_projection(f_bev.moveaxis(1, -1)).moveaxis(-1, 1)
        else:
            pred_bev = pred["bev"] = self.bev_net({"input": f_bev})
            f_bev = pred_bev["output"]

        # Check if we have panorama views and process accordingly
        is_panorama = self._is_panorama_batch(data)
        if is_panorama:
            # Panorama processing: concatenate 3 views into equilateral triangle
            # This implements the core requirement to combine BEV features from
            # view1 (0°), view2 (120°), view3 (240°) into a single triangular
            # feature map
            f_bev_concat, valid_bev_concat = self._concatenate_panorama_bev_features(
                f_bev, valid_bev
            )

            # Get yaw angles for view1 (every 3rd view starting from 0)
            # This determines the orientation of the concatenated features
            if "roll_pitch_yaw" in data:
                yaw_all = data["roll_pitch_yaw"][..., -1]  # Get yaw component
                yaw_view1 = yaw_all[::3]  # Every 3rd element (view1 panorama)

                # Rotate concatenated features to align with map tiles
                # This ensures the triangular features match the map coordinate
                # system
                f_bev_final, valid_bev_final = self._rotate_features_to_map(
                    f_bev_concat, valid_bev_concat, yaw_view1
                )
            else:
                # If no yaw data available, use concatenated features without
                # rotation
                f_bev_final, valid_bev_final = f_bev_concat, valid_bev_concat

            # Store both individual and concatenated features for potential
            # analysis/debugging
            pred["features_bev_individual"] = f_bev
            pred["valid_bev_individual"] = valid_bev
            pred["features_bev_concatenated"] = f_bev_concat
            pred["valid_bev_concatenated"] = valid_bev_concat
        else:
            # Standard single-view processing (backwards compatible)
            f_bev_final, valid_bev_final = f_bev, valid_bev

        # Use the final BEV features for exhaustive voting
        scores = self.exhaustive_voting(
            f_bev_final, f_map, valid_bev_final, pred_bev.get("confidence")
        )
        scores = scores.moveaxis(1, -1)  # B,H,W,N
        if "log_prior" in pred_map and self.conf.apply_map_prior:
            scores = scores + pred_map["log_prior"][0].unsqueeze(-1)
        # pred["scores_unmasked"] = scores.clone()
        if "map_mask" in data:
            scores.masked_fill_(~data["map_mask"][..., None], -np.inf)
        if "yaw_prior" in data:
            mask_yaw_prior(scores, data["yaw_prior"], self.conf.num_rotations)
        log_probs = log_softmax_spatial(scores)
        with torch.no_grad():
            uvr_max = argmax_xyr(scores).to(scores)
            uvr_avg, _ = expectation_xyr(log_probs.exp())

        return {
            **pred,
            "scores": scores,
            "log_probs": log_probs,
            "uvr_max": uvr_max,
            "uv_max": uvr_max[..., :2],
            "yaw_max": uvr_max[..., 2],
            "uvr_expectation": uvr_avg,
            "uv_expectation": uvr_avg[..., :2],
            "yaw_expectation": uvr_avg[..., 2],
            "features_image": f_image,
            "features_bev": f_bev_final,
            "valid_bev": (
                valid_bev_final.squeeze(1)
                if valid_bev_final.dim() == 3
                else valid_bev_final
            ),
            "is_panorama": is_panorama,
        }

    def loss(self, pred, data):
        # Handle panorama batches - use ground truth from view1 of each panorama
        if pred.get("is_panorama", False):
            # For panorama batches, take ground truth from view1 (every 3rd element)
            xy_gt = data["uv"][::3]  # view1 ground truth positions
            yaw_gt = data["roll_pitch_yaw"][::3, -1]  # view1 ground truth yaw
            mask = data.get("map_mask")
            if mask is not None:
                mask = mask[::3]  # view1 masks
        else:
            # Standard single-view processing
            xy_gt = data["uv"]
            yaw_gt = data["roll_pitch_yaw"][..., -1]
            mask = data.get("map_mask")

        if self.conf.do_label_smoothing:
            nll = nll_loss_xyr_smoothed(
                pred["log_probs"],
                xy_gt,
                yaw_gt,
                self.conf.sigma_xy / self.conf.pixel_per_meter,
                self.conf.sigma_r,
                mask=mask,
            )
        else:
            nll = nll_loss_xyr(pred["log_probs"], xy_gt, yaw_gt)
        loss = {"total": nll, "nll": nll}
        if self.training and self.conf.add_temperature:
            loss["temperature"] = self.temperature.expand(len(nll))
        return loss

    def metrics(self):
        return {
            "xy_max_error": Location2DError("uv_max", self.conf.pixel_per_meter),
            "xy_expectation_error": Location2DError(
                "uv_expectation", self.conf.pixel_per_meter
            ),
            "yaw_max_error": AngleError("yaw_max"),
            "xy_recall_2m": Location2DRecall(2.0, self.conf.pixel_per_meter, "uv_max"),
            "xy_recall_5m": Location2DRecall(5.0, self.conf.pixel_per_meter, "uv_max"),
            "yaw_recall_2°": AngleRecall(2.0, "yaw_max"),
            "yaw_recall_5°": AngleRecall(5.0, "yaw_max"),
        }
