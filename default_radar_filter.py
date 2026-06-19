#!/usr/bin/env python3
"""Standalone default radar filtering helper.

This module mirrors the Radar-QPE-Tool default FILT_* cleaning step without
requiring a hosted app session. The main callable is `filter_radar_volume()`.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


FILTERED_FIELD_NAMES = (
    "reflectivity_filtered",
    "cross_correlation_ratio_filtered",
    "differential_reflectivity_filtered",
)


@dataclass(frozen=True)
class DefaultFilterConfig:
    filt_rhohv_min: float = 0.70
    filt_rhohv_enabled: bool = True
    filt_ref_min: float = -20.0
    filt_ref_enabled: bool = True
    filt_vel_tex: float = 7.0
    filt_vel_tex_enabled: bool = True
    filt_phidp_tex: float = 40.0
    filt_phidp_tex_enabled: bool = True
    filt_zdr_tex: float = 4.0
    filt_zdr_tex_enabled: bool = True
    wind_ngates: int = 7
    src_ref_field: str = "reflectivity"
    src_rho_field: str = "cross_correlation_ratio"
    src_phi_field: str = "differential_phase"
    src_zdr_field: str = "differential_reflectivity"
    src_vel_field: str = "velocity"
    transfer_duplicate_masks: bool = True


def _enabled(value: float, enabled_flag: bool = True) -> bool:
    if not bool(enabled_flag):
        return False
    try:
        fval = float(value)
    except Exception:
        return False
    return bool(np.isfinite(fval) and fval != -999.0)


def _circular_argmin(source_az_deg: np.ndarray, targets_az_deg: np.ndarray) -> np.ndarray:
    delta = np.abs(
        ((source_az_deg[:, np.newaxis] - targets_az_deg[np.newaxis, :] + 180.0) % 360.0)
        - 180.0
    )
    return np.argmin(delta, axis=0)


def _require_fields(radar, cfg: DefaultFilterConfig) -> None:
    required = [cfg.src_ref_field, cfg.src_rho_field, cfg.src_zdr_field]
    if _enabled(cfg.filt_phidp_tex, cfg.filt_phidp_tex_enabled):
        required.append(cfg.src_phi_field)
    if _enabled(cfg.filt_vel_tex, cfg.filt_vel_tex_enabled):
        required.append(cfg.src_vel_field)
    missing = [name for name in required if name not in radar.fields]
    if missing:
        available = ", ".join(sorted(radar.fields.keys()))
        raise ValueError(
            "Filtering requires fields {} but missing {}. Available: {}".format(
                required, missing, available
            )
        )


def _apply_duplicate_mask_transfer(radar, excluded: np.ndarray, cfg: DefaultFilterConfig) -> np.ndarray:
    if not bool(cfg.transfer_duplicate_masks):
        return excluded
    try:
        fixed_angles = np.asarray(radar.fixed_angle["data"], dtype=np.float64)
    except Exception:
        return excluded
    if fixed_angles.size < 2:
        return excluded

    ancillary_fields = [cfg.src_rho_field]
    if _enabled(cfg.filt_phidp_tex, cfg.filt_phidp_tex_enabled):
        ancillary_fields.append(cfg.src_phi_field)
    if _enabled(cfg.filt_zdr_tex, cfg.filt_zdr_tex_enabled):
        ancillary_fields.append(cfg.src_zdr_field)

    ref_data = radar.fields[cfg.src_ref_field]["data"]
    score_ratio_threshold = 0.2

    for angle in np.unique(np.round(fixed_angles, 3)):
        sweep_indices = np.where(np.isclose(fixed_angles, angle, atol=1.0e-3))[0].astype(np.int64)
        if sweep_indices.size < 2:
            continue

        scores = []
        for sweep_idx in sweep_indices:
            sweep_slice = radar.get_slice(int(sweep_idx))
            ref_valid = ~np.ma.getmaskarray(ref_data[sweep_slice])
            denom = int(np.count_nonzero(ref_valid))
            if denom == 0:
                scores.append(0.0)
                continue
            field_fracs = []
            for field_name in ancillary_fields:
                field_valid = ~np.ma.getmaskarray(radar.fields[field_name]["data"][sweep_slice])
                field_fracs.append(float(np.count_nonzero(field_valid & ref_valid)) / float(denom))
            scores.append(float(np.mean(field_fracs)))

        donor_pos = int(np.argmax(np.asarray(scores, dtype=np.float64)))
        donor_idx = int(sweep_indices[donor_pos])
        donor_score = float(scores[donor_pos])
        if donor_score <= 0.0:
            continue

        donor_slice = radar.get_slice(donor_idx)
        donor_az = np.asarray(radar.get_azimuth(donor_idx), dtype=np.float64)
        donor_mask = excluded[donor_slice, :]

        for sweep_idx, score in zip(sweep_indices, scores):
            recipient_idx = int(sweep_idx)
            if recipient_idx == donor_idx or score >= score_ratio_threshold * donor_score:
                continue
            recipient_slice = radar.get_slice(recipient_idx)
            recipient_az = np.asarray(radar.get_azimuth(recipient_idx), dtype=np.float64)
            ray_map = _circular_argmin(donor_az, recipient_az)
            excluded[recipient_slice, :] = np.logical_or(excluded[recipient_slice, :], donor_mask[ray_map, :])

    return excluded


def add_default_filtered_fields(radar, config: Optional[DefaultFilterConfig] = None):
    """Add default filtered fields to an already-loaded Py-ART Radar object.

    Returns the same Radar object after adding/replacing:
    - reflectivity_filtered
    - cross_correlation_ratio_filtered
    - differential_reflectivity_filtered

    Texture fields used for filtering are also added when their corresponding
    filters are enabled.
    """
    import pyart

    cfg = config or DefaultFilterConfig()
    _require_fields(radar, cfg)

    gatefilter = pyart.correct.GateFilter(radar)
    gatefilter.include_all()

    if _enabled(cfg.filt_rhohv_min, cfg.filt_rhohv_enabled):
        gatefilter.exclude_below(cfg.src_rho_field, cfg.filt_rhohv_min, exclude_masked=False)
    if _enabled(cfg.filt_ref_min, cfg.filt_ref_enabled):
        gatefilter.exclude_below(cfg.src_ref_field, cfg.filt_ref_min, exclude_masked=False)

    if _enabled(cfg.filt_phidp_tex, cfg.filt_phidp_tex_enabled):
        phi_tex = pyart.util.texture_along_ray(radar, var=cfg.src_phi_field, wind_size=cfg.wind_ngates)
        phi_tex = np.ma.array(phi_tex, copy=False)
        phi_tex.mask = np.ma.getmaskarray(radar.fields[cfg.src_phi_field]["data"]) | (phi_tex > 999).data
        field = pyart.config.get_metadata("differential_phase_texture")
        field["data"] = phi_tex
        radar.add_field("differential_phase_texture", field, replace_existing=True)
        gatefilter.exclude_above("differential_phase_texture", cfg.filt_phidp_tex, exclude_masked=False)

    if _enabled(cfg.filt_zdr_tex, cfg.filt_zdr_tex_enabled):
        zdr_tex = pyart.util.texture_along_ray(radar, var=cfg.src_zdr_field, wind_size=cfg.wind_ngates)
        zdr_tex = np.ma.array(zdr_tex, copy=False)
        zdr_tex.mask = np.ma.getmaskarray(radar.fields[cfg.src_zdr_field]["data"]) | (zdr_tex > 999).data
        field = pyart.config.get_metadata("differential_reflectivity_texture")
        field["data"] = zdr_tex
        radar.add_field("differential_reflectivity_texture", field, replace_existing=True)
        gatefilter.exclude_above("differential_reflectivity_texture", cfg.filt_zdr_tex, exclude_masked=False)

    if _enabled(cfg.filt_vel_tex, cfg.filt_vel_tex_enabled):
        vel_tex = pyart.util.texture_along_ray(radar, var=cfg.src_vel_field, wind_size=cfg.wind_ngates)
        vel_tex = np.ma.array(vel_tex, copy=False)
        vel_tex.mask = np.ma.getmaskarray(radar.fields[cfg.src_vel_field]["data"]) | (vel_tex > 999).data
        field = pyart.config.get_metadata("velocity_texture")
        field["data"] = vel_tex
        radar.add_field("velocity_texture", field, replace_existing=True)
        gatefilter.exclude_above("velocity_texture", cfg.filt_vel_tex, exclude_masked=False)

    excluded = np.array(gatefilter.gate_excluded, copy=True)
    excluded = _apply_duplicate_mask_transfer(radar, excluded, cfg)

    ref_filt = radar.fields[cfg.src_ref_field].copy()
    ref_filt["data"] = np.ma.masked_where(excluded, radar.fields[cfg.src_ref_field]["data"])
    ref_filt["long_name"] = "Filtered reflectivity"
    radar.add_field("reflectivity_filtered", ref_filt, replace_existing=True)

    rho_filt = radar.fields[cfg.src_rho_field].copy()
    rho_filt["data"] = np.ma.masked_where(excluded, radar.fields[cfg.src_rho_field]["data"])
    rho_filt["long_name"] = "Filtered cross correlation ratio"
    radar.add_field("cross_correlation_ratio_filtered", rho_filt, replace_existing=True)

    zdr_filt = radar.fields[cfg.src_zdr_field].copy()
    zdr_filt["data"] = np.ma.masked_where(excluded, radar.fields[cfg.src_zdr_field]["data"])
    zdr_filt["long_name"] = "Filtered differential reflectivity"
    radar.add_field("differential_reflectivity_filtered", zdr_filt, replace_existing=True)

    return radar


def filter_radar_volume(
    volume_file: str,
    config: Optional[DefaultFilterConfig] = None,
    *,
    include_original: bool = False,
    include_textures: bool = False,
    return_radar: bool = False,
    output_file: Optional[str] = None,
) -> Dict[str, object]:
    """Read a radar volume file and return default-filtered fields.

    Parameters
    ----------
    volume_file:
        Path to a radar volume readable by Py-ART.
    config:
        Optional `DefaultFilterConfig` override. Defaults match the app UI.
    include_original:
        Include the original source reflectivity, RHOHV, and ZDR arrays.
    include_textures:
        Include generated texture arrays when those filters are enabled.
    return_radar:
        Include the modified Py-ART Radar object in the returned dictionary.
    output_file:
        Optional path to write a CfRadial file containing the original and
        added filtered fields.

    Returns
    -------
    dict
        Keys are field names and values are numpy masked arrays. If
        `return_radar=True`, the key `radar` contains the modified Radar.
    """
    import pyart

    cfg = config or DefaultFilterConfig()
    path = os.path.abspath(str(volume_file))
    radar = pyart.io.read(path)
    add_default_filtered_fields(radar, cfg)

    out: Dict[str, object] = {
        name: np.ma.array(radar.fields[name]["data"], copy=True)
        for name in FILTERED_FIELD_NAMES
        if name in radar.fields
    }

    if include_original:
        for name in (cfg.src_ref_field, cfg.src_rho_field, cfg.src_zdr_field):
            if name in radar.fields:
                out[name] = np.ma.array(radar.fields[name]["data"], copy=True)

    if include_textures:
        for name in (
            "velocity_texture",
            "differential_phase_texture",
            "differential_reflectivity_texture",
        ):
            if name in radar.fields:
                out[name] = np.ma.array(radar.fields[name]["data"], copy=True)

    if output_file:
        pyart.io.write_cfradial(os.path.abspath(str(output_file)), radar)

    if return_radar:
        out["radar"] = radar

    return out


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply Radar-QPE-Tool default FILT_* cleaning to one radar volume."
    )
    parser.add_argument("volume_file", help="Radar volume readable by Py-ART")
    parser.add_argument("--output", "-o", help="Optional filtered CfRadial output path")
    parser.add_argument("--include-original", action="store_true", help="Also return/source-copy original core fields")
    parser.add_argument("--no-duplicate-transfer", action="store_true", help="Disable duplicate-sweep mask transfer")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    cfg = DefaultFilterConfig(transfer_duplicate_masks=not bool(args.no_duplicate_transfer))
    fields = filter_radar_volume(
        args.volume_file,
        cfg,
        include_original=bool(args.include_original),
        output_file=args.output,
    )
    print("Filtered fields:")
    for name, arr in fields.items():
        data = np.ma.array(arr, copy=False)
        valid = int(np.count_nonzero(~np.ma.getmaskarray(data)))
        print("  {} shape={} valid_gates={}".format(name, tuple(data.shape), valid))
    if args.output:
        print("Wrote {}".format(os.path.abspath(args.output)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
