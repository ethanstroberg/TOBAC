#!/usr/bin/env python3
"""
QuerySoundings.py
=================
Queries all available soundings over a given time window and prints a summary
to the terminal.

Run with Tailscale active:
    python QuerySoundings.py
"""

import datetime
from anvil import soundings as _snd_mod
import pathlib
import datetime

UTC = datetime.timezone.utc

# ── Time window ────────────────────────────────────────────────────────────────
T_START = datetime.datetime(2026, 1, 24,  0, 30, 0, tzinfo=UTC)
T_END   = datetime.datetime(2026, 1, 24,  1, 50, 0, tzinfo=UTC)
# ──────────────────────────────────────────────────────────────────────────────

# ── Optional download ──────────────────────────────────────────────────────────
# Set DOWNLOAD_INDEX to the # shown in the printed table (1-based) to download
# that sounding to DOWNLOAD_DIR.  Set to None to skip downloading.
DOWNLOAD_INDEX = None
DOWNLOAD_DIR   = "/Users/ethan1/Desktop/vs_code/Rainmaker/Soundings/"
# ──────────────────────────────────────────────────────────────────────────────


def main():
    print(f"Querying soundings: {T_START:%Y-%m-%d %H:%MZ} → {T_END:%Y-%m-%d %H:%MZ}")
    print("=" * 70)

    try:
        recs = _snd_mod.list_soundings(start=T_START, end=T_END)
    except Exception as e:
        print(f"[error] list_soundings failed: {e}")
        return

    if not recs:
        print("No soundings found in this window.")
        return

    sorted_recs = sorted(recs, key=lambda r: r.timestamp)

    print(f"Found {len(sorted_recs)} sounding(s):\n")
    print(f"  {'#':<4}  {'Timestamp (UTC)':<22}  {'Lat':>8}  {'Lon':>10}  S3 Key")
    print(f"  {'-'*4}  {'-'*22}  {'-'*8}  {'-'*10}  {'-'*40}")

    for i, rec in enumerate(sorted_recs, start=1):
        ts  = datetime.datetime.fromtimestamp(
            rec.timestamp.timestamp(), tz=UTC).strftime('%Y-%m-%d %H:%M:%S')
        lat = f"{float(rec.lat):.4f}" if rec.lat is not None else "     n/a"
        lon = f"{float(rec.lon):.4f}" if rec.lon is not None else "       n/a"
        print(f"  {i:<4}  {ts:<22}  {lat:>8}  {lon:>10}  {rec.s3_key}")

    print(f"\nTotal: {len(sorted_recs)} sounding(s) between "
          f"{T_START:%Y-%m-%d %H:%MZ} and {T_END:%Y-%m-%d %H:%MZ}")

    # ── Download ──────────────────────────────────────────────────────────────
    if DOWNLOAD_INDEX is not None:
        if 1 <= DOWNLOAD_INDEX <= len(sorted_recs):
            rec  = sorted_recs[DOWNLOAD_INDEX - 1]
            dest = pathlib.Path(DOWNLOAD_DIR)
            dest.mkdir(parents=True, exist_ok=True)
            print(f"\nDownloading #{DOWNLOAD_INDEX}: {rec.s3_key} → {dest}")
            try:
                result = _snd_mod.download_sounding(rec.s3_key, dest)
                print(f"  ✓ Saved: {result}")
            except Exception as e:
                print(f"  [error] download failed: {e}")
        else:
            print(f"\n[error] DOWNLOAD_INDEX {DOWNLOAD_INDEX} out of range "
                  f"(1–{len(sorted_recs)})")


if __name__ == "__main__":
    main()