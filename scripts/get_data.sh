#!/usr/bin/env bash
# Fetch the source corpora. Nothing is redistributed in this repository;
# see LICENSE-DATA.md for the terms attached to each.
set -euo pipefail

DATA_DIR="${1:-data}"
mkdir -p "$DATA_DIR"

echo "==> PhysioNet/CinC 2015 (training set, 750 records)"
if [ -d "$DATA_DIR/cinc2015" ]; then
  echo "    already present, skipping"
else
  wget -r -N -c -np -nH --cut-dirs=3 \
       -P "$DATA_DIR/cinc2015" \
       https://physionet.org/files/challenge-2015/1.0.0/training/
  echo "    done. 627 of 750 records carry the ECG+PPG core pair."
fi

cat <<'NOTE'

==> WUSTL-EHMS-2020  (manual step)
    Not publicly downloadable. Request it from the dataset authors, then place
    the CSV at:
        data/wustl/wustl-ehms-2020.csv
    Used only for the leakage audit.

==> MIMIC-III Waveform  (optional)
    Characterised but NOT injected in this release. Only needed if you are
    extending the benchmark to long recordings.

NOTE
echo "Next: make paper"
