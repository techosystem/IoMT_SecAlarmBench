# Data terms

**The code here is MIT-licensed (see `LICENSE`). The data is not ours to
license, and none of it is redistributed in this repository.**

## What this repository does *not* contain

No raw or derived physiological waveforms are committed. `.dat`, `.hea`,
`.npz`, `.h5` and similar are gitignored by design. This matches the paper's
commitment: *"Restricted raw waveforms are not redistributed, and access to
the original signals is provided through the source datasets and their
respective terms."*

Users regenerate the benchmark locally from the source corpora. Generation is
deterministic under a fixed seed, so a regenerated benchmark is identical to
the one the reported results were computed on.

## Source corpora and their terms

| Corpus | Role here | Access | License |
|---|---|---|---|
| PhysioNet/CinC 2015 | physiological spine (627 records) | physionet.org/content/challenge-2015/ | Open Data Commons Attribution (ODC-BY) v1.0 — **verify on the landing page** |
| WUSTL-EHMS-2020 | dual-modality bridge, leakage audit | request from the dataset authors | per the authors' stated terms |
| MIMIC-III Waveform | characterised, not injected | physionet.org/content/mimic3wdb/ | ODC-BY v1.0 — **verify** |

You must comply with each source corpus's terms independently of the MIT
licence on this code, and cite the source datasets as well as this work.

## If you publish a derived artifact

Any windowed benchmark you generate is a derivative of the source database and
inherits its terms, including ODC-BY's attribution requirement.
