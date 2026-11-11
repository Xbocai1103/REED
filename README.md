# REED

**Refining Evidence from Encoding to Decision in Medical Time Series Classification**  

## Overview

REED refines time-series representations through three components:

- **HESO:** organizes temporal and channel evidence, optionally combining current states with local transitions.
- **CLUC:** applies bounded feedback based on differences between adjacent layers' updates.
- **DNSS:** regularizes classifier-related residual feature energy during training.

## Installation

Install a compatible PyTorch build, then run:

```bash
pip install -r requirements.txt
```

## Data

The paper evaluates **ADFTD, APAVA, TDBRAIN, PTB, PTB-XL, FLAAP, and UCI-HAR**. Prepare the benchmark-processed `.npy` files in `dataset/<DATASET>/` as expected by `data_provider/data_loader.py`. Data files are not included.

## Run

For the provided APAVA configuration (five seeds, starting from 42):

```bash
bash scripts/APAVA.sh
```

The main training entry point is `run.py`; see `python run.py --help` for available arguments. Other dataset configurations are described in Table II of the paper, but their launch scripts are not included in this release.


