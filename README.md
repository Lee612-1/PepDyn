# PepDyn

## Overview

PepDyn is a processed molecular dynamics (MD) dataset for protein-peptide complexes.
Each sample corresponds to one protein-peptide complex trajectory and stores structural,
dynamic, and energetic annotations for downstream machine learning or data analysis.


Dataset statistics:

- Total samples in LMDB: `2979`
- Training keys: `2829`
- Test keys: `150`

## Quick Start

### 1. Create the conda environment

```bash
conda env create -f environment.yml
conda activate pepdyn
```

### 2. Download the PepDyn dataset from Hugging Face

The processed dataset can be downloaded from the Hugging Face dataset repository `lyt12/PepDyn`.

```bash
pip install -U huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='lyt12/PepDyn', repo_type='dataset', local_dir='PepDyn_dataset')"
```

After download, the local layout should look like:

```text
PepDyn/
├── PepDyn_dataset/
│   ├── PepDyn_dataset.lmdb/
│   ├── train_keys.txt
│   └── test_keys.txt
├── configs/
├── pepdyn/
└── scripts/
```

### 3. Update config paths

Before training, set the dataset paths in `configs/rmsf_full.yaml` and `configs/mmgbsa_full.yaml` to the downloaded PepDyn files, for example:

```yaml
data:
  lmdb_path: PepDyn/PepDyn_dataset.lmdb

split:
  train_keys_path: PepDyn/train_keys.txt
  test_keys_path: PepDyn/test_keys.txt
```

### 4. Run the AI baselines

Train the RMSF baseline:

```bash
cd /home/lyt/misato-dataset/pepdyn_code
python scripts/train_rmsf.py --config configs/rmsf_full.yaml
```

Train the MM/GBSA baseline:

```bash
cd /home/lyt/misato-dataset/pepdyn_code
python scripts/train_mmgbsa.py --config configs/mmgbsa_full.yaml
```

## Data File Layout

```text
PepDyn/
├── PepDyn_dataset.lmdb/
│   ├── data.mdb
│   └── lock.mdb
├── train_keys.txt
├── test_keys.txt
└── README.md
```

## Split Files

The two key files list sample IDs that match LMDB entry keys exactly.

- `train_keys.txt`: sample IDs used for the training split
- `test_keys.txt`: sample IDs used for the test split

Each line contains one sample key, for example:

```text
A_B_pdb1awu
H_A_pdb7m53
```

You can use these files to build deterministic train/test subsets after loading the LMDB.

## Data Format

Each LMDB entry is stored as a Python dictionary serialized with `pickle`.

```python
{
  "metadata": {...},
  "coords": np.ndarray,           # shape: (n_frames, n_atoms, 3)
  "atom_rmsf": np.ndarray,        # shape: (n_atoms,)
  "frame_features": {...},
  "residue_features": {...},
  "pair_features": {...}
}
```

## Field Description

### 1. Metadata

```python
"metadata": {
  "sample_id": str,
  "pdbid": str,
  "protein_chain": str,
  "peptide_chain": str,
  "n_frames": int,
  "n_atoms": int,
  "protein_atom_count": int,
  "peptide_atom_count": int,
  "peptide_start_index": int,
  "chain_start_indices": List[int],
  "atom_names": List[str],
  "residue_names": List[str],
  "residue_ids": List[int]
}
```

### 2. Coordinates

```python
"coords": np.ndarray  # (n_frames, n_atoms, 3)
```

- 3D coordinates for all atoms across all trajectory frames
- Unit: Angstrom (`A`)

### 3. Atom-level Dynamics

```python
"atom_rmsf": np.ndarray  # (n_atoms,)
```

- Per-atom root mean square fluctuation computed from the trajectory

### 4. Frame-level Features

```python
"frame_features": {
  "COM_distance": List[float],
  "RMSD_complex": List[float],
  "RMSD_peptide": List[float],
  "contacts_count": List[int],
  "min_heavy_dist": List[float],
  "interface_res_pairs_count": List[int],
  "buried_SASA": List[float],
  "mmgbsa": List[float],
}
```

- Each list has length `n_frames`
- Values describe frame-wise geometric or energetic properties along the trajectory

### 5. Residue-level Features

```python
"residue_features": {
  "protein": {
    "resid": List[int],
    "resname": List[str],
    "rmsf_A": List[float],
    "contact_occupancy": List[float]
  },
  "peptide": {
    "resid": List[int],
    "resname": List[str],
    "rmsf_A": List[float],
    "contact_occupancy": List[float]
  }
}
```

- Separate summaries are provided for receptor residues and peptide residues
- `contact_occupancy` describes the fraction of frames in which a residue participates in interface contact

### 6. Residue-pair Features

```python
"pair_features": {
  "receptor_resid": List[int],
  "receptor_resname": List[str],
  "peptide_resid": List[int],
  "peptide_resname": List[str],
  "residue_pair_occupancy": List[float]
}
```

- Residue-pair interface statistics between receptor and peptide chains
- `residue_pair_occupancy` records the fraction of frames in which the residue pair is in contact

## LMDB Special Keys

The LMDB also includes a few reserved keys:

| Key | Format | Description |
| --- | --- | --- |
| `__keys__` | pickled `List[str]` | List of all sample IDs |
| `__len__` | UTF-8 bytes | Number of samples |
| `__meta__` | JSON string | Dataset-level metadata |

For this release, `__meta__` is:

```json
{"num_samples": 2979, "format": "pickle"}
```

## How to Load

```python
import json
import lmdb
import pickle

env = lmdb.open("PepDyn_dataset.lmdb", subdir=True, readonly=True, lock=False)

with env.begin() as txn:
    keys = pickle.loads(txn.get(b"__keys__"))
    num_samples = int(txn.get(b"__len__").decode())
    meta = json.loads(txn.get(b"__meta__").decode())
    sample = pickle.loads(txn.get(keys[0].encode()))

print(num_samples)
print(meta)
print(sample["metadata"]["sample_id"])
print(sample["coords"].shape)
```

To use the provided split files:

```python
with open("train_keys.txt") as f:
    train_keys = [line.strip() for line in f if line.strip()]

with env.begin() as txn:
    train_sample = pickle.loads(txn.get(train_keys[0].encode()))
```

## Notes

- Sample IDs in `train_keys.txt` and `test_keys.txt` are aligned with LMDB keys exactly
- The dataset is distributed as a preprocessed LMDB for efficient random access
- This repository is intended as a dataset release; model training instructions are not included in this card
