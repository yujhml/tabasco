"""Unguided TABASCO sampling.
"""

import torch

from tabasco.chem.convert import MoleculeConverter


def sample_unguided(lightning_module, num_mols, num_steps, batch_size=500):
    mol_converter = MoleculeConverter()
    all_mols = []
    remaining = num_mols

    while remaining > 0:
        bs = min(batch_size, remaining)
        with torch.no_grad():
            batch = lightning_module.sample(batch_size=bs, num_steps=num_steps)
        mols = mol_converter.from_batch(batch)
        all_mols.extend(mols)
        remaining -= bs

    return {
        "mols": all_mols,
        "total_forward_calls": num_mols * num_steps,
    }
