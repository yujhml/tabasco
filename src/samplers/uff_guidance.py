"""UFF-bound gradient guidance sampler.
"""

import torch

from tabasco.callbacks.ema import EMAOptimizer
from tabasco.chem.convert import MoleculeConverter
from tabasco.sample.guided_sampling import GuidedSampling
from tabasco.sample.interpolant_guidance import UFFBoundGuidance


def sample_uff_guidance(
    lightning_module,
    batch_size,
    num_steps,
    step_switch=90,
    guidance_lr=0.01,
    guidance_steps=1,
    regress_to_center=True,
    ema_strength=0.999,
):
    device = next(lightning_module.parameters()).device
    mol_converter = MoleculeConverter()

    adam_opt = torch.optim.Adam(lightning_module.model.net.parameters())
    ema_optimizer = EMAOptimizer(adam_opt, device, ema_strength)

    uff_bound_guidance = UFFBoundGuidance(
        mol_converter=mol_converter,
        lr=guidance_lr,
        n_steps=guidance_steps,
        regress_to_center=regress_to_center,
    )

    unguided_sampling = GuidedSampling(
        lightning_module=lightning_module,
        inpaint_function=None,
        steer_interpolant=None,
        ema_optimizer=ema_optimizer,
    )

    guided_sampling = GuidedSampling(
        lightning_module=lightning_module,
        inpaint_function=None,
        steer_interpolant=uff_bound_guidance,
        ema_optimizer=ema_optimizer,
    )

    noisy_batch = lightning_module.model._sample_noise_like_batch(
        batch_size=batch_size
    )

    T = lightning_module.model._get_sample_schedule(num_steps=num_steps)
    T = T.to(noisy_batch.device)[:, None]
    T = T.repeat(1, noisy_batch["coords"].shape[0])

    partially_denoised = unguided_sampling.sample(
        x_t=noisy_batch, timesteps=T[:step_switch]
    )

    boosted_batch = guided_sampling.sample(
        x_t=partially_denoised.detach().clone(),
        timesteps=T[step_switch:],
    )

    generated_mols = mol_converter.from_batch(boosted_batch)

    unguided_calls = step_switch * batch_size
    guided_calls = (num_steps - step_switch) * batch_size * (1 + guidance_steps)
    total_forward_calls = unguided_calls + guided_calls

    return {
        "mols": generated_mols,
        "total_forward_calls": total_forward_calls,
    }
