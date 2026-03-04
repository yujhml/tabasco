"""Bayesian Predictive Potentials for FK Steering (BPP-FK).
"""

import numpy as np
import torch
from scipy.stats import norm as _norm_dist
from tensordict import TensorDict

from sklearn.linear_model import BayesianRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tabasco.chem.convert import MoleculeConverter
from .rewards import REWARD_FUNCTIONS


@torch.no_grad()
def extract_features(x_t: TensorDict, t_scalar: float) -> np.ndarray:
    coords = x_t["coords"]
    atomics = x_t["atomics"]
    pad_mask = x_t["padding_mask"]

    B, N, C = atomics.shape
    valid_mask = ~pad_mask

    feats_list = []
    for b in range(B):
        vm = valid_mask[b]
        n_atoms = vm.sum().float().item()
        if n_atoms < 1:
            n_atoms = 1.0

        c = coords[b][vm]
        a = atomics[b][vm]

        c_mean = c.mean(dim=0)
        c_std = c.std(dim=0).clamp(min=1e-6)
        norms = c.norm(dim=1)
        norm_mean = norms.mean()
        norm_std = norms.std().clamp(min=1e-6)

        if c.shape[0] > 1:
            dists = torch.cdist(c.unsqueeze(0), c.unsqueeze(0)).squeeze(0)
            triu_idx = torch.triu_indices(c.shape[0], c.shape[0], offset=1)
            pw = dists[triu_idx[0], triu_idx[1]]
            mean_pw = pw.mean()
            max_pw = pw.max()
        else:
            mean_pw = torch.tensor(0.0, device=c.device)
            max_pw = torch.tensor(0.0, device=c.device)

        a_probs = a.float().softmax(dim=-1).mean(dim=0)

        feat = torch.cat([
            torch.tensor([t_scalar, n_atoms], device=c.device),
            c_mean, c_std,
            mean_pw.unsqueeze(0), max_pw.unsqueeze(0),
            norm_mean.unsqueeze(0), norm_std.unsqueeze(0),
            a_probs,
        ])
        feats_list.append(feat)

    return torch.stack(feats_list).cpu().numpy().astype(np.float32)


@torch.no_grad()
def evaluate_particles(model, mol_converter, x_t, t_batch, reward_fn):
    N = x_t["coords"].shape[0]
    pred = model._call_net(x_t, t_batch)
    mols = mol_converter.from_batch(pred)
    rewards = np.zeros(N)
    for i, mol in enumerate(mols):
        rewards[i] = reward_fn(mol)
    return rewards



def integrate_segment(model, x_t, T, step_start, step_end, N):
    n_calls = 0
    for i in range(step_start, step_end):
        t = T[i].expand(N)
        dt = (T[i + 1] - T[i]).expand(N)
        x_t = model._step(x_t, t, dt)
        n_calls += N
    return x_t, n_calls



def resample_tensordict(x_t, indices):
    if isinstance(indices, np.ndarray):
        idx = torch.from_numpy(indices).long().to(x_t.device)
    else:
        idx = indices.long().to(x_t.device)
    return TensorDict(
        {
            "coords": x_t["coords"][idx].clone(),
            "atomics": x_t["atomics"][idx].clone(),
            "padding_mask": x_t["padding_mask"][idx].clone(),
        },
        batch_size=len(indices),
    )



class TwistPredictor:
    def __init__(self, name: str):
        self.name = name

    def fit(self, U: np.ndarray, y: np.ndarray):
        pass

    def predict_mean_std(self, U: np.ndarray):
        pass


class BLRPredictor(TwistPredictor):

    def __init__(self):
        super().__init__("BLR")
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("blr", BayesianRidge()),
        ])

    def fit(self, U, y):
        self.model.fit(U, y)

    def predict_mean_std(self, U):
        m, s = self.model.predict(U, return_std=True)
        return m.astype(np.float32), np.maximum(s, 1e-6).astype(np.float32)


class GPPredictor(TwistPredictor):

    def __init__(self, random_state=42):
        super().__init__("GP")
        kernel = (
            ConstantKernel(1.0, (1e-2, 1e2))
            * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
            + WhiteKernel(1e-3, (1e-6, 1e-1))
        )
        self.scaler = StandardScaler()
        self.gp = GaussianProcessRegressor(
            kernel=kernel, alpha=0.0, normalize_y=True,
            random_state=random_state,
        )

    def fit(self, U, y):
        Us = self.scaler.fit_transform(U)
        self.gp.fit(Us, y)

    def predict_mean_std(self, U):
        Us = self.scaler.transform(U)
        m, s = self.gp.predict(Us, return_std=True)
        return m.astype(np.float32), np.maximum(s, 1e-6).astype(np.float32)


class RFPredictor(TwistPredictor):

    def __init__(self, n_estimators=500, random_state=42):
        super().__init__("RF")
        self.scaler = StandardScaler()
        self.rf = RandomForestRegressor(
            n_estimators=n_estimators, min_samples_leaf=3,
            random_state=random_state, n_jobs=-1,
        )

    def fit(self, U, y):
        Us = self.scaler.fit_transform(U)
        self.rf.fit(Us, y)

    def predict_mean_std(self, U):
        Us = self.scaler.transform(U)
        preds = np.stack(
            [tree.predict(Us) for tree in self.rf.estimators_], axis=0
        )
        m = preds.mean(axis=0)
        s = preds.std(axis=0)
        return m.astype(np.float32), np.maximum(s, 1e-6).astype(np.float32)


class TabPFNPredictor(TwistPredictor):

    def __init__(self):
        super().__init__("TabPFN")
        from tabpfn import TabPFNRegressor
        self.scaler = StandardScaler()
        _dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = TabPFNRegressor(device=_dev, ignore_pretraining_limits=True)

    def fit(self, U, y):
        Us = self.scaler.fit_transform(U)
        self.model.fit(Us, y)

    def predict_mean_std(self, U):
        Us = self.scaler.transform(U)
        m = self.model.predict(Us)
        s = np.full_like(m, 0.25)
        return m.astype(np.float32), np.maximum(s, 1e-6).astype(np.float32)


ESTIMATOR_REGISTRY = {
    "blr": BLRPredictor,
    "gp": GPPredictor,
    "rf": RFPredictor,
    "tabpfn": TabPFNPredictor,
}


def make_predictor(name: str, random_state: int = 42) -> TwistPredictor:
    key = name.lower()
    cls = ESTIMATOR_REGISTRY[key]
    if key == "gp":
        return cls(random_state=random_state)
    elif key == "rf":
        return cls(random_state=random_state)
    else:
        return cls()


def compute_raw_bpp_quantile(
    predictor: TwistPredictor,
    U: np.ndarray,
    alpha: float = 0.25,
    clip_c: float = 2.0,
):
    m, s = predictor.predict_mean_std(U)
    z_alpha = _norm_dist.ppf(alpha)
    q = m + z_alpha * s

    q = q - np.mean(q)
    q = np.clip(q, -clip_c, clip_c)
    return q, m, s


def normalize_log_weights(log_w: np.ndarray) -> np.ndarray:
    lw = np.asarray(log_w, dtype=np.float64)
    lw = lw - np.max(lw)
    w = np.exp(lw)
    return w / w.sum()


def systematic_resample(weights: np.ndarray) -> np.ndarray:
    n = len(weights)
    cdf = np.cumsum(weights)
    positions = (np.random.rand() + np.arange(n)) / n
    idx = np.zeros(n, dtype=np.int64)
    i, j = 0, 0
    while i < n:
        if positions[i] <= cdf[j]:
            idx[i] = j
            i += 1
        else:
            j += 1
    return idx


def sample_bpp(
    lightning_module,
    estimator_name="rf",
    n_particles=100,
    n_warm=200,
    resample_times=(0.70, 0.80, 0.88, 0.93, 0.96, 0.98, 0.99),
    lmbda=7.0,
    alpha=0.25,
    clip_c=2.0,
    tau_max=1.0,
    reward="qed",
    num_steps=100,
    batch_size=100,
):

    device = next(lightning_module.parameters()).device
    mol_converter = MoleculeConverter()
    reward_fn = REWARD_FUNCTIONS[reward]
    model = lightning_module.model

    T = model._get_sample_schedule(num_steps=num_steps).to(device)
    T_np = T.cpu().numpy()

    resample_steps = []
    for rt in sorted(resample_times):
        step_idx = int(np.argmin(np.abs(T_np - rt)))
        step_idx = max(1, min(step_idx, len(T_np) - 2))
        if step_idx not in resample_steps:
            resample_steps.append(step_idx)
    resample_steps = sorted(resample_steps)
    actual_resample_times = [float(T_np[s]) for s in resample_steps]

    total_forward_calls = 0
    training_forward_calls = 0

    all_train_features = []
    all_train_labels = []

    remaining = n_warm
    while remaining > 0:
        bs = min(batch_size, remaining)
        x_t = model._sample_noise_like_batch(batch_size=bs)

        prev_step = 1
        feat_blocks = []
        for rs in resample_steps:
            x_t, calls = integrate_segment(model, x_t, T, prev_step, rs, bs)
            total_forward_calls += calls
            training_forward_calls += calls
            feat_blocks.append(extract_features(x_t, float(T_np[rs])))
            prev_step = rs

        x_t, calls = integrate_segment(model, x_t, T, prev_step, len(T) - 1, bs)
        total_forward_calls += calls
        training_forward_calls += calls

        mols = mol_converter.from_batch(x_t)
        rewards = np.array([reward_fn(m) for m in mols], dtype=np.float32)
        y = lmbda * rewards

        for fb in feat_blocks:
            all_train_features.append(fb)
            all_train_labels.append(y)

        remaining -= bs

    U_train = np.concatenate(all_train_features, axis=0)
    y_train = np.concatenate(all_train_labels, axis=0)

    predictor = make_predictor(estimator_name, random_state=42)
    predictor.fit(U_train, y_train)

    mu_check, _ = predictor.predict_mean_std(U_train[:min(500, len(U_train))])
    y_check = y_train[:min(500, len(y_train))]
    r2 = float(1.0 - np.var(mu_check - y_check) / (np.var(y_check) + 1e-12))

    def _guided_sample():
        nonlocal total_forward_calls
        N = n_particles
        x_t = model._sample_noise_like_batch(batch_size=N)

        ess_hist = []
        tau_hist = []
        unique_hist = []

        prev_step = 1
        for k, rs in enumerate(resample_steps):
            x_t, calls = integrate_segment(model, x_t, T, prev_step, rs, N)
            total_forward_calls += calls

            t_val = float(T_np[rs])
            U = extract_features(x_t, t_val)

            q_raw, _m, _s = compute_raw_bpp_quantile(
                predictor, U, alpha=alpha, clip_c=clip_c,
            )

            tau_k = tau_max
            log_potential = tau_k * q_raw
            w = normalize_log_weights(log_potential)
            ess = 1.0 / (np.sum(w ** 2) + 1e-12)
            ess_hist.append(float(ess))
            tau_hist.append(float(tau_k))

            if tau_k > 1e-6:
                idx = systematic_resample(w)
                x_t = resample_tensordict(x_t, idx)
                n_unique = int(len(np.unique(idx)))
            else:
                n_unique = N
            unique_hist.append(n_unique)

            prev_step = rs

        x_t, calls = integrate_segment(
            model, x_t, T, prev_step, len(T) - 1, N
        )
        total_forward_calls += calls

        return x_t, ess_hist, tau_hist, unique_hist

    x_t, ess_hist, tau_hist, unique_hist = _guided_sample()
    all_mols = mol_converter.from_batch(x_t)
    final_rewards = np.array(
        [reward_fn(m) for m in all_mols], dtype=np.float64
    )

    inference_forward_calls = total_forward_calls - training_forward_calls

    return {
        "mols": all_mols,
        "final_rewards": final_rewards.tolist(),
        "total_forward_calls": total_forward_calls,
        "training_forward_calls": training_forward_calls,
        "inference_forward_calls": inference_forward_calls,
        "n_resample_checkpoints": len(resample_steps),
        "actual_resample_times": actual_resample_times,
        "ess_history": ess_hist,
        "tau_history": tau_hist,
        "n_unique_history": unique_hist,
        "surrogate_train_r2": r2,
        "estimator_name": estimator_name,
        "mean_ess": float(np.mean(ess_hist)) if ess_hist else 0.0,
        "mean_tau": float(np.mean(tau_hist)) if tau_hist else 0.0,
        "min_n_unique": int(np.min(unique_hist)) if unique_hist else 0,
    }
