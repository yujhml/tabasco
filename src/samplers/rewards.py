from rdkit.Chem import Crippen, Descriptors, Lipinski


def reward_qed(mol):
    if mol is None:
        return 0.0
    return Descriptors.qed(mol)


def reward_lipinski(mol):
    if mol is None:
        return 0.0
    score = 0
    if Descriptors.ExactMolWt(mol) < 500:
        score += 1
    if Lipinski.NumHDonors(mol) <= 5:
        score += 1
    if Lipinski.NumHAcceptors(mol) <= 10:
        score += 1
    logp = Crippen.MolLogP(mol)
    if -2 <= logp <= 5:
        score += 1
    if Lipinski.NumRotatableBonds(mol) <= 10:
        score += 1
    return score / 5.0


def reward_logp_target(mol, lo=-2.0, hi=5.0):
    if mol is None:
        return 0.0
    logp = Crippen.MolLogP(mol)
    if lo <= logp <= hi:
        return 1.0
    return max(0.0, 1.0 - 0.1 * min(abs(logp - lo), abs(logp - hi)))


REWARD_FUNCTIONS = {
    "qed": reward_qed,
    "lipinski": reward_lipinski,
    "logp": reward_logp_target,
}
