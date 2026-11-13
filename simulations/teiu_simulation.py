"""
TEIU - Simulation 01 : Electro-Intelligence
Theory of Universal Electro-Intelligence (TEIU)
Auteur : Gregory Lombardo - Version 1.0 - 12 novembre 2026

Objectif :
Montrer que dans un reseau electrique simple, l'augmentation de
l'integration (Phi proxy) et de la complexite topologique (C)
ameliore la prediction, tandis que la dissipation (D) la degrade.
C'est la Prediction 1 du manifeste TEIU.

Formule testee : I = Phi * log(1+C) * 1/(1+D)

Usage :
    python teiu_simulation.py
    python teiu_simulation.py --n 120 --steps 800 --plot
"""

import argparse
import numpy as np

def build_network(n=100, p_recurrent=0.12, seed=0):
    rng = np.random.default_rng(seed)
    # Topologie : matrice d'adjacence aleatoire, asymetrique
    W = (rng.random((n, n)) < p_recurrent).astype(float)
    np.fill_diagonal(W, 0.0)
    # Poids synaptiques electriques : excitateurs et inhibiteurs
    weights = rng.normal(0.0, 1.0, size=(n, n)) * W
    # Normalisation spectrale pour stabilite
    spectral = np.max(np.abs(np.linalg.eigvals(weights))) + 1e-9
    weights = weights / (spectral * 1.15)
    return weights, W

def complexity_proxy(W):
    """C : complexite topologique. Degre moyen + clustering + diversite des motifs."""
    n = W.shape[0]
    deg = W.sum(axis=1) + W.sum(axis=0)
    mean_deg = float(np.mean(deg))
    # Clustering simple : triangles / triplets possibles
    W_bin = (W > 0).astype(float)
    triangles = float(np.trace(W_bin @ W_bin @ W_bin) / 6.0)
    density = float(W_bin.sum() / (n * (n - 1) + 1e-9))
    C = mean_deg * (1.0 + density * 10.0) + np.log1p(triangles)
    return float(C), float(mean_deg), float(density)

def phi_proxy(states):
    """
    Phi proxy : information integree approximee.
    On compare l'information mutuelle du systeme entier vs la somme
    des informations des deux moities coupees.
    states : (T, n) binaire ou continu seuille.
    """
    # Binarisation par mediane pour calcul d'entropie
    b = (states > np.median(states, axis=0, keepdims=True)).astype(int)
    T, n = b.shape
    def entropy_of_partition(cols):
        # Entropie jointe approximee par histogramme des motifs
        if len(cols) == 0:
            return 0.0
        # On limite a 12 colonnes pour eviter explosion combinatoire
        cols = cols[:12]
        codes = np.zeros(T, dtype=int)
        for i, c in enumerate(cols):
            codes += (b[:, c].astype(int) << i)
        _, counts = np.unique(codes, return_counts=True)
        p = counts / counts.sum()
        return float(-np.sum(p * np.log2(p + 1e-12)))
    total = entropy_of_partition(list(range(n)))
    half = n // 2
    left = entropy_of_partition(list(range(half)))
    right = entropy_of_partition(list(range(half, n)))
    phi = max(0.0, total - 0.5 * (left + right))
    # Normalisation par taille
    return float(phi)

def run_network(weights, steps=600, input_signal=None, dissipation=0.15):
    n = weights.shape[0]
    V = np.zeros(n)  # potentiel electrique
    states = np.zeros((steps, n))
    # Dynamique type Hodgkin-Huxley simplifiee : integrate-and-fire electrique
    tau = 8.0
    for t in range(steps):
        ext = 0.0
        if input_signal is not None:
            ext = input_signal[t] * 0.8
        # Courant recurrent + entree + bruit thermique (dissipation)
        I_rec = weights @ np.tanh(V)
        noise = np.random.randn(n) * dissipation
        dV = (-V + I_rec + ext + noise) / tau
        V = V + dV
        # Non-linearite : tanh comme potentiel d'action adouci
        states[t] = np.tanh(V)
    return states

def prediction_score(states, target):
    """Score de prediction : correlation entre etat du reseau et cible future."""
    # On utilise une regression lineaire simple : lecture (readout) electrique
    T = states.shape[0]
    # On predit target[t+1] a partir de states[t]
    X = states[:-1]
    y = target[1:]
    # Ridge
    lam = 1.0
    W_read = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)
    pred = X @ W_read
    # R2
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2) + 1e-9
    r2 = 1.0 - ss_res / ss_tot
    return float(max(0.0, r2)), W_read

def experiment():
    print("="*64)
    print("TEIU - Simulation 01 : I = Phi * log(1+C) * 1/(1+D)")
    print("="*64)
    rng = np.random.default_rng(42)
    steps = 800
    t = np.arange(steps)
    # Signal cible : combinaison de sinusoides (monde a predire)
    target = np.sin(0.05*t) + 0.5*np.sin(0.13*t + 1.0) + 0.3*np.sin(0.021*t)
    target = target / np.std(target)

    configs = [
        {"label": "Reseau pauvre : faible recurrence", "n": 80, "p": 0.03, "diss": 0.35},
        {"label": "Reseau moyen : recurrence moyenne", "n": 100, "p": 0.12, "diss": 0.15},
        {"label": "Reseau riche : haute recurrence, basse dissipation", "n": 120, "p": 0.22, "diss": 0.08},
        {"label": "Reseau riche mais dissipatif", "n": 120, "p": 0.22, "diss": 0.40},
    ]
    results = []
    for cfg in configs:
        W, Wbin = build_network(n=cfg["n"], p_recurrent=cfg["p"], seed=1)
        C, deg, dens = complexity_proxy(Wbin)
        states = run_network(W, steps=steps, input_signal=target, dissipation=cfg["diss"])
        phi = phi_proxy(states)
        r2, _ = prediction_score(states, target)
        I = phi * np.log1p(C) * 1.0/(1.0 + cfg["diss"]*5.0)
        results.append((cfg["label"], phi, C, cfg["diss"], r2, I))
        print(f"\n{cfg['label']}")
        print(f"  Phi proxy = {phi:.3f} | C = {C:.2f} | D = {cfg['diss']:.2f} | R2 prediction = {r2:.3f} | I TEIU = {I:.3f}")

    print("\n" + "-"*64)
    print("Interpretation TEIU : I eleve = prediction elevee. D eleve casse I meme si C est haut.")
    print("C'est la Prediction 1 et la Prediction 4 (substrat memristif/photonique).")
    print("Pour reproduire : voir README_GitHub_TEIU.md")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    res = experiment()
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            labels = [r[0].split(":")[0] for r in res]
            Is = [r[5] for r in res]
            R2s = [r[4] for r in res]
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            ax[0].bar(labels, Is, color="#1B6B7A")
            ax[0].set_title("I TEIU par configuration")
            ax[0].tick_params(axis='x', rotation=18)
            ax[1].bar(labels, R2s, color="#0F2A44")
            ax[1].set_title("Score de prediction R2")
            ax[1].tick_params(axis='x', rotation=18)
            plt.tight_layout()
            plt.savefig("teiu_sim01_result.png", dpi=180)
            print("Figure sauvegardee : teiu_sim01_result.png")
        except Exception as e:
            print("Plot echoue :", e)
