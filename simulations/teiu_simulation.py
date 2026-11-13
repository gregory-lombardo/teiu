"""
TEIU - Simulation 01 v1.1 CORRIGEE : I = Phi x log(1+C) x 1/(1+D)
Theory of Universal Electro-Intelligence (TEIU)
Auteur : Gregory Lombardo - v1.1 - 13 novembre 2026

CORRECTIONS par rapport a v1.0 :
- v1.0 : tache trop facile (sinusoides pures) -> R2 ~ 0.99 partout, ne distingue rien.
  v1.1 : tache chaotique (logistique chaotique + sinus + bruit) -> R2 distingue les reseaux.
- v1.0 : proxy de Phi par binarisation -> Phi = 0 pour les grands reseaux, sature.
  v1.1 : proxy de Phi gaussien par covariance (entropie) -> Phi monte avec C, ne tombe plus a 0.
- v1.0 : I = 0 pour 3 reseaux sur 4 a cause de Phi = 0.
  v1.1 : I doit monter avec C quand D est basse, et chuter quand D est haute.

Usage :
    python teiu_simulation_v1.1.py
    python teiu_simulation_v1.1.py --plot
    # Pour remplacer la v1.0 sur GitHub : renommer en teiu_simulation.py et pousser
"""

import argparse
import numpy as np

def build_network(n=60, k=6, p_rewire=0.12, seed=3):
    rng = np.random.default_rng(seed)
    W = np.zeros((n, n))
    for i in range(n):
        for d in range(1, k//2 + 1):
            j = (i + d) % n
            W[i, j] = 1.0
            W[j, i] = 1.0
    # Rewire small-world
    for i in range(n):
        for j in range(n):
            if W[i, j] > 0 and rng.random() < p_rewire:
                W[i, j] = 0.0; W[j, i] = 0.0
                new = rng.integers(0, n)
                if new != i:
                    W[i, new] = 1.0; W[new, i] = 1.0
    np.fill_diagonal(W, 0.0)
    return W

def complexity_proxy(W):
    n = W.shape[0]
    deg = W.sum(axis=1) + W.sum(axis=0)
    mean_deg = float(np.mean(deg))
    density = float(W.sum() / (n * (n - 1) + 1e-9))
    Wb = (W > 0).astype(float)
    triangles = float(np.trace(Wb @ Wb @ Wb) / 6.0)
    C = mean_deg * (1.0 + density * 10.0) + np.log1p(triangles)
    return float(C)

def phi_proxy_gaussian(states, max_nodes=24, seed=0):
    """
    v1.1 : Phi par entropie gaussienne, pas par binarisation.
    Phi = H(left) + H(right) - H(total)
    Si les deux moities sont independantes, H(total) = H(left)+H(right) -> Phi = 0.
    Si elles sont integrees, H(total) < somme -> Phi > 0.
    Ne tombe plus a 0 quand n grandit.
    """
    rng = np.random.default_rng(seed)
    T, n = states.shape
    # Sous-echantillonne pour stabiliser la covariance
    if n > max_nodes:
        idx = rng.choice(n, size=max_nodes, replace=False)
        X = states[:, idx]
    else:
        X = states
        idx = np.arange(n)
    n2 = X.shape[1]
    half = n2 // 2
    def entropy(Y):
        # Y : T x m
        Yc = Y - np.mean(Y, axis=0, keepdims=True)
        # Covariance avec regularisation
        cov = np.cov(Yc, rowvar=False) + np.eye(Y.shape[1]) * 1e-3
        # logdet stable
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            return 0.0
        m = Y.shape[1]
        # Entropie gaussienne : 0.5 * log det(2 pi e Sigma)
        return float(0.5 * (logdet + m * np.log(2 * np.pi * np.e)))
    H_total = entropy(X)
    H_left = entropy(X[:, :half])
    H_right = entropy(X[:, half:])
    phi = H_left + H_right - H_total
    return float(max(0.0, phi))

def chaotic_target(steps, seed=0):
    """
    v1.1 : cible chaotique, pas sinusoides pures.
    Logistique chaotique r=3.99 + sinus + bruit.
    Difficile a predire parfaitement, donc R2 distingue les reseaux.
    """
    rng = np.random.default_rng(seed)
    x = 0.5
    logi = np.zeros(steps)
    for t in range(steps):
        x = 3.99 * x * (1 - x)
        logi[t] = x
    t = np.arange(steps)
    sine = 0.4 * np.sin(0.05 * t) + 0.2 * np.sin(0.13 * t + 1.0)
    noise = rng.standard_normal(steps) * 0.15
    target = logi * 1.2 + sine + noise
    target = (target - np.mean(target)) / (np.std(target) + 1e-9)
    return target

def run_reservoir(W, steps, target, dissipation=0.15, seed=0, rho_target=0.9):
    rng = np.random.default_rng(seed)
    n = W.shape[0]
    # v1.1b : normalisation par rayon spectral, sinon le reseau riche sature et perd la memoire
    # On calcule le rayon spectral de W, on le ramene a rho_target < 1
    try:
        eig = np.linalg.eigvals(W.astype(float))
        rho = np.max(np.abs(eig)) + 1e-9
    except:
        rho = 1.0
    Ws = W * (rho_target / rho)
    # Poids aleatoires signes pour eviter la saturation tanh : +/- 
    Ws = Ws * rng.choice([-1.0, 1.0], size=Ws.shape) * 0.5
    # On garde la topologie : 0 reste 0
    Ws = Ws * (W > 0)
    X = np.zeros(n)
    states = np.zeros((steps, n))
    for t in range(steps):
        ext = target[t] * 0.5
        rec = Ws @ X
        noise = rng.standard_normal(n) * dissipation
        # Leaky : 0.7 memoire + 0.3 nouveau, evite la saturation
        X = X * 0.7 + np.tanh(rec + ext + noise * 0.1) * 0.3
        states[t] = X
    return states

def prediction_score(states, target):
    X = states[:-1]
    y = target[1:]
    lam = 1.0
    W_read = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)
    pred = X @ W_read
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2) + 1e-9
    r2 = 1.0 - ss_res / ss_tot
    return float(r2)  # peut etre negatif si tres mauvais, on garde tel quel pour voir

def experiment(n=60, steps=800):
    print("="*72)
    print("TEIU - Simulation 01 v1.1 CORRIGEE : I = Phi x log(1+C) x 1/(1+D)")
    print("Corrections : cible chaotique + Phi gaussien (pas de binarisation)")
    print("="*72)
    target = chaotic_target(steps, seed=0)

    configs = [
        ("Reseau pauvre : faible recurrence", 2, 0.35, 1, 0.30),
        ("Reseau moyen : recurrence moyenne", 6, 0.15, 2, 0.70),
        ("Reseau riche : haute recurrence, basse dissipation", 12, 0.08, 3, 0.95),
        ("Reseau riche mais dissipatif", 12, 0.40, 4, 0.95),
    ]
    results = []
    for name, k, D, seed, rho in configs:
        W = build_network(n=n, k=k, p_rewire=0.12, seed=seed)
        C = complexity_proxy(W)
        states = run_reservoir(W, steps, target, dissipation=D, seed=seed, rho_target=rho)
        phi = phi_proxy_gaussian(states)
        r2 = prediction_score(states, target)
        I = phi * np.log1p(C) * 1.0 / (1.0 + D)
        results.append((name, phi, C, D, r2, I))
        print(f"\n{name}")
        print(f"  Phi = {phi:.3f} | C = {C:.2f} | D = {D:.2f} | R2 = {r2:.3f} | I = {I:.3f}")

    print("\n" + "-"*72)
    print("Interpretation v1.1 :")
    # Verifications automatiques
    phis = [r[1] for r in results]
    r2s = [r[4] for r in results]
    Is = [r[5] for r in results]
    if phis[2] > phis[0]:
        print("=> OK : Phi riche > Phi pauvre. Le proxy ne s'effondre plus.")
    else:
        print("=> ATTENTION : Phi riche <= Phi pauvre. Le proxy a encore un probleme.")
    if max(r2s) - min(r2s) > 0.05:
        print(f"=> OK : R2 distingue les reseaux (ecart {max(r2s)-min(r2s):.3f}). La tache n'est plus trop facile.")
    else:
        print("=> ATTENTION : R2 trop similaire partout. Tache encore trop facile ou trop dure.")
    if Is[2] > Is[0] and Is[2] > Is[3]:
        print("=> OK : I riche basse dissipation > I pauvre et > I riche dissipatif. D casse I comme prevu.")
    else:
        print("=> ATTENTION : I ne suit pas la prediction. A investiguer.")
    print("Pour reproduire : voir README et TEIU_arXiv.tex Section 5")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    res = experiment(n=args.n, steps=args.steps)
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            labels = ["Pauvre", "Moyen", "Riche", "Riche\ndissipatif"]
            Is = [r[5] for r in res]
            R2s = [r[4] for r in res]
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].bar(labels, Is, color=["#9AA8B5", "#1B6B7A", "#0F2A44", "#9AA8B5"], edgecolor="#0F2A44")
            axes[0].set_title("I TEIU par configuration (v1.1)")
            axes[0].set_ylabel("I")
            axes[1].bar(labels, R2s, color=["#9AA8B5", "#1B6B7A", "#0F2A44", "#9AA8B5"], edgecolor="#0F2A44")
            axes[1].set_title("Score de prediction R2 (v1.1)")
            axes[1].set_ylabel("R2")
            axes[1].set_ylim(min(0, min(R2s)-0.1), 1.0)
            plt.tight_layout()
            plt.savefig("teiu_sim01_v1.1_result.png", dpi=180)
            print("Figure sauvegardee : teiu_sim01_v1.1_result.png")
        except Exception as e:
            print("Plot echoue :", e)
