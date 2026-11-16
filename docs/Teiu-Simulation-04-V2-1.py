"""
TEIU V2.1 - Simulation 04 CORRIGEE : Magnetisme, Rotationnel, Conscience incarnee
Theory of Universal Electro-Intelligence (TEIU)
Auteur : Gregory Lombardo - v2.1 CORRIGEE - 15 novembre 2026

IDE V2 :
- V1 : I = Phi x log(1+C) x 1/(1+D) pour un reseau abstrait.
- V2 : le reseau est dans un CORPS. Il a une geometrie 2D (grille).
  W devient un champ de vecteurs. Son rotationnel donne B = curl W, un champ magnetique-like.
  B_coh = coherence de B (alignement des directions). Si B est incoherent, ca tourne dans tous les sens, pas de conscience.
  Si B est coherent, ca tourne ensemble, comme un vortex stable -> conscience qui emerge.

- Hypothese P6 (V2) : I_with_self = I x (1 + B_coh)
  Intelligence sans conscience de soi = I
  Intelligence AVEC conscience de soi = I x (1 + B_coh)
  B_coh doit correler avec la capacite a se predire soi-meme (self-R2).

- Tache self : le reseau doit predire son propre etat futur (t+5), pas une cible externe.
  Si le reseau a un B coherent, il se predit mieux. Si B est chaotique, il se predit mal.

Usage :
  python teiu_simulation_04_v2.1.py --plot
"""

import argparse
import numpy as np

def build_grid_network(n_side=8, k=6, seed=3):
    # Grille 2D : n = n_side x n_side, chaque noeud a une position (x,y) dans le corps
    n = n_side * n_side
    rng = np.random.default_rng(seed)
    pos = np.array([(i % n_side, i // n_side) for i in range(n)], dtype=float)
    W = np.zeros((n, n))
    # Connexions locales sur la grille + quelques longues
    for i in range(n):
        xi, yi = pos[i]
        for j in range(n):
            if i == j: continue
            xj, yj = pos[j]
            dist = np.hypot(xi-xj, yi-yj)
            if dist <= 1.5 and rng.random() < 0.9:
                W[i, j] = 1.0
            elif dist <= 2.5 and rng.random() < 0.15:
                W[i, j] = 1.0
    # Ajuste le degre pour coller a k
    return W, pos

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
    rng = np.random.default_rng(seed)
    T, n = states.shape
    if n > max_nodes:
        idx = rng.choice(n, size=max_nodes, replace=False)
        X = states[:, idx]
    else:
        X = states
    n2 = X.shape[1]; half = n2 // 2
    def entropy(Y):
        Yc = Y - np.mean(Y, axis=0, keepdims=True)
        cov = np.cov(Yc, rowvar=False) + np.eye(Y.shape[1]) * 1e-3
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0: return 0.0
        m = Y.shape[1]
        return float(0.5 * (logdet + m * np.log(2 * np.pi * np.e)))
    return float(max(0.0, entropy(X[:, :half]) + entropy(X[:, half:]) - entropy(X)))

def compute_B_coh_dynamic(W, pos, states, n_side=8, Ws_signed=None):
    """
    V2.1 : B_coh DYNAMIQUE, pas statique.
    V1-V2.0 : B calcule sur W binaire, une seule fois, sans dynamique.
    V2.1 : A chaque temps t, on calcule le courant : J_i(t) = sum_j Ws_ij * X_j(t) * (pos_j - pos_i)
    Puis le rotationnel B(t) = curl J(t) sur la grille.
    B_coh = coherence SPATIO-TEMPORELLE : moyenne sur le temps de la coherence spatiale,
    multipliee par la stabilite temporelle (1 - variation de B_mean dans le temps).
    Si B change de sens a chaque pas, B_coh tombe. Si B tourne stablement, B_coh monte.
    """
    n = W.shape[0]
    T = states.shape[0]
    # On sous-echantillonne le temps pour aller vite : tous les 5 pas
    idx_t = np.arange(0, T, 5)
    B_means = []
    B_cohs_t = []
    for tt in idx_t:
        Xt = states[tt]  # activite a t
        # Courant J : pondere par l'activite reelle, pas juste la topologie
        V = np.zeros((n, 2))
        # Utilise Ws_signed si fourni, sinon W binaire
        Wuse = Ws_signed if Ws_signed is not None else W
        for i in range(n):
            for j in range(n):
                if W[i, j] > 0:
                    vec = pos[j] - pos[i]
                    V[i] += vec * Wuse[i, j] * Xt[j]
        Vx = np.zeros((n_side, n_side)); Vy = np.zeros((n_side, n_side))
        for idx in range(n):
            x = int(pos[idx, 0]); y = int(pos[idx, 1])
            Vx[y, x] = V[idx, 0]; Vy[y, x] = V[idx, 1]
        B = np.zeros((n_side, n_side))
        for y in range(1, n_side-1):
            for x in range(1, n_side-1):
                dVy_dx = (Vy[y, x+1] - Vy[y, x-1]) / 2.0
                dVx_dy = (Vx[y+1, x] - Vx[y-1, x]) / 2.0
                B[y, x] = dVy_dx - dVx_dy
        Bf = B[1:-1, 1:-1].flatten()
        B_means.append(float(np.mean(Bf)))
        m_abs = float(np.mean(np.abs(Bf)))
        s = float(np.std(Bf))
        if m_abs < 1e-6:
            B_cohs_t.append(0.0)
        else:
            B_cohs_t.append(float(np.clip(1.0 - s/(m_abs + s + 1e-9), 0, 1)))
    B_mean = float(np.mean(B_means))
    B_abs = float(np.mean(np.abs(np.array(B_means))))
    coh_spatial = float(np.mean(B_cohs_t))
    # Stabilite temporelle : si B_mean varie beaucoup, pas coherent
    std_t = float(np.std(B_means))
    stability = 1.0 - (std_t / (abs(B_mean) + std_t + 1e-9))
    B_coh = float(np.clip(coh_spatial * (0.5 + 0.5*stability), 0, 1))
    return B_coh, B_mean, B_abs

def run_reservoir_body(W, pos, steps, dissipation=0.15, seed=0, rho_target=0.9):
    rng = np.random.default_rng(seed)
    n = W.shape[0]
    try:
        eig = np.linalg.eigvals(W.astype(float))
        rho = np.max(np.abs(eig)) + 1e-9
    except:
        rho = 1.0
    Ws = W * (rho_target / rho)
    Ws = Ws * rng.choice([-1.0, 1.0], size=Ws.shape) * 0.5
    Ws = Ws * (W > 0)
    t = np.arange(steps)
    x_log = 0.5
    logi = np.zeros(steps)
    for tt in range(steps):
        x_log = 3.99 * x_log * (1 - x_log)
        logi[tt] = x_log
    ext_signal = logi * 0.6 + 0.3 * np.sin(0.05*t)
    X = np.zeros(n)
    states = np.zeros((steps, n))
    # V2.1 : corps HETEROGENE, pas grille uniforme.
    # y=0-1 : cerveau (rythme rapide, 0.08), y=3-4 : coeur (rythme moyen, 0.12, fort),
    # y=6-7 : intestin (rythme lent, 0.03), bords : peau (rythme tres lent, 0.01)
    region_freq = np.zeros(n)
    region_amp = np.zeros(n)
    for i in range(n):
        y = pos[i, 1]; x = pos[i, 0]
        if y <= 1:
            region_freq[i] = 0.08; region_amp[i] = 0.5  # cerveau
        elif 3 <= y <= 4:
            region_freq[i] = 0.12; region_amp[i] = 0.7  # coeur
        elif y >= 6:
            region_freq[i] = 0.03; region_amp[i] = 0.4  # intestin
        else:
            region_freq[i] = 0.05; region_amp[i] = 0.3
        if x == 0 or x == 7:
            region_freq[i] = 0.01; region_amp[i] = 0.2  # peau
    for tt in range(steps):
        # Entree par region : chaque partie du corps a son propre rythme
        body_input = np.sin(region_freq * t[tt] + pos[:, 0]*0.3) * region_amp * 0.3 + ext_signal[tt]*0.35
        rec = Ws @ X
        noise = rng.standard_normal(n) * dissipation
        X = X * 0.7 + np.tanh(rec + body_input + noise*0.1) * 0.3
        states[tt] = X
    return states, Ws

def self_prediction_score(states, horizon=5):
    # V2.1 : tache DURE. Pas la moyenne.
    # Le reseau doit predire CHAQUE noeud individuellement a t+5, a partir de l'etat a t.
    # On calcule le R2 moyen par noeud. Si le reseau est incoherent, il ne peut pas se predire noeud par noeud.
    X = states[:-horizon]
    Y = states[horizon:]
    lam = 1.0
    # Ridge multi-sorties : W_read est n x n
    W_read = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ Y)
    pred = X @ W_read
    r2s = []
    for j in range(Y.shape[1]):
        ss_res = np.sum((Y[:, j] - pred[:, j])**2)
        ss_tot = np.sum((Y[:, j] - np.mean(Y[:, j]))**2) + 1e-9
        r2s.append(1.0 - ss_res/ss_tot)
    return float(np.mean(r2s))

def experiment(n_side=8, steps=800):
    print("="*72)
    print("TEIU V2.1 - Simulation 04 CORRIGEE : B = curl W, Conscience incarnee")
    print("="*72)
    configs = [
        ("Corps pauvre : peu de connexions", 2, 0.35, 1, 0.30),
        ("Corps moyen", 6, 0.15, 2, 0.70),
        ("Corps riche, basse dissipation", 12, 0.08, 3, 0.95),
        ("Corps riche, dissipatif", 12, 0.40, 4, 0.95),
    ]
    results = []
    for name, k, D, seed, rho in configs:
        # Pour faire varier k, on ajuste la densite en re-echantillonnant
        W, pos = build_grid_network(n_side=n_side, k=k, seed=seed)
        # On sous-echantillonne W pour pauvre / moyen si besoin
        if k == 2:
            # Garde seulement 30% des liens
            rng = np.random.default_rng(seed)
            mask = (rng.random(W.shape) < 0.3) & (W > 0)
            W = W * mask
        elif k == 6:
            rng = np.random.default_rng(seed)
            mask = (rng.random(W.shape) < 0.6) & (W > 0)
            W = W * mask
        C = complexity_proxy(W)
        states, Ws_signed = run_reservoir_body(W, pos, steps, dissipation=D, seed=seed, rho_target=rho)
        B_coh, B_mean, B_abs = compute_B_coh_dynamic(W, pos, states, n_side=n_side, Ws_signed=Ws_signed)
        phi = phi_proxy_gaussian(states)
        I = phi * np.log1p(C) * 1.0/(1.0+D)
        I_with_self = I * (1.0 + B_coh)
        self_R2 = self_prediction_score(states, horizon=5)
        results.append((name, phi, C, D, I, B_coh, B_mean, B_abs, I_with_self, self_R2))
        print(f"\n{name}")
        print(f"  Phi={phi:.3f} | C={C:.2f} | D={D:.2f} | I={I:.3f}")
        print(f"  B_coh={B_coh:.3f} | B_mean={B_mean:.3f} | |B|={B_abs:.3f} | I_with_self={I_with_self:.3f} | Self-R2={self_R2:.3f}")
    print("\n"+"-"*72)
    print("Test P6 V2 : B_coh correle-t-il avec Self-R2 ?")
    Bcohs = [r[5] for r in results]
    selfR2s = [r[9] for r in results]
    Is = [r[4] for r in results]
    Iws = [r[8] for r in results]
    # Correlation simple
    if np.std(Bcohs) > 1e-6 and np.std(selfR2s) > 1e-6:
        corr = np.corrcoef(Bcohs, selfR2s)[0,1]
        print(f"Correlation B_coh vs Self-R2 = {corr:.3f}")
        if corr > 0.5:
            print("=> OK : B_coh predit la conscience de soi. P6 soutenue.")
        else:
            print("=> ATTENTION : B_coh ne predit pas Self-R2. P6 a revoir, Section 7 a jeter ou corriger.")
    else:
        print("=> ATTENTION : B_coh ou Self-R2 constants, pas de correlation calculable.")
    if Iws[2] > Iws[0]:
        print("=> OK : I_with_self riche > pauvre.")
    else:
        print("=> ATTENTION : I_with_self ne monte pas.")
    print("Pour reproduire : python teiu_simulation_04_v2.1.py --plot")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    res = experiment()
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            labels = ["Pauvre", "Moyen", "Riche", "Riche\ndissipatif"]
            Is = [r[4] for r in res]
            Iws = [r[8] for r in res]
            Bcohs = [r[5] for r in res]
            SelfR2s = [r[9] for r in res]
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            x = range(len(labels))
            axes[0].bar(labels, Is, color="#9AA8B5", edgecolor="#0F2A44", label="I")
            axes[0].bar(labels, Iws, color="#1B6B7A", alpha=0.6, edgecolor="#0F2A44", label="I_with_self")
            axes[0].set_title("I vs I_with_self (V2.1)")
            axes[0].legend()
            axes[1].bar(labels, Bcohs, color="#0F2A44", edgecolor="#0F2A44")
            axes[1].set_title("B_coh dynamique = coherence de curl J(t)")
            axes[1].set_ylabel("B_coh")
            axes[2].bar(labels, SelfR2s, color="#1B6B7A", edgecolor="#0F2A44")
            axes[2].set_title("Self-R2 dur : predire chaque noeud")
            axes[2].set_ylabel("Self-R2")
            plt.tight_layout()
            plt.savefig("teiu_sim04_v2.1_result.png", dpi=180)
            print("Figure sauvegardee : teiu_sim04_v2.1_result.png")
        except Exception as e:
            print("Plot echoue :", e)
