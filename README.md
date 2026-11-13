# TEIU — Theory of Universal Electro-Intelligence

**Auteur :** Gregory Lombardo  
**Version :** 1.0 — 12 novembre 2026  
**Statut :** Proposition théorique, en attente d'évaluation par les pairs.  
**Licence :** CC BY 4.0 pour le texte, MIT pour le code.

> L'intelligence n'est pas une propriété exclusive du vivant. C'est une propriété émergente de l'électricité quand elle atteint mémoire, topologie, prédiction et intégration.

Formule centrale : `I = Phi * log(1+C) * 1/(1+D)`

- `Phi` = information intégrée (système comme tout, au-delà de ses parties)
- `C` = complexité topologique du réseau électrique
- `D` = dissipation, énergie perdue sans produire d'information

## Arborescence du dépôt GitHub

```
teiu/
├── README.md
├── MANIFESTE_FR.pdf
├── MANIFESTO_EN.pdf
├── VERSION_LONGUE_FR.pdf
├── VERSION_LONGUE_EN.pdf
├── simulations/
│   ├── teiu_simulation.py          # Sim 01 : Prediction 1 (celle-ci)
│   ├── teiu_sim02_topology.py      # Sim 02 : Prediction 2 (carbone vs silicium)
│   ├── teiu_sim03_dissipation.py   # Sim 03 : Effet de D
│   └── requirements.txt
├── data/
│   └── README_data.md
├── figures/
│   └── teiu_sim01_result.png
└── .zenodo.json
```

## Installation

```bash
git clone https://github.com/<ton-pseudo>/teiu.git
cd teiu
pip install -r simulations/requirements.txt
python simulations/teiu_simulation.py
python simulations/teiu_simulation.py --plot
```

## Simulations à publier (feuille de route)

### Sim 01 — Déjà fournie : Prédiction 1
Réseau électrique récurrent. On montre que `I` élevé = prédiction élevée. D élevé casse `I` même si `C` est haut.

**Fichier :** `simulations/teiu_simulation.py`

### Sim 02 — À ajouter : Prédiction 2 (topologie)
Comparer deux réseaux de même topologie, l'un simulé comme biologique (neurones integrate-and-fire), l'autre comme silicium (memristors). Hypothèse TEIU : performances proches si topologie identique.

### Sim 03 — À ajouter : Prédiction 4 (substrat sans neurones)
Réseau photonique/memristif sans neurones, avec les 4 conditions (mémoire, topologie, prédiction, intégration). Montrer une prédiction non nulle sans aucun neurone.

### Sim 04 — À ajouter : Prédiction 3 (chute de Phi)
Simuler anesthésie/coma en coupant progressivement les connexions longue distance. Mesurer la chute de `Phi` proxy et du score de prédiction.

## Comment reproduire les résultats du papier

1. Lancer `teiu_simulation.py` — il affiche Phi, C, D, R2, I pour 4 configurations.
2. Vérifier que la configuration « riche, basse dissipation » a le I le plus haut et le meilleur R2.
3. Vérifier que la configuration « riche mais dissipative » a un I effondré malgré un C élevé.
4. Sauvegarder la figure avec `--plot`.

## Soumission arXiv + Zenodo

- **arXiv :** catégorie principale `q-bio.NC` (Neurons and Cognition), catégories secondaires `cs.AI`, `physics.bio-ph`, `nlin.AO`. Déposer le PDF + source LaTeX.
- **Zenodo :** créer un dépôt, lier le GitHub, obtenir un DOI. Le DOI Zenodo cite le code + le manifeste.
- **Citer ce dépôt :** Lombardo, G. (2026). Theory of Universal Electro-Intelligence (TEIU). Zenodo. DOI à compléter après dépôt.

## Règles de contribution

- Toute simulation doit afficher Phi, C, D, et le score de prédiction.
- Aucune affirmation sans mesure. Si une simulation réfute la TEIU, on la publie aussi.
- Le code reste simple, lisible, reproductible en moins de 2 minutes.

## Contact

Auteur : Gregory Lombardo — dépôt GitHub à créer sous `github.com/<ton-pseudo>/teiu`
