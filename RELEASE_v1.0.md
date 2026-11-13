# TEIU v1.0 - Premier jet - Release pour Zenodo

Date : 13 novembre 2026
Auteur : Gregory Lombardo

## Ce que contient cette release
- README.md
- simulations/teiu_simulation.py - Sim 01 : I = Phi x log(1+C) x 1/(1+D)
- simulations/requirements.txt
- requirements.txt
- docs/TEIU_Version_Longue_FR.pdf
- docs/TEIU_Version_Longue_EN.pdf
- figures/teiu_sim01_result.png - Figure generee par : python simulations/teiu_simulation.py --plot
- LICENSE-CODE-MIT.txt - Code sous MIT
- LICENSE-TEXT-CC-BY-4.0.txt - Textes sous CC BY 4.0

## Licences
- Code : MIT - voir LICENSE-CODE-MIT.txt
- Textes : CC BY 4.0 - voir LICENSE-TEXT-CC-BY-4.0.txt

## Comment reproduire la figure
```
pip install -r requirements.txt
python simulations/teiu_simulation.py --plot
```
La figure teiu_sim01_result.png est generee dans figures/.

## Pour Zenodo
Cette release v1.0 est celle a lier a Zenodo.
Sur Zenodo : Creer un depot, choisir GitHub, selectionner le depot teiu, tag v1.0.
Le DOI Zenodo sera : 10.5281/zenodo.XXXXXXX (a completer apres publication Zenodo)

## Prochaines etapes
- v1.1 : ajout DOI Zenodo dans README et PDFs
- v1.2 : ajout Sim 02 et Sim 03
