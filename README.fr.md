<div align="center">

# 🎮 Tetris avec Deep Q-Learning

[![en](https://img.shields.io/badge/lang-en-red.svg)](https://github.com/Tisma09/Tetris_RL/blob/master/README.md)
[![fr](https://img.shields.io/badge/lang-fr-blue.svg)](https://github.com/Tisma09/Tetris_RL/blob/master/README.fr.md)

Un jeu Tetris avec une IA qui apprend à jouer toute seule !

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

</div>

---

## 🚀 Présentation

Ce projet combine Tetris avec du Deep Q-Learning, créant une IA qui apprend à jouer au jeu par renforcement.

### ✨ Fonctionnalités
- 🎮 Jeu Tetris avec Pygame
- 🧠 Implémentation Deep Q-Learning
- 🚀 Entraînement multi-processus
- 💾 Sauvegarde/Chargement des modèles

## 🛠️ Installation
```bash
# Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

## 🎯 Utilisation

Lancer le programme :
```bash
python main.py
```

Puis :
- Taper 'y' pour entraîner une nouvelle IA
- Taper 'n' pour regarder l'IA jouer

## 📁 Structure du Projet
```
├── main.py           # Point d'entrée
├── tetris_game.py    # Implémentation du jeu
├── dql_agent.py      # Agent IA
├── train.py          # Fonctions d'entraînement
├── tetris_env.py     # Environnement Gym
├── config.py         # Paramètres du jeu
└── plot_scores.py    # Graphiques d'entraînement
```

---

<div align="center">

## Contributeurs

👤 [@Tismo](https://github.com/Tisma09)  
👤 [@Alexis](https://github.com/4lexisGo)  
👤 [@J0lataupe](https://github.com/J0lataupe)

</div>