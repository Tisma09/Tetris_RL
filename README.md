<div align="center">

# 🎮 Tetris with Deep Q-Learning

[![en](https://img.shields.io/badge/lang-en-red.svg)](https://github.com/Tisma09/Tetris_RL/blob/master/README.md)
[![fr](https://img.shields.io/badge/lang-fr-blue.svg)](https://github.com/Tisma09/Tetris_RL/blob/master/README.fr.md)

An AI-powered Tetris implementation that learns to play by itself!

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

</div>

---

## 🚀 Overview

This project combines Tetris with Deep Q-Learning, creating an AI that learns to play the game through reinforcement learning.

### ✨ Features
- 🎮 Tetris game built with Pygame
- 🧠 Deep Q-Learning implementation
- 🚀 Multi-process training 
- 💾 Save/Load AI models

## 🛠️ Installation
```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

## 🎯 How to Use

Start the program:
```bash
python main.py
```

Then:
- Type 'y' to train a new AI
- Type 'n' to watch the trained AI play

## 📁 Project Structure
```
├── main.py           # Entry point
├── tetris_game.py    # Game implementation
├── dql_agent.py      # AI agent
├── train.py          # Training functions
├── tetris_env.py     # Gym environment
├── config.py         # Game settings
└── plot_scores.py    # Training graphs
```

---

<div align="center">

## Contributors

👤 [@Tismo](https://github.com/Tisma09)  
👤 [@Alexis](https://github.com/4lexisGo)  
👤 [@J0lataupe](https://github.com/J0lataupe)

</div>