import matplotlib.pyplot as plt
import numpy as np
import csv
import os

def read_scores_from_file(filename):
    """ Lire les scores depuis un fichier CSV. Chaque ligne contient un numéro d'épisode et son score. """
    episodes = []
    scores = []
    if os.path.exists(filename):
        with open(filename, mode='r') as f:
            reader = csv.reader(f)
            for row in reader:
                try:
                    episode = int(row[0])  # Numéro de l'épisode
                    score = float(row[1])  # Score de l'épisode
                    episodes.append(episode)
                    scores.append(score)
                except ValueError as e:
                    print(f"Erreur de conversion pour la ligne '{row}': {e}")
    return episodes, scores

def plot_scores(num_simulations):
    """ Générer un graphique des scores combinés de toutes les simulations """
    all_episodes = []
    all_scores = []

    # Lire les scores de toutes les simulations et les combiner
    for i in range(num_simulations):
        log_file = f'logs/scores_sim_{i}.csv'  # Charger depuis le dossier 'logs'
        episodes, scores = read_scores_from_file(log_file)
        
        all_episodes.extend(episodes)  # Ajouter tous les épisodes
        all_scores.extend(scores)  # Ajouter tous les scores

    # Créer un nuage de points pour toutes les simulations combinées
    plt.figure(figsize=(10, 6))

    # Tracer les croix (points) pour tous les épisodes combinés
    plt.scatter(all_episodes, all_scores, marker='x', s=50, alpha=0.7, label="Scores combinés")

    # Ajouter une courbe de tendance globale (régression linéaire)
    episodes_array = np.array(all_episodes)
    scores_array = np.array(all_scores)

    # Calcul de la régression linéaire (ajustement de polynôme de degré 1)
    coefficients = np.polyfit(episodes_array, scores_array, 1)
    polynomial = np.poly1d(coefficients)

    # Tracer la courbe de tendance
    trendline = polynomial(episodes_array)
    plt.plot(episodes_array, trendline, color='red', label="Courbe de tendance", linestyle='--')

    # Ajouter des détails au graphique
    plt.xlabel("Numéro d'épisode")
    plt.ylabel("Score")
    plt.title("Scores combinés de toutes les simulations avec courbe de tendance")
    plt.legend()
    plt.grid(True)  # Ajouter la grille
    plt.ylim(-2500, 1000)  # Cadre des y entre -2500 et 1000
    plt.show()

if __name__ == "__main__":
    plot_scores(2)  # Afficher les résultats pour 20 simulations
