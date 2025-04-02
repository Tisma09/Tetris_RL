import matplotlib.pyplot as plt
import numpy as np
import csv
import os
from datetime import datetime

def creation_repertoire_logs(parent_folder, num_simulations, num_episodes):
    """Créer un dossier pour enregistrer les fichiers de logs, avec un nom basé sur la date et l'heure actuelles et les paramètres passés"""
    # Obtenir la date et l'heure actuelles
    current_time = datetime.now()
    # Formater la date et l'heure au format JJ_MM_AAAA__HH_MM
    formatted_time = current_time.strftime("%d_%m_%Y__%H_%M")
    
    # Ajouter les variables num_simulations et num_episodes à la fin du nom du dossier
    folder_name = f"{formatted_time}_sim_{num_simulations}_ep_{num_episodes}"
    
    # Spécifier le chemin du dossier à créer à l'intérieur du répertoire parent
    folderpath = os.path.join(parent_folder, folder_name)
    
    # Créer le dossier
    os.makedirs(folderpath, exist_ok=True)

    return folderpath

def creation_fichier_scores(folderpath, sim_id):
    """Créer un fichier CSV pour enregistrer les scores des épisodes de chaque simulations"""
    # Définir le nom du fichier CSV pour cette simulation
    path_log_file = f'{folderpath}/scores_sim_{sim_id}.csv'

    return path_log_file

def sauvegarde_scores(path_log_file, episode, total_reward):
    """Enregistrer le score final de l'épisode dans le fichier CSV si un fichier est spécifié"""
    if path_log_file:
        with open(path_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode + 1, total_reward])  # Enregistrer le score final de l'épisode
    else:
        print("Aucun fichier de log spécifié. Le score n'a pas été enregistré.")

def read_scores_from_file(filename):
    """Lire les scores depuis un fichier CSV. Chaque ligne contient un numéro d'épisode et son score."""
    """Utiliser uniquement pour l'éxécution de ce fichier isolé."""
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

def plot_scores_csv(folderpath):
    """Générer un graphique des scores combinés de toutes les simulations"""
    all_episodes = []
    all_scores = []

    # Compter le nombre de simulations
    # Lister tous les fichiers dans le dossier
    fichiers = os.listdir(folderpath)
    # Filtrer uniquement les fichiers avec l'extension '.csv'
    fichiers_csv = [fichier for fichier in fichiers if fichier.endswith('.csv')]
    num_simulations = len(fichiers_csv)

    # Lire les scores de toutes les simulations et les combiner
    for i in range(num_simulations):
        log_file = os.path.join(folderpath, f'scores_sim_{i}.csv')  # Charger depuis le dossier 'folderpath'
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
    #plt.ylim(-2500, 1000)  # Cadre des y entre -2500 et 1000
    plt.show()

def plot_scores_list(matrix_reward):
    # Obtenir le nombre de simulations et d'épisodes
    num_simulations, num_episodes = matrix_reward.shape

    # Créer une liste d'épisodes (répéter les indices d'épisodes pour chaque simulation)
    all_episodes = np.tile(np.arange(num_episodes), num_simulations)

    # Applatir la matrice de scores pour obtenir tous les scores
    all_scores = matrix_reward.flatten()

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
    plt.show

def dernier_dossier_cree(parent_folder='logs'):
    """Trouver le dernier dossier créé dans le répertoire parent."""
    """Utiliser uniquement pour l'éxécution de ce fichier isolé."""
    # Lister tous les éléments dans le dossier parent
    all_entries = os.listdir(parent_folder)
    
    # Filtrer uniquement les dossiers (pas les fichiers)
    folders = [entry for entry in all_entries if os.path.isdir(os.path.join(parent_folder, entry))]
    
    # Vérifier si la liste de dossiers est vide
    if not folders:
        print(f"Aucun dossier trouvé dans {parent_folder}")
        return None
    
    # Trier les dossiers par date de création, du plus récent au plus ancien
    folders.sort(key=lambda folder: os.path.getctime(os.path.join(parent_folder, folder)), reverse=True)
    
    # Le dernier dossier créé est maintenant le premier de la liste triée
    dernier_dossier = folders[0]
    return dernier_dossier

if __name__ == "__main__":
    folderpath = dernier_dossier_cree()  # Récupérer le dernier dossier créé
    if folderpath:
        # Exécuter plot_scores en passant le folderpath pour accéder aux fichiers dans le dossier spécifique
        plot_scores_csv(folderpath)
