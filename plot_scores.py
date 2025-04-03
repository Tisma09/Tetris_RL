import matplotlib.pyplot as plt
import numpy as np
import csv
import os
from datetime import datetime

def create_dirs_logs(policy_name, parent_folder='logs'):
    """Créer un dossier pour enregistrer les fichiers de logs, avec un nom basé sur la date et l'heure actuelles et les paramètres passés"""
    # Obtenir la date et l'heure actuelles
    current_time = datetime.now()

    # Formater la date et l'heure au format JJ_MM_AAAA__HH_MM
    formatted_time = current_time.strftime("%d_%m_%Y__%H_%M")
    
    # Ajouter le nom de la politique d'entrainement
    folder_name = f"{formatted_time}_policy_{policy_name}"
    
    # Spécifier le chemin du dossier à créer à l'intérieur du répertoire parent
    folderpath = os.path.join(parent_folder, folder_name)
    
    # Créer le dossier
    os.makedirs(folderpath, exist_ok=True)

    print(f"Dossier créé : {folderpath}")

    return folderpath

def create_files_scores(folderpath, matrix_reward):
    """Créer un fichier CSV et enregistre les scores des épisodes de chaque simulation."""
    
    # Vérifier que le dossier existe, sinon le créer
    os.makedirs(folderpath, exist_ok=True)

    # Obtenir le nombre de simulations et d'épisodes
    num_simulations, num_episodes = matrix_reward.shape

    # Sauvegarde des fichiers CSV
    for sim in range(num_simulations):
        filename = os.path.join(folderpath, f"simulation_{sim + 1}.csv")  # Chemin du fichier

        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Episode", "Reward"])  # En-tête

            for episode in range(num_episodes):
                # Convertir en entier et flottant pour assurer un formatage compatible avec `read_scores_from_file`
                writer.writerow([int(episode + 1), float(matrix_reward[sim, episode])])

    print(f"{num_simulations} créés fichiers CSV avec succès dans {folderpath} !")

# Fonction pour lire les scores depuis un fichier CSV
def read_scores_from_file(filename):
    """Lire les scores depuis un fichier CSV. Chaque ligne contient un numéro d'épisode et son score."""
    """Utiliser uniquement pour l'exécution de ce fichier isolé."""
    
    episodes = []
    scores = []
    
    if os.path.exists(filename):
        with open(filename, mode='r', newline='') as f:
            reader = csv.reader(f)
            next(reader, None)  # Ignorer l'en-tête
            
            for row in reader:
                try:
                    episode = int(row[0])  # Numéro de l'épisode
                    score = float(row[1])  # Score de l'épisode
                    episodes.append(episode)
                    scores.append(score)
                except (ValueError, IndexError) as e:
                    print(f"Erreur de conversion pour la ligne '{row}': {e}")
    
    return episodes, scores

def extract_csv(folderpath):
    """Extraire les informations des fichiers CSV"""
    """Utiliser uniquement pour l'éxécution de ce fichier isolé."""
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

    all_episodes += 1
    plot_scores(all_episodes, all_scores)

def extract_list(matrix_reward):
    """Extraire les informations de la liste de liste"""
    """Utiliser uniquement pour l'éxécution de ce fichier isolé."""

    # Obtenir le nombre de simulations et d'épisodes
    num_simulations, num_episodes = matrix_reward.shape

    # Créer une liste d'épisodes (répéter les indices d'épisodes pour chaque simulation)
    all_episodes = np.tile(np.arange(num_episodes), num_simulations)

    # Applatir la matrice de scores pour obtenir tous les scores
    all_scores = matrix_reward.flatten()

    plot_scores(all_episodes, all_scores)

def plot_scores(all_episodes, all_scores):
    """Tracer les scores de chaque épisodes de toutes les simulations"""
    # Tracer les croix (points) pour tous les épisodes combinés
    plt.scatter(all_episodes, all_scores, marker='x', s=50, alpha=0.7, label="Scores combinés")

    if len(all_episodes) > 1:
        # Ajouter une courbe de tendance globale (régression linéaire)
        episodes_array = np.array(all_episodes)
        scores_array = np.array(all_scores)

        # Calcul de la régression linéaire (ajustement de polynôme de degré 1)
        coefficients = np.polyfit(episodes_array, scores_array, 1)
        polynomial = np.poly1d(coefficients)

        # Tracer la courbe de tendance
        trendline = polynomial(episodes_array)
        plt.plot(episodes_array, trendline, color='red', label="Courbe de tendance", linestyle='--')

    stats(all_scores)

    # Ajouter des détails au graphique
    plt.xlabel("Numéro d'épisode")
    plt.ylabel("Score")
    plt.title("Scores combinés de toutes les simulations avec courbe de tendance")
    plt.legend()
    plt.grid(True)  # Ajouter la grille
    plt.show()

def stats(all_scores):
    """Calculer les statistiques de base sur les scores"""
    """Utiliser uniquement pour l'éxécution de ce fichier isolé."""

    # Obtenir le nombre de score supérieurs à 0
    num_positive_scores = (all_scores > 0).sum()
    print(f"Nombre de scores supérieurs à 0 : {num_positive_scores} sur {len(all_scores)}")

    # Trouver le maximum
    min_score = np.min(all_scores)
    max_score = np.max(all_scores)
    
    print("Score minimum:", min_score)
    print("Score maximum:", max_score)

    # Calculer la moyenne, la médiane et l'écart type
    mean_score = np.mean(all_scores)
    median_score = np.median(all_scores)
    std_dev_score = np.std(all_scores)

    # Afficher les résultats
    print(f"Moyenne des scores : {mean_score:.2f}")
    print(f"Médiane des scores : {median_score:.2f}")
    print(f"Écart type des scores : {std_dev_score:.2f}")


def last_folder_created(parent_folder='logs'):
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
    last_folder = folders[0]
    return last_folder

if __name__ == "__main__":
    folderpath = last_folder_created()  # Récupérer le dernier dossier créé
    if folderpath:
        # Exécuter plot_scores en passant le folderpath pour accéder aux fichiers dans le dossier spécifique
        extract_csv(folderpath)
