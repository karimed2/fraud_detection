# Fraud Detection – Détection de Fraudes Bancaires

## Description du projet
Ce projet vise à développer une **solution complète de détection de fraudes bancaires** en combinant machine learning et interface web.  
L’objectif est de permettre à un utilisateur (banque ou client) de charger ses transactions et d’identifier automatiquement celles qui sont suspectes.

Le projet se compose de trois parties principales :  
1. **Modèle de Machine Learning**  
   - Détecte les transactions frauduleuses à partir d’un jeu de données historique.
   - Algorithmes utilisés : Random Forest, XGBoost, Réseaux de Neurones.
   - Gestion du déséquilibre entre transactions normales et frauduleuses.

2. **Interface Web**  
   - Permet à l’utilisateur de charger un fichier CSV contenant ses transactions.
   - Affiche les résultats sous forme de tableau avec les transactions suspectes mises en évidence.
   - Possibilité de télécharger un rapport des transactions frauduleuses.

3. **API REST avec FastAPI**  
   - Expose le modèle ML via des endpoints `/predict_one` et `/predict_batch`.
   - Rend le modèle exploitable par d’autres applications bancaires.
   - Permet une intégration rapide dans les systèmes d’information des banques.

---

## Technologies utilisées
| Technologie | Usage |
|-------------|-------|
| Python      | Développement du modèle et de l’API |
| FastAPI     | Création de l’API REST pour exposer le modèle |
| Pandas / NumPy | Manipulation des données et prétraitement |
| Joblib      | Sauvegarde et chargement du modèle ML |
| Google Colab | Entraînement et test du modèle ML |
| HTML / CSS / JS | Interface utilisateur pour la visualisation des résultats |

---

## Structure du projet
