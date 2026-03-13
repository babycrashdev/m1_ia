# Journal de Bord - Projet M1 IA : Classification de Séries Temporelles

Ce fichier sert à documenter chaque étape du projet pour faciliter la rédaction du rapport final de 25 pages.

## Plan Global
1. [ ] Configuration de l'environnement (Colab + GitHub)
2. [ ] Exploration et Préparation des Données (EDA + Preprocessing)
3. [ ] Modèle 1 : LSTM (RNN)
4. [ ] Modèle 2 : CNN 1D
5. [ ] Modèle 3 : Transformers
6. [ ] Évaluation et Comparaison
7. [ ] Rédaction du Rapport Final

---

## État d'avancement

### 📜 Étape 1 : Configuration de l'environnement (13/03/2026)
*   **Action :** Création du Notebook `Projet_ia.ipynb` et du Journal de bord.
*   **Infrastructure :** Activation du GPU T4 sur Google Colab pour accélérer l'entraînement des modèles.
*   **Données :** Récupération via le dépôt [ts-data-for-workload-classification](https://github.com/phuongntmse/ts-data-for-workload-classification).
### 🔍 Étape 2 : Exploration et Préparation des Données (13/03/2026)

#### Choix des caractéristiques (Features)
Nous avons sélectionné 4 variables clés (colonnes) pour représenter l'état du système :
1. `CpuALL_usage` : Charge globale du processeur.
2. `memory_usage_perce` : Occupation de la RAM.
3. `read_rate_during_time` : Vitesse de lecture disque.
4. `write_rate_during_time` : Vitesse d'écriture disque.

#### Normalisation (Scaling)
Utilisation du `MinMaxScaler` pour ramener toutes les valeurs dans l'intervalle **[0, 1]**. Cela empêche une variable (comme la mémoire en octets) de dominer une autre (comme le CPU en %) à cause de son échelle numérique plus grande.

#### Stratégie de Fenêtrage (Sliding Window)
*   **Taille de fenêtre :** 50 timesteps.
*   **Raison :** Permettre aux modèles (LSTM, CNN) de capturer la dynamique temporelle. Une activité ne se définit pas par un instant T, mais par son évolution sur une durée.

#### Encodage des Labels
*   **CPU : 0** | **DISK : 1** | **MEMORY : 2**
*   Transformation des dossiers de données en classes numériques pour l'apprentissage supervisé.

*   **Objectif Suivant :** Séparation Train/Test et création du modèle LSTM (Étape 3).
