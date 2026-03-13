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

### 🧠 Étape 3 : Modèle 1 - LSTM (Long Short-Term Memory) (13/03/2026)

#### Pourquoi le LSTM ?
Le LSTM est un type de réseau de neurones récurrent (RNN) conçu pour traiter des séquences de données. Sa particularité est de posséder des "portes" (gates) qui lui permettent de décider quelles informations du passé garder ou oublier. C'est idéal pour notre projet car une activité système (ex: une lecture disque) laisse une trace qui s'étale dans le temps.

#### Architecture du modèle
*   **Couche d'entrée :** Reçoit des tenseurs de forme (50, 4).
*   **Couche LSTM (64 unités) :** Analyse la dynamique de la séquence.
*   **Dropout (20%) :** Technique de régularisation consistant à désactiver aléatoirement certains neurones pour forcer le modèle à être plus robuste et éviter le "sur-apprentissage" (overfitting).
*   **Couche Dense (32 neurones, ReLU) :** Aide à la prise de décision complexe.
*   **Couche de Sortie (3 neurones, Softmax) :** Produit une distribution de probabilité sur les 3 classes (CPU, DISK, MEMORY).

#### Entraînement
*   **Fonction de perte :** `sparse_categorical_crossentropy` (adaptée aux labels entiers).
*   **Optimiseur :** `Adam` (standard efficace pour ajuster les poids du réseau).

#### Résultats de l'entraînement
*   **Précision finale (Accuracy) :** ~90.7%
*   **Précision de validation (Val_Accuracy) :** ~91.2%
*   **Observation :** La précision de validation est légèrement supérieure à la précision d'entraînement, ce qui indique que le modèle n'est pas en sur-apprentissage (pas d'overfitting) et qu'il généralise très bien. La courbe de perte (loss) a montré une descente régulière, validant le choix du taux d'apprentissage et de l'optimiseur Adam.

### 📊 Étape 4 : Analyse de performance et Matrice de Confusion (13/03/2026)

#### Matrice de Confusion
La matrice de confusion permet de visualiser les erreurs spécifiques du modèle. 
*   **Diagonale :** Représente les bonnes prédictions.
*   **Hors-diagonale :** Montre les confusions (ex: une activité Disque prédite comme CPU).
*   **Interprétation :** Une forte concentration sur la diagonale confirme la capacité du LSTM à extraire des signatures temporelles uniques pour chaque type d'activité système.

#### Métriques de Classification
*   **Précision (Precision) :** Capacité du modèle à ne pas prédire une classe à tort.
*   **Rappel (Recall) :** Capacité du modèle à détecter tous les exemples d'une classe.
*   **F1-Score :** Moyenne harmonique de la précision et du rappel, donnant une vision globale de la performance par classe.

#### Analyse de l'erreur (LSTM)
L'analyse fine du rapport de classification montre une faiblesse sur la catégorie **MEMORY** (Recall de 0.57). 
*   **Interprétation :** Une partie importante des traces mémoire est confondue avec l'activité CPU. Cela s'explique par la signature hybride de certaines opérations de gestion de mémoire vive qui sollicitent le processeur de manière similaire à une tâche de fond.
*   **Point fort :** La catégorie **DISK** est parfaitement identifiée (F1-score de 1.00), montrant une signature temporelle très discriminante.

### 🖼️ Étape 5 : Modèle 2 - CNN 1D (Convolutional Neural Network) (13/03/2026)

#### Pourquoi le CNN 1D ?
Habituellement utilisé pour l'image (2D), le CNN en 1D applique des filtres glissants le long de la série temporelle. Au lieu de s'appuyer sur une "mémoire" (comme le LSTM), il cherche des **signatures locales** (formes, pics, transitions rapides). Cette approche est souvent plus performante pour détecter des comportements brusques.

#### Architecture du modèle
*   **Conv1D (64 filtres, kernel 3) :** Applique 64 filtres différents pour détecter des motifs simples.
*   **MaxPooling1D :** Réduit la résolution temporelle par deux en ne conservant que les valeurs maximales (les plus significatives).
*   **Flatten :** Transforme les cartes de caractéristiques en un vecteur plat.
*   **Dense & Dropout :** Couches classiques de classification finale.

#### Résultats de l'entraînement (CNN 1D)
*   **Précision finale :** ~92.4%
*   **Précision de validation :** ~92.1%
*   **Observation :** Le CNN 1D montre une meilleure capacité à différencier la mémoire (Recall 0.61 contre 0.57 pour le LSTM). Les filtres de convolution semblent plus efficaces pour extraire des signatures fréquentielles spécifiques aux accès RAM.

### ⚡ Étape 6 : Modèle 3 - Time Series Transformer (Attention Mechanism) (13/03/2026)

#### Pourquoi le Transformer ?
Contrairement aux modèles précédents, le Transformer utilise le mécanisme d'**Attention**. Cela permet au modèle de pondérer l'importance de chaque point dans la fenêtre de 50. Par exemple, si un pic de lecture disque apparaît au début de la fenêtre, l'Attention peut décider que c'est l'information la plus cruciale pour classer la séquence entière.

#### Architecture
*   **Multi-Head Attention :** Utilise 4 "têtes" d'attention travaillant en parallèle pour regarder les données sous différents angles.
*   **GlobalAveragePooling1D :** Condense l'information de toute la séquence de manière plus globale que le MaxPooling du CNN.
*   **LayerNormalization :** Technique cruciale pour stabiliser l'apprentissage des Transformers en normalisant les activations entre les couches.

#### Résultats de l'entraînement (Transformer)
*   **Précision finale (Accuracy) :** ~90.6%
*   **Précision de validation (Val_Accuracy) :** ~91.0%
*   **Analyse du F1-Score :** Le modèle avec mécanisme d'attention (Transformer) atteint un F1-score de 0.92 pour le CPU et 1.00 pour le DISK. Cependant, pour la catégorie MEMORY, le rappel (recall) est de 0.59.
*   **Observation :** Bien que les Transformers soient extrêmement puissants sur des données textuelles ou des séries temporelles longues, sur cette tâche spécifique de classification de traces CPU/Disque/Mémoire, l'approche locale par filtres (CNN 1D) reste la plus performante. Le Transformer semble avoir des difficultés similaires au LSTM pour distinguer les accès mémoire complexes de l'activité CPU résiduelle.

### 🏆 Étape 7 : Conclusion et Synthèse des Modèles (13/03/2026)

#### Récapitulatif des Modèles
Le meilleur modèle global, avec la meilleure capacité de discernement sur l'ensemble des classes (y compris la catégorie difficile "MEMORY"), est le **Modèle 2 (CNN 1D)**.
1.  **CNN 1D :** Val_Accuracy ~92%, excellent pour identifier rapidement les signatures de pics à l'aide de ses filtres glissants.
2.  **LSTM :** Val_Accuracy ~91%, très bon mais peine légèrement plus sur la différenciation CPU vs Memory.
3.  **Transformer (Attention) :** Val_Accuracy ~91%, performances comparables au LSTM. Le mécanisme d'attention global n'a pas apporté de plus-value déterminante par rapport à la détection locale de motifs du CNN.

*   **Objectif Suivant :** Étape finale, sauvegarde des modèles et perspectives (application future).

