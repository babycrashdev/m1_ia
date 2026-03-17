# Plan de Rédaction - Rapport Projet M1 IA

Le rapport [rapport.tex](file:///d:/Documents/BaByDev/m1_ia/rapport.tex) sera structuré pour répondre aux exigences du cahier des charges (~25 pages, méthodologie détaillée, analyse des résultats).

## 1. Structure du Document LaTeX
- **Préambule** : Configuration des packages (geometry, biblatex, graphicx, hyperref, listings pour le code, etc.).
- **Page de Garde** : Titre, noms, institution (M1 IA), date.
- **Table des Matières**.

## 2. Introduction (2-3 pages)
- Présentation du sujet : Importance de la supervision système.
- Objectif : Classification automatique des traces (CPU, Disque, Mémoire).
- Présentation des données : Provenance (GitHub de Mme Thi Mai Phuong).

## 3. Analyse des Données et Préparation (4-5 pages)
- **Analyse Exploratoire (EDA)** : Visualisation des signatures types (pics CPU, flux disque).
- **Choix des Caractéristiques** : Pourquoi `CpuALL_usage`, `memory_usage_perce`, etc.
- **Prétraitement** : 
    - Normalisation (MinMaxScaler) et importance de l'échelle.
    - Fenêtrage (Sliding Window de 50) : Transformation du problème en séquences temporelles.
    - Encodage des étiquettes.

## 4. Modélisation et Architectures (6-8 pages)
- **Approche 1 : LSTM (RNN)**
    - Théorie : Cellules de mémoire, oubli et mise à jour.
    - Architecture implémentée (nb neurones, Dropout).
- **Approche 2 : CNN 1D**
    - Théorie : Extraction de motifs locaux par filtres de convolution.
    - Architecture implémentée (Noyaux, Pooling).
- **Approche 3 : Transformer (Attention)**
    - Théorie : Mécanisme d'attention globale (Multi-Head Attention).
    - Architecture implémentée (LayerNorm, GlobalPooling).

## 5. Analyse des Résultats et Comparaisons (5-6 pages)
- Présentation des tableaux de performance (Accuracy, F1-Score).
- **Courbes d'apprentissage** : Analyse de la convergence et de l'overfitting.
- **Matrices de Confusion** : Analyse fine des erreurs (confusion entre Mémoire et CPU).
- Synthèse : Pourquoi le CNN 1D est-il plus robuste sur ces données ?

## 6. Perspectives et Scénario Réel (2-3 pages)
- Proposition d'un **Modèle Hiérarchique** :
    - Niveau 1 : Classification des micro-tâches (notre travail actuel).
    - Niveau 2 : Identification de l'application (Navigateur, Compilateur) par macro-séquences.
- Pistes d'optimisation (Window Size, Early Stopping).

## 7. Conclusion
- Synthèse du projet et compétences acquises.
- Bibliographie et Sources (IA utilisées, documentations Keras/TensorFlow).

---

## Étapes de Travail
1. [ ] **Étape 1** : Création du squelette LaTeX (Préambule et Page de garde).
2. [ ] **Étape 2** : Rédaction de l'Introduction et de la section Données.
3. [ ] **Étape 3** : Développement technique des sections Modèles (LSTM, CNN, Transformer).
4. [ ] **Étape 4** : Intégration des résultats (tableaux, descriptions des métriques).
5. [ ] **Étape 5** : Rédaction des Perspectives et de la Conclusion.
6. [ ] **Étape 6** : Relecture, ajustement du style pour le volume des 25 pages.
