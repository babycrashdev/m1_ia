- le but du projet est de faire de la classification de séries temporelles qui représentent des traces d'activités CPU, disque, mémoire, etc, obtenu avec profilant un certain nombre d'exécution d'outils de tests de charge (benchmark)

- il s'agit donc d'entraîner un modèle afin qu'il reconnaisse si une trace est de type CPU, disque ou mémoire

- les .csv des traces sont disponibles ici: https://github.com/phuongntmse/ts-data-for-workload-classification, avec une explication. Ces traces ont été produites par Mme Thi Mai Phuong qui est une étudiante en thèse co-dirigée par M. Adel Noureddine et moi-même.

- il est demandé d'utiliser 3 approches:

    1- des modèles de réseaux de neurones récurrent de type LSTM (approche assez classique pour des séries temporelles)
    2- des modèles de réseaux de neurones convolutionels, plus adaptés pour les images, mais qui peuvent être quand même utilisé pour des séries temporelles avec certaines adaptations
    3- des modèles plus récent de type Attention et Transformer: Time Series Transformers

- il faudra évaluer les performances de la classification avec des métriques comme le MSE, F1 score, etc

- vous pouvez utiliser des IAs pour vous aider. Il faudra indiquer toutes les sources.

- vous devez rendre un rapport en .pdf de 25 pages environ qui explique en détail ce que vous avez fait et les principales étapes et résultats. Je dois être convaincu que vous avez compris ce que vous avez fait.

- vous devez rendre un lien vers un Colab de votre projet qui doit être fonctionnel

- dans un cas d'application réel où on aurait une trace d'exécution d'une application, l'objectif serait de pouvoir identifier des séquences de traces que l'on pourrait associer à un type d'activités (CPU, disque ou mémoire) donnée. On pourrait ensuite entraîner un modèle plus général qui pourrait identifier si des séquences de trace d'exécution sont issues d'un navigateur web, d'un compilateur, un traitement de texte, etc

- en perspective dans le rapport, indiquer quelles seraient les pistes pour tendre vers ce scénario d'utilisation