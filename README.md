<div align="center">

#  AI Diabetes Intelligence Dashboard

**Analyse prédictive et visualisation interactive des facteurs de risque du diabète**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.18%2B-3F4F75?logo=plotly&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?logo=scikitlearn&logoColor=white)
![Licence](https://img.shields.io/badge/Licence-MIT-green)
![Statut](https://img.shields.io/badge/Statut-Projet%20acad%C3%A9mique-blue)

*Mini-Projet DataViz : Module Visualisation de Données*
*Dataset : Pima Indians Diabetes (Kaggle / UCI)*

</div>

---

##  Sommaire

- [Aperçu](#-aperçu)
- [Fonctionnalités](#-fonctionnalités)
- [Captures d'écran](#-captures-décran)
- [Installation et lancement](#-installation-et-lancement)
- [Structure du projet](#-structure-du-projet)
- [Dataset](#-dataset)
- [Préparation des données](#-préparation-des-données)
- [Modèle de prédiction](#-modèle-de-prédiction)
- [Insights clés](#-insights-clés)
- [Limites et avertissement](#️-limites-et-avertissement)
- [Pistes d'amélioration](#-pistes-damélioration)
- [Auteur](#-auteur)
- [Licence](#-licence)

---

##  Aperçu

Ce projet est un **tableau de bord interactif** construit avec Streamlit et Plotly pour explorer le dataset *Pima Indians Diabetes* (768 patientes). Il combine :

-  **une exploration visuelle** pilotée par des filtres dynamiques,
-  **un modèle Random Forest** qui estime en temps réel la probabilité de diabète à partir de 8 mesures cliniques,
-  **un guide santé** qui situe les mesures de l'utilisateur par rapport aux seuils cliniques reconnus (OMS / ADA).

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  AI Diabetes Intelligence Dashboard                          Accuracy 77.9%  │
│  Analyse prédictive avancée des facteurs de risque diabétiques               │
│  [ Machine Learning ]  [ DataViz ]  [ Santé Publique ]                       │
├──────────────┬──────────────┬──────────────┬──────────────┬──────────────────┤
│ Total        │ Cas          │ Taux         │ Glucose      │ IMC              │
│ Patientes    │ Diabétiques  │ Diabète      │ Moyen        │ Moyen            │
│    768       │    268       │   34.9%      │  121.7 mg/dL │   32.0 kg/m²     │
├──────────────┴──────────────┴──────────────┴──────────────┴──────────────────┤
│  SIDEBAR FILTRES          │  GRAPHIQUES PRINCIPAUX                            │
│  ─────────────────────    │  ──────────────────────────────────────────────  │
│  Statut Diabétique        │  Distribution Glucose  │  IMC par Statut          │
│  Âge                      │  Âge vs Glucose        │  Matrice Corrélation     │
│  IMC                      │  Diabète / Groupe d'âge│  Pie Chart               │
│  Glucose                  │  Score de Risque       │  Sankey Flow             │
│  Pression Artérielle      │                                                   │
│  Insuline                 │  Prédiction IA + Guide Santé Personnelle          │
│  Grossesses               │  Importance des Variables (Random Forest)         │
└───────────────────────────────────────────────────────────────────────────────┘
```

---

## Fonctionnalités

### Filtres interactifs (sidebar)

Statut diabétique (Tous / Diabétique / Non-Diabétique), plage d'âge, IMC, glucose, pression artérielle, insuline et nombre de grossesses.

### KPIs dynamiques

Total de patientes, cas diabétiques, taux de diabète, glucose moyen et IMC moyen, recalculés à chaque changement de filtre.

### Visualisations

| Graphique | Type | Description |
|---|---|---|
| Distribution Glucose | Histogramme | Avec seuils cliniques à 100 et 126 mg/dL |
| IMC par Statut | Violin plot | Seuils Surpoids (25) et Obésité (30) |
| Âge vs Glucose | Scatter | Taille = IMC, couleur = statut diabétique |
| Corrélations | Heatmap | Matrice de corrélation inter-variables |
| Diabète par Âge | Barplot | Prévalence par tranche d'âge |
| Répartition | Pie chart | Proportion diabétiques / non-diabétiques |
| Score de Risque | Line chart | Évolution du score composite par âge |
| Parcours de Risque | Sankey | Flux Glucose → IMC → Diagnostic |

### Prédiction IA

- Modèle **Random Forest** (100 arbres, split train/test 80/20, `random_state=42`)
- Entrée : 8 mesures cliniques, sortie : probabilité de diabète en temps réel
- Métriques affichées : Accuracy, Précision, Rappel, F1-Score
- Importance relative des variables

### Guide Santé Personnelle

- Tableau de référence avec seuils médicaux reconnus (OMS / ADA)
- Évaluation automatique : **Normal / Vigilance / Élevé**
- Recommandation personnalisée selon le profil détecté

---

## Captures d'écran

| KPIs et Distribution | Âge vs Glucose et Heatmap |
|---|---|
| ![KPIs](Capture%20d'ecran/KPIs%20Cards.png) | ![Age Glucose](Capture%20d'ecran/Age%20vs%20Glucose%20et%20heatmap.png) |

| Diabète par Âge et Répartition | Prédiction IA |
|---|---|
| ![Age Groups](Capture%20d'ecran/Diabete%20par%20Groupe%20d'Age%20et%20repartition%20globale%20.png) | ![Prediction](Capture%20d'ecran/Prediction%20IA%20%E2%80%94%20Risque%20Diabetique%20Individuel.png) |

| Diagramme Sankey | Guide Santé |
|---|---|
| ![Sankey](Capture%20d'ecran/Diagramme%20de%20Flux%20(Sankey)%20%E2%80%94%20Parcours%20de%20Risque.png) | ![Guide](Capture%20d'ecran/Guide%20Sante%20%E2%80%94%20Comprendre%20vos%20Mesures.png) |

| Importance des Variables | Distribution Glucose et IMC |
|---|---|
| ![Importance](Capture%20d'ecran/Importance%20relative.png) | ![Distribution](Capture%20d'ecran/Distribution%20de%20glucose%20et%20IMC%20.png) |

---

## Installation et lancement

**Prérequis :** Python 3.10 ou supérieur et `pip`.

**1. Cloner le projet**

```bash
git clone https://github.com/votre-username/Mini_Projet_Diabete.git
cd Mini_Projet_Diabete
```

**2. (Recommandé) Créer un environnement virtuel**

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

**3. Installer les dépendances**

```bash
pip install -r requirements.txt
```

**4. Lancer l'application**

```bash
python -m streamlit run app.py
```

L'application s'ouvre automatiquement sur **http://localhost:8501**.

>  Pour explorer les visualisations statiques et la démarche d'analyse, ouvrez `analysis.ipynb` avec Jupyter.

### Dépendances

```
streamlit>=1.32.0
pandas>=2.0.0
plotly>=5.18.0
numpy>=1.26.0
scikit-learn>=1.4.0
```

---

## Structure du projet

```
Mini_Projet_Diabete/
│
├── app.py                # Application Streamlit principale
├── analysis.ipynb        # Notebook d'analyse et visualisations statiques
├── diabetes.csv          # Dataset Pima Indians (768 patientes)
├── requirements.txt      # Dépendances Python
├── README.md             # Documentation du projet
├── LICENSE               # Licence MIT
│
└── Capture d'ecran/      # Captures utilisées dans ce README
    ├── KPIs Cards.png
    ├── Distribution de glucose et IMC .png
    ├── Age vs Glucose et heatmap.png
    ├── Diabete par Groupe d'Age et repartition globale .png
    ├── Diagramme de Flux (Sankey) — Parcours de Risque.png
    ├── Prediction IA — Risque Diabetique Individuel.png
    ├── Guide Sante — Comprendre vos Mesures.png
    └── Importance relative.png
```

---

## Dataset

Le dataset **Pima Indians Diabetes** (UCI / Kaggle) contient des mesures médicales de **768 patientes** d'origine Pima (Arizona, USA). C'est un benchmark classique pour la prédiction du diabète de type 2.

| Variable | Description | Unité |
|---|---|---|
| `Pregnancies` | Nombre de grossesses | — |
| `Glucose` | Concentration plasmatique en glucose | mg/dL |
| `BloodPressure` | Pression artérielle diastolique | mmHg |
| `SkinThickness` | Épaisseur du pli cutané tricipital | mm |
| `Insulin` | Insuline sérique à 2 h | µU/mL |
| `BMI` | Indice de masse corporelle | kg/m² |
| `DiabetesPedigreeFunction` | Fonction de pedigree (influence génétique) | — |
| `Age` | Âge de la patiente | années |
| `Outcome` | Diagnostic (0 = Non-diabétique, 1 = Diabétique) | — |

**Prévalence :** 34.9 % de cas diabétiques (268 / 768).

---

## Préparation des données

1. Remplacement des **zéros impossibles** (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`) par la médiane de la colonne
2. Suppression des **doublons**
3. Création de variables dérivées : `AgeGroup`, `GlucoseLevel`, `BMILevel`, `RiskScore`
4. Score de risque composite pondéré :

| Composante | Poids |
|---|---|
| Glucose | 35 % |
| IMC | 20 % |
| Âge | 15 % |
| Insuline | 15 % |
| Pedigree | 15 % |

---

## Modèle de prédiction

| Paramètre | Valeur |
|---|---|
| Algorithme | Random Forest (`RandomForestClassifier`) |
| Nombre d'arbres | 100 |
| Split train / test | 80 % / 20 % |
| Graine aléatoire | `random_state=42` |
| Accuracy | ~77–80 % |

---

## Insights clés

1. **Le glucose** est le prédicteur le plus discriminant (importance RF > 20 %).
2. Le glucose médian atteint ~140 mg/dL chez les diabétiques contre ~107 chez les non-diabétiques.
3. L'IMC médian des diabétiques (~34) dépasse le seuil d'obésité (30).
4. Le taux de diabète augmente fortement avec l'âge : ~22 % chez les 20-30 ans, jusqu'à ~60 % après 50 ans.
5. La combinaison **glucose élevé + obésité** est le parcours dominant vers le diagnostic (Sankey).
6. Le Random Forest atteint ~78 % d'accuracy avec seulement 8 variables cliniques.

---

## Limites et avertissement

> **Cet outil est un projet pédagogique. Il ne constitue en aucun cas un dispositif médical ni un avis de santé.** Les résultats ne remplacent pas l'avis d'un professionnel de santé. En cas de doute, consultez un médecin.

Limites techniques à garder en tête :

- **Population restreinte** : le dataset ne contient que des femmes d'origine Pima âgées d'au moins 21 ans ; les résultats ne se généralisent pas à d'autres populations.
- **Taille réduite** : 768 observations, ce qui limite la robustesse du modèle.
- **Imputation par la médiane** : simple et reproductible, mais elle réduit la variance et peut biaiser certaines distributions (notamment `Insulin`, très affectée par les valeurs manquantes).
- **Évaluation sur un seul split** : l'accuracy dépend du découpage ; une validation croisée donnerait une estimation plus stable.
- **Déséquilibre de classes** (65 / 35) : l'accuracy seule est trompeuse, d'où l'affichage de la précision, du rappel et du F1-score.
- **Score de risque** : ses pondérations sont heuristiques, non issues d'une étude clinique.

---

## Pistes d'amélioration

- [ ] Validation croisée et optimisation d'hyperparamètres (`GridSearchCV`)
- [ ] Comparaison avec d'autres modèles (régression logistique, XGBoost, SVM)
- [ ] Imputation avancée (KNN, MICE)
- [ ] Explicabilité des prédictions individuelles (SHAP)
- [ ] Courbe ROC et matrice de confusion dans le dashboard
- [ ] Déploiement sur Streamlit Community Cloud
- [ ] Export des résultats filtrés en CSV / PDF

---

## Auteur

Projet réalisé dans le cadre du module **Visualisation de Données**
2ᵉ année DUT, Semestre 4, 2025-2026

---

## Licence

Le code source de ce projet est distribué sous licence **MIT**. Voir le fichier [LICENSE](LICENSE) pour le texte complet.

Le dataset *Pima Indians Diabetes* reste soumis aux conditions de ses sources d'origine (UCI Machine Learning Repository / Kaggle).
