# AI Diabetes Intelligence Dashboard

> Mini-Projet DataViz — Module Visualisation de Données  
> Dataset : Pima Indians Diabetes | Source : Kaggle / UCI  
> Technologies : Python · Streamlit · Plotly · Scikit-learn · Pandas · NumPy

---

## Apercu de l'application

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  AI Diabetes Intelligence Dashboard                          Accuracy 77.9%  │
│  Analyse predictive avancee des facteurs de risque diabetiques               │
│  [ Machine Learning ]  [ DataViz ]  [ Sante Publique ]                       │
├──────────────┬──────────────┬──────────────┬──────────────┬──────────────────┤
│ Total        │ Cas          │ Taux         │ Glucose      │ IMC              │
│ Patientes    │ Diabetiques  │ Diabete      │ Moyen        │ Moyen            │
│    768       │    268       │   34.9%      │  121.7 mg/dL │   32.0 kg/m2    │
├──────────────┴──────────────┴──────────────┴──────────────┴──────────────────┤
│  SIDEBAR FILTRES          │  GRAPHIQUES PRINCIPAUX                            │
│  ─────────────────────    │  ──────────────────────────────────────────────  │
│  Statut Diabetique        │  Distribution Glucose  │  IMC par Statut          │
│  Age                      │  Age vs Glucose        │  Matrice Correlation     │
│  IMC                      │  Diabete / Age Group   │  Pie Chart               │
│  Glucose                  │  Score de Risque       │  Sankey Flow             │
│  Pression Arterielle      │                                                   │
│  Insuline                 │  Prediction IA + Guide Sante Personnelle          │
│  Grossesses               │  Importance des Variables (Random Forest)         │
└───────────────────────────────────────────────────────────────────────────────┘
```

---

## Captures d'ecran

| KPIs et Distribution | Age vs Glucose et Heatmap |
|---|---|
| ![KPIs](Capture%20d'ecran/KPIs%20Cards.png) | ![Age Glucose](Capture%20d'ecran/Age%20vs%20Glucose%20et%20heatmap.png) |

| Diabete par Age et Repartition | Prediction IA |
|---|---|
| ![Age Groups](Capture%20d'ecran/Diabete%20par%20Groupe%20d'Age%20et%20repartition%20globale%20.png) | ![Prediction](Capture%20d'ecran/Prediction%20IA%20%E2%80%94%20Risque%20Diabetique%20Individuel.png) |

| Diagramme Sankey | Guide Sante |
|---|---|
| ![Sankey](Capture%20d'ecran/Diagramme%20de%20Flux%20(Sankey)%20%E2%80%94%20Parcours%20de%20Risque.png) | ![Guide](Capture%20d'ecran/Guide%20Sante%20%E2%80%94%20Comprendre%20vos%20Mesures.png) |

| Importance des Variables | Distribution Glucose et IMC |
|---|---|
| ![Importance](Capture%20d'ecran/Importance%20relative.png) | ![Distribution](Capture%20d'ecran/Distribution%20de%20glucose%20et%20IMC%20.png) |

---

## Structure du projet

```
Mini_Projet_Diabete/
│
├── app.py                          # Application Streamlit principale
├── analysis.ipynb                  # Notebook analyse et visualisations statiques
├── diabetes.csv                    # Dataset Pima Indians (768 patientes)
├── requirements.txt                # Dependances Python
├── README.md                       # Documentation du projet
│
└── Capture d'ecran/
    ├── KPIs Cards.png
    ├── Distribution de glucose et IMC .png
    ├── Age vs Glucose et heatmap.png
    ├── Diabete par Groupe d'Age et repartition global....png
    ├── Diagramme de Flux (Sankey) — Parcours de Ris....png
    ├── Prediction IA — Risque Diabetique Individuel.png
    ├── Guide Sante — Comprendre vos Mesures.png
    └── Importance relative.png
```

---

## Dataset

Le dataset **Pima Indians Diabetes** contient des mesures medicales de **768 patientes feminines** d'origine Pima (Arizona, USA). Il est largement utilise comme benchmark en machine learning pour la prediction du diabete de type 2.

| Variable | Description | Unite |
|---|---|---|
| Pregnancies | Nombre de grossesses | — |
| Glucose | Concentration plasmatique en glucose | mg/dL |
| BloodPressure | Pression arterielle diastolique | mmHg |
| SkinThickness | Epaisseur du pli cutane tricipital | mm |
| Insulin | Insuline serique a 2h | uU/mL |
| BMI | Indice de masse corporelle | kg/m2 |
| DiabetesPedigreeFunction | Fonction de pedigree (influence genetique) | — |
| Age | Age de la patiente | annees |
| Outcome | Diagnostic (0 = Non-Diabetique, 1 = Diabetique) | — |

**Prevalence :** 34.9% de cas diabetiques (268 / 768)

---

## Preparation des donnees

- Remplacement des **zeros impossibles** (Glucose, BloodPressure, SkinThickness, Insulin, BMI) par la mediane
- Suppression des **doublons**
- Creation de variables derivees : `AgeGroup`, `GlucoseLevel`, `BMILevel`, `RiskScore`
- Score de risque composite pondere : Glucose (35%) + IMC (20%) + Age (15%) + Insuline (15%) + Pedigree (15%)

---

## Fonctionnalites du Dashboard

### Filtres interactifs (sidebar)
- Statut diabetique (Tous / Diabetique / Non-Diabetique)
- Plage d'age, IMC, glucose, pression arterielle, insuline, grossesses

### KPIs dynamiques
- Total patientes, cas diabetiques, taux de diabete, glucose moyen, IMC moyen

### Visualisations

| Graphique | Type | Description |
|---|---|---|
| Distribution Glucose | Histogramme | Avec seuils cliniques 100 et 126 mg/dL |
| IMC par Statut | Violin | Seuils Surpoids (25) et Obesite (30) |
| Age vs Glucose | Scatter | Taille = IMC, couleur = statut diabetique |
| Correlation | Heatmap | Matrice des correlations inter-variables |
| Diabete par Age | Barplot | Prevalence par tranche d'age |
| Repartition | Pie Chart | Proportion diabetiques / non-diabetiques |
| Score de Risque | Line Chart | Evolution du score composite par age |
| Parcours de Risque | Sankey | Flux Glucose → IMC → Diagnostic |

### Prediction IA
- Modele **Random Forest** (100 arbres, train/test 80/20, `random_state=42`)
- Accuracy : **~77-80%**
- Entree : 8 mesures cliniques → probabilite de diabete en temps reel
- Metriques : Accuracy, Precision, Rappel, F1-Score

### Guide Sante Personnelle
- Tableau de reference clinique avec seuils medicaux reconnus (OMS / ADA)
- Evaluation automatique : Normal / Vigilance / Eleve
- Recommandation personnalisee selon le profil detecte

---

## Installation et lancement

**1. Cloner le projet**

```bash
git clone https://github.com/votre-username/Mini_Projet_Diabete.git
cd Mini_Projet_Diabete
```

**2. Installer les dependances**

```bash
pip install -r requirements.txt
```

**3. Lancer l'application**

```bash
python -m streamlit run app.py
```

L'application s'ouvre automatiquement sur `http://localhost:8501`

---

## Dependances

```
streamlit>=1.32.0
pandas>=2.0.0
plotly>=5.18.0
numpy>=1.26.0
scikit-learn>=1.4.0
```

---

## Insights cles

1. **Glucose** est le predicteur le plus discriminant (importance RF > 20%)
2. Les patientes diabetiques ont un glucose median ~140 mg/dL vs ~107 chez les non-diabetiques
3. La mediane de l'IMC chez les diabetiques (~34) depasse le seuil d'obesite (30)
4. Le taux de diabete augmente fortement avec l'age : ~22% chez les 20-30 ans vers ~60% apres 50 ans
5. La combinaison Glucose eleve + Obesite est le parcours dominant vers le diagnostic (Sankey)
6. Le modele Random Forest atteint ~78% d'accuracy avec seulement 8 variables cliniques

---

## Auteur

Projet realise dans le cadre du module **Visualisation de Donnees**  
2eme annee DUT — Semestre 4 — 2025-2026
