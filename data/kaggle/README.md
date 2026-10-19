# Real datasets (Kaggle-origin)

Used by topics 24-28. Files are the standard public copies of well-known Kaggle datasets, downloaded from public GitHub mirrors (raw files, unmodified), because Kaggle itself requires a login/API key.

| File | Kaggle dataset | Mirror used | Rows |
|---|---|---|---|
| `titanic.csv` | Titanic - Machine Learning from Disaster (`train.csv`) | github.com/datasciencedojo/datasets | 891 |
| `pima_diabetes.csv` | Pima Indians Diabetes Database | github.com/plotly/datasets | 768 |
| `wine_quality_red.csv` | Red Wine Quality (UCI) | github.com/plotly/datasets | 1,599 |
| `air_passengers.csv` | Air Passengers (Box-Jenkins airline data) | github.com/jbrownlee/Datasets | 144 |
| `heart_disease.csv` | Heart Disease Dataset (Cleveland) | github.com/kb22/Heart-Disease-Prediction | 303 |

To use the official Kaggle versions instead: `kaggle datasets download -d <owner>/<dataset>` and replace the file with the same name (column names are unchanged).

**Known data issues (handled in the analysis):** Titanic `Age` is 20% missing; Pima uses `0` for missing glucose/BP/skin/insulin/BMI; heart-disease `target` appears inverted relative to its description (see topic 28).
