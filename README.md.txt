# FireRisk AI — Statistics Mini Project

## What this project contains
- `fire_risk_dataset_bejaia.csv` — 122-row Bejaia-region working dataset.
- `FireRisk_AI_Statistics_Project.ipynb` — complete Python notebook.
- `FireRisk_AI_Statistics_Report.docx` — editable report.
- `FireRisk_AI_Statistics_Report.pdf` — submission-ready PDF.
- `requirements.txt` — Python packages.

## Project objective
Classify a daily observation as `fire` or `not fire` using weather and Fire Weather Index variables.

## Dataset
Original source: UCI Machine Learning Repository — Algerian Forest Fires Dataset.
https://archive.ics.uci.edu/dataset/547/algerian+forest+fire+dataset

The original dataset contains 244 observations from Bejaia and Sidi Bel-abbes. This project uses the 122 observations from Bejaia.

## Models
- Logistic Regression
- Random Forest Classifier

## Fixed test split results
Both models: 96.0% accuracy, 100% precision, 91.7% recall, 95.7% F1-score.
ROC-AUC: Logistic Regression 0.994; Random Forest 0.987.

## Important
This is a statistical/ML educational prototype. It is not an operational fire-warning system. Do not claim that it can reliably predict real-world fires in India.
