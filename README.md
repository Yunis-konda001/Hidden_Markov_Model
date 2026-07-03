# HMM Activity Recognition

**Author:** Kumi Yunis Konda  
**Course:** Machine Learning Techniques II  
**Assignment:** Formative 2 — Hidden Markov Models

I built a Hidden Markov Model to recognize human activities from smartphone accelerometer and gyroscope data.

## Activities

- standing
- walking
- jumping
- still

## Project structure

```
Hidden Markov Models/
├── data/                          # labelled sensor CSV files
│   ├── standing/
│   ├── walking/
│   ├── jumping/
│   └── still/
├── figures/                       # plots saved from the notebook   
├── hmm_activity.ipynb             # main notebook
├── requirements.txt
└── README.md
```

## Data

- Recorded with **Sensor Logger** on iPhone
- ~**99.9 Hz** sampling rate
- **79 CSV files** (accel + gyro pairs); `walking_07_accel.csv` is missing and excluded from training
- Naming pattern: `{activity}_{01-10}_{accel|gyro}.csv`

## What the notebook does

1. Load and merge accelerometer + gyroscope CSVs
2. Split recordings into **1-second windows** (0.5 s step)
3. Extract **10 features** (accel time-domain, gyro time-domain, FFT)
4. Z-score normalize using training data only
5. Train a **4-state Gaussian HMM** (hmmlearn)
6. Decode unseen test data with **Viterbi**
7. Save figures and evaluation metrics to `figures/`

**Train / test split:** recordings `01–08` for training, `09–10` for unseen testing.

## How to run

```bash
pip install -r requirements.txt
python -m jupyter notebook hmm_activity.ipynb
```

Run all cells top to bottom. Figures and metrics save to `figures/`.

To rebuild the PDF report:

```bash
python generate_report_pdf.py
```

## Results (unseen test data)

| Activity | Samples | Sensitivity | Specificity | Accuracy |
|----------|---------|-------------|-------------|----------|
| Standing | 30 | 0.933 | 1.000 | 0.933 |
| Walking | 33 | 1.000 | 0.979 | 1.000 |
| Jumping | 33 | 1.000 | 1.000 | 1.000 |
| Still | 32 | 1.000 | 1.000 | 1.000 |

**Overall test accuracy: 98.4%** (126 / 128 windows)

## Files

| File | Description |
|------|-------------|
| `hmm_activity.ipynb` | Main analysis notebook |
| `data/` | Labelled sensor recordings |
| `report/formative2_hmm_report.pdf` | Submission report |
| `figures/` | Plots (confusion matrix, transition matrix, etc.) |
| `figures/evaluation_metrics.csv` | Metrics table from the notebook |

## Figures generated

- `raw_sensor_samples.png` — accel and gyro per activity
- `baum_welch_convergence.png` — Baum–Welch log-likelihood
- `initial_state_probs.png` — initial state probabilities (π)
- `transition_matrix.png` — transition matrix (A)
- `emission_means.png` — emission means per state (B)
- `decoded_sequence.png` — Viterbi decode on unseen test recording
- `confusion_matrix.png` — test set confusion matrix
