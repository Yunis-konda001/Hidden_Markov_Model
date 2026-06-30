# Formative 2: Modeling Human Activity States Using Hidden Markov Models

**Student:** [Your Name]  
**Course:** [Course Name]  
**Date:** June 30, 2026

---

## 1. Background and Motivation

I want to build a simple activity monitor for daily health tracking using only a smartphone. Many people carry their phone in a pocket while moving around at home or outdoors. If I can detect whether someone is standing, walking, jumping, or completely still from accelerometer and gyroscope signals, I can estimate activity level without extra hardware. This is useful for fitness apps, fall detection, and reminders to move after long sitting periods. Human activity recognition is hard because the true activity is hidden behind noisy sensor readings. A Hidden Markov Model (HMM) is a good fit because it models hidden activity states, observations from sensor features, and transitions between activities over time.

---

## 2. Data Collection and Preprocessing

### 2.1 Data collection

I used the **Sensor Logger** app on my smartphone to record four activities:

| Activity | Duration per recording | Notes |
|----------|------------------------|-------|
| Standing | 5–10 seconds | Phone held steady at waist level |
| Walking | 5–10 seconds | Consistent walking pace |
| Jumping | 5–10 seconds | Continuous jumps |
| Still | 5–10 seconds | Phone placed on a flat surface |

I collected **10 recordings per activity** (accel + gyro CSV files each), stored in `data/standing/`, `data/walking/`, `data/jumping/`, and `data/still/`.

### 2.2 File format and sampling rate

Each CSV file has columns: `time`, `seconds_elapsed`, `z`, `y`, `x`.

I measured the mean time step between rows and found a sampling rate of about **99.9 Hz** (~100 Hz). All my files used the same phone, so I did **not** need resampling. I documented this rate in my notebook.

**Note:** `walking_07_accel.csv` was missing, so I excluded recording 07 from walking analysis. I also trimmed one extra row from `still_03_gyro.csv` so accel and gyro lengths matched.

### 2.3 Windowing

I split each recording into overlapping windows:

| Setting | Value | Reason |
|---------|-------|--------|
| Window size | 1.0 second (99 samples) | Captures about one walking step cycle at 100 Hz |
| Step size | 0.5 second (50% overlap) | Gives more windows from short 8-second clips |

### 2.4 Sample plots

See Figure 1 (raw accelerometer magnitude for each activity) in `figures/raw_sensor_samples.png`.

---

## 3. Feature Extraction

I computed **8 features** per window from accelerometer data (time and frequency domain).

### 3.1 Time-domain features

| Feature | Description | Why it helps |
|---------|-------------|--------------|
| mean_mag | Mean accelerometer magnitude | Still has low mean; jumping has higher mean |
| std_mag | Standard deviation of magnitude | Walking and jumping show more variation |
| var_mag | Variance of magnitude | Separates active vs inactive states |
| sma | Signal magnitude area: mean of \|x\|+\|y\|+\|z\| | Simple motion intensity measure |
| corr_xy | Correlation between accel x and y | Captures axis coupling during movement |

### 3.2 Frequency-domain features (FFT)

| Feature | Description | Why it helps |
|---------|-------------|--------------|
| dom_freq | Dominant FFT frequency | Walking has rhythmic step frequency |
| spec_energy | Sum of squared FFT magnitudes | High for periodic activities |
| fft_peak | Maximum FFT magnitude (excluding DC) | Highlights strong periodic motion |

### 3.3 Normalization

I used **z-score normalization** fit on **training windows only**:

`z = (x - mean) / std`

This stops large-scale features (like spectral energy) from dominating the HMM.

---

## 4. HMM Setup and Implementation

### 4.1 Model components

| Element | My definition |
|---------|---------------|
| Hidden states (Z) | 4 states: standing, walking, jumping, still |
| Observations (X) | 8-dimensional feature vector per window |
| Transition matrix (A) | 4×4 probabilities of moving between activities |
| Emission (B) | Gaussian distribution per state (diag covariance) |
| Initial probabilities (π) | Learned during training |

### 4.2 Algorithms

I used the **hmmlearn** `GaussianHMM` library:

- **Baum–Welch (EM):** `model.fit()` with `tol=1e-4` as convergence check. Training converged in 2 iterations.
- **Viterbi decoding:** `model.predict()` to find the most likely state sequence on unseen test data.

I initialized emission means from each activity's average training features to help states align with activities.

### 4.3 Train / test split

| Set | Recordings | Purpose |
|-----|------------|---------|
| Train | 01–08 (walking skips 07) | Fit HMM |
| Test | 09 and 10 | Unseen evaluation |

Test recordings were **not used in training**. They were recorded in the same environment as training data but on different sessions.

---

## 5. Results and Interpretation

### 5.1 Visualizations

- **Transition matrix:** `figures/transition_matrix.png` — shows how likely the model moves between activities.
- **Emission means:** `figures/emission_means.png` — shows feature patterns per activity state.
- **Decoded sequence:** `figures/decoded_sequence.png` — Viterbi predictions over time on unseen `walking_09`.
- **Confusion matrix:** `figures/confusion_matrix.png` — test set classification results.
- **Baum–Welch convergence:** `figures/baum_welch_convergence.png`

### 5.2 Evaluation table (unseen test data)

| State (Activity) | Number of Samples | Sensitivity | Specificity | Overall Accuracy |
|------------------|-------------------|-------------|-------------|------------------|
| Standing | 30 | 0.967 | 1.000 | 0.992 |
| Walking | 33 | 1.000 | 0.989 | 0.992 |
| Jumping | 33 | 1.000 | 1.000 | 1.000 |
| Still | 32 | 1.000 | 1.000 | 1.000 |

**Overall test accuracy:** 99.6% (127 of 128 windows correct)

The model generalizes well to unseen recordings 09 and 10. Jumping and still were easiest to classify (high variance vs near-zero motion). Standing was occasionally confused with walking (1 window misclassified).

---

## 6. Discussion and Conclusion

### 6.1 Easiest and hardest activities

- **Easiest:** Jumping and still — very different motion levels and frequency content.
- **Hardest:** Standing vs walking — both involve upright posture with moderate accel values; one standing window was misclassified as walking.

### 6.2 Transition probabilities

The transition matrix shows high self-transition probabilities (staying in the same activity), which matches real behavior within a short 8-second clip where activity rarely changes.

### 6.3 Sensor noise and sampling rate

At ~100 Hz, my window size of 1 second gave stable features. Sensor noise had little effect because I averaged over 99 samples per window. A lower sampling rate would require longer windows.

### 6.4 Possible improvements

- Re-record missing `walking_07_accel.csv` and add more data to exceed 90 seconds per activity.
- Add more sensors (magnetometer, barometer).
- Try longer windows or deep learning models.
- Collect test data in a different environment to test robustness.

---

## 7. Task Allocation (Individual Work)

| Task | Person | Description |
|------|--------|-------------|
| Data collection | Me | Recorded 10 sessions × 4 activities with Sensor Logger |
| Data cleaning | Me | Organized CSV files, fixed still_03 mismatch |
| Feature extraction | Me | Implemented 8 time/frequency features with z-score |
| HMM implementation | Me | Trained GaussianHMM, Viterbi decode, Baum–Welch |
| Evaluation and plots | Me | Confusion matrix, metrics table, all figures |
| Report writing | Me | This report |

---

## References

- Sensor Logger app (iOS/Android)
- hmmlearn documentation: https://hmmlearn.readthedocs.io/
- Course materials on Hidden Markov Models
