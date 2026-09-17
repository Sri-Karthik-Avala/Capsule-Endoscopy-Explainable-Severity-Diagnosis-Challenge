# Capsule Endoscopy Explainable Severity Diagnosis Challenge

| | |
| --- | --- |
| Final rank | #7 |
| Domain | Computer Vision |
| Difficulty | Hard |
| Scoring | ↑ Higher is better |
| Compute | A10G |
| Challenge status | Accepted / closed |
| Solutions submitted | 4 |
| Last submission | 2026-06-26 |

## Problem statement

### Overview

Capsule endoscopy is a non-invasive diagnostic procedure in which a patient swallows a miniature camera capsule that captures thousands of gastrointestinal tract images during its passage through the digestive system. Reviewing these frames manually is time-consuming, repetitive, and clinically demanding.

Most automated systems focus only on predicting a class label. In real clinical workflows, specialists also need to understand **why a lesion was detected**, **why the finding may be severe**, and **which cases should be reviewed first**before trusting AI-assisted decisions.

This challenge asks participants to build explainable computer vision systems that analyze capsule endoscopy images and generate clinically meaningful predictions.

For each frame, participants must identify the correct diagnosis category, provide a structured visual explanation, estimate severity, explain the severity cause, and assign review priority.

The dataset includes realistic variability such as blur, illumination changes, compression artifacts, fluid occlusion, and reduced visibility to better simulate real deployment environments.

This is a multi-target medical imaging challenge designed to evaluate diagnostic accuracy, explanation quality, severity reasoning, ranking performance, calibration, and robustness.

---

### Task

For each test image, predict:

- **label** → anatomical or lesion category
- **reason_tag** → structured explanation describing why the label was predicted
- **severity_reason** → explanation describing why the case is severe or low risk
- **urgent_prob** → probability urgent specialist review is required *(0 to 1)*
- **severity_prob** → estimated severity score *(0 to 1)*
- **review_priority** → specialist review rank *(1 to 5, where 5 is highest priority)*

---

### Evaluation

Final leaderboard score is a weighted combination of the following metrics:

```
| Metric                                         | Weight |
| ---------------------------------------------- | -----: |
| Macro F1 Score on `label`                      |    40% |
| Macro F1 Score on `reason_tag`                 |    15% |
| Macro F1 Score on `severity_reason`            |    15% |
| ROC AUC on `urgent_prob`                       |    10% |
| Severity Score on `severity_prob`              |    10% |
| Spearman Rank Correlation on `review_priority` |    10% |
```

Severity Score

First compute RMSE between true and predicted severity values:

```
RMSE = sqrt( (1/N) * Σ(y_true - y_pred)^2 )
```

Then transform it into a bounded score:

```
Severity_Score = max(0, 1 - RMSE)
```

Perfect severity predictions receive **1.0**.

### Final Score

```
Final_Score =
0.40 * F1_label
+ 0.15 * F1_reason_tag
+ 0.15 * F1_severity_reason
+ 0.10 * AUC_urgent
+ 0.10 * Severity_Score
+ 0.10 * Spearman_priority
```

Higher scores are better.

---

### Dataset Structure

```
public/
├── train_images/
├── test_images/
├── train.csv
├── test.csv
└── sample_submission.csv

private/
└── answers.csv
```

---

### File Descriptions

`train_images/`

Contains labeled training images used for model development.

`test_images/`

Contains unlabeled evaluation images used for leaderboard scoring.

`train.csv`

```
| Column          | Type    | Description                     |
| --------------- | ------- | ------------------------------- |
| id              | string  | Image filename                  |
| label           | string  | Ground-truth diagnosis category |
| reason_tag      | string  | Structured visual explanation   |
| severity_reason | string  | Severity explanation tag        |
| urgent_prob     | float   | Urgency target between 0 and 1  |
| severity_prob   | float   | Severity target between 0 and 1 |
| review_priority | integer | Priority rank from 1 to 5       |
```

test.csv

```
| Column | Type   | Description         |
| ------ | ------ | ------------------- |
| id     | string | Test image filename |
```

`sample_submission.csv`

```
| Column          | Type    |
| --------------- | ------- |
| id              | string  |
| label           | string  |
| reason_tag      | string  |
| severity_reason | string  |
| urgent_prob     | float   |
| severity_prob   | float   |
| review_priority | integer |
```

---

### Example Submission

```
id,label,reason_tag,severity_reason,urgent_prob,severity_prob,review_priority
img_001.jpg,Polyp,raised_round_mass,large_protruding_lesion,0.84,0.80,4
img_002.jpg,Normal clean mucosa,normal_texture,no_high_risk_pattern,0.03,0.10,1
img_003.jpg,Blood - fresh,red_fluid_region,active_bleeding_visible,0.99,0.96,5
```

---

### Diagnosis Labels

Possible categories include:

- Ampulla of vater
- Angiectasia
- Blood - fresh
- Blood - hematin
- Erosion
- Erythema
- Foreign body
- Ileocecal valve
- Lymphangiectasia
- Normal clean mucosa
- Polyp
- Pylorus
- Reduced mucosal view
- Ulcer

---

### Reason Tags

Possible values for `reason_tag` include:

- `red_fluid_region`
- `dark_blood_residue`
- `raised_round_mass`
- `white_crater_lesion`
- `vascular_red_spots`
- `inflamed_red_surface`
- `foreign_object_shape`
- `landmark_ring_structure`
- `blurred_visibility`
- `white_spot_pattern`
- `normal_texture`

---

### Severity Reason Tags

Possible values for `severity_reason` include:

- `active_bleeding_visible`
- `deep_ulceration_pattern`
- `large_protruding_lesion`
- `suspicious_vascular_cluster`
- `widespread_inflammation`
- `retained_foreign_object`
- `unclear_view_needs_review`
- `low_risk_appearance`
- `no_high_risk_pattern`

---

### Challenge Goals

Participants are encouraged to build systems that are:

- Accurate across common and rare findings
- Interpretable and clinically trustworthy
- Strong at severity estimation and triage ranking
- Robust to noisy or low-quality images
- Well-calibrated for probability outputs
- Efficient and generalizable to real-world workflows
