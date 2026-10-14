# Video Streaming Quality of Experience Analysis

A data science project exploring **Quality of Experience (QoE) in mobile video streaming** using crowdsourced measurements from the PoqMoN/PoqeMoN study around Paris.

The analysis examines how application, network, device, user, and subjective feedback variables relate to **Mean Opinion Score (MOS)**, then builds machine-learning models to identify sessions at risk of poor user experience.

## Project goals

- Explore the main drivers of perceived video-streaming quality.
- Quantify relationships between technical QoS/QoA metrics and MOS.
- Detect unusual streaming sessions and network behavior.
- Convert MOS into an actionable at-risk / retained classification target.
- Compare baseline and ensemble classification models.
- Translate model results into practical QoE decisions.

## Repository structure

```text
.
├── DataScienceProject_updated.ipynb
├── data/
│   └── video_streaming_qoe.csv
├── docs/
│   └── Analysis of Quality of Experience in Video Streaming Report.pdf
└── README.md
```

- **DataScienceProject_updated.ipynb** — main exploratory analysis and modeling notebook.
- **data/video_streaming_qoe.csv** — streaming-session dataset used in the analysis.
- **docs/** — final project report.

## Dataset

The dataset contains roughly **1,540 streaming sessions** from **181 participants**, covering multiple Android devices and French mobile operators.

The response variable is:

- **MOS (Mean Opinion Score):** subjective quality rating from 1 (poor) to 5 (excellent).

Feature groups include:

| Group | Examples |
| --- | --- |
| QoA — Application | resolution, bitrate, framerate, dropped frames, buffering count/time |
| QoS — Service/network | service type, operator |
| QoD — Device | model, OS version, API level |
| QoU — User | age, sex, study group |
| QoF — Feedback | audio and video quality ratings |

## Analysis workflow

1. Load and inspect the session-level data.
2. Check data types, missing values, ranges, and category balance.
3. Explore MOS and its relationship with technical and subjective variables.
4. Detect anomalous sessions using statistical and machine-learning methods.
5. Define an actionable binary outcome:
   - **Retained:** MOS ≥ 4
   - **At risk:** MOS < 4
6. Prepare features for modeling.
7. Compare classification models including Logistic Regression, Random Forest, XGBoost, and LightGBM.
8. Interpret feature importance and connect model findings to QoE decisions.

## Code preview

A minimal starting point for reproducing the analysis:

```python
import pandas as pd

DATA_PATH = "data/video_streaming_qoe.csv"

df = pd.read_csv(DATA_PATH)

print(df.shape)
print(df.dtypes)
print(df["MOS"].value_counts().sort_index())

# Turn subjective QoE into an actionable binary target.
df["retained"] = (df["MOS"] >= 4).astype(int)

# Example QoE indicators to inspect first.
qoe_features = [
    "QoA_VLCbitrate",
    "QoA_VLCframerate",
    "QoA_BUFFERINGcount",
    "QoA_BUFFERINGtime",
    "QoF_audio",
    "QoF_video",
]

print(df[qoe_features + ["MOS", "retained"]].describe())
```

## Main findings reported in the project

- Buffering time and buffering count are strongly associated with lower MOS.
- Higher framerate generally corresponds to better perceived quality.
- Bitrate alone is not a sufficient explanation of user satisfaction.
- The data are imbalanced across devices, OS versions, and operators.
- MOS is concentrated toward higher ratings.
- Audio/video subjective feedback and buffering variables are among the strongest predictive features.

The notebook also compares several classifiers, with reported accuracy around 90%. Accuracy should not be treated as the only decision metric because the retained/at-risk target is imbalanced; precision, recall, F1, ROC-AUC/PR-AUC, calibration, and business costs should also be considered.

## Limitations

- MOS is subjective and can vary across users.
- Device, OS, and operator representation is uneven.
- Sessions from the same user may not be statistically independent.
- The original workflow does not deeply model temporal effects such as congestion by time of day.
- A binary MOS threshold simplifies a naturally ordinal outcome.
- Predictive performance does not by itself establish causal effects.

## Suggested next steps

This repository can be rebuilt into a stronger end-to-end data science project by:

- separating raw data, notebooks, reusable source code, and model artifacts;
- introducing user-aware train/test splitting to reduce leakage;
- documenting preprocessing and feature definitions;
- adding proper baseline models and imbalance-aware evaluation;
- comparing classification with ordinal/regression approaches for MOS;
- adding calibration, threshold selection, and decision-cost analysis;
- making every conclusion traceable to a plot, statistical test, or model result.

## Tech stack

- Python
- pandas / NumPy
- matplotlib / seaborn
- scikit-learn
- XGBoost
- LightGBM
- Jupyter Notebook

## Report

See [Analysis of Quality of Experience in Video Streaming Report](docs/Analysis%20of%20Quality%20of%20Experience%20in%20Video%20Streaming%20Report.pdf).

## Author

**Pratistha Thapa**  
MSc Data Science & Network Intelligence  
Télécom SudParis
