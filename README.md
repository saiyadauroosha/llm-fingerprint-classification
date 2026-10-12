# LLM Fingerprint Classification

**CMPSC 448 Midterm Project**

## Project
**Who Wrote It? Identifying LLMs from Their Responses**

This project investigates whether neural text classifiers can identify the LLM family that generated a response.

We compare:
- Convolutional Neural Networks (CNN)
- Recurrent Neural Networks using LSTM

### LLM Families
- GPT
- Claude
- Gemini

## Research Questions

- **RQ1:** Can we identify which LLM generated a response using only the generated output?
- **RQ2:** How does performance change when using the prompt only, response only, or prompt + response?
- **RQ3:** Do LLM-family identification signals generalize across unseen task domains?
- **RQ4:** What linguistic and structural characteristics help distinguish different LLM families?

## Dataset

The project uses the publicly available **LMArena Arena Human Preference 140K** dataset as the source dataset.

After cleaning, filtering, balancing, and leakage control, the final curated dataset contains:

- 2,000 GPT examples
- 2,000 Claude examples
- 2,000 Gemini examples
- **6,000 examples total**

The dataset contains four controlled domains:

- General
- Code
- Math
- Creative Writing

### Fixed Data Splits

All CNN and RNN/LSTM experiments use the same fixed train/validation/test splits:

- `data/splits/train.parquet`
- `data/splits/validation.parquet`
- `data/splits/test.parquet`

The same exact prompt is kept within a single split to reduce prompt leakage between training, validation, and test data.

See `reports/dataset_summary.md` for additional dataset preparation details.

## Models

### CNN
The CNN uses:
- 200-dimensional learned embeddings
- Convolution kernel sizes 3, 4, and 5
- 100 filters per kernel size
- Max pooling
- Dropout = 0.5
- Final 3-class classification layer

### RNN/LSTM
The RNN model uses:
- 200-dimensional learned embeddings
- Single-layer LSTM
- 128 hidden units
- Packed sequences for variable-length inputs
- Dropout = 0.5
- Final 3-class classification layer

## Training Setup

Both models use:

- Adam optimizer
- Learning rate = 0.001
- Cross-entropy loss
- Batch size = 64
- 10 training epochs
- Random seed = 42
- Validation accuracy for model selection

## Evaluation Metrics

The project reports:

- Accuracy
- Macro Precision
- Macro Recall
- Macro-F1
- Confusion matrices

## Main Results

For response-only LLM-family identification:

- **CNN:** 91.96% accuracy, 92.01% Macro-F1
- **RNN/LSTM:** 76.00% accuracy, 75.94% Macro-F1

RQ2 showed that most of the predictive signal comes from the generated response rather than the user prompt.

RQ3 showed that generalization performance depends on both the model architecture and the held-out domain.

RQ4 showed that response structure, punctuation, formatting, local phrasing, and response length all provide useful signals for LLM-family classification.

## Final Project Report

The final project report is available here:

[CMPSC448 Midterm Project Report](CMPSC448_Midterm_Project_Report.pdf)

## Repository Structure

- `data/` - curated dataset and fixed data splits
- `notebooks/` - dataset exploration and preparation
- `src/cnn/` - CNN implementation and experiments
- `src/rnn/` - RNN/LSTM implementation and experiments
- `src/evaluation/` - shared evaluation utilities
- `results/` - experiment outputs, metrics, and result files
- `reports/` - dataset and project documentation
- `CMPSC448_Midterm_Project_Report.pdf` - final project report

## Team Members

- Saiyada Uroosha Shahnawaz — Team Lead
- Vanisha Gupta
- Ariana Sookoo
- Erin Jibu
- Ananya Muppirala

## Notes

All reported numerical results were generated from the team's project code and fixed dataset splits.