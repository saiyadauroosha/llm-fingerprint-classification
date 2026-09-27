# LLM Fingerprint Classification

CMPSC 448 Midterm Project

## Project
**Who Wrote It? Identifying LLMs from Their Responses**

We investigate whether CNN and RNN/LSTM classifiers can identify the LLM family that generated a response.

### LLM Families
- GPT
- Claude
- Gemini

### Research Questions
- RQ1: Can we identify which LLM generated a response?
- RQ2: Does the user's prompt help identify the LLM?
- RQ3: Do LLM fingerprints generalize across tasks/domains?
- RQ4: What characteristics distinguish different LLMs?

## Dataset

Source: LMArena Arena Human Preference 140K.

Final curated dataset:
- 2,000 GPT examples
- 2,000 Claude examples
- 2,000 Gemini examples
- 6,000 examples total

Fixed train/validation/test splits are stored in:

- `data/splits/train.parquet`
- `data/splits/validation.parquet`
- `data/splits/test.parquet`

All team members must use these same splits.

See `reports/dataset_summary.md` for dataset preparation details.

## Repository Structure

- `data/` - curated dataset and fixed splits
- `notebooks/` - dataset exploration/preparation
- `src/cnn/` - CNN implementation
- `src/rnn/` - RNN/LSTM implementation
- `src/evaluation/` - shared evaluation code
- `results/` - experiment outputs and figures
- `reports/` - project documentation

