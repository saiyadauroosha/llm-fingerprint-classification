
# CMPSC 448 LLM Fingerprint Dataset Summary

## Source
LMArena Arena Human Preference 140K
https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k

## Final Dataset
Total examples: 6000

### Families
Claude    2000
GPT       2000
Gemini    2000

### Splits
train         4188
validation     916
test           896

## Dataset Construction

1. Used LMArena Arena Human Preference 140K as the source dataset.
2. Kept GPT, Claude, and Gemini families.
3. Restricted the main experiment to English.
4. Used single-turn conversations only:
   one user prompt + one assistant response.
5. Flattened Arena model-A/model-B battles into individual response examples.
6. Removed 23 empty responses.
7. Removed 47 exact duplicate examples.
8. Removed 153 obvious direct model self-identification responses.
9. Preserved punctuation, capitalization, Markdown, formatting, and response length.
10. Created prompt_group_id so identical prompts cannot appear across different train/validation/test splits.
11. Selected three representative model versions from each family.
12. Created exclusive domain labels:
    general, code, math, creative_writing.
13. Excluded mixed-domain examples from the final sample.
14. Created a balanced dataset of 2,000 examples per family.

## Domain Sampling Per Family

General: 800
Code: 500
Math: 350
Creative writing: 350

Total per family: 2,000

## Train / Validation / Test

Grouped approximately 70/15/15 using prompt_group_id.

Train: 4188
Validation: 916
Test: 896

Prompt-group overlap across splits: 0.

## Important Usage Rule

CNN and RNN/LSTM experiments must use the provided fixed
train/validation/test files.

Do not create new random splits.

Tokenizer/vocabulary should be fitted on TRAINING data only.

## Files

data/processed/final_dataset.parquet
data/processed/final_dataset.csv
data/processed/model_family_mapping.csv

data/splits/train.parquet
data/splits/validation.parquet
data/splits/test.parquet
