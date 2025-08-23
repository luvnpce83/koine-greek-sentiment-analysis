# Emotions in the Identity of Paul: A Sentiment Analysis Approach to Paul’s Jewish Identity

**Author**: Young Un Kang  
**Affiliation**: College of Theology, Yonsei University

[![Build Status](https://github.com/luvnpce83/koine-greek-sentiment-analysis/actions/workflows/main.yml/badge.svg)](https://github.com/luvnpce83/koine-greek-sentiment-analysis/actions/workflows/main.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📖 Overview

This repository contains the code, data, and fine-tuned models for the doctoral dissertation, "Emotions in the Identity of Paul." This project applies state-of-the-art Natural Language Processing (NLP) to analyze emotional expression in the Pauline Epistles of the New Testament.

The core of this work involves two specialized models fine-tuned on bespoke Koine Greek datasets:
1.  An **Emotion Classifier** for fine-grained analysis based on Plutchik’s Wheel of Emotions.
2.  A **Valence Analyzer** for Aspect-Based Sentiment Analysis (ABSA), assessing the positive, negative, or neutral sentiment of a given text.

This project emphasizes methodological transparency and reproducibility. All code, data, training scripts, and final models are publicly available.

## 🤖 Models & Datasets

This study developed two primary models, each trained on a custom-built "gold standard" dataset.

### 1. Emotion Classifier

A multi-class classification model fine-tuned to analyze texts for Plutchik’s eight basic emotions (Joy, Trust, Fear, Surprise, Sadness, Disgust, Anger, Anticipation).

-   **Model**: `pranaydeeps/Ancient-Greek-BERT` fully fine-tuned.
-   **Hugging Face Hub**: [luvnpce83/ancient-greek-emotion-bert](https://huggingface.co/luvnpce83/ancient-greek-emotion-bert)
-   **Performance**: Achieved a **Macro F1 score of 0.626** on the hold-out test set.

#### Methodology Overview

The training corpus was constructed through a rigorous, multi-stage protocol to ensure the highest quality annotations for the low-resource context of Koine Greek.

1.  **Candidate Sourcing**: Candidate verses were systematically identified from the *Greek-English Lexicon of the New Testament Based on Semantic Domains* (Louw-Nida), focusing on Domain 25 (Attitudes and Emotions) and domains representing "social-emotional acts" (33, 87.B, 88).
2.  **Human-in-the-Loop (HITL) Annotation**: A Large Language Model (LLM) provided a baseline analysis, which was then meticulously reviewed, corrected, and validated by the primary researcher.
3.  **Corpus Finalization**: This process resulted in a "golden standard" corpus of 884 samples with clear emotional expressions.
4.  **Data Augmentation**: To ensure robust training, the dataset was expanded to 2,616 samples using back-translation and generative augmentation techniques.

### 2. Valence Analyzer

A regression model that performs nuanced Aspect-Based Sentiment Analysis (ABSA) by predicting a valence score from -1.0 (Negative) to +1.0 (Positive).

-   **Model**: `pranaydeeps/Ancient-Greek-BERT` fine-tuned for regression.
-   **Hugging Face Hub**: [luvnpce83/ancient-greek-valence-bert](https://huggingface.co/luvnpce83/ancient-greek-valence-bert)
-   **Performance**: Achieved a **Pearson correlation of ~0.64** on the in-domain test set.

#### Methodology Overview

To our knowledge, this work represents the first creation of a "gold standard" sentiment corpus for the New Testament. The model was trained on a unified dataset from two distinct sources, building upon the precedent set by prior work in Homeric Greek.

1.  **Homeric Dataset**:
    -   This component utilizes the sentiment dataset from the work of Pavlopoulos et al. on the Homeric epics, which involved 16 expert annotators and resulted in 611 samples.
    -   **Citation**: Pavlopoulos, J., et al. (2022). "Sentiment Analysis of Homeric Text: The 1st Book of Iliad." *LREC 2022*. ([Link](https://aclanthology.org/2022.lrec-1.765/))
    -   **Source**: [https://github.com/ipavlopoulos/sentiment_in_homeric_text](https://github.com/ipavlopoulos/sentiment_in_homeric_text)

2.  **New Testament Dataset**:
    -   A new, bespoke corpus of **82 samples** was created for this dissertation.
    -   The dataset was annotated by **eight domain experts** (Th.D. graduates, Ph.D. students, or Ph.D. graduates in New Testament studies from Yonsei University).
    -   Annotators classified the felt emotion of each verse as 'Positive', 'Negative', or 'Neutral', which was used to calculate a final valence score.

3.  **Corpus Merging & Augmentation**: The smaller New Testament corpus was first expanded using the same data augmentation techniques as the emotion classifier. It was then merged with the Homeric data to form the final, robust training corpus after an Exploratory Data Analysis (EDA) confirmed the statistical compatibility of the two sources.

## ⚙️ Setup and Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/luvnpce83/koine-greek-sentiment-analysis.git
    cd koine-greek-sentiment-analysis
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Run Inference:** To predict the emotion of a Koine Greek sentence, use `predict.py`:
    ```bash
    python src/inference/predict.py --text "ὦ ἀνόητοι Γαλάται"
    ```

## 📊 Experiment Tracking

This project uses [Weights & Biases](https://wandb.ai) for experiment tracking. To log your own experiments, sign up for a free account and run `wandb login` before executing a training script.

## 📄 Citation

If you use the models, data, or code from this project in your research, please cite the following dissertation:

```bibtex
@phdthesis{kang2025emotions,
  author    = {Kang, Young Un},
  title     = {Emotions in the Identity of Paul: A Sentiment Analysis Approach to Paul’s Jewish Identity},
  school    = {School of Theology, Yonsei University},
  year      = {2025}
}
```

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
