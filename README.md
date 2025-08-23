# Emotions in the Identity of Paul: A Sentiment Analysis Approach to Paul’s Jewish Identity

**Author**: Young Un Kang  
**Affiliation**: College of Theology, Yonsei University

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📖 Overview

This repository contains the code, data, and fine-tuned models for the doctoral dissertation, "Emotions in the Identity of Paul." The project focuses on applying Natural Language Processing (NLP) techniques to analyze emotional expression in the Pauline Epistles. The core of this work is a custom-trained transformer model, fine-tuned on a bespoke dataset of Koine Greek, to classify texts according to Plutchik’s wheel of emotions.

## 📂 Repository Structure

-   **/data**: Contains the raw and processed datasets used for training.
    -   `raw`: The initial 884 samples curated from the Louw-Nida lexicon.
    -   `processed`: The final, augmented dataset of 2,616 samples.
-   **/src**: Includes all Python source code.
    -   `/augmentation`: Scripts for data augmentation (back-translation, generative).
    -   `/training`: Scripts for fine-tuning the models.
    -   `/inference`: A script (`predict.py`) for running predictions on new text.
-   `requirements.txt`: A list of all necessary Python packages.

## 🤖 Models

This study developed two primary models for different levels of textual analysis:

1.  **Emotion Classifier**
    -   **Task**: An 8-class classifier for Plutchik's basic emotions (Joy, Trust, Fear, Surprise, Sadness, Disgust, Anger, Anticipation).
    -   **Performance**: Achieved a **Macro F1 score of 0.626** on a strictly held-out validation set.
    -   **Status**: Complete. Model weights are available in `/models/emotion_classifier`.

2.  **Valence Analyzer**
    -   **Task**: A binary classifier for sentiment polarity (Positive/Negative).
    -   **Purpose**: To provide a broader sentiment overview, complementing the granular emotion analysis.
    -   **Status**: In development.

The fine-tuned model is available on the Hugging Face Hub. This model was fine-tuned from the `pranaydeeps/Ancient-Greek-BERT` base model.

Emotion Classifier: [luvnpce83/ancient-greek-emotion-bert](https://huggingface.co/luvnpce83/ancient-greek-emotion-bert)

## 💾 Dataset

The model was trained on a custom dataset of 2,616 annotated sentences in Koine Greek. The creation process involved:
1.  **Initial Curation**: A "golden standard" corpus of 884 samples was manually created based on the semantic domains of the Louw-Nida lexicon.
2.  **Data Augmentation**: To overcome the low-resource nature of Ancient Greek, the dataset was expanded using back-translation and generative augmentation via a large language model.

## ⚙️ Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [Your-Repository-URL]
    cd [repository-folder]
    ```

2.  **Create a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r src/training/requirements.txt
    ```

## 🚀 Usage

### Training

To replicate the training process for the emotion classifier, run the training script with the optimal hyperparameters:

```bash
python src/training/train_emotion_classifier.py --epochs 25 --learning_rate 5e-5
```

### Inference

To predict the emotion of a new Koine Greek sentence, use the `predict.py` script:

```bash
python src/inference/predict.py --text "ὦ ἀνόητοι Γαλάται"
```
**Expected Output:**
```
✅ Predicted Emotion: Anger

--- Probability Distribution ---
Anger: 0.8521
Disgust: 0.0893
...
```

## 📊 Experiment Tracking & Visualization

This project is integrated with [Weights & Biases](https://wandb.ai) for experiment tracking. All training scripts, including `train_emotion_classifier.py` and `train_valence_regressor.py`, will automatically log metrics, losses, and model performance to your `wandb` workspace.

To use this feature:
1.  **Sign up** for a free account at [wandb.ai](https://wandb.ai).
2.  **Log in** from your terminal before running the training script:
    ```bash
    wandb login
    ```
    You will be prompted to enter the API key found in your `wandb` profile settings.
3.  Once you are logged in, simply run the training script. The console output will include a link to view your experiment live.

If you do not wish to use `wandb`, the training will still proceed normally, but no metrics will be logged online.

## 📊 Results

The final emotion classification model achieved a **Macro F1 score of 0.626**. The fine-tuned Valence Analyzer achieved a **Pearson correlation of ~0.64** on the primary in-domain test set, indicating a strong ability to predict sentiment trends.

This result was obtained using a rigorous evaluation protocol where the validation set was kept strictly separate from the data augmentation pipeline. This provides a more reliable measure of the model's ability to generalize to unseen data, a critical consideration in a low-resource ancient language context.

For detailed logs, training curves, and a full history of all experiments, please refer to the project's [Weights & Biases workspace](https://wandb.ai/luvnpce-yonsei-university/huggingface/workspace?nw=nwuserluvnpce).

## 📄 Citation

If you use the code or data from this project in your research, please cite the following dissertation:

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
