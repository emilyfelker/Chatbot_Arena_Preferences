import re
import zipfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from textstat import textstat
from time import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

## TODO:
# plot to check for bias in dataset (preference for model a or b)
# unbias dataset
# try different models
# feature importance plot
# update ipython notebook for kaggle

start_time = time()  # to time entire program

def load_data(zip_file_path):
    try: # for local environment
        with zipfile.ZipFile(zip_file_path, 'r') as z:
            with z.open('train.csv') as train_file:
                train_df = pd.read_csv(train_file)
            with z.open('test.csv') as test_file:
                test_df = pd.read_csv(test_file)
    except (FileNotFoundError, zipfile.BadZipFile): # for Kaggle environment
        train_df = pd.read_csv('/kaggle/input/lmsys-chatbot-arena/train.csv')
        test_df = pd.read_csv('/kaggle/input/lmsys-chatbot-arena/test.csv')
    print(f"Data loaded. Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    return train_df, test_df


def create_target_column(df):
    df['target'] = df.apply(
        lambda row: 0 if row['winner_model_a'] == 1 else (1 if row['winner_model_b'] == 1 else 2), axis=1
    )
    return df


def plot_bias_in_dataset(train_df, filename='bias_distribution.png'):
    plt.figure(figsize=(6, 4))
    blue_shade = sns.color_palette("Blues")[4]
    sns.countplot(x='target', data=train_df, color=blue_shade, order=[0, 2, 1])
    plt.title("Distribution of User Preferences")
    plt.xlabel("Preference")
    plt.ylabel("Count")
    plt.xticks([0, 1, 2], ['Model A', 'Tie', 'Model B'])
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.clf()
    print(f"Bias distribution plot saved as {filename}")


def calculate_features(df):
    inputs = ['prompt', 'response_a', 'response_b']
    feature_dict = {}

    # Pre-compile regex patterns
    pronoun_patterns = {
        'pronoun_I_count': re.compile(r'\bI\b', re.IGNORECASE),
        'pronoun_you_count': re.compile(r'\byou\b', re.IGNORECASE),
        'pronoun_we_count': re.compile(r'\bwe\b', re.IGNORECASE)
    }

    for col in inputs:
        print(f"Calculating features for {col}...")

        # Character Count
        feature_dict[f'{col}_char_count'] = df[col].str.len()

        # Word List and Word Count
        word_list = df[col].str.findall(r'\b\w+\b')
        feature_dict[f'{col}_word_count'] = word_list.str.len()

        # Sentence Count
        sentence_count = df[col].str.count(r'[.!?]+').replace(0, 1)
        feature_dict[f'{col}_sentence_count'] = sentence_count

        # Average Word Length
        feature_dict[f'{col}_avg_word_length'] = (
            feature_dict[f'{col}_char_count'] / feature_dict[f'{col}_word_count'].replace(0, np.nan)
        )

        # Average Sentence Length (in words)
        feature_dict[f'{col}_avg_sentence_length'] = feature_dict[f'{col}_word_count'] / sentence_count

        # Punctuation Counts
        punctuations = {
            'exclamation_count': '!',
            'question_count': r'\?',
            'comma_count': ',',
            'period_count': r'\.',
            'semicolon_count': ';',
            'colon_count': ':'
        }

        for punct_name, punct_char in punctuations.items():
            feature_dict[f'{col}_{punct_name}'] = df[col].str.count(punct_char)

        # Pronoun Counts using pre-compiled patterns
        for pronoun_name, pattern in pronoun_patterns.items():
            feature_dict[f'{col}_{pronoun_name}'] = df[col].str.count(pattern)

        # Type-Token Ratio using vectorized operations
        all_words = word_list.explode()
        total_word_counts = feature_dict[f'{col}_word_count']
        unique_word_counts = all_words.groupby(level=0).nunique()
        feature_dict[f'{col}_type_token_ratio'] = unique_word_counts / total_word_counts.replace(0, np.nan)

        # Readability Scores
        texts = df[col].fillna('')
        # real ones commented out for now because they are a processing bottleneck
        # feature_dict[f'{col}_flesch_reading_ease'] = texts.map(textstat.flesch_reading_ease)
        # feature_dict[f'{col}_flesch_kincaid_grade'] = texts.map(textstat.flesch_kincaid_grade)
        feature_dict[f'{col}_flesch_reading_ease'] = texts.map(lambda *x : 0.0)
        feature_dict[f'{col}_flesch_kincaid_grade'] = texts.map(lambda *x : 0.0)


    # Convert the feature dictionary to a DataFrame and concatenate
    feature_df = pd.DataFrame(feature_dict)
    df = pd.concat([df.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)

    return df


def calculate_differences_and_ratios(df):
    print("Calculating differences and ratios between inputs...")

    pairs = [('prompt', 'response_a'), ('prompt', 'response_b'), ('response_a', 'response_b')]

    basic_features = ['char_count', 'word_count', 'sentence_count', 'avg_word_length', 'avg_sentence_length',
                      'exclamation_count', 'question_count', 'comma_count', 'period_count', 'semicolon_count',
                      'colon_count', 'pronoun_I_count', 'pronoun_you_count', 'pronoun_we_count', 'type_token_ratio',
                      'flesch_reading_ease', 'flesch_kincaid_grade']

    diff_ratio_dict = {}

    for feature in basic_features:
        for col1, col2 in pairs:
            diff = df[f'{col1}_{feature}'] - df[f'{col2}_{feature}']
            ratio = df[f'{col1}_{feature}'] / df[f'{col2}_{feature}'].replace(0, np.nan)

            diff_ratio_dict[f'{col1}_{col2}_{feature}_difference'] = diff
            diff_ratio_dict[f'{col1}_{col2}_{feature}_ratio'] = ratio

    diff_ratio_df = pd.DataFrame(diff_ratio_dict)
    diff_ratio_df.index = df.index

    df = pd.concat([df.reset_index(drop=True), diff_ratio_df.reset_index(drop=True)], axis=1)
    print("Finished calculating differences and ratios.")
    return df


def add_basic_features(df):
    df = calculate_features(df)
    df = calculate_differences_and_ratios(df)
    return df


def prepare_data(train_df):
    print("Preparing data for training...")
    feature_cols = [col for col in train_df.columns if any(keyword in col for keyword in [
        '_char_count', '_word_count', '_sentence_count',
        '_avg_word_length', '_avg_sentence_length',
        '_exclamation_count', '_question_count', '_comma_count', '_period_count',
        '_semicolon_count', '_colon_count', '_pronoun_I_count', '_pronoun_you_count',
        '_pronoun_we_count', '_type_token_ratio', '_flesch_reading_ease', '_flesch_kincaid_grade',
        '_difference', '_ratio'
    ])]

    train_df['target'] = train_df.apply(
        lambda row: 0 if row['winner_model_a'] == 1 else (1 if row['winner_model_b'] == 1 else 2), axis=1
    )
    X = train_df[feature_cols]
    y = train_df['target']
    X = X.fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Data prepared: {X_train.shape[0]} training samples, {X_val.shape[0]} validation samples.")
    return X_train, X_val, y_train, y_val, scaler


def train_model(X_train, y_train):
    print("Training model...")
    model = LogisticRegression(solver='lbfgs', max_iter=2000)
    model.fit(X_train, y_train)
    print("Model training complete.")
    return model


def evaluate_model(model, X_val, y_val):
    print("Evaluating model...")
    y_val_pred_proba = model.predict_proba(X_val)
    loss = log_loss(y_val, y_val_pred_proba)
    print(f'Validation Log Loss: {loss}')
    y_val_pred = model.predict(X_val)
    cm = confusion_matrix(y_val, y_val_pred)
    plot_confusion_matrix(cm)
    print("Model evaluation complete.")


def plot_confusion_matrix(cm, filename='confusion_matrix.png'):
    reordered_indices = [0, 2, 1]  # to put the tie case in the middle
    reordered_cm = cm[reordered_indices][:, reordered_indices]
    cm_relative = reordered_cm.astype('float') / reordered_cm.sum(axis=1)[:, np.newaxis]
    labels = ['Model A Wins', 'Tie', 'Model B Wins']
    sns.heatmap(cm_relative, annot=True, fmt='.2%', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Relative Counts)')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.clf()
    print(f"Confusion matrix saved as {filename}.")


def make_predictions(model, test_df, scaler):
    print("Making predictions on the test set...")
    feature_cols = [col for col in test_df.columns if any(keyword in col for keyword in [
        '_char_count', '_word_count', '_sentence_count',
        '_avg_word_length', '_avg_sentence_length',
        '_exclamation_count', '_question_count', '_comma_count', '_period_count',
        '_semicolon_count', '_colon_count', '_pronoun_I_count', '_pronoun_you_count',
        '_pronoun_we_count', '_type_token_ratio', '_flesch_reading_ease', '_flesch_kincaid_grade',
        '_difference', '_ratio'
    ])]
    X_test = test_df[feature_cols].fillna(0)
    X_test_scaled = scaler.transform(X_test)
    y_test_pred_proba = model.predict_proba(X_test_scaled)
    submission_df = test_df[['id']].copy()
    submission_df['winner_model_a'] = y_test_pred_proba[:, 0]
    submission_df['winner_model_b'] = y_test_pred_proba[:, 1]
    submission_df['winner_model_tie'] = y_test_pred_proba[:, 2]
    print("Test set predictions complete.")
    return submission_df


def create_submission_file(submission_df, filename='submission.csv'):
    submission_df.to_csv(filename, index=False)
    print(f"Submission file saved as {filename}")


def main():
    train_df, test_df = load_data('data/lmsys-chatbot-arena.zip')
    train_df = create_target_column(train_df)
    plot_bias_in_dataset(train_df, 'bias_distribution.png')
    train_df = add_basic_features(train_df)
    test_df = add_basic_features(test_df)
    X_train, X_val, y_train, y_val, scaler = prepare_data(train_df)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_val, y_val)
    submission = make_predictions(model, test_df, scaler)
    create_submission_file(submission)

    # Print total runtime
    print(f"Total runtime: {time() - start_time:.2f} seconds")

if __name__ == '__main__':
    main()