import pandas as pd
import zipfile


def load_data(zip_file_path):
    zip_file_path = 'data/lmsys-chatbot-arena.zip'
    with zipfile.ZipFile(zip_file_path, 'r') as z:
        with z.open('train.csv') as train_file:
            train_df = pd.read_csv(train_file)
        with z.open('test.csv') as test_file:
            test_df = pd.read_csv(test_file)
    return train_df, test_df


def add_basic_features(df):
    df['response_a_length'] = df['response_a'].apply(len)
    df['response_b_length'] = df['response_b'].apply(len)
    return df

if __name__ == '__main__':
    # Load dataset
    train_df, test_df = load_data('data/lmsys-chatbot-arena.zip')

    # Add basic features like response length
    train_df = add_basic_features(train_df)
    test_df = add_basic_features(test_df)

    # Set pandas to display all columns for inspection
    pd.set_option('display.max_columns', None)

    # Inspect the modified training data with new features
    print(train_df.head())

# Feature Engineering:
# extract useful features from the text data, such as:
# text embeddings (using pre-trained models like BERT, GPT, etc.)
# handcrafted features like response length, sentiment analysis, verbosity, or other response-level characteristics.
# consider including metadata (e.g., the prompt, position of the response)

# model selection for classification

# train model
# use techniques such as cross-validation to ensure the model is generalizing well.
# monitor the model's performance on the validation set using log loss as the evaluation metric

# evaluate model
# test the model on unseen data and compute the log loss to measure performance.
# compare the predicted probabilities against the actual user preferences to assess accuracy

