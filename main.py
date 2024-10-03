import pandas as pd
import zipfile


def load_data(zip_file_path):
    zip_file_path = 'data/lmsys-chatbot-arena.zip'
    with zipfile.ZipFile(zip_file_path, 'r') as z:
        with z.open('train.csv') as train_file:
            train_df = pd.read_csv(train_file)
        with z.open('test.csv') as test_file:
            test_df = pd.read_csv(test_file)
    pd.set_option('display.max_columns', None)
    print(train_df.head())
    print(test_df.head())


if __name__ == '__main__':
    # data pre-processing:
    # load dataset
    load_data('data/lmsys-chatbot-arena.zip')

    # inspect and clean data including proper formatting
    # no need to clean because of how this dataset was constructed (no missing values)

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

