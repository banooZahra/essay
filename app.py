import numpy as np
import pandas as pd
import collections
from nltk.tokenize import word_tokenize
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from flask import Flask, send_file, request


def tokenize(texts):
    """Tokenize texts, build vocabulary and find maximum sentence length.

    Args:
        texts (List[str]): List of text data

    Returns:
        tokenized_texts (List[List[str]]): List of list of tokens
        word2idx (Dict): Vocabulary built from the corpus
        max_len (int): Maximum sentence length
    """

    max_len = 0
    tokenized_texts = []
    word2idx = {}

    # Add <pad> and <unk> tokens to the vocabulary
    word2idx['<pad>'] = 0
    word2idx['<unk>'] = 1

    # Building our vocab from the corpus starting from index 2
    idx = 2
    for sent in texts:
        tokenized_sent = word_tokenize(sent)

        # Add `tokenized_sent` to `tokenized_texts`
        tokenized_texts.append(tokenized_sent)

        # Add new token to `word2idx`
        for token in tokenized_sent:
            if token not in word2idx:
                word2idx[token] = idx
                idx += 1

        # Update `max_len`
        max_len = max(max_len, len(tokenized_sent))

    return tokenized_texts, word2idx, max_len


def encode(tokenized_texts, word2idx, max_len):
    """Pad each sentence to the maximum sentence length and encode tokens to
    their index in the vocabulary.

    Returns:
        input_ids (np.array): Array of token indexes in the vocabulary with
            shape (N, max_len). It will the input of our CNN model.
    """

    input_ids = []
    for tokenized_sent in tokenized_texts:
        # Pad sentences to max_len
        tokenized_sent += ['<pad>'] * (max_len - len(tokenized_sent))

        # Encode tokens to input_ids
        input_id = [word2idx.get(token) for token in tokenized_sent]
        input_ids.append(input_id)

    return np.array(input_ids)


class CNN_NLP(nn.Module):
    """A 1D Convulational Neural Network for Sentence Classification."""

    def __init__(self,
                 pretrained_embedding=None,
                 freeze_embedding=False,
                 filter_sizes=None,
                 num_filters=None,
                 num_classes=2,
                 dropout=0.5):
        """
        The constructor for CNN_NLP class.

        Args:
            pretrained_embedding (torch.Tensor): Pretrained embeddings with
                shape (vocab_size, embed_dim)
            freeze_embedding (bool): Set to False to fine-tune pretraiend
                vectors. Default: False
            filter_sizes (List[int]): List of filter sizes. Default: [3, 4, 5]
            num_filters (List[int]): List of number of filters, has the same
                length as `filter_sizes`. Default: [100, 100, 100]
            num_classes (int): Number of classes. Default: 2
            dropout (float): Dropout rate. Default: 0.5
        """

        super(CNN_NLP, self).__init__()
        # Embedding layer
        if num_filters is None:
            num_filters = [100, 100, 100]
        if filter_sizes is None:
            filter_sizes = [3, 4, 5]
        if pretrained_embedding is not None:
            self.vocab_size, self.embed_dim = pretrained_embedding.shape
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding,
                                                          freeze=freeze_embedding)

        # Conv Network
        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(in_channels=self.embed_dim,
                      out_channels=num_filters[i],
                      kernel_size=filter_sizes[i])
            for i in range(len(filter_sizes))
        ])
        # Fully-connected layer and Dropout
        self.fc1 = nn.Linear(np.sum(num_filters), 50)
        self.fc2 = nn.Linear(50, num_classes)
        self.dropout = nn.Dropout(p=dropout, inplace=False)

    def forward(self, input_ids, lang):
        """Perform a forward pass through the network.

        Args:
            input_ids (torch.Tensor): A tensor of token ids with shape
                (batch_size, max_sent_length)
            lang: input language

        Returns:
            logits (torch.Tensor): Output logits with shape (batch_size,n_classes)
        """

        # Get embeddings from `input_ids`. Output shape: (b, max_len, embed_dim)
        x_embed = self.embedding(input_ids).float()

        # Permute `x_embed` to match input shape requirement of `nn.Conv1d`.
        # Output shape: (b, embed_dim, max_len)
        x_reshaped = x_embed.permute(0, 2, 1)

        # Apply CNN and ReLU. Output shape: (b, num_filters[i], L_out)
        x_conv_list = [F.relu(conv1d(x_reshaped)) for conv1d in self.conv1d_list]

        # Max pooling. Output shape: (b, num_filters[i], 1)
        x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
                       for x_conv in x_conv_list]

        # Concatenate x_pool_list to feed the fully connected layer.
        # Output shape: (b, sum(num_filters))
        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list],
                         dim=1)

        # Compute logits. Output shape: (b, n_classes)
        fc1 = self.fc1(self.dropout(x_fc))
        fc1 = F.relu(fc1)
        logits = self.fc2(self.dropout(fc1))

        return logits


def make_plot(personality_type, predictions):
    colors_0 = ['royalblue', 'lightpink', 'mediumpurple', 'goldenrod']
    colors_1 = ['midnightblue', 'palevioletred', 'rebeccapurple', 'darkgoldenrod']
    # read_descriptions()
    plt.rcdefaults()
    fig, ax = plt.subplots(figsize=(12, 6))

    types = ['Extrovert(E)', 'Intuitive(N)', 'Feeler(F)', 'Judger(J)']
    y_pos = np.arange(len(types))
    performance = [prediction * 100 for prediction in predictions]

    # Plot a solid vertical grid line to highlight the median position
    ax.axvline(50, color='grey', alpha=0.4)
    ax.xaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)
    rects_0 = ax.barh(y_pos, performance, align='center', height=0.6)
    rects_1 = ax.barh(y_pos, [100 - per for per in performance], left=performance, align='center', height=0.6)
    ax.set_xticks(np.arange(0, 101, 10))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(types)
    title = 'Your Personality Type: ' + personality_type
    ax.set_title(title)
    # set labels
    i = 0
    for rect_0, rect_1 in zip(rects_0, rects_1):
        rect_0.set_color(colors_0[i])
        rect_1.set_color(colors_1[i])
        width_0 = int(rect_0.get_width())
        width_1 = int(rect_0.get_width())
        # Center the text vertically in the bar
        yloc_0 = rect_0.get_y() + rect_0.get_height() / 2
        yloc_1 = rect_1.get_y() + rect_1.get_height() / 2
        ax.annotate(str(100 - int(predictions[i] * 100)) + "%", xy=(width_0, yloc_0), xytext=(5, 0),
                    textcoords="offset points",
                    horizontalalignment='left', verticalalignment='center', color='white', weight='bold', clip_on=True)
        ax.annotate(str(int(predictions[i] * 100)) + "%", xy=(width_1, yloc_1), xytext=(-5, 0),
                    textcoords="offset points",
                    horizontalalignment='right', verticalalignment='center', color='black', weight='bold', clip_on=True)
        i += 1

    ax2 = ax.twinx()
    ax2.set_yticks(y_pos)
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticklabels(['Introvert(I)', 'Sensor(S)', 'Thinker(T)', 'Perceiver(P)'])
    # plt.show()
    plt.savefig('static/result.png')


def test_evaluate(language, model, val_dataloader):
    """After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    # Tracking variables
    label_predictions = []
    user_index = []

    # For each batch in our validation set...
    for batch in val_dataloader:
        # Load batch to GPU
        b_input_ids, b_user_index = tuple(t for t in batch)

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, language)
            preds = F.softmax(logits, dim=1)
            print(b_user_index, b_input_ids, preds[:, 1])
            label_predictions.extend(preds[:, 1].cpu().detach().tolist())
            user_index.extend(b_user_index.detach().tolist())
    print(label_predictions)
    print(user_index)
    return user_index, label_predictions


def calculate_personality(language):
    test_df = pd.read_csv('anvil_test.csv')
    test_df['u_idx'] = [i for i in range(0, len(test_df))]
    test_df['text'], word2idx, max_len = tokenize(test_df['text'])
    test_df = pd.DataFrame(test_df)
    test_input_ids = encode(test_df['text'], word2idx, max_len)
    test_inputs = np.array(test_input_ids)
    test_inputs = torch.tensor(test_inputs)
    test_idx = torch.tensor(test_df['u_idx'])
    batch_size = 1
    test_data = TensorDataset(test_inputs, test_idx)
    test_sampler = RandomSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
    DIMENSIONS = ['EI', 'NS', 'FT', 'JP']
    for k in range(len(DIMENSIONS)):
        model = torch.load(
            'D:/B/XLM-RoBERTa/models/cnn_(kaggle+essays-test)_' + language + '_fast_xlm_' + DIMENSIONS[k][0] + '.pth',
            map_location=torch.device('cpu'))
        u_idx, label_pred = test_evaluate(language, model, test_dataloader)
        test_df['predict_{}'.format(DIMENSIONS[k][0])] = [pred for _, pred in sorted(zip(u_idx, label_pred))]

    predicted_type = []
    types = collections.defaultdict(int)
    for _, row in test_df.iterrows():
        p_type = ''
        for k in range(len(DIMENSIONS)):
            if row['predict_{}'.format(DIMENSIONS[k][0])] > 0.5:
                p_type += DIMENSIONS[k][0]
                types[DIMENSIONS[k][0]] += 1
            else:
                p_type += DIMENSIONS[k][1]
                types[DIMENSIONS[k][1]] += 1
        predicted_type.append(p_type)
    test_df['predicted_type'] = predicted_type
    print(test_df)
    user_predictions = []
    user_type = ''
    for k in range(len(DIMENSIONS)):
        post_number = len(test_df)
        avg_prediction = test_df['predict_{}'.format(DIMENSIONS[k][0])].sum() / post_number
        user_predictions.append(avg_prediction)
        if avg_prediction > 0.5:
            user_type += DIMENSIONS[k][0]
        else:
            user_type += DIMENSIONS[k][1]
    return test_df, user_type, user_predictions


def predict(lang):
    df, u_type, u_predictions = calculate_personality(lang)
    print(u_type, u_predictions)
    make_plot(u_type, u_predictions)


app = Flask(__name__)


@app.route("/pd/textInput", methods=['POST'])
def textInput():
    expdata = {'text': []}
    content = request.form['m']
    expdata['text'].append(content)
    df = pd.DataFrame.from_dict(expdata)
    df.to_csv('anvil_test.csv', index=False)
    lang = request.form['lang']
    predict(lang)
    return send_file('static/result.png', mimetype='image/gif')


@app.route("/pd/fileInput", methods=['POST'])
def fileInput():
    expdata = {'text': []}
    uploaded_files = request.files.getlist("file[]")
    print(uploaded_files)
    for f in uploaded_files:
        content = f.read().decode('utf-8')
        print('content: ', content)
        expdata['text'].append(content)
    df = pd.DataFrame.from_dict(expdata)
    df.to_csv('anvil_test.csv', index=False)
    lang = request.form['lang']
    predict(lang)
    return send_file('static/result.png', mimetype='image/gif')


if __name__ == "__main__":
    app.run(host='46.209.4.217', port=8001)
