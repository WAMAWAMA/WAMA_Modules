# ref: CNN-RNN: A Unified Framework for Multi-label Image Classification, 2016
# https://github.com/AmrMaghraby/CNN-RNN-A-Unified-Framework-for-Multi-label-Image-Classification
# https://blog.csdn.net/weixin_43436958/article/details/107827306
# https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning todo prefered this one

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from wama_modules.Encoder import ResNetEncoder
from wama_modules.BaseModule import GlobalMaxPool
from demo.multi_label.generate_multilabel_dataset import label_category_dict, label_name, dataset


class Vocabulary():
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


class CNNRNN(nn.Module):
    def __init__(self, in_channel, embed_dim, hidden_dim, vocab_size, num_layers, max_seq_length=20, dim=2):
        """
        :param embed_size:
        :param hidden_size:
        :param vocab_size:
        :param num_layers:
        :param max_seq_length:
        """
        """Set the hyper-parameters and build the layers."""
        super().__init__()

        f_channel_list = [64, 128, 256, embed_dim]
        self.img_embed = ResNetEncoder(
            in_channel,
            stage_output_channels=f_channel_list,
            stage_middle_channels=f_channel_list,
            blocks=[1, 2, 3, 4],
            type='131',
            downsample_ration=[0.5, 0.5, 0.5, 0.8],
            dim=dim)
        self.pool = GlobalMaxPool()

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, vocab_size)
        self.max_seg_length = max_seq_length

    def forward(self, image, captions, lengths):
        # encoder
        features = self.pool(self.img_embed(image)[-1])

        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)

        packed = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        """
        In training phase (forward), for the decoder part, source and target texts are predefined and shifted. 
        For example, if the image description is "Giraffes standing next to each other", 
        1) input: the source sequence is a list containing :
        ['<start>', 'A', 'B', 'C'] 
        2) output: the target sequence is a list containing 
        ['A', 'B', 'C', '<end>']

        But, in real implement,
        1) output: the target sequence is a list containing 
        ['<start>', 'A', 'B', 'C', '<end>']
        lengths = 5
        2) input: the source sequence is a list containing :
        [image_features,'<start>', 'A', 'B', 'C','<end>'] 
        because the image_features is the first token, which make a shift of the sequence
        and !, at the same time, we use the "pack_padded_sequence" with arg lengths = 5,
        so its equal to:
        [image_features,'<start>', 'A', 'B', 'C']
        """

        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs

    def inference(self, image, states=None):
        # encoder
        features = self.pool(self.img_embed(image)[-1])

        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)  # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))  # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)  # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)  # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)  # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)  # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids



if __name__ == '__main__':
    image_1D_tensor = (torch.tensor(np.stack([case['img_1D'].astype(np.float32) for case in dataset], 0))).permute(0, 2, 1)
    image_2D_tensor = (torch.tensor(np.stack([case['img_2D'].astype(np.float32) for case in dataset], 0))).permute(0, 3, 1, 2)
    image_3D_tensor = (torch.tensor(np.stack([case['img_3D'].astype(np.float32) for case in dataset], 0))).permute(0, 4, 1, 2, 3)

    # CNNRNN only support binary labels
    label_category_dict['weather'] = 2  # 5 → 2, force binary classification


    # make vocabulary
    # Create a vocab wrapper and add some special default tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # add labels to vocabulary
    for label in label_name:
        vocab.add_word(label)

    # make input data
    """
    targets: torch tensor of shape (batch_size, padded_length).
    lengths: list; valid length for each padded caption.
    """

    """ REORDER the label name
    the input label order is important for training phase
    as the original paper said (section 3.4. Training)
    https://openaccess.thecvf.com/content_cvpr_2016/papers/Wang_CNN-RNN_A_Unified_CVPR_2016_paper.pdf
    they used occurrence frequencies to fix the order, that is, 
    let the More frequent labels appear earlier than the
    less frequent ones, which corresponds to the intuition that 
    easier objects should be predicted first to help predict more difficult objects
    and they also tried to randomly permute the label
    orders in each mini-batch, but it makes the training very
    difficult to converge.

    However, I prefer to use a strategy which called minimum loss
    that is, we can compute the loss between pred_logits and each labels in all possible order
    and then choose the min loss to backward

    e.g.
    loss = []
    loss.append(criterion([1,4,5], gt = [1,5,2]))
    loss.append(criterion([1,4,5], gt = [1,2,5]))
    loss.append(criterion([1,4,5], gt = [2,1,5]))
    loss.append(criterion([1,4,5], gt = [2,5,1]))
    loss.append(criterion([1,4,5], gt = [5,2,1]))
    loss.append(criterion([1,4,5], gt = [5,1,2]))
    loss = min(loss)

    But, for huge amount of labels, this method is not efficient, so
    the Hungarian matching is preferred (see DETR)
    """

    # let more frequent labels appear earlier than the less frequent ones
    all_labels = np.array([case['label_value'] for case in dataset])
    positive_label_frequency = np.zeros(shape=(len(label_name)))
    for index in range(all_labels.shape[0]):
        _labels = all_labels[index]
        for i in range(all_labels.shape[1]):
            if _labels[i] == 1:
                positive_label_frequency[i] += 1
    positive_label_frequency_add_labelindex = np.stack([positive_label_frequency,np.array(list(range(len(positive_label_frequency))))])
    positive_label_frequency_add_labelindex = positive_label_frequency_add_labelindex[:, positive_label_frequency_add_labelindex[0].argsort()]
    positive_label_frequency_add_labelindex = positive_label_frequency_add_labelindex[:,::-1]
    new_label_name = [label_name[int(i)] for i in positive_label_frequency_add_labelindex[1]]

    # add positive_labels
    for case in dataset:
        case['positive_labels'] = []
        for label in new_label_name:
            if case['label_value'][label_name.index(label)] == 1:
                case['positive_labels'].append(label)

    # add caption
    for case in dataset:
        caption = []  # word index list
        caption.append(vocab('<start>'))
        caption.extend([vocab(p_label) for p_label in case['positive_labels']])
        caption.append(vocab('<end>'))
        case['caption'] = torch.Tensor(caption)

    # test 1D models --------------------------------------------------------------
    images = image_1D_tensor
    dim = 1
    print('-' * 22, 'build 1D model', '-'*18)
    print('input image batch shape:', images.shape)

    embed_size = 256  # default in paper
    hidden_size = 512  # default in paper
    num_layers = 1  # default in paper
    model = CNNRNN(
        in_channel=2,
        embed_dim=512,
        hidden_dim=hidden_size,
        vocab_size=len(vocab),
        num_layers=num_layers,
        dim=dim)

    # train phase
    lengths = [len(case['caption']) for case in dataset]
    captions = torch.zeros(len(dataset), max(lengths)).long()  # zero is the padding index, start 1, end 2
    for case_i, case in enumerate(dataset):
        end = lengths[case_i]
        captions[case_i, :end] = case['caption'][:end]

    targets = pack_padded_sequence(captions, lengths, batch_first=True, enforce_sorted=False)[0]

    outputs = model(images, captions, lengths)

    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, targets)

    # inference phase
    pre_label_idx_batch = model.inference(images)
    for case_i in range(pre_label_idx_batch.shape[0]):
        pre_label_idx = pre_label_idx_batch[case_i].cpu().numpy()  # select the 1st case in the minibatch
        positive_labels = []
        for word_id in pre_label_idx:
            word = vocab.idx2word[word_id]
            positive_labels.append(word)
            if word == '<end>':
                break
        print('case_i:', case_i, 'positive label', positive_labels)


    # test 2D models --------------------------------------------------------------
    images = image_2D_tensor
    dim = 2
    print('-' * 22, 'build 2D model', '-'*18)
    print('input image batch shape:', images.shape)

    embed_size = 256  # default in paper
    hidden_size = 512  # default in paper
    num_layers = 1  # default in paper
    model = CNNRNN(
        in_channel=2,
        embed_dim=512,
        hidden_dim=hidden_size,
        vocab_size=len(vocab),
        num_layers=num_layers,
        dim=dim)

    # train phase

    lengths = [len(case['caption']) for case in dataset]
    captions = torch.zeros(len(dataset), max(lengths)).long()  # zero is the padding index, start 1, end 2
    for case_i, case in enumerate(dataset):
        end = lengths[case_i]
        captions[case_i, :end] = case['caption'][:end]

    targets = pack_padded_sequence(captions, lengths, batch_first=True, enforce_sorted=False)[0]

    outputs = model(images, captions, lengths)

    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, targets)

    # inference phase
    pre_label_idx_batch = model.inference(images)
    for case_i in range(pre_label_idx_batch.shape[0]):
        pre_label_idx = pre_label_idx_batch[case_i].cpu().numpy()  # select the 1st case in the minibatch
        positive_labels = []
        for word_id in pre_label_idx:
            word = vocab.idx2word[word_id]
            positive_labels.append(word)
            if word == '<end>':
                break
        print('case_i:', case_i, 'positive label', positive_labels)


    # test 3D models --------------------------------------------------------------
    images = image_3D_tensor
    dim = 3
    print('-' * 22, 'build 3D model', '-'*18)
    print('input image batch shape:', images.shape)

    embed_size = 256  # default in paper
    hidden_size = 512  # default in paper
    num_layers = 1  # default in paper
    model = CNNRNN(
        in_channel=2,
        embed_dim=512,
        hidden_dim=hidden_size,
        vocab_size=len(vocab),
        num_layers=num_layers,
        dim=dim)

    # train phase

    lengths = [len(case['caption']) for case in dataset]
    captions = torch.zeros(len(dataset), max(lengths)).long()  # zero is the padding index, start 1, end 2
    for case_i, case in enumerate(dataset):
        end = lengths[case_i]
        captions[case_i, :end] = case['caption'][:end]

    targets = pack_padded_sequence(captions, lengths, batch_first=True, enforce_sorted=False)[0]

    outputs = model(images, captions, lengths)

    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, targets)

    # inference phase
    pre_label_idx_batch = model.inference(images)
    for case_i in range(pre_label_idx_batch.shape[0]):
        pre_label_idx = pre_label_idx_batch[case_i].cpu().numpy()  # select the 1st case in the minibatch
        positive_labels = []
        for word_id in pre_label_idx:
            word = vocab.idx2word[word_id]
            positive_labels.append(word)
            if word == '<end>':
                break
        print('case_i:', case_i, 'positive label', positive_labels)


