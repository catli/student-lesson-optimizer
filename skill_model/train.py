'''
    Create a matrix of all learners
'''

from skill import Skill
from skill_model.process_data import split_train_and_test_data, convert_token_to_matrix, extract_content_map
from model.gru import GRU_MODEL as gru_model

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import pdb
import yaml

# create a vector of ability for each session based on 
# the RNN model 
# and then multiply the t(ability)* mask 
# and the weight of difficulty*mask
# train the difficulty weight (gradient)

# based on 
# probability of correct =
#      exp(-1) + 1+exp(-(a-d)

# probability of correct + exp(-1) +1 = exp(a) + exp(d)
# probability of correct * exp(d) ~ exp(a)

def train(skill_model, rnn_model, optimizer, train_data, loader,
          train_keys, epoch, content_dim):
    # set in training node
    model.train()
    train_loss = []

    for step, batch_x in enumerate(loader):  # batch_x: index of batch data
        print('Epoch: ', epoch, ' | Iteration: ', step+1)
        # convert token data to matrix
        # need to convert batch_x from tensor flow object to numpy array
        # before converting to matrix
        input_padded, label_padded, label_mask, seq_lens = convert_token_to_matrix(
            batch_x[0].numpy(), train_data, train_keys, content_dim)
        # Variable, used to set tensor, but no longer necessary
        # Autograd automatically supports tensor with requires_grade=True
        #  https://pytorch.org/docs/stable/autograd.html?highlight=autograd%20variable
        padded_input = Variable(torch.Tensor(
            input_padded), requires_grad=False)  # .cuda()
        padded_label = Variable(torch.Tensor(
            label_padded), requires_grad=False)  # .cuda()
        padded_mask = Variable(torch.Tensor(
            label_mask), requires_grad=False)  # .cuda()

        # clear gradients and hidden state
        optimizer.zero_grad()
        # is this equivalent to generating prediction
        # what is the label generated?
        pred_ability = rnn_model(padded_input, seq_lens)
        y_pred = skill_model(pred_ability, padded_mask)  # .cuda()
        loss = model.loss(y_pred, padded_label, padded_mask)  # .cuda()

        print('Epoch %s: The %s-th iteration: %s loss \n' %(
            str(epoch), str(step+1), str(loss.data[0].numpy())))
        loss.backward()
        optimizer.step()
        # append the loss after converting back to numpy object from tensor
        train_loss.append(loss.data[0].numpy())
    pdb.set_trace()
    average_loss = np.mean(train_loss)
    return average_loss

def run_train_and_evaluate():
    content_index_filename = loaded_params['content_index_filename']
    rnn_model_filename = loaded_params['model_filename']
    nb_lstm_units = loaded_params['nb_lstm_units']
    nb_lstm_layers = loaded_params['nb_lstm_layers']

    exercise_to_index_map, content_dim = extract_content_map(
        content_index_filename)
    rnn_model = gru_model(input_dim=input_dim,
                      output_dim=content_dim,
                      nb_lstm_layers=nb_lstm_layers,
                      nb_lstm_units=nb_lstm_units,
                      batch_size=batchsize)
    rnn_model.load_state_dict(torch.load( rnn_model_filename ))
    skill_model = Skill(content_dim)
    optimizer = torch.optim.Adam([skill_model.skill_difficulty], lr = learning_rate)


if __name__ == '__main__':
    # set hyper parameters
    loaded_params = yaml.load(open('input/predict_params.yaml', 'r'))
    batchsize = loaded_params['batchsize']
    learning_rate = loaded_params['learning_rate']
    data_name = loaded_params['data_name']
    # perc_sample_print = loaded_params['perc_sample_print']
    run_train_and_evaluate(loaded_params)