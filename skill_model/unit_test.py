# Unit test: run with pytest
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from gru import GRU_MODEL as gru_model
from skill import Skill
from skill_model.process_data import split_train_and_test_data, convert_token_to_matrix, extract_content_map
from skill_model.train import train
import pdb



def test_train_and_evaluate():
    exercise_filename = 'data/fake_tokens'
    content_index_filename = 'data/exercise_index_all'
    rnn_model_filename = 'output/model_unit50layer1bsize20thresh02_MaxGrowthObjective'
    train_keys, val_keys, full_data = split_train_and_test_data(
        exercise_filename, content_index_filename, 0)
    exercise_to_index_map, content_dim = extract_content_map(
        content_index_filename)
    input_dim = content_dim*2
    rnn_model = gru_model(input_dim=input_dim,
                      output_dim=content_dim,
                      nb_lstm_layers=1,
                      nb_lstm_units=50,
                      batch_size=1)
    skill_model = Skill(content_dim)

    rnn_model.load_state_dict(torch.load( rnn_model_filename ))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    data_index = torch.IntTensor(range(len(train_keys)))
    torch_data_index = Data.TensorDataset(data_index)
    loader = Data.DataLoader(dataset=torch_data_index,
                                       batch_size=1,
                                       drop_last=True)
    loss = train(skill_model, rnn_model, optimizer, full_data, loader, train_keys, epoch = 1,
          content_dim = content_dim)
    # eval_loss, total_predicted, total_label, total_correct, \
    #   total_sessions = evaluate_loss(model, full_data,
    #       loader, train_keys, content_dim)
    epoch_result = 'Epoch %d: %d loss' % (1, loss)
    print(epoch_result)
    print(skill_model.skill_difficulty)
    assert model, "UH OH"
    print("PASS UNIT TEST")
