# Unit test: run with pytest
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from gru import GRU_MODEL as gru_model
from process_data import split_train_and_test_data, convert_token_to_matrix, extract_content_map
from train import train
from evaluate import evaluate_loss
import pdb


def test_train_and_evaluate():
    exercise_filename = 'data/fake_tokens'
    content_index_filename = 'data/exercise_index_all'
    train_keys, val_keys, full_data = split_train_and_test_data(
        exercise_filename, content_index_filename, 0)
    exercise_to_index_map, content_dim = extract_content_map(
        content_index_filename)
    input_dim = content_dim*2
    model = gru_model(input_dim=input_dim,
                      output_dim=content_dim,
                      nb_lstm_layers=1,
                      nb_lstm_units=50,
                      batch_size=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    data_index = torch.IntTensor(range(len(train_keys)))
    torch_data_index = Data.TensorDataset(data_index)
    loader = Data.DataLoader(dataset=torch_data_index,
                                       batch_size=1,
                                       drop_last=True)
    train(model, optimizer, full_data, loader, train_keys, epoch = 1,
          content_dim = content_dim)
    eval_loss, total_predicted, total_label, total_correct, \
      total_sessions = evaluate_loss(model, full_data,
          loader, train_keys, content_dim)
    epoch_result = 'Epoch %d unit test: %d / %d  precision \
                    and %d / %d  recall with %d sessions  \n' % (
            1, total_correct, total_predicted,
            total_correct, total_label, total_sessions)
    print(epoch_result)
    assert model, "UH OH"
    print("PASS UNIT TEST")


def test_train_split():
    # make sure the training validiton split
    # is working as expected
    exercise_filename = 'data/fake_tokens'
    content_index_filename = 'data/exercise_index_all'
    train_keys, val_keys, full_data = split_train_and_test_data(
        exercise_filename, content_index_filename, 0.2)
    assert len(train_keys) == 4
    assert len(val_keys) == 1
    print("PASS TRAIN VALIDATION SPLIT")


def test_convert_token_to_matrix():
    # test a couple of things
    # (1) number of student in batch match expected
    # (2) the number of session for the first student match expected
    # (2) the number of activities for the first session match expected
    # (3) the perc correct match expected
    exercise_filename = 'data/fake_tokens'
    content_index_filename = 'data/exercise_index_all'
    train_keys, val_keys, full_data = split_train_and_test_data(
        exercise_filename, content_index_filename, 0)
    exercise_to_index_map, content_dim = extract_content_map(
        content_index_filename)

    batch = np.array([0,1])
    batch_train_keys = [key[0] for key in train_keys[0:2]]

    input_padded, label_padded, label_mask, seq_lens = convert_token_to_matrix(
            batch, full_data, train_keys, content_dim)

    assert len(input_padded) == len(batch)
    assert len(label_padded) == len(batch)
    assert len(label_mask) == len(batch)
    print("PASS BATCH NUM")

    student_data = full_data[batch_train_keys[0]]
    assert len(input_padded[0,:,:]) == len(student_data)-1
    assert len(label_padded[0,:,:]) == len(student_data)-1
    assert len(label_mask[0,:,:]) == len(student_data)-1
    print("PASS SESS NUM")

    student_data = full_data[batch_train_keys[0]]
    first_sesh_skills = np.unique([x[0] for x in student_data['1']])
    sec_sesh_skills = np.unique([x[0] for x in student_data['2']])
    assert np.sum(input_padded[0,0,:content_dim]>0) == len(first_sesh_skills)
    assert np.sum(label_padded[0,0,:]>0) == len(sec_sesh_skills)
    assert np.sum(label_mask[0,0,:]) == len(sec_sesh_skills)
    print("PASS SKILL NUM")



# if __name__ == '__main__':
#     # set hyper parameters
#     test_train_split()
#     test_convert_token_to_matrix()
#     test_train_and_evaluate()
