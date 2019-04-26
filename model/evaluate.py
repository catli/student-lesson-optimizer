
'''
   To evaluate the loss of prediction on validation data
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data as Data
import numpy as np
from process_data import split_train_and_test_data, convert_token_to_matrix
import random
import csv
import pdb



def evaluate_loss(model, val_data, loader, val_keys, content_dim, threshold):
    '''
      # output_sample_filename, epoch, exercise_to_index_map, 
      # perc_sample_print, ):
        set in training node
        perc_sample_print = 0.05 # set the percent sample
    '''
    model.eval()
    val_loss = []
    total_predicted = 0
    total_label = 0
    total_correct = 0
    total_no_predicted = 0
    total_sessions = 0
    for step, batch_x in enumerate(loader):  # batch_x: index of batch data
        print('Evaluate Loss | Iteration: ', step+1)
        # convert token data to matrix
        # need to convert batch_x from tensor flow object to numpy array
        # before converting to matrix

        input_padded, label_padded, label_mask, seq_lens = convert_token_to_matrix(
            batch_x[0].numpy(), val_data, val_keys, content_dim)
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
        model.hidden = model.init_hidden()
        # is this equivalent to generating prediction
        # what is the label generated?

        y_pred = model(padded_input, seq_lens)  # .cuda()
        loss = model.loss(y_pred, padded_label, padded_mask)  # .cuda()
        # append the loss after converting back to numpy object from tensor
        val_loss.append(loss.data.numpy())
        threshold_output, correct_ones = find_max_predictions(
            y_pred, padded_mask, input_padded, content_dim, threshold)  # .cuda()
        total_predicted += len(torch.nonzero(threshold_output))
        total_label += len(torch.nonzero(padded_label))
        total_correct += len(torch.nonzero(correct_ones))
        # total_no_predicted += num_no_pred
        total_sessions += np.sum(seq_lens)

    average_loss = np.mean(val_loss)
    # of label=1 that were predicted accurately
    return average_loss, total_predicted, total_label, \
        total_correct, total_sessions


def mask_padded_errors(threshold_output, seq_lens):
    # [TODO SLO]: 
    #     (1) move mask_padded_error to loss
    num_no_pred = 0
    for i, output in enumerate(threshold_output):
        # the full size of threshold
        num_sess = threshold_output[i].shape[0]
        seq_len = seq_lens[i]
        # calculate the number of sessions with no prediction
        sess_with_pred = np.sum(
            threshold_output[i][:seq_len, ].detach().numpy(),
            axis=1)
        num_no_pred += int(np.sum(sess_with_pred == 0))
        threshold_output[i][:seq_len, ]
        for sess_i in range(seq_len, num_sess):
            threshold_output[i][sess_i] = 0
    return threshold_output, num_no_pred




def find_max_predictions(output, label, input_padded, content_dim, threshold):
    '''
        for each session, select k matching content where k is
        the number of content the student actual completed
        the k matching content is based on predict score growth

        once generated, compare the predicted and the actual content
        and return location of correct predictions
    '''
    rel_thresh_output = torch.zeros(output.shape)
    for stud, _ in enumerate(output):
        # init total correct and total answer, sum up from input_padded
        num_corrects = np.zeros(content_dim)
        num_answers = np.zeros(content_dim)
        for sess, _ in enumerate(output[stud]):
            # add the number of correct answers and total answers
            # from the previous session (in input padded)
            num_answers += input_padded[stud, sess, :content_dim]
            # num correct  = num_answers * perc_correct
            num_corrects += (input_padded[stud, sess, :content_dim]*
                input_padded[stud, sess, content_dim:])
            # number of predicted activity will match actual number completed
            # assume that students will complete the same number of activities
            # in this prediction scenario
            k = torch.sum(label[stud, sess]>0) # number of content completed
            if k==0:
                continue
            else:
                # create the denominator from num answers
                denom = num_answers.copy()
                denom[denom==0] = 1
                mastery = np.divide(num_corrects, denom)
                growth_vals = output[stud, sess].detach().numpy() - mastery
                # cap threshold to target only those in proximal learning
                capped_growth_vals = growth_vals[growth_vals<=threshold]
                # pick the threshold for k-th highest growth threshold
                rel_thresh =  sorted(capped_growth_vals)[-k] # threshold of content
                # if the output greater growth threshold, set to 1
                # otherwise, all other skills set to 0
                rel_thresh_output[stud, sess] = torch.tensor((
                    (growth_vals >=rel_thresh)*(growth_vals<threshold)
                    ).astype('float'))
    # find the difference between label and prediction
    # where prediction is incorrect (label is one and
    # threshold output 0), then the difference would be 1
    predict_diff = label - rel_thresh_output
    # if label =1 and threshold output =0
    incorrect_ones = F.threshold(predict_diff, 0.999, 0)
    # if label = 1 and incorrect = 0
    correct_ones = label - incorrect_ones
    return rel_thresh_output, correct_ones

