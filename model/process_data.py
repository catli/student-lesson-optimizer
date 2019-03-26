import numpy as np
import json
from sklearn.model_selection import train_test_split
import torch.nn.utils as utils
import pdb


def convert_token_to_matrix(batch_index, json_data, json_keys, content_num):
    '''
        convert the token to a multi-hot vector
        from student session activity json_data
        convert this data in batch form
    '''
    # [TODO SLO]:
    #     [Optional] change input so that each
    #        skill worked on is fed in sequential order

    # number of students in the batch
    num_sess = []
    # max number of sessions in batch
    for student_index in batch_index:
        # return the key pairs (student_id, seq_len)
        # and the first item of pair as student id
        student_key = json_keys[student_index][0]
        num_sess.append(len(json_data[student_key].keys())-1)
    max_seq = np.max(num_sess) + 1
    seq_lens = num_sess

    input_padded, label_padded, label_mask = create_padded_matrix_with_correct(
        batch_index, json_data, json_keys, content_num, max_seq)

    # assign the number of sessions as sequence length for each student
    # this will feed be used later to tell the model
    # which sessions are padded
    return input_padded, label_padded, label_mask, seq_lens




def create_padded_matrix_with_correct(batch_index, json_data, json_keys,
                                      content_num, max_seq):
    '''
        input:
            json_data: {student_key: session_id: [(skill_id, is_correct}]}
            batch_index = the student id in the batch
        output:
            but input and label will translate row for each session
            and column for each skill, populating with perc correct
            for each data. A problem not worked on is set to 0
            [[ 0 0.2 0 0.85 ]]
        steps:
            create an empty matrix for the padded input /output
            populated with the count/binomial state, concatenated
                with the percent correct
        output vectors populated with the binomial state
    '''
    batchsize = len(batch_index)
    # placeholder for padded input and label
    input_padded = np.zeros((batchsize, int(max_seq), content_num), int)
    correct_padded = np.zeros((batchsize, int(max_seq), content_num), int)
    label_mask_padded = np.zeros((batchsize, int(max_seq), content_num), int)
    # populate student_padded
    for stud_num, student_index in enumerate(batch_index):
        # return the key pairs (student_id, seq_len)
        # and the first item of pair as student id
        student_key = json_keys[student_index][0]
        sessions = sorted(json_data[student_key].keys())
        for sess_num, session in enumerate(sessions):
            # sessions data, with tuples of student activity
            #    content_items = (exercise_id , is_correct)
            content_items = json_data[student_key][session]
            for item_num, item in enumerate(content_items):
                exercise_id = item[0]
                is_correct = item[1]
                label_mask_padded[stud_num, sess_num, exercise_id-1] = 1
                input_padded[stud_num, sess_num, exercise_id-1]+= 1
                correct_padded[stud_num, sess_num, exercise_id-1]+= is_correct
    concat_input_padded, perc_correct_padded = concat_perc_correct(
        correct_padded, input_padded)
    # take first n-1 sessions for input and last n-1 sessions for output
    concat_input_padded = concat_input_padded[:, :-1]
    # generate the labels and mask by averaging over multiple sessions
    # default set to no averaging (num_next = 1)
    label_padded, label_mask = max_next_sessions(perc_correct_padded, label_mask_padded,
        num_next = 1)
    return concat_input_padded, label_padded, label_mask


def concat_perc_correct(correct_padded, input_padded):
    '''
        calculate the perc correct for activtiies worked on
        and then concatenate with input matrix
    '''
    # create denominator
    correct_denom = input_padded.copy()
    # set 0 to 1 for divisbility
    correct_denom[correct_denom == 0] = 1
    # divide correct by denom
    perc_correct_padded = correct_padded/correct_denom
    # concatenate the input and ocrrect
    concat_input_padded = np.concatenate((input_padded, perc_correct_padded),
        axis=2)
    return concat_input_padded, perc_correct_padded



def max_next_sessions(perc_correct_padded, label_padded, num_next):
    '''
        For the next x sessions, create a new
        output that returns the max of _num_next_ sessions.
        Only works if dim(perc_correct_padded) = dim(label_padded)
    '''
    perc_correct_padded = perc_correct_padded[:, 1:]
    label_padded = label_padded[:, 1:]
    next_label_mask = label_padded.copy()
    next_correct_padded = perc_correct_padded.copy()
    for b, _ in enumerate(label_padded):
        for i, _ in enumerate(label_padded[b]):
            next_label_mask[b, i, :] = np.max(
                label_padded[b, i:(i+num_next), :], axis=0)
            next_correct_padded[b, i, :] = np.max(
                perc_correct_padded[b, i:(i+num_next), :], axis=0)
    return next_correct_padded, next_label_mask


def extract_content_map(content_index_filename):
    '''

    '''
    index_reader = open(content_index_filename, 'r')
    exercise_to_index_map = json.load(index_reader)
    content_num = len(exercise_to_index_map.keys())
    return exercise_to_index_map, content_num


def split_train_and_test_data(exercise_filename, content_index_filename,
                              test_perc = 0 ):
    '''
        split the data into training and test by learners
        input: exercise file with json data
            {'anon_student_id': session_1: [(skill_key, %_correct),
                (skill_key, %_correct)], session_2:
                [(skill_key, %_correct), (skill_key, %_correct)]}
    '''
    exercise_reader = open(exercise_filename, 'r')
    full_data = json.load(exercise_reader)
    train_data = {}
    val_data = {}
    ordered_train_keys, ordered_val_keys = split_train_and_test_ids(
        json_data=full_data,
        test_perc=test_perc)
    # to expose the json file
    index_reader = open(content_index_filename, 'r')
    exercise_to_index_map = json.load(index_reader)
    return ordered_train_keys, ordered_val_keys, full_data 


def split_train_and_test_ids(json_data, test_perc):
    '''
        split anon ids into test_perc % in test dataset
        and 1-test_perc % in training dataset
    '''
    student_ids = [student for student in json_data]
    train_ids, val_ids = train_test_split(student_ids,
                                          test_size=test_perc)
    ordered_train_keys = create_ordered_sequence_list(train_ids, json_data)
    ordered_val_keys = create_ordered_sequence_list(val_ids, json_data)
    return ordered_train_keys, ordered_val_keys


def create_ordered_sequence_list(set_ids, exercise_json):
    '''
        create ordered sequence length
        will be used for batching to efficiently
        run through the sequence ids
    '''
    key_seq_pair = create_key_seqlen_pair(set_ids, exercise_json)
    key_seq_pair.sort(key=lambda x: x[1], reverse=True)
    return key_seq_pair


def create_key_seqlen_pair(set_ids, json_data):
    '''
        create a tuple with learner id and the sequence length,
        i.e. number of sessions per learner
            ('$fd@w', 4)
    '''
    key_seq_pair = []
    for id in set_ids:
        seq_len = len(json_data[id])
        key_seq_pair.append((id, seq_len))
    return key_seq_pair



