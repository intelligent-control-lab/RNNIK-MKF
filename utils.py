import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def create_inout_sequences(input_data, seq_len, input_size, output_step):
    """
    Creates input sequence for training. [(sequence, label)]
    """
    inout_seq = []
    for data in input_data:
        data_seq = []
        L = len(data)
        for i in range(L-seq_len-output_step+1):
            train_seq = data[i:i+seq_len, :input_size]
            train_label = data[i+seq_len:i+seq_len+output_step, :input_size]
            data_seq.append((train_seq ,train_label))
        inout_seq.append(data_seq)
    return inout_seq

def load_scale_data():
    """
    Load and combine all data sets.
    """
    data_all = np.zeros((1, 3))
    num_human = 1
    num_task = 5
    num_trials = 2
    for human_id in range(num_human):
        for task_id in range(num_task):
            for trial in range(num_trials):
                data = np.loadtxt('./data/subject'+str(human_id)+'_'+str(task_id)+'_'+str(trial)+'_instruct/data_avg.txt')
                data = data[:, :3]
                data_all = np.concatenate((data_all, data), axis=0)
    data_all = data_all[1:, :]
    return data_all

def load_data_set(data_set):
    """ 
    Load data specified by a set
    """
    data_all = []
    for data_fname in data_set:
        human_id = data_fname[0]
        task_id = data_fname[1]
        trial = data_fname[2]

        data = np.loadtxt('data/subject'+str(human_id)+'_'+str(task_id)+'_'+str(trial)+'_instruct/data_avg.txt')
        data = data[:, :3]
        data_all.append(data)
    return data_all

def load_data(data_fname):
    """
    Load a single data file
    """
    human_id = data_fname[0]
    task_id = data_fname[1]
    trial = data_fname[2]
    data = np.loadtxt('data/subject'+str(human_id)+'_'+str(task_id)+'_'+str(trial)+'_instruct/data_avg.txt')
    return data

def create_scaler(min_, max_):
    return MinMaxScaler(feature_range=(min_, max_))

def fit_data(scaler_in, data):
    return scaler_in.fit(data)

def scale_data(data_all, scaler):
    data_normalized_all = []
    for data in data_all:
        data_normalized = scaler.transform(data.reshape(-1, 3))
        data_normalized = torch.FloatTensor(data_normalized).view(-1, 3)
        data_normalized_all.append(data_normalized)
    return data_normalized_all

def inverse_scale_data(data, scaler):
    data_inverse = scaler.inverse_transform(np.array(data).reshape(-1, 3))
    return data_inverse

def calculate_rmse(test_data, actual_predictions):
    count = 0
    error_sum_l = 0
    error_sum_r = 0
    data_len = len(test_data)
    pred_step = actual_predictions.shape[1]

    for i in range(data_len):
        for j in range(pred_step):
            pred_l = actual_predictions[i, j, :3]
            pred_r = actual_predictions[i, j, 3:]
            try:
                ground_truth_l = test_data[i+j+1, :3]
                ground_truth_r = test_data[i+j+1, 3:]
            except Exception as e:
                continue
            #print(pred, ground_truth)
            error_l = rmse(pred_l, ground_truth_l)
            error_r = rmse(pred_r, ground_truth_r)
            #print(error)
            error_sum_l += error_l
            error_sum_r += error_r
            count += 1

    return error_sum_l/count, error_sum_r/count

def rmse(a, b):
    err = np.sqrt(((a - b) ** 2).mean())
    return err

def visualize_wrist_predictions(test_data, actual_predictions, predict_step, input_step):
    # Visualize Results
    fig, axs = plt.subplots(3) # visualize w_x, w_y, w_z
    fig.suptitle('Wrist Prediction Step: '+str(predict_step))

    axs[0].set(ylabel='Wrist X')
    axs[0].grid(True)
    axs[0].autoscale(axis='x', tight=True)
    axs[0].plot(test_data[:, 0], label='Test Data')
    if(predict_step <= 1):
        axs[0].plot(actual_predictions[:, :, 0], label='Predictions')
    else:
        for i in range(input_step, len(actual_predictions), predict_step+5):
            x = np.arange(i+1, i+1+predict_step, 1)
            if(i == input_step):
                axs[0].plot(x, actual_predictions[i, :, 0], label='Predictions')
            else:
                axs[0].plot(x, actual_predictions[i, :, 0])
    axs[0].legend(loc="upper right", shadow=True, fancybox=True)

    axs[1].set(ylabel='Wrist Y')
    axs[1].grid(True)
    axs[1].autoscale(axis='x', tight=True)
    axs[1].plot(test_data[:, 1])
    if(predict_step <= 1):
        axs[1].plot(actual_predictions[:, :, 1])
    else:
        for i in range(input_step, len(actual_predictions), predict_step+5):
            x = np.arange(i+1, i+1+predict_step, 1)
            axs[1].plot(x, actual_predictions[i, :, 1])

    axs[2].set(ylabel='Wrist Z')
    axs[2].grid(True)
    axs[2].autoscale(axis='x', tight=True)
    axs[2].plot(test_data[:, 2])
    if(predict_step <= 1):
        axs[2].plot(actual_predictions[:, :, 2])
    else:
        for i in range(input_step, len(actual_predictions), predict_step+5):
            x = np.arange(i+1, i+1+predict_step, 1)
            axs[2].plot(x, actual_predictions[i, :, 2])

def visualize_arm_predictions(test_data, actual_predictions, predict_step, input_step):
    # Visualize Results
    fig, axs = plt.subplots(6) # visualize wx, wy, wz, ex, ey, ez
    fig.suptitle('Full Arm Prediction Step: '+str(predict_step))

    axs[0].set(ylabel=' Wrist X')
    axs[0].grid(True)
    axs[0].autoscale(axis='x', tight=True)
    axs[0].plot(test_data[:, 0], label='Test Data')
    if(predict_step <= 1):
        axs[0].plot(actual_predictions[:, :, 0], label='Predictions')
    else:
        for i in range(input_step, len(actual_predictions), predict_step+5):
            x = np.arange(i+1, i+1+predict_step, 1)
            if(i == input_step):
                axs[0].plot(x, actual_predictions[i, :, 0], label='Predictions')
            else:
                axs[0].plot(x, actual_predictions[i, :, 0])
    axs[0].legend(loc="upper right", shadow=True, fancybox=True)

    axs[1].set(ylabel='Wrist Y')
    axs[1].grid(True)
    axs[1].autoscale(axis='x', tight=True)
    axs[1].plot(test_data[:, 1])
    if(predict_step <= 1):
        axs[1].plot(actual_predictions[:, :, 1])
    else:
        for i in range(input_step, len(actual_predictions), predict_step+5):
            x = np.arange(i+1, i+1+predict_step, 1)
            axs[1].plot(x, actual_predictions[i, :, 1])

    axs[2].set(ylabel='Wrist Z')
    axs[2].grid(True)
    axs[2].autoscale(axis='x', tight=True)
    axs[2].plot(test_data[:, 2])
    if(predict_step <= 1):
        axs[2].plot(actual_predictions[:, :, 2])
    else:
        for i in range(input_step, len(actual_predictions), predict_step+5):
            x = np.arange(i+1, i+1+predict_step, 1)
            axs[2].plot(x, actual_predictions[i, :, 2])

    axs[3].set(ylabel='Elbow X')
    axs[3].grid(True)
    axs[3].autoscale(axis='x', tight=True)
    axs[3].plot(test_data[:, 6])
    if(predict_step <= 1):
        axs[3].plot(actual_predictions[:, :, 3])
    else:
        for i in range(input_step, len(actual_predictions), predict_step+5):
            x = np.arange(i+1, i+1+predict_step, 1)
            axs[3].plot(x, actual_predictions[i, :, 3])

    axs[4].set(ylabel='Elbow Y')
    axs[4].grid(True)
    axs[4].autoscale(axis='x', tight=True)
    axs[4].plot(test_data[:, 7])
    if(predict_step <= 1):
        axs[4].plot(actual_predictions[:, :, 4])
    else:
        for i in range(input_step, len(actual_predictions), predict_step+5):
            x = np.arange(i+1, i+1+predict_step, 1)
            axs[4].plot(x, actual_predictions[i, :, 4])

    axs[5].set(xlabel='Time', ylabel='Elbow Z')
    axs[5].grid(True)
    axs[5].autoscale(axis='x', tight=True)
    axs[5].plot(test_data[:, 8])
    if(predict_step <= 1):
        axs[5].plot(actual_predictions[:, :, 5])
    else:
        for i in range(input_step, len(actual_predictions), predict_step+5):
            x = np.arange(i+1, i+1+predict_step, 1)
            axs[5].plot(x, actual_predictions[i, :, 5])

def show_visualizations():
    plt.show()