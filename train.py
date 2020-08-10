import torch
import torch.nn as nn
import time
import utils as util
import numpy as np
import RNN_LSTM as LSTM
import yaml

train_data_set = [[0, 0, 1], [0, 1, 1], [0, 2, 0]]
test_data_set = [[0, 0, 1], [0, 1, 1], [0, 2, 1]]

with open('params.yaml') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
train = data['Train_wrist_predictor']['train'] # True: train model before evaluating. False: evaluate model
prediction_step = data['prediction_step']
sequence_length = data['wrist_predictor']['input_seq_len']
input_size = data['wrist_predictor']['input_size']
hidden_size = data['wrist_predictor']['hidden_size']
num_layers = data['wrist_predictor']['num_layers']
output_size = data['wrist_predictor']['output_size']
output_step = 1
num_epochs = data['Train_wrist_predictor']['num_epochs']
learning_rate = data['Train_wrist_predictor']['learning_rate']

save_model = data['Train_wrist_predictor']['save_model'] # If true, save trained model every 20 epoch
output_folder = data['Train_wrist_predictor']['output_folder']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_fname = 'trained_model/model_norm_1layer_left/model56.tar' # Specify model filename if save_model false

# Scale data
scaler = util.create_scaler(-5, 5)
scale_data = util.load_scale_data()
scaler = util.fit_data(scaler, scale_data)

# Load data
train_data = util.load_data_set(train_data_set)
test_data = util.load_data_set(test_data_set)

#scaler.data_min_, scaler.data_max_ = util.fit_scaler()
train_data_normalized = util.scale_data(train_data, scaler)
train_inout_seq = util.create_inout_sequences(train_data_normalized, sequence_length, input_size, output_step)
test_data_normalized = util.scale_data(test_data, scaler)

model = LSTM.RNN(input_size, sequence_length, hidden_size, num_layers, output_step, output_size, device).to(device)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if(train):
    print('Start training!')
    model.train()
    min_loss = 1000
    time_start = time.time()
    for i in range(num_epochs):
        ti = time.time()
        for inout_seq in train_inout_seq:
            for seq, labels in inout_seq:
                seq_tensor = torch.Tensor(seq)
                labels_tensor = torch.Tensor(labels)

                seq_tensor = seq_tensor.view(1, sequence_length, input_size).to(device)
                labels_tensor = labels_tensor.to(device)
                y_pred = model(seq_tensor)
                
                loss = loss_function(y_pred, labels_tensor)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        print("epocho "+str(i)+f' {time.time()-ti:1.3f}s', f'loss: {loss.item():10.10f}')
        cur_loss = loss.item()
        if(cur_loss < min_loss and save_model):
            torch.save({'epoch': i,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                        }, output_folder+'model'+str(i)+'.tar')
            min_loss = cur_loss
    print(f'epoch: {i:1} loss: {loss.item():10.10f}')
    if(save_model):
        torch.save({'epoch': i,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }, output_folder+'model'+str(i)+'.tar')
    print(f'Training time: {time.time()-time_start:1.3f}')
else:
    model = LSTM.RNN(input_size, sequence_length, hidden_size, num_layers, output_step, output_size, device).to(device)
    loss_function = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    checkpoint = torch.load(model_fname)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

# Evaluate
print('Start Evaluating!')
model.eval()

# Evaluate for each traj profile
for test_idx in range(len(test_data_set)):
    test_inputs = test_data_normalized[test_idx][:sequence_length, :].tolist()
    test_data_size = len(test_data_normalized[test_idx])
    test_output = []
    for i in range(test_data_size):
        test_inputs.append(test_data_normalized[test_idx][i, :].tolist())
        tmp_inputs = test_inputs[:][:]
        cur_predict = []
        for step in range(prediction_step):
            seq = torch.FloatTensor(tmp_inputs[-sequence_length:]).view(1, sequence_length, input_size).to(device)
            with torch.no_grad():
                predict = model(seq)
                for l in range(output_step):
                    step_predict = []
                    for j in range(output_size):
                        step_predict.append(predict[l, j].item())
                    cur_predict.append(step_predict)
            tmp_inputs.append(step_predict)
        test_output.append(cur_predict)      
    actual_predictions = util.inverse_scale_data(test_output, scaler)
    actual_predictions = np.asarray(actual_predictions).reshape(test_data_size, prediction_step, output_size)

    # Visualize Results
    util.visualize_wrist_predictions(test_data[test_idx], actual_predictions, prediction_step, sequence_length)
util.show_visualizations()