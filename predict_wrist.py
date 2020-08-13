import torch
import yaml
import utils as util
import RNN_LSTM as LSTM
import numpy as np
import time
from MKF import MKF

class WristPredictor():
    def __init__(self, param_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with open(param_path) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        self.pred_step = data['prediction_step']
        self.sequence_length = data['wrist_predictor']['input_seq_len']
        self.input_size = data['wrist_predictor']['input_size']
        self.hidden_size = data['wrist_predictor']['hidden_size']
        self.num_layers = data['wrist_predictor']['num_layers']
        self.output_size = data['wrist_predictor']['output_size']
        self.output_step = 1

        # Load model
        model_path = data['wrist_predictor']['model_path']
        self.model = LSTM.RNN(self.input_size, self.sequence_length, self.hidden_size, self.num_layers, self.output_step, self.output_size, self.device).to(self.device)
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Load adaptor
        self.enable_adapt = data['wrist_predictor']['adapt']
        self.K = np.transpose(np.asarray(self.model.state_dict()['linear.weight'].cpu())) # parameter matrix to be adapted
        self.pre_x = 0 # Previous x state
        lamda = data['wrist_predictor']['lambda'] # Forgetting factor
        EMA_v = data['wrist_predictor']['EMA_v']
        EMA_p = data['wrist_predictor']['EMA_p']
        self.MKF = MKF(self.K, np.identity(1), lamda, EMA_v, EMA_p)
        self.K_list = [] 
        self.jt_list = []

        # Multi-epoch Adaptation params
        self.enable_multiepoch = data['wrist_predictor']['enable_multi_epoch']
        self.ep1 = data['wrist_predictor']['multi_epoch_epsilon1']
        self.ep2 = data['wrist_predictor']['multi_epoch_epsilon2']
        
    def predict(self, input_seq, pred_y, adapt):
        """
        This function outputs the prediction given the input sequence.
        When adapt is enabled, pred_y is the prediction of the current step from last step.
        """
        inputs = input_seq[:][:]
        cur_predict = []
        for i in range(self.pred_step):
            seq = torch.FloatTensor(inputs[-self.sequence_length:]).view(1, self.sequence_length, self.input_size).to(self.device)
            if(i == 0 and adapt):
                jt = np.linalg.norm(np.asarray(input_seq[-1][:]).reshape(3, 1) - np.asarray(pred_y).reshape(3, 1))
                self.jt_list.append(jt)
                if(self.enable_adapt):
                    kt = 1
                    if(self.enable_multiepoch):
                        if(jt > self.ep1 and jt <= self.ep2):
                            kt = 2
                        elif(jt > self.ep2):
                            kt = 0
                    for idx in range(kt):
                        err = (np.asarray(input_seq[-1][:]).reshape((3, 1)) - np.asarray(pred_y).reshape((3, 1)))
                        self.K = self.MKF.adapt(self.pre_x, err)
                        pred_y = np.matmul(np.transpose(self.K), self.pre_x)
                    self.model.state_dict()['linear.weight'][:, :] = torch.FloatTensor(np.transpose(self.K[:, :])).to(self.device)
                    self.K_list.append(np.reshape(self.K, (1, self.K.shape[0]*self.K.shape[1])))
                    
            with torch.no_grad():
                predict = self.model(seq)
                step_predict = []
                for j in range(self.output_size):
                    step_predict.append(predict[0, j].item())
                cur_predict.append(step_predict)
                if(i == 0):
                    self.pre_x = np.asarray(self.model.x_pre)
            inputs.append(step_predict)
        return cur_predict

if __name__ == "__main__":
    test_data_set = [0, 3, 1]
    test_data = util.load_data(test_data_set)[:, :3]
    scaler = util.create_scaler(-1, 1)
    scale_data = util.load_scale_data()
    scaler = util.fit_data(scaler, scale_data)
    test_data_normalized = util.scale_data([test_data[:, :3]], scaler)[0]
    wristPredictor = WristPredictor('params.yaml')

    # Evaluate
    test_inputs = test_data_normalized[:wristPredictor.sequence_length, :].tolist()
    horizon = len(test_data_normalized)
    test_output = []
    for i in range(horizon):
        test_inputs.append(test_data_normalized[i, :].tolist())
        if(i == 0):
            wrist_prediction = wristPredictor.predict(test_inputs, 0, 0)
        else:
            wrist_prediction = wristPredictor.predict(test_inputs, wrist_prediction[0], 1)
        test_output.append(wrist_prediction)      
    actual_predictions = util.inverse_scale_data(test_output, scaler)
    actual_predictions = np.asarray(actual_predictions).reshape(horizon, wristPredictor.pred_step, wristPredictor.output_size)

    # Visualize Results
    util.visualize_wrist_predictions(test_data, actual_predictions, wristPredictor.pred_step, wristPredictor.sequence_length)
    if(wristPredictor.enable_adapt):
        np.savetxt('k_list_wrist.txt', np.reshape(wristPredictor.K_list, (horizon-1, 128*3)))
    np.savetxt('jt_list.txt', np.reshape(wristPredictor.jt_list, (horizon-1, 1)))
    np.savetxt('prediction.txt', np.reshape(actual_predictions, (horizon*wristPredictor.pred_step, 3)))
    util.show_visualizations()
