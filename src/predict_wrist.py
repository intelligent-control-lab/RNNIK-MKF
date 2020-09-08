import torch
import yaml
import RNN_LSTM as LSTM
import numpy as np
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
        self.output_step = 1 # N-to-1

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
        self.MKF = MKF(self.K, np.identity(self.hidden_size), lamda, EMA_v, EMA_p)
        self.K_list = [] 
        self.err_list = []

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
        
        seq = torch.FloatTensor(inputs[-self.sequence_length:]).view(1, self.sequence_length, self.input_size).to(self.device)
        if(adapt and not isinstance(self.pre_x, int)):
            err = (np.asarray(input_seq[-1][:]).reshape((3, 1)) - np.asarray(pred_y).reshape((3, 1)))
            self.err_list.append(err.reshape((1, 3)))

            jt = np.linalg.norm(np.asarray(input_seq[-1][:]).reshape(3, 1) - np.asarray(pred_y).reshape(3, 1))
            if(self.enable_adapt):
                kt = 1
                if(self.enable_multiepoch):
                    if(jt > self.ep1 and jt <= self.ep2):
                        kt = 2
                    elif(jt > self.ep2):
                        kt = 0
                while(kt > 0):
                    err = (np.asarray(input_seq[-1][:]).reshape((3, 1)) - np.asarray(pred_y).reshape((3, 1)))
                    self.K = self.MKF.adapt(self.pre_x, err)
                    pred_y = np.matmul(np.transpose(self.K), self.pre_x)
                    kt -= 1
                self.model.state_dict()['linear.weight'][:, :] = torch.FloatTensor(np.transpose(self.K[:, :])).to(self.device)
                self.K_list.append(np.reshape(self.K, (1, self.K.shape[0]*self.K.shape[1])))
                    
        with torch.no_grad():
            predict = self.model(seq)
            step_predict = []
            for j in range(self.output_size):
                step_predict.append(predict[0, j].item())
            cur_predict.append(step_predict)
            if(adapt):
                self.pre_x = np.asarray(self.model.x_pre)
        return cur_predict
