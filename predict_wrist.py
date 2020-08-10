import torch
import yaml
import utils as util
import RNN_LSTM as LSTM
import numpy as np
from RLS import RLS

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
        self.output_step = 1
        self.output_size = data['wrist_predictor']['output_size']

        # Load model
        model_path = data['wrist_predictor']['model_path']
        self.model = LSTM.RNN(self.input_size, self.sequence_length, self.hidden_size, self.num_layers, self.output_step, self.output_size, self.device).to(self.device)
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # RLS
        self.enable_adapt = data['wrist_predictor']['adapt']
        lamda = data['wrist_predictor']['lambda'] # Forgetting factor
        self.K = np.transpose(np.asarray(self.model.state_dict()['linear.weight'].cpu())) # parameter matrix to be adapted
        self.RLS = RLS(self.K, 100000*np.identity(self.hidden_size), lamda) # RLS adaptor
        self.K_list = [] 
        self.pre_x = 0 # Previous x state

    def predict(self, input_seq, pred_y, adapt):
        """
        This function outputs the prediction given the input sequence.
        When adapt is enabled, pred_y is the prediction of the current step from last step.
        """
        inputs = input_seq[:][:]
        cur_predict = []
        for i in range(self.pred_step):
            seq = torch.FloatTensor(inputs[-self.sequence_length:]).view(1, self.sequence_length, self.input_size).to(self.device)
            if(i==0 and self.enable_adapt and adapt):
                err = (np.asarray(input_seq[-1][:]) - np.asarray(pred_y)).reshape((3, 1))
                self.K = self.RLS.adapt(self.pre_x, err)
                self.K_list.append(np.reshape(self.K, (1, self.K.shape[0]*self.K.shape[1])))
                self.model.state_dict()['linear.weight'][:, :] = torch.FloatTensor(np.transpose(self.K[:, :])).to(self.device)
            with torch.no_grad():
                predict = self.model(seq)
                step_predict = []
                for j in range(self.output_size):
                    step_predict.append(predict[0, j].item())
                cur_predict.append(step_predict)
                if(i==0):
                    self.pre_x = np.asarray(self.model.x_pre)
            inputs.append(step_predict)
        return cur_predict

if __name__ == "__main__":
    test_data_set = [0, 0, 1]
    input_model_path = 'trained_model/model_norm_1layer_left/model56.tar'
    test_data = util.load_data(test_data_set)[:, :3]
    scaler = util.create_scaler(-5, 5)
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
    util.show_visualizations()
