import numpy as np
import time
from predict_arm import ArmPredictor
from predict_wrist import WristPredictor
import utils as util
import copy

param_path = 'params.yaml'
test_data = [0, 0, 0] # [human_id, task_id, trial]

# Load data
test_data = util.load_data(test_data) # data: [lw, rw, le, re, ls, rs] nx18
scale_data = util.load_scale_data()
scaler = util.create_scaler(-1, 1)
scaler = util.fit_data(scaler, scale_data)
wrist_data = test_data[:, 0:3]
elbow_data = test_data[:, 6:9]
shoulder_data = test_data[:, 12:15]
wrist_data_normalized = util.scale_data([wrist_data], scaler)[0]
saveData = []

# Create wrist and arm prediction modules
wristPredictor = WristPredictor(param_path)
armPredictor = ArmPredictor(param_path)
armPredictor.set_arm_len(shoulder_data[0, :3], elbow_data[0, :3], wrist_data[0, :3])
shoulder_start = shoulder_data[0, :]
H = np.matrix([[1, 0, 0, shoulder_start[0]],
               [0, 1, 0, shoulder_start[1]],
               [0, 0, 1, shoulder_start[2]],
               [0, 0, 0, 1]]) # Transformation from camera frame to shoulder frame

test_inputs = wrist_data_normalized[:wristPredictor.sequence_length, :].tolist()
horizon = len(wrist_data) - wristPredictor.pred_step
test_output = []

print("Time horizon: ", horizon)
for i in range(horizon):
    t1 = time.time()
    test_inputs.append(wrist_data_normalized[i, :].tolist())
    inputs = copy.deepcopy(test_inputs)
    cur_th = armPredictor.IK(np.reshape(wrist_data[i, :3], (3, 1)), np.reshape(elbow_data[i, :3], (3, 1)), H)
    for j in range(wristPredictor.pred_step):
        if(j == 0 and i != 0):
            # update
            wrist_prediction = wristPredictor.predict(inputs, pre_wrist_pred, 1)
            wrist_pred = util.inverse_scale_data(wrist_prediction, scaler) # Un-normalize wrist prediction
            elbow_prediction, cur_th = armPredictor.predict_arm(np.asarray(wrist_pred[0]), cur_th, H, 1) # First time. No adaptation
        else:
            wrist_prediction = wristPredictor.predict(inputs, 0, 0) 
            wrist_pred = util.inverse_scale_data(wrist_prediction, scaler)
            elbow_prediction, cur_th = armPredictor.predict_arm(np.asarray(wrist_pred[0]), cur_th, H, 0)
        if(j == 0):
            pre_wrist_pred = wrist_prediction[0]
        inputs.append(wrist_prediction[0])
        step_predict = [wrist_pred[0][0], wrist_pred[0][1], wrist_pred[0][2], 
                        elbow_prediction[0, 0], elbow_prediction[0, 1], elbow_prediction[0, 2],
                        shoulder_start[0], shoulder_start[1], shoulder_start[2]]
        test_output.append(step_predict)
    print(i, time.time()-t1)
actual_predictions = np.asarray(test_output).reshape(horizon, wristPredictor.pred_step, 9)

# Visualize Results
util.visualize_arm_predictions(test_data, actual_predictions, wristPredictor.pred_step, wristPredictor.sequence_length)
if(wristPredictor.enable_adapt):
    np.savetxt('k_list_wrist.txt', np.reshape(wristPredictor.K_list, (len(wristPredictor.K_list), wristPredictor.hidden_size*wristPredictor.output_size)))
if(armPredictor.enable_adapt):
    np.savetxt('k_list_elbow.txt', np.reshape(armPredictor.K_list, (len(armPredictor.K_list), 25)))
np.savetxt('prediction.txt', np.reshape(actual_predictions, (horizon*wristPredictor.pred_step, 9)))
np.savetxt('wrist_adapt_err.txt', np.reshape(np.asarray(wristPredictor.jt_list), (len(wristPredictor.jt_list), 3)))
np.savetxt('elbow_adapt_err.txt', np.reshape(np.asarray(armPredictor.err_th_list), (len(armPredictor.err_th_list), 5)))
armPredictor.shutdown_matlab()
util.show_visualizations()