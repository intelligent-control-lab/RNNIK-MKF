import numpy as np
import time
from predict_arm import ArmPredictor
from predict_wrist import WristPredictor
import utils as util

param_path = 'params.yaml'
test_data = [0, 0, 1] # [human_id, task_id, trial]

test_data = util.load_data(test_data) # data: [lw, rw, le, re, ls, rs] nx18
scaler = util.create_scaler(-5, 5)
scale_data = util.load_scale_data()
scaler = util.fit_data(scaler, scale_data)

wrist_data = test_data[:, 0:3]
elbow_data = test_data[:, 6:9]
shoulder_data = test_data[:, 12:15]
wrist_data_normalized = util.scale_data([wrist_data], scaler)[0]
saveData = []

wristPredictor = WristPredictor(param_path)
armPredictor = ArmPredictor(param_path)

# Set arm length
armPredictor.set_arm_len(shoulder_data[0, :3], elbow_data[0, :3], wrist_data[0, :3])

test_inputs = wrist_data_normalized[:wristPredictor.sequence_length, :].tolist()
horizon = len(wrist_data) - wristPredictor.pred_step
test_output = []
print("Time horizon: ", horizon)
for i in range(horizon):
    t1 = time.time()
    test_inputs.append(wrist_data_normalized[i, :].tolist())
    if(i == 0):
        wrist_prediction = wristPredictor.predict(test_inputs, 0, 0) # First time. No adaptation
    else:
        wrist_prediction = wristPredictor.predict(test_inputs, wrist_prediction[0], 1)

    shoulder_start = shoulder_data[i, :]
    H = np.matrix([[1, 0, 0, shoulder_start[0]],
                   [0, 1, 0, shoulder_start[1]],
                   [0, 0, 1, shoulder_start[2]],
                   [0, 0, 0, 1]]) # Transformation from camera frame to shoulder frame

    cur_th = armPredictor.IK(np.reshape(wrist_data[i, :3], (3, 1)), np.reshape(elbow_data[i, :3], (3, 1)), H)
    wrist_pred = util.inverse_scale_data(wrist_prediction, scaler) # Un-normalize wrist prediction
    #wrist_pred = wrist_data[i+1:i+1+prediction_step, :]
    if(i==0):
        elbow_prediction = armPredictor.predict_arm(np.asarray(wrist_pred), cur_th, H, 0) # First time. No adaptation
    else:
        elbow_prediction = armPredictor.predict_arm(np.asarray(wrist_pred), cur_th, H, 1)
    
    # Record prediction
    for idx in range(wristPredictor.pred_step):
        step_predict = [wrist_pred[idx][0], wrist_pred[idx][1], wrist_pred[idx][2], 
                        elbow_prediction[idx, 0], elbow_prediction[idx, 1], elbow_prediction[idx, 2],
                        shoulder_start[0], shoulder_start[1], shoulder_start[2]]
        test_output.append(step_predict)
    print(i, time.time()-t1)
actual_predictions = np.asarray(test_output).reshape(horizon, wristPredictor.pred_step, 9)

# Visualize Results
util.visualize_arm_predictions(test_data, actual_predictions, wristPredictor.pred_step, wristPredictor.sequence_length)
if(wristPredictor.enable_adapt):
    np.savetxt('k_list_wrist.txt', np.reshape(wristPredictor.K_list, (horizon-1, wristPredictor.hidden_size*wristPredictor.output_size)))
if(armPredictor.enable_adapt):
    np.savetxt('k_list_elbow.txt', np.reshape(armPredictor.K_list, (horizon-1, 25)))
np.savetxt('prediction.txt', np.reshape(actual_predictions, (horizon*wristPredictor.pred_step, 9)))
armPredictor.shutdown_matlab()
util.show_visualizations()