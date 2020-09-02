import numpy as np
import matlab.engine
from MKF import MKF
import yaml

class ArmPredictor():
    def __init__(self, param_path):
        self.matlab_eng = matlab.engine.start_matlab()
        self.matlab_eng.addpath(r'matlab_src/')
        self.matlab_eng.init_params(nargout=0)
        self.l1_len = 0 # Upper arm
        self.l2_len = 0 # Lower arm
        self.K = np.identity(5) # 5 DOF arm

        with open(param_path) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        lamda = data['arm_predictor']['lambda']
        self.enable_adapt = data['arm_predictor']['adapt']
        self.pred_step = data['prediction_step']
        EMA_v = data['arm_predictor']['EMA_v']
        EMA_p = data['arm_predictor']['EMA_p']
        self.MKF = MKF(self.K, np.identity(5), lamda, EMA_v, EMA_p, sig_r=0.005, sig_q=0.005)
        
        self.K_list = []
        self.pre_cur_pos = 0
        self.pre_cur_th = 0
        self.pre_J = 0
        self.err_th_list = []
    
    def predict_arm(self, wrist_prediction, cur_th, shoulder_trans, adapt):
        cur_th = np.reshape(cur_th, (5, 1)) # 5 DOF arm
        prediction = []

        # Transform position from camera frame to shoulder frame
        next_pos_world = wrist_prediction[:3]
        next_pos = self.trans_to_frame(next_pos_world, shoulder_trans)

        # Calculate Jacobian and position at current step
        J = self.matlab_eng.jacobian(float(cur_th[0]), float(cur_th[1]), float(cur_th[2]), float(cur_th[3]), float(cur_th[4]), 
                                     self.l1_len, self.l2_len, float(np.pi), nargout=1)
        fk = self.matlab_eng.FK(float(cur_th[0]), float(cur_th[1]), float(cur_th[2]), float(cur_th[3]), float(cur_th[4]), 
                                self.l1_len, self.l2_len, float(np.pi), nargout=1)
        cur_pos = np.asarray(fk).reshape((3, 1))

        if(adapt and self.enable_adapt and not isinstance(self.pre_J, int)):
            dw = cur_pos-self.pre_cur_pos
            xk = np.matmul(np.transpose(self.pre_J), dw)
            pred_cur_th = self.pre_cur_th+np.matmul(np.transpose(self.K), xk)
            err = cur_th-pred_cur_th
            self.err_th_list.append(err.reshape((1, 5)))
            self.K = self.MKF.adapt(xk, err)
            self.K_list.append(np.reshape(self.K, (1, self.K.shape[0]*self.K.shape[1])))
        if(adapt and self.enable_adapt):
            self.pre_cur_pos = cur_pos
            self.pre_cur_th = cur_th
            self.pre_J = J
            
        dw = next_pos[:3, :]-cur_pos[:3, :]
        next_th = cur_th + np.matmul(np.matmul(np.transpose(self.K), np.transpose(J)), dw)
        cur_th = next_th
            
        # Transform prediction in joint space to Cartesian
        fk_elbow = self.matlab_eng.FK_elbow(float(cur_th[0]), float(cur_th[1]), float(cur_th[2]), float(cur_th[3]), float(cur_th[4]), 
                                            self.l1_len, self.l2_len, float(np.pi), nargout=1)
        fk_elbow = np.asarray(fk_elbow).reshape((3, 1))
        fk_elbow = np.append(fk_elbow, np.array([[1]]), axis=0)
        pos_elbow = np.matmul(shoulder_trans, fk_elbow)
        prediction.append(np.reshape(pos_elbow[:3, :], (1, 3)))
        prediction = np.asarray(prediction)
        prediction = np.reshape(prediction, (1, 3))
        return prediction, next_th

    def IK(self, wrist_target, elbow_target, H, step=0.3, epsilon=0.0001):
        """
        This function calculates the joint space configuration of arm with the given joint positions
        Parameters: 
            wrist_target: An array of length 3. The desired [x, y, z] position of wrist.
            elbow_target: An array of length 3. The desired [x, y, z] position of elbow.
          
        Returns: 
            cur_th (5x1 matrix): The joint space configuration of the arm with the given cartesian position.
        """
        # Transform positions from camera frame to shoulder frame
        elbow_target = self.trans_to_frame(elbow_target[:3], H)
        wrist_target = self.trans_to_frame(wrist_target[:3], H)
        target_pos = np.asarray([elbow_target[:3], wrist_target[:3]]).reshape((6, 1))
        err_pos = target_pos.copy()
        cur_th = np.zeros((5, 1))
        count = 0
        while(not self.reach_target(err_pos, epsilon) and count<500):
            J = self.matlab_eng.jacobian_ew(float(cur_th[0]), float(cur_th[1]), float(cur_th[2]), float(cur_th[3]), float(cur_th[4]), 
                                            self.l1_len, self.l2_len, float(np.pi), nargout=1)
            J_inv = self.matlab_eng.J_inv(J)
            d_th = np.matmul(J_inv, err_pos)
            cur_th = cur_th + step*d_th

            pos = self.matlab_eng.FK_ew(float(cur_th[0]), float(cur_th[1]), float(cur_th[2]), float(cur_th[3]), float(cur_th[4]), 
                                        self.l1_len, self.l2_len, float(np.pi), nargout=1)
            cur_pos = np.asarray(pos).reshape((6, 1))
            err_pos = target_pos - cur_pos
            count+=1
        return cur_th
    
    def reach_target(self, error, eps):
        return (error[0]<eps and error[1]<eps and error[2]<eps and error[3]<eps and error[4]<eps and error[5]<eps)

    def set_arm_len(self, p1, p2, p3):
        """
        This function sets the arm lenght of the model. p1: shoulder, p2: elbow, p3: wrist.
        """
        self.l1_len = self.calc_len(p1, p2)
        self.l2_len = self.calc_len(p2, p3)

    def calc_len(self, p1, p2):
        """
        This function calculates the length between p1 and p2
        """
        length = np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2+(p1[2]-p2[2])**2)
        return float(length)

    def trans_to_frame(self, point, H):
        """
        This function transforms the input point to the frame specified by the transformation H
        """
        pt = np.matrix([[point[0]], [point[1]], [point[2]], [1]])
        new_pt = np.matmul(np.linalg.inv(H), pt)
        return new_pt

    def shutdown_matlab(self):
        self.matlab_eng.quit()
