import numpy as np
import matlab.engine
from RLS import RLS
import yaml

class ArmPredictor():
    def __init__(self, param_path):
        self.matlab_eng = matlab.engine.start_matlab()
        self.matlab_eng.addpath(r'matlab_src/')
        self.matlab_eng.init_params(nargout=0)
        self.l1_len = 0 # Upper arm
        self.l2_len = 0 # Lower arm
        self.K = np.identity(5)

        with open(param_path) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        lamda = data['arm_predictor']['lambda']
        self.enable_adapt = data['arm_predictor']['adapt']
        self.pred_step = data['prediction_step']
        self.RLS = RLS(self.K, 100000*np.identity(5), lamda)

        self.error_list = []
        self.K_list = []
        self.pre_cur_pos = 0
        self.pre_cur_th = 0
        self.pre_J = 0
    
    def predict_arm(self, wrist_prediction, cur_th, shoulder_trans, adapt):
        cur_th = np.reshape(cur_th, (5, 1))
        prediction = []
        for i in range(self.pred_step):
            next_pos_world = wrist_prediction[i, 0:3]
            next_pos = self.trans_to_frame(next_pos_world, shoulder_trans)
            J = self.matlab_eng.jacobian(float(cur_th[0]), float(cur_th[1]), float(cur_th[2]), float(cur_th[3]), float(cur_th[4]), 
                                         self.l1_len, self.l2_len, float(np.pi), nargout=1)
            fk = self.matlab_eng.FK(float(cur_th[0]), float(cur_th[1]), float(cur_th[2]), float(cur_th[3]), float(cur_th[4]), 
                                    self.l1_len, self.l2_len, float(np.pi), nargout=1)
            cur_pos = np.asarray(fk).reshape((3, 1))
            
            # RLS
            if(i==0):
                if(adapt and self.enable_adapt):
                    eck = cur_pos-self.pre_cur_pos
                    xk = np.matmul(np.transpose(self.pre_J), eck)
                    pred_cur_th = self.pre_cur_th+np.matmul(np.transpose(self.K), xk)
                    ek = cur_th-pred_cur_th

                    self.K = self.RLS.adapt(xk, ek)
                    self.K_list.append(np.reshape(self.K, (1, 25)))
                else:
                    self.pre_cur_pos = cur_pos
                    self.pre_cur_th = cur_th
                    self.pre_J = J
            
            dw = next_pos[0:3, :]-cur_pos[0:3, :]
            next_th = cur_th + np.matmul(np.matmul(np.transpose(self.K), np.transpose(J)), dw)
            cur_th = next_th
            
            fk_elbow = self.matlab_eng.FK_elbow(float(cur_th[0]), float(cur_th[1]), float(cur_th[2]), float(cur_th[3]), float(cur_th[4]), 
                                                self.l1_len, self.l2_len, float(np.pi), nargout=1)
            fk_elbow = np.asarray(fk_elbow).reshape((3, 1))
            fk_elbow = np.append(fk_elbow, np.array([[1]]), axis=0)
            pos_elbow = np.matmul(shoulder_trans, fk_elbow)
            prediction.append(np.reshape(pos_elbow[:3, :], (1, 3)))
        prediction = np.asarray(prediction)
        prediction = np.reshape(prediction, (self.pred_step, 3))
        return prediction

    def IK(self, wrist_target, elbow_target, H, step=0.3, epsilon=0.0005):
        """
        This function calculates the joint space configuration of arm with the given joint positions
        Parameters: 
            wrist_target: An array of length 3. The desired [x, y, z] position of wrist.
            elbow_target: An array of length 3. The desired [x, y, z] position of elbow.
          
        Returns: 
            cur_th (5x1 matrix): The joint space configuration of the arm with the given cartesian position.
        """
        # Transform positions to shoulder frame
        elbow_tar = self.trans_to_frame(elbow_target[0:3], H)
        wrist_tar = self.trans_to_frame(wrist_target[0:3], H)
        target_pos = np.asarray([elbow_tar[0], elbow_tar[1], elbow_tar[2], wrist_tar[0], wrist_tar[1], wrist_tar[2]])
        target_pos = np.reshape(target_pos, (6, 1))
        ep = target_pos.copy()
        cur_th = np.matrix([[0], [0], [0], [0], [0]])
        count = 0
        while((abs(ep[0])>epsilon or abs(ep[1])>epsilon or abs(ep[2])>epsilon or 
               abs(ep[3])>epsilon or abs(ep[4])>epsilon or abs(ep[5])>epsilon) and count<1000):
            J = self.matlab_eng.jacobian_ew(float(cur_th[0]), float(cur_th[1]), float(cur_th[2]), float(cur_th[3]), float(cur_th[4]), 
                                            self.l1_len, self.l2_len, float(np.pi), nargout=1)
            J_inv = self.matlab_eng.J_inv(J)
            d_th = np.matmul(J_inv, ep)
            cur_th = cur_th + step*d_th

            pos = self.matlab_eng.FK_ew(float(cur_th[0]), float(cur_th[1]), float(cur_th[2]), float(cur_th[3]), float(cur_th[4]), 
                                        self.l1_len, self.l2_len, float(np.pi), nargout=1)
            cur_pos = np.asarray(pos).reshape((6, 1))
            ep = target_pos-cur_pos
            count+=1
        return cur_th

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
