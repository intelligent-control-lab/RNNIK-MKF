import numpy as np
import matlab.engine
import time
from RLS import RLS

class ArmPredictor():
    def __init__(self, prediction_step):
        self.matlab_eng = matlab.engine.start_matlab()
        self.matlab_eng.addpath(r'matlab_src/')
        self.matlab_eng.init_params(nargout=0)
        self.pred_step = prediction_step
        self.l1_len = 0 # Upper arm
        self.l2_len = 0 # Lower arm
        self.K = np.identity(5)
        self.RLS = RLS(self.K, 100000*np.identity(5), 1)

        self.error_list = []
        self.K_list = []
    
    def predict_arm(self, wrist_prediction, cur_th, pre_cur_pos, pre_cur_th, pre_J, shoulder_trans, idx, adapt):
        cur_th = np.reshape(cur_th, (5, 1))
        prediction = []
        for i in range(self.pred_step):
            next_pos_world = wrist_prediction[i, 0:3]
            next_pos = self.trans_to_frame(next_pos_world, shoulder_trans)
            J = self.matlab_eng.jacobian(float(cur_th[0]), float(cur_th[1]), float(cur_th[2]), float(cur_th[3]), float(cur_th[4]), 
                                         self.l1_len, self.l2_len, float(np.pi), nargout=1)
            fk = self.matlab_eng.FK(float(cur_th[0]), float(cur_th[1]), float(cur_th[2]), float(cur_th[3]), float(cur_th[4]), 
                                    self.l1_len, self.l2_len, float(np.pi), nargout=1)
            cur_pos = np.matrix([[fk[0][3]], [fk[1][3]], [fk[2][3]]])
            
            # RLS
            if(i==0 and adapt):
                eck = cur_pos-pre_cur_pos
                xk = np.matmul(np.transpose(pre_J), eck)
                pred_cur_th = pre_cur_th+np.matmul(np.transpose(self.K), xk)
                ek = cur_th-pred_cur_th
                self.K = self.RLS.adapt(xk, ek)
                self.K_list.append(np.reshape(self.K, (1, 25)))
            if(i==0):
                pre_cur_pos = cur_pos
                pre_cur_th = cur_th
                pre_J = J
            
            dw = next_pos[0:3, :]-cur_pos[0:3, :]
            next_th = cur_th + np.matmul(np.matmul(np.transpose(self.K), np.transpose(J)), dw)
            cur_th = next_th
            
            fk_elbow = self.matlab_eng.FK_elbow(float(cur_th[0]), float(cur_th[1]), float(cur_th[2]), float(cur_th[3]), float(cur_th[4]), 
                                                self.l1_len, self.l2_len, float(np.pi), nargout=1)
            pos_elbow = np.matmul(shoulder_trans, np.matrix([[fk_elbow[0][3]], [fk_elbow[1][3]], [fk_elbow[2][3]], [1]]))
            prediction.append(np.reshape(pos_elbow[:3, :], (1, 3)))
        prediction = np.asarray(prediction)
        prediction = np.reshape(prediction, (self.pred_step, 3))
        return prediction, pre_cur_pos, pre_cur_th, pre_J

    def IK(self, wrist_target, elbow_target, H, step=0.3, epsilon=0.0001):
        """
        This function calculates the joint space configuration of arm with the given joint positions
        """
        # Transfer positions to shoulder frame
        elbow_tar = self.trans_to_frame(elbow_target[0:3], H)
        wrist_tar = self.trans_to_frame(wrist_target[0:3], H)
        target_pos = np.asarray([elbow_tar[0], elbow_tar[1], elbow_tar[2], wrist_tar[0], wrist_tar[1], wrist_tar[2]])
        target_pos = np.reshape(target_pos, (6, 1))
        ep = target_pos.copy()
        cur_th = np.matrix([[0], [0], [0], [0], [0]])

        while(abs(ep[0])>epsilon or abs(ep[1])>epsilon or abs(ep[2])>epsilon or abs(ep[3])>epsilon or abs(ep[4])>epsilon or abs(ep[5])>epsilon):
            J = self.matlab_eng.jacobian_ew(float(cur_th[0]), float(cur_th[1]), float(cur_th[2]), float(cur_th[3]), float(cur_th[4]), 
                                            self.l1_len, self.l2_len, float(np.pi), nargout=1)
            J_inv = self.matlab_eng.J_inv(J)
            d_th = np.matmul(J_inv, ep)
            cur_th = cur_th + step*d_th

            pos = self.matlab_eng.FK_ew(float(cur_th[0]), float(cur_th[1]), float(cur_th[2]), float(cur_th[3]), float(cur_th[4]), 
                                        self.l1_len, self.l2_len, float(np.pi), nargout=1)
            cur_pos = np.matrix([[pos[0][3]], [pos[1][3]], [pos[2][3]], [pos[4][3]], [pos[5][3]], [pos[6][3]]])
            ep = target_pos-cur_pos
        return cur_th

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

    def shutdown(self):
        self.matlab_eng.quit()

