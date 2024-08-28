#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# 40m
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import logging
import numpy as np
import random
from gym import spaces
from numpy.core._multiarray_umath import ndarray
from scipy.stats import norm
from gym.envs.classic_control import rendering
import gym
import time
import copy

logger = logging.getLogger(__name__)
gym.logger.set_level(40)

PLACEHOLDER = 0

# In[ ]:;
class Base():
    def __init__(self):
        self.t = 0.1
        self.road_len = 300
        self.road_width = 7.5
        self.v_len = 20
        self.v_width = 5
        
        self.car_length = 5.0
        self.car_width = 2.0  # 车宽
      
    
class HCVcar(Base):
    def __init__(self, init_x, init_y, init_v, p):
        super(HCVcar, self).__init__()
        self.old_x = init_x
        self.old_y = init_y
        self.old_vx = init_v
        
        self.x = init_x
        self.y = init_y
        self.vx = init_v
        self.vy = 0
        self.p = 1 # p=0 不服从控制，p=1 服从控制
        
    def IDM(self, v, delta_v, s):
        T = 1.0
        s_0 = 2
        delta = 4
        a = 3
        b = 1.5
        v_desired = 30
        #print('safe', v * T + (v * delta_v) / (2 * np.power(a * b, 0.5)))
        s_star = s_0 + np.maximum(0, (v * T + (v * delta_v) / (2 * np.power(a * b, 0.5))))
        a0 = a * (1 - np.power((v / v_desired), delta) - np.power((s_star / s), 2))
        #print('delta_v', delta_v, 'headway', s, 'a0', a0)
        v_update = v + a0 * self.t
        if v_update < 0:
            v_update = 0
            a0 = -v / self.t
        return a0, v_update
    
    def step_forward(self, acc_x, speediff, headway):

        dice = np.random.uniform()
        if dice <= self.p:
            self.state = "cv"
            self.ax = acc_x
            self.x += self.vx * self.t + 0.5 * self.ax * self.t ** 2
            self.vx += self.ax * self.t
            
        else:
            self.state = "hcv"
            self.ax, vx = self.IDM(self.vx, speediff, headway)
            self.x += self.vx * self.t + 0.5 * self.ax * self.t ** 2
            self.vx = vx
            
        self.x = np.clip(self.x, 0, self.road_len)
        self.vx = np.clip(self.vx, 0, self.v_len)  
        
    def reset(self):
        self.x = self.old_x + 1 * np.random.uniform(0, 1)
        self.y = self.old_y + 0.5 * np.random.uniform(0, 1)
        self.vx = self.old_vx + 2 * np.random.uniform(0, 1)
        self.vy = 0
    
    def represent(self):
        return [self.x, self.vx]

    
class OBJcar(HCVcar):
    def __init__(self, init_x, init_y, init_v):
        super(OBJcar, self).__init__(init_x, init_y, init_v, p=1)
        
    def step_forward(self, acc_x, acc_y):
        self.x += self.vx * self.t + 0.5 * acc_x * self.t ** 2
        self.y += self.vy * self.t + 0.5 * acc_y * self.t ** 2
        self.vx += acc_x * self.t
        self.vy += acc_y * self.t
        
        self.x = np.clip(self.x, 0, self.road_len)
        self.y = np.clip(self.y, 0, self.road_width)
        self.vx = np.clip(self.vx, 0, self.v_len)  
        self.vy = np.clip(self.vy, -self.v_width, self.v_width)
        
    def represent(self):
        return [self.x, self.y, self.vx, self.vy]
    

class LaneChangeEnv(gym.Env):
    state: ndarray
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self):
        self.a_longitude_max = 3  # 纵向加速度上限
        self.a_longitude_min = -3  # 纵向加速度下限
        self.a_latitude_max = 1  # 横向加速度上限
        self.a_latitude_min = -1  # 横向加速度下限
        self.road_length = 300  # 道路长度
        self.road_width = 3.75  # 车道宽度
        self._max_episode_steps = 500
        
        self.leader_y = 3.75 * 1.5  # 引导车辆的纵坐标
        self.object_y = 3.75 / 2  # 目标车辆不换道的纵坐标
        
        self.original_state = np.array([70,  # 0-- object_x
                                        3.75 / 2,  # 1-- object_y
                                        20,  # 2-- object_xv
                                        0,  # 3-- object_yv
                                        90,  # 4-- leader_x
                                        20,  # 5-- leader_v
                                        50,  # 6-- follower_x
                                        20,  # 7-- follower_v
                                        130,  # 8-- around11_x
                                        20,  # 9-- around11_v
                                        10,  # 10-- around12_x
                                        20,  # 11-- around12_v
                                        110,  # 12-- around21_x
                                        20,  # 13-- around21_v 
                                        30,  # 14-- around22_x
                                        20,  # 15-- around22_v
                                        ])  # 初始状态 

        self.mean_state = np.array([150, 3.75, 20, 2.5, 150, 20, 150, 20, 150, 20, 150, 20, 150, 20, 150, 20])
        self.maxmin_state = np.array([300, 7.5, 25, 5, 300, 25, 300, 25, 300, 25, 300, 25, 300, 25, 300, 25])
        
        self.obj_veh = OBJcar(self.original_state[0], self.object_y, self.original_state[2])
        self.leader_veh = HCVcar(self.original_state[4], self.leader_y, self.original_state[5], 1)
        self.lagger_veh = HCVcar(self.original_state[6], self.leader_y, self.original_state[7], 1)
        self.env_veh1 = HCVcar(self.original_state[8], self.leader_y, self.original_state[9], 0)
        self.env_veh2 = HCVcar(self.original_state[10], self.leader_y, self.original_state[11], 0)
        self.env_veh3 = HCVcar(self.original_state[12], self.object_y, self.original_state[13], 0)
        self.env_veh4 = HCVcar(self.original_state[14], self.object_y, self.original_state[15], 0)
        self.all_veh = [self.obj_veh, self.leader_veh, self.lagger_veh,self.env_veh1, 
                        self.env_veh2, self.env_veh3, self.env_veh4]
        
        self.action_space = spaces.Box(np.array([self.a_longitude_min, self.a_latitude_min,
                                                 self.a_longitude_min, self.a_longitude_min]),
                                       np.array([self.a_longitude_max, self.a_latitude_max,
                                                 self.a_longitude_max, self.a_longitude_max]), dtype=np.float32)
        # 动作空间，四个加速度在上下界范围内的排列，object_ax,1object_ay,leader_a, follower_a
        self.observation_space = spaces.Box(np.zeros(16),
                                            np.zeros(16), dtype=np.float32)
        self.reset()
        self.viewer = None
        
    def get_repr(self):
        all_repr = []
        for veh in self.all_veh:
            all_repr += veh.represent()
        return (np.array(all_repr) - self.mean_state) / self.maxmin_state
    
    def reset(self):
        for veh in self.all_veh:
            veh.reset()
        
        self.object_x = self.obj_veh.x
        self.leader_x = self.leader_veh.x
        self.follower_x = self.lagger_veh.x
        return self.get_repr(), False, np.array([0, 0, 0, 0, 0])

    def step(self, action):
        done = 0
        crash_flag = 0
        bingo_flag = 0
        end_flag = 0
        action[1] = action[1] / 3
        
        self.obj_veh.step_forward(action[0], action[1])
        self.leader_veh.step_forward(action[2], self.env_veh1.vx - self.leader_veh.vx, self.env_veh1.x - self.leader_veh.x)
        self.lagger_veh.step_forward(action[3], random.randint(-1,1), random.randint(30,35))
        #print('leader', self.leader_veh.state)
        #print('lagger', self.lagger_veh.state)
        self.env_veh1.step_forward(PLACEHOLDER, random.randint(-1,1), random.randint(40,45))
        self.env_veh2.step_forward(PLACEHOLDER, self.lagger_veh.vx - self.env_veh2.vx, self.lagger_veh.x - self.env_veh2.x)
        self.env_veh3.step_forward(PLACEHOLDER, random.randint(-1,1), random.randint(40,45))
        self.env_veh4.step_forward(PLACEHOLDER, self.obj_veh.vx - self.env_veh4.vx, self.obj_veh.x - self.env_veh4.x)

        reward_00 = self.obj_veh.x - self.object_x
        reward_01 = self.leader_veh.x - self.leader_x
        reward_02 = self.lagger_veh.x - self.follower_x
        reward_0 = 1000 * (reward_00 + reward_01 + reward_02)  # move on para_a0 = 0.5,1,3
        self.object_x = self.obj_veh.x
        self.leader_x = self.leader_veh.x
        self.follower_x = self.lagger_veh.x

        reward_1 = 2000 * (self.obj_veh.y - self.object_y)  # 鼓励换道趋向 para_a1 = 80,100,150

        reward_2 = - 300 * np.abs(np.arctan(action[1] - action[0]))  # 转向角，舒适度 para_a2=-300,-100,-200
        
        reward_3 = 0
        if action[0] >= 0 and action[1] >= 0: 
            reward_3 += 5 * 1.13 / 44000 * (2500 * (1 / np.sqrt(self.obj_veh.vx ** 2 + self.obj_veh.vy ** 2))
                                          + 3785.0374 * 1 + 3.2895 * np.sqrt(self.obj_veh.vx ** 2 + self.obj_veh.vy ** 2)
                                          + 4852.9412 * np.sqrt(action[0] ** 2 + action[1] ** 2))
        elif action[0] < 0 and action[1] >= 0: 
            action[0] = 0
            reward_3 += 5 * 1.13 / 44000 * (2500 * (1 / np.sqrt(self.obj_veh.vx ** 2 + self.obj_veh.vy ** 2))
                                          + 3785.0374 * 1 + 3.2895 * np.sqrt(self.obj_veh.vx ** 2 + self.obj_veh.vy ** 2)
                                          + 4852.9412 * np.sqrt(action[0] ** 2 + action[1] ** 2))
        elif action[0] >= 0 and action[1] < 0: 
            action[1] = 0
            reward_3 += 5 * 1.13 / 44000 * (2500 * (1 / np.sqrt(self.obj_veh.vx ** 2 + self.obj_veh.vy ** 2))
                                          + 3785.0374 * 1 + 3.2895 * np.sqrt(self.obj_veh.vx ** 2 + self.obj_veh.vy ** 2)
                                          + 4852.9412 * np.sqrt(action[0] ** 2 + action[1] ** 2))
        elif action[0] < 0 and action[1] < 0: 
            action[0] = 0
            action[1] = 0
            reward_3 += 5 * 1.13 / 44000 * (2500 * (1 / np.sqrt(self.obj_veh.vx ** 2 + self.obj_veh.vy ** 2))
                                          + 3785.0374 * 1 + 3.2895 * np.sqrt(self.obj_veh.vx ** 2 + self.obj_veh.vy ** 2)
                                          + 4852.9412 * np.sqrt(action[0] ** 2 + action[1] ** 2))
               
        for veh in self.all_veh[1:]:
            if veh.ax >= 0:
                reward_3 += 5 * 1.13 / 44000 * (2500 * (1 / (veh.vx + 1e-6)) + 
                                            3785.0374 * 1 + 3.2895 * veh.vx + 4852.9412 * veh.ax)
            elif veh.ax < 0:
                veh.ax = 0
                reward_3 += 5 * 1.13 / 44000 * (2500 * (1 / (veh.vx + 1e-6)) + 
                                            3785.0374 * 1 + 3.2895 * veh.vx + 4852.9412 * veh.ax)

        reward_3 = - 200 * 4 * reward_3  # fuel & emission para_a3 = -0.007; 70;1000；300

#         reward_3 = 0.132 * np.sqrt(self.obj_veh.vx ** 2 + self.obj_veh.vy ** 2) + 1.1 * np.sqrt(self.obj_veh.vx ** 2 + self.obj_veh.vy ** 2) * np.sqrt(action[0] ** 2 + action[1] ** 2) + 0.0003202 * (np.sqrt(self.obj_veh.vx ** 2 + self.obj_veh.vy ** 2))** 3
#         for veh in self.all_veh[1:]:
#             reward_3 += 0.132 * veh.vx + 1.1 * veh.vx * veh.ax +0.0003202 * veh.vx ** 3
#         reward_3 = -10 * reward_3
        
#         VSP = 0.132∙v+1.1∙v∙a+0.0003202∙v^3

        reward_4 = -10
        if (self.leader_veh.x - self.obj_veh.x < 6 or 
            self.obj_veh.x - self.lagger_veh.x < 6 or 
            self.env_veh3.x - self.obj_veh.x < 6 or
            self.obj_veh.x - self.env_veh4.x < 6 or 
            self.env_veh1.x - self.leader_veh.x < 6 or 
            self.lagger_veh.x - self.env_veh2.x < 6):  # bump
            reward_4 = -1000000
            print('crash')
       
            crash_flag = 1
            done = 1

        elif self.leader_y - self.obj_veh.y <= 0.5:  # 靠近中心线奖励
            print('bingo')
            reward_4 = 1 * (2046 * (self.leader_y - self.obj_veh.y - 1) ** 2 - 92) # 2046,-1,-46; 3000,50;
            bingo_flag = 1
            done = 1
            
#         elif 0.5 < self.leader_y - self.obj_veh.y <= 3.75:  # 换道奖励
#             reward_4 = 40 * (self.leader_y - self.obj_veh.y - 1) ** 2 - 10 # 2046,-1,-46

        elif self.obj_veh.x >= self.road_length:
            print('end')
            end_flag = 1
            done = 1

        rewards = reward_0 + reward_1 + reward_2 + reward_3 + reward_4 #+ reward_5
        info = {'reward': np.array([reward_0,reward_1,reward_2,reward_3,reward_4]), 'crash':crash_flag, 'bingo':bingo_flag, 'end': end_flag}
        return self.get_repr(), rewards, done, info #, 11, 12, 13
    
    def create_veh_render(self, l, r, t, b, color):
        car_obj = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        car_obj.add_attr(rendering.Transform(translation=(0, 280)))
        obj_trans = rendering.Transform()
        car_obj.add_attr(obj_trans)
        car_obj.set_color(color[0], color[1], color[2])
        return car_obj, obj_trans
        
    
    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        screen_width = 1500  # 界面宽
        screen_height = 500  # 界面高
        scale = screen_width / 250  # 显示比例
        if self.viewer is None:
            
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.line1 = rendering.Line((0, 280), (300 * scale, 280))
            self.line2 = rendering.Line((0, 280 + 3.75 * scale), (300 * scale, 280 + 3.75 * scale))  # 中间的线，应为虚线
            self.line3 = rendering.Line((0, 280 + 7.5 * scale), (300 * scale, 280 + 7.5 * scale))  # 画车道的三条线
            self.line1.set_color(0, 0, 0)
            self.line2.set_color(0, 0, 0)
            self.line3.set_color(0, 0, 0)  # 设置车道线的颜色

            l, r, t, b = (- scale * self.obj_veh.car_length / 2, scale * self.obj_veh.car_length / 2, 
                          -scale * self.obj_veh.car_width / 2, scale * self.obj_veh.car_width / 2)

            car_obj, self.obj_trans = self.create_veh_render(l, r, t, b, (1., .9, 0))
            car_leader, self.leader_trans = self.create_veh_render(l, r, t, b, (1., 0, 0))
            car_follower, self.follower_trans = self.create_veh_render(l, r, t, b, (1., 0, 0))
            car_around11, self.around11_trans = self.create_veh_render(l, r, t, b, (0.5, 0.5, 0.5))
            car_around12, self.around12_trans = self.create_veh_render(l, r, t, b, (0.5, 0.5, 0.5))
            car_around21, self.around21_trans = self.create_veh_render(l, r, t, b, (0.5, 0.5, 0.5))
            car_around22, self.around22_trans = self.create_veh_render(l, r, t, b, (0.5, 0.5, 0.5))

            self.viewer.add_geom(self.line1)
            self.viewer.add_geom(self.line2)
            self.viewer.add_geom(self.line3)
            self.viewer.add_geom(car_leader)
            self.viewer.add_geom(car_obj)
            self.viewer.add_geom(car_follower)
            self.viewer.add_geom(car_around11)
            self.viewer.add_geom(car_around12)
            self.viewer.add_geom(car_around21)
            self.viewer.add_geom(car_around22)

            # print('self.state[0]', self.state[0])
            # print('self.state[1]', self.state[1])

        obj_x = self.obj_veh.x
        obj_y = self.obj_veh.y
        leader_x = self.leader_veh.x
        leader_y = self.leader_veh.y
        follower_x = self.lagger_veh.x
        follower_y = self.lagger_veh.y
        around11_x = self.env_veh1.x
        around11_y = self.env_veh1.y
        around12_x = self.env_veh2.x
        around12_y = self.env_veh2.y
        around21_x = self.env_veh3.x
        around21_y = self.env_veh3.y
        around22_x = self.env_veh4.x
        around22_y = self.env_veh4.y

        self.obj_trans.set_translation(obj_x * scale, obj_y * scale)
        self.leader_trans.set_translation(leader_x * scale, leader_y * scale)
        self.follower_trans.set_translation(follower_x * scale, follower_y * scale)
        self.around11_trans.set_translation(around11_x * scale, around11_y * scale)
        self.around12_trans.set_translation(around12_x * scale, around12_y * scale)
        self.around21_trans.set_translation(around21_x * scale, around21_y * scale)
        self.around22_trans.set_translation(around22_x * scale, around22_y * scale)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

