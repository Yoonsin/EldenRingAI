import cv2
import json
import numpy as np
import pytesseract
import time
import requests
import os
from tensorboardX import SummaryWriter
import tensorflow as tf

TOTAL_ACTIONABLE_TIME = 120
HP_CHART = {}

debug_dir = "debug_hp"
os.makedirs(debug_dir, exist_ok=True)

#vigor_chart.csv -> 생명력에 맞는 체력량을 저장하는 딕셔너리
with open('vigor_chart.csv', 'r') as v_chart:
    for line in v_chart.readlines():
        stat_point = int(line.split(',')[0])
        hp_amount = int(line.split(',')[1])
        HP_CHART[stat_point] = hp_amount


class EldenReward:
    def __init__(self, char_slot, logdir) -> None:
        self.previous_runes_held = None
        self.current_runes_held = None

        self.previous_stats = None
        self.current_stats = None
        self.seen_boss = False

        self.max_hp = None
        self.prev_hp = None
        self.curr_hp = None
        self.debug_idx =0

        self.hp_ratio = 0.403

        self.character_slot = char_slot

        self.death_ratio = 0.005

        self.time_since_death = time.time()
        self.time_since_seen_boss = time.time()

        self.death = False
        self.curr_boss_hp = 3200
        #self.prev_boss_hp = None

        self.agent_ip = 'localhost'
        self._request_stats()

        self.boss_max_hp = 3200
        self.logger = SummaryWriter(os.path.join(logdir, 'PPO_0'))
        self.iteration = 0
        self.boss_hp = 1
        self.time_since_last_hp_change = time.time()
        self.time_since_last_boss_hp_change = time.time()
        self.boss_hp_history = []
        self.boss_hp_target_range = 10
        self.boss_hp_target_window = 5
        self.time_till_fight = 120
        self.time_since_reset = time.time()
        self.min_boss_hp = 1
        self.time_since_check_for_boss = time.time()
        self.hp_history = []
        


    def _request_stats(self):
        headers = {"Content-Type": "application/json"}
        response = requests.get(f"http://{self.agent_ip}:6000/stats/{self.character_slot}", headers=headers)
        stats = response.json()
        print(stats)

        self.previous_stats = self.current_stats
        self.current_stats = [stats['vigor'],
                              stats['mind'],
                              stats['endurance'],
                              stats['strength'],
                              stats['dexterity'],
                              stats['intelligence'],
                              stats['faith'],
                              stats['arcane']]
        self.max_hp = HP_CHART[self.current_stats[0]]
        print(f"max_hp : {self.max_hp}")
        self.time_alive_multiplier = 1


    def _get_runes_held(self, frame):
        runes_image = frame[1133:1166, 1715:1868]
        #cv2.imwrite("debug_hp/runes_debug.png", frame[1133:1166, 1715:1868])
        runes_image = cv2.resize(runes_image, (154*3, 30*3))
        runes_held = pytesseract.image_to_string(runes_image,  lang='eng',config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789')
        if runes_held != "":
            return int(runes_held)
        else:
            return self.previous_runes_held


    def _get_boss_name(self, frame):

        debug_frame = frame.copy()
        x1, y1, x2, y2 = 440, 830, 600, 885
        hp_image = frame[y1:y2, x1:x2]

        debug_frame = frame.copy()
        cv2.rectangle(debug_frame,(x1, y1), (x2, y2), (0, 0, 255), 2)  # 빨간색 사각형
        #cv2.imwrite(f"debug_hp/boss_name_debug{self.debug_idx}.png", debug_frame)
        self.debug_idx += 1

        boss_name = frame[y1:y2, x1:x2]
        boss_name = cv2.resize(boss_name, ((x2 - x1) * 3, (y2 - y1) * 3))
        boss_name = pytesseract.image_to_string(boss_name, lang='kor+eng', config='--psm 6 --oem 3')

        if boss_name != "":
            print(boss_name.strip())
            return boss_name.strip()
        else:
            return None

    
    def _get_boss_dmg(self, frame):
        debug_frame = frame.copy()
        cv2.rectangle(debug_frame, (1410, 840), (1480, 860), (255, 0, 0), 2)  # 시각적 확인
        #cv2.imwrite(f"debug_hp/boss_dmg_debug{self.debug_idx}.png", debug_frame)
        self.debug_idx += 1
        boss_dmg = frame[840:860, 1410:1480]
        boss_dmg = cv2.resize(boss_dmg, ((1480-1410)*3, (860-840)*3))
        boss_dmg = pytesseract.image_to_string(boss_dmg,  lang='eng',config='--psm 6 --oem 3')
        print(f"boss dmg: {boss_dmg}")
        if boss_dmg != "":
            return int(boss_dmg)
        else:
            return 0

    def get_player_hp(self, frame):

        # 1920x1200 해상도에서 파란색 체력바 위치 추정
        hp_bar_region = frame[64:81, 160:280]

        # RGB → HSV
        hsv = cv2.cvtColor(hp_bar_region, cv2.COLOR_RGB2HSV)

        # 파란색 HSV 범위
        lower_blue = np.array([100, 100, 50])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # 체력 비율 계산
        blue_pixel_count = cv2.countNonZero(blue_mask)
        total_pixel_count = blue_mask.size
        hp_ratio = blue_pixel_count / total_pixel_count

        return hp_ratio
        
    def update(self, frame):
        self.iteration += 1
        # self.previous_runes_held = self.current_runes_held
        # try:
        #     self.current_runes_held = self._get_runes_held(frame)
        # except:
        #     pass
        
        # if not self.previous_runes_held is None and not self.current_runes_held is None:
        #     runes_reward = self.current_runes_held - self.previous_runes_held
        # else:
        #     runes_reward = 0

        if self.curr_hp is None:
            self.curr_hp = self.max_hp
        stat_reward = 0
        # if not self.previous_stats is None and not self.current_stats is None:
        #     if runes_reward < 0:
                # self._request_stats()
                # for i, stat in enumerate(self.current_stats):
                #     if self.current_stats[i] != self.previous_stats[i]:
                #         stat_reward += (self.current_stats[i] - self.previous_stats[i]) * 10000


        hp_reward = 0
        total_hp_reward = 0
        if not self.death:
            t0 = time.time()

            # if self.time_since_last_hp_change > 1.0:
            # hp_image = frame[51:55, 155:155 + int(self.max_hp * self.hp_ratio) - 20]
            #
            # debug_frame = frame.copy()
            # cv2.rectangle(debug_frame, (155, 51), (155 + int(self.max_hp * self.hp_ratio) - 20, 55), (255, 0, 0), 2)  # 시각적 확인
            # cv2.imwrite(f"debug_hp/hp_debug{self.debug_idx}.png", debug_frame)
            # self.debug_idx += 1

            debug_frame = frame.copy()
            x1,y1,x2,y2 = 150,45,360,62
            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 빨간색 사각형
            #cv2.imwrite(f"debug_hp/hp_debug{self.debug_idx}.png", debug_frame)
            self.debug_idx += 1

            hp_image = frame[y1:y2, x1:x2]


            lower = np.array([0,150,75])
            upper = np.array([150,255,125])
            hsv = cv2.cvtColor(hp_image, cv2.COLOR_RGB2HSV)
            mask = cv2.inRange(hsv, lower, upper)
            matches = np.argwhere(mask==255)
            self.prev_hp = self.curr_hp
            self.curr_hp = (len(matches) / (hp_image.shape[1] * hp_image.shape[0])) * self.max_hp

            print(f"curr hp: {self.curr_hp}")

            # check for 1 hp
            if (self.curr_hp / self.max_hp) < self.death_ratio:
                lower = np.array([0,0,150])
                upper = np.array([255,255,255])
                hsv = cv2.cvtColor(hp_image, cv2.COLOR_RGB2HSV)
                mask = cv2.inRange(hsv, lower, upper)
                matches = np.argwhere(mask==255)
                self.prev_hp = self.curr_hp
                self.curr_hp = (len(matches) / (hp_image.shape[1] * hp_image.shape[0])) * self.max_hp
            
            t1 = time.time()

            if not self.prev_hp is None and not self.curr_hp is None:
                hp_reward = (self.curr_hp - self.prev_hp) / self.max_hp
                if hp_reward != 0:
                    self.time_since_last_hp_change = time.time()
                if hp_reward > 0:
                    hp_reward /= 10
                self.hp_history.append(hp_reward)
                # Use the hp history to effect reward, hopefully making taking damage more punishing
                num_samples = 15 if len(self.hp_history) > 15 else len(self.hp_history)
                for i in range(num_samples):
                    total_hp_reward += self.hp_history[-(i + 1)]


            #debugging
                # 디버깅 인덱스 초기화
                # if not hasattr(self, 'debug_idx'):
                #     self.debug_idx = 0
                # else:
                #     self.debug_idx += 1
                #
                # # 시각화용 복사
                # #debug_frame = frame.copy()
                # tmp = self.get_player_hp(frame)
                #
                # cv2.rectangle(debug_frame, (160, 100), (540, 115), (255, 0, 0), 2)  # 시각적 확인
                # cv2.putText(debug_frame, f"HP Ratio (Blue): {self.hp_ratio:.3f}", (30, 50),
                #             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
                # cv2.imwrite("debug_hp/frame_debug.png", debug_frame)
                #
                # print(f"[DEBUG HP] Frame {self.debug_idx:05d} - HP Ratio: {tmp:.3f}")

                # 이후 원래 로직 계속 진행됨...

            boss_name = ""
            if not self.seen_boss and time.time() - self.time_since_check_for_boss > 2:
                boss_name = self._get_boss_name(frame)
                self.time_since_check_for_boss = time.time()

            boss_dmg_reward = 0
            boss_find_reward = 0
            boss_timeout = 2.5
            # set_hp = False
            print(f"boss name: {boss_name}")
            if not boss_name is None and ('트리 가드' in boss_name or 'Tree Sentinel' in boss_name):
                if not self.seen_boss:
                    self.time_till_fight = 1 - ((time.time() - self.time_since_reset) / TOTAL_ACTIONABLE_TIME)
                self.seen_boss = True
                self.time_since_seen_boss = time.time()
            
            if not self.time_since_seen_boss is None:
                time_since_boss = time.time() - self.time_since_seen_boss
                boss_find_reward = 0
                # if time_since_boss < boss_timeout:
                #     if not self.seen_boss:
                #         boss_find_reward = -time_since_boss / TOTAL_ACTIONABLE_TIME
                #     else:
                #         boss_find_reward = 0
                #     try:
                #         #dmg = self._get_boss_dmg(frame)
                #         #self.curr_boss_hp -= dmg
                #         #boss_dmg_reward = 0
                #         pass
                #     except:
                #         pass
                # else:
                #     boss_find_reward = -time_since_boss / TOTAL_ACTIONABLE_TIME
                
            t2 = time.time()
            # if p_count < 10:
            #     self.prev_hp = self.curr_hp
            #     self.curr_hp = self.max_hp
        self.logger.add_scalar('curr_hp', self.curr_hp / self.max_hp, self.iteration)
        
        boss_hp = 1
        if self.seen_boss and not self.death:


            boss_hp_image = frame[869:873, 475:1460]
            debug_frame = frame.copy()
            cv2.rectangle(debug_frame, (475, 869), (1460, 873), (255, 0, 0), 2)  # 시각적 확인
            #cv2.imwrite(f"debug_hp/boss_boss_hp{self.debug_idx}.png", debug_frame)
            self.debug_idx += 1

            lower = np.array([0,0,75])
            upper = np.array([150,255,255])
            hsv = cv2.cvtColor(boss_hp_image, cv2.COLOR_RGB2HSV)
            mask = cv2.inRange(hsv, lower, upper)
            matches = np.argwhere(mask==255)
            boss_hp = len(matches) / (boss_hp_image.shape[1] * boss_hp_image.shape[0])

        if self.boss_hp is None:
            self.boss_hp = 1

        if abs(boss_hp - self.boss_hp) < 0.08 and self.time_since_last_boss_hp_change > 1.0:
            boss_dmg_reward = (self.boss_hp - boss_hp) * 5
            if boss_dmg_reward < 0:
                boss_dmg_reward = 0
            self.boss_hp = boss_hp
            self.boss_hp_history.append(self.boss_hp)
            self.time_since_last_boss_hp_change = time.time()

        t3 = time.time()

        percent_through_fight_reward = 0
        if not self.death:
            if len(self.boss_hp_history) >= self.boss_hp_target_window:
                boss_max = None
                boss_min = None
                for i in range(self.boss_hp_target_window):
                    if boss_max is None:
                        boss_max = self.boss_hp_history[-(i + 1)]
                    elif boss_max < self.boss_hp_history[-(i + 1)]:
                        boss_max = self.boss_hp_history[-(i + 1)]
                    if boss_min is None:
                        boss_min = self.boss_hp_history[-(i + 1)]
                    elif boss_min > self.boss_hp_history[-(i + 1)]:
                        boss_min = self.boss_hp_history[-(i + 1)]
                if abs(boss_max - boss_min) < self.boss_hp_target_range:
                    percent_through_fight_reward = (1 - self.boss_hp) * 0.75
                    self.time_alive_multiplier = (1 - self.boss_hp) * 0.25
                    if self.boss_hp < self.min_boss_hp:
                        self.min_boss_hp = self.boss_hp
                else:
                    percent_through_fight_reward = 0
            else:
                percent_through_fight_reward = 0
        else:
            percent_through_fight_reward = 0
        self.logger.add_scalar('boss_hp', self.boss_hp, self.iteration)

        t_end = time.time()
        # print("Reward Player HP: {:.5f}".format(t1 - t0))
        # print("Reward Boss Find: {:.5f}".format(t2 - t1))
        # print("Reward Boss HP: {:.5f}".format(t3 - t2))
        # print("Reward Percent through: {:.5f}".format(t_end - t3))


        if not self.death and not self.curr_hp is None:
            self.death = (self.curr_hp / self.max_hp) <= self.death_ratio
            time_alive = time.time() - self.time_since_death
            if self.seen_boss:
                time_alive_reward = (time_alive * 0.01) * (self.time_alive_multiplier)
            else:
                time_alive_reward = 0
            if self.death:
                hp_reward = -1
                self.time_since_death = time.time()
                #self.curr_hp = self.max_hp
                self.death = False
                self.seen_boss = False
                self.time_since_last_hp_change = time.time()
                self.boss_hp_history = []
                return time_alive_reward, percent_through_fight_reward, total_hp_reward, True, boss_dmg_reward, boss_find_reward, self.time_since_seen_boss
            else:
                return time_alive_reward, percent_through_fight_reward, total_hp_reward, self.death, boss_dmg_reward, boss_find_reward, self.time_since_seen_boss