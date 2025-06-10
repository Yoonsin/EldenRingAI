import os
import cv2
import gym
import time
import json
import wave
import pyaudio
import requests
import threading
import numpy as np
import pytesseract
import subprocess
from gym import spaces
import tensorflow as tf
from mss import mss as mss
from EldenReward import EldenReward
from tensorboardX import SummaryWriter


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)


TOTAL_ACTIONABLE_TIME = 120

DISCRETE_ACTIONS = {'w': 'run_forwards',
                    's': 'run_backwards',
                    'a': 'run_left',
                    'd': 'run_right',
                    'release_wasd': 'release_wasd',
                    'space': 'dodge',
                    'm1': 'attack',
                    'shift+m1': 'strong_attack',
                    'm2': 'guard',
                    'shift+m2': 'skill',
                    'r': 'use_item',
                    'space_hold': 'sprint',
                    'f': 'jump'}

N_DISCRETE_ACTIONS = len(DISCRETE_ACTIONS)
N_CHANNELS = 3
IMG_WIDTH = 1920
IMG_HEIGHT = 1200
MODEL_HEIGHT = 450#500
MODEL_WIDTH = 800
CLASS_NAMES = ['successful_parries', 'missed_parries']
HP_CHART = {}
with open('vigor_chart.csv', 'r') as v_chart:
    for line in v_chart.readlines():
        stat_point = int(line.split(',')[0])
        hp_amount = int(line.split(',')[1])
        HP_CHART[stat_point] = hp_amount
#print(HP_CHART)

def timer_callback(t_start):
    while True:
        with open('obs_timer.txt', 'w') as f:
            duration = time.gmtime(time.time() - t_start)
            days = int(np.floor((time.time() - t_start) / (24 * 60 * 60)))
            f.write(str(days).zfill(2) + ":")
            f.write(str(time.strftime('%H:%M:%S', duration)))
        time.sleep(1)

def audio_to_fft(audio):
    # Since tf.signal.fft applies FFT on the innermost dimension,
    # we need to squeeze the dimensions and then expand them again
    # after FFT
    audio = tf.squeeze(audio, axis=-1)
    fft = tf.signal.fft(
        tf.cast(tf.complex(real=audio, imag=tf.zeros_like(audio)), tf.complex64)
    )
    fft = tf.expand_dims(fft, axis=-1)

    # Return the absolute value of the first half of the FFT
    # which represents the positive frequencies
    return tf.math.abs(fft[:, : (32000 // 2), :])

def path_to_audio(path):
    """Reads and decodes an audio file."""
    audio = tf.io.read_file(path)
    audio, _ = tf.audio.decode_wav(audio, 1, 2 * 16000)
    return audio

class AudioRecorder():
    # Audio class based on pyAudio and Wave
    def __init__(self):
        self.open = True
        self.rate = 48000
        self.frames_per_buffer = 1024
        self.channels = 2
        self.format = pyaudio.paInt16
        self.audio_filename = "parry.wav"
        self.audio = pyaudio.PyAudio()

        # for i in range(self.audio.get_device_count()):
        #     info = self.audio.get_device_info_by_index(i)
        #     print(f"Device {i}: {info}")

        device_index = self.get_device_index_by_name("스테레오 믹스")
        if device_index is None:
            raise RuntimeError("스테레오 믹스 장치를 찾을 수 없습니다!")

        self.stream = self.audio.open(format=self.format,
                                      channels=self.channels,
                                      rate=self.rate,
                                      input=True,
                                      frames_per_buffer = self.frames_per_buffer,
                                      input_device_index=device_index)
        self.audio_frames = []
        self.active = False

    def get_device_index_by_name(self,name_contains):
        p = pyaudio.PyAudio()
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if name_contains.lower() in info['name'].lower():
                return i
        return None

    # Audio starts being recorded
    def record(self, iter):
        # AudioRecorder 클래스 내에서 파일 쓰기 전에 폴더 생성
        os.makedirs("parries", exist_ok=True)

        self.active = True
        data = self.stream.read(self.frames_per_buffer) 
        self.audio_frames.append(data)
        file_name = os.path.join('parries', self.audio_filename[:-4] + "_" + str(iter).zfill(7) + '.wav')
        waveFile = wave.open(file_name, 'wb')
        waveFile.setnchannels(self.channels)
        waveFile.setsampwidth(self.audio.get_sample_size(self.format))
        waveFile.setframerate(self.rate)
        waveFile.writeframes(b''.join(self.audio_frames))
        waveFile.close()
        self.active = False



    def get_audio(self):
        if len(self.audio_frames) > 0:
            return self.audio_frames.pop()
        else:
            return None


    def close(self):
        self.active = False
        self.stream.close()
        self.audio.terminate()

        waveFile = wave.open(self.audio_filename, 'wb')
        waveFile.setnchannels(self.channels)
        waveFile.setsampwidth(self.audio.get_sample_size(self.format))
        waveFile.setframerate(self.rate)
        waveFile.writeframes(b''.join(self.audio_frames))
        waveFile.close()

    # Launches the audio recording function using a thread
    def start(self, iter):
        if not self.active:
            audio_thread = threading.Thread(target=self.record, args=(iter,))
            audio_thread.start()


class EldenEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self, logdir):
        super(EldenEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        print(f"cnt : {N_DISCRETE_ACTIONS}")
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(MODEL_HEIGHT, MODEL_WIDTH, N_CHANNELS), dtype=np.uint8)

        self.agent_ip = 'localhost'
        self.logger = SummaryWriter(os.path.join(logdir, 'PPO_0'))
        self.debug_idx = 0
        
        # Start Elden Ring
        headers = {"Content-Type": "application/json"}
        requests.post(f"http://{self.agent_ip}:6000/action/start_elden_ring", headers=headers)
        time.sleep(70)

        # Load save file
        headers = {"Content-Type": "application/json"}
        requests.post(f"http://{self.agent_ip}:6000/action/load_save", headers=headers)
        
        self.reward = 0
        self.rewardGen = EldenReward(1, logdir)
        self.death = False
        self.t_start = time.time()
        self.done = False
        self.iteration = 0
        self.first_step = False
        self.consecutive_deaths = 0
        self.locked_on = False
        self.num_runs = 0
        self.max_reward = None
        self.time_since_r = time.time()
        
        self.reward_history = []
        self.parry_dict = {'vod_duration':None,
                           'parries': []}
        self.t_since_parry = None
        self.parry_detector =  tf.saved_model.load("parry_detector")

        self.prev_step_end_ts = time.time()
        self.last_fps = []
        self.sct = mss()
        self.audio_cap = AudioRecorder()
        self.boss_hp_end_history = []

        #Timer thread to keep track of time
        threading.Thread(target=timer_callback, args=(time.time(),)).start()

        #subprocess.Popen(['python', 'timer.py', '>', 'obs_timer.txt'])

    def grab_screen_shot(self):
        for num, monitor in enumerate(self.sct.monitors[1:], 1):
            # Get raw pixels from the screen
            sct_img = self.sct.grab(monitor)

            # Create the Image
            #decoded = cv2.imdecode(np.frombuffer(sct_img, np.uint8), -1)
            #BGRA to RGB conversion (NumPy array)
            return cv2.cvtColor(np.asarray(sct_img), cv2.COLOR_BGRA2RGB)

    def step(self, action):
        #model takes an action
        print('step start')

        #tensorboard logging - action per iteration
        self.logger.add_scalar('chosen_action', int(action), self.iteration)
        print(f"[ACTION SELECTED]: {int(action)}")

        #now time stamp
        t0 = time.time()
        if not self.first_step:
            # If the last step was more than 5 seconds ago, we need to reset the game
            if t0 - self.prev_step_end_ts > 5:
                headers = {"Content-Type": "application/json"}
                #requests.post(f"http://{self.agent_ip}:6000/recording/stop", headers=headers)
                requests.post(f"http://{self.agent_ip}:6000/action/custom/{4}", headers=headers)
                requests.post(f"http://{self.agent_ip}:6000/action/release_keys/{1}", headers=headers)
                time.sleep(10)
                requests.post(f"http://{self.agent_ip}:6000/action/return_to_grace", headers=headers)
                time.sleep(5)
                requests.post(f"http://{self.agent_ip}:6000/action/release_keys/{1}", headers=headers)
                self.done = True
            self.logger.add_scalar('time_between_steps', t0 - self.prev_step_end_ts, self.iteration)

        parry_reward = 0
        if int(action) == 9: #guard action
            # Start recording audio for parry detection
            self.audio_cap.start(self.iteration)
            self.parry_dict['parries'].append(time.time() - self.t_start)
        
        # if int(action) == 9:
        #     if self.t_since_parry is None or (time.time() - self.t_since_parry) > 2.1:
        #         headers = {"Content-Type": "application/json"}
        #         requests.post(f"http://{self.agent_ip}:6000/recording/start", headers=headers)
        #         self.t_since_parry = time.time()
        # if not self.t_since_parry is None and (time.time() - self.t_since_parry) > 2:
        #     headers = {"Content-Type": "application/json"}
        #     response = requests.post(f"http://{self.agent_ip}:6000/recording/stop", headers=headers)
        # headers = {"Content-Type": "application/json"}
        # response = requests.post(f"http://{self.agent_ip}:6000/recording/get_num_files", headers=headers)
        # if response.json()['num_files'] > 0:
        #     try:
        #         response = requests.post(f"http://{self.agent_ip}:6000/recording/get_parry", headers=headers)
        #         print(response.json())
        #         parry_sound_bytes = response.json()['parry_sound_bytes']
        #         decoded_bytes = base64.b64decode(parry_sound_bytes)
        #         byte_io = io.BytesIO(decoded_bytes)
        #         AudioSegment.from_raw(byte_io, 16000*2, 16000, 1).export(f'parries/{self.iteration}.wav', format='wav')
        #         audio = np.expand_dims(byte_io, axis=0)
        #         fft = audio_to_fft(audio)
        #         y_pred = self.parry_detector(fft)
        #         labels = np.squeeze(y_pred)
        #         index = np.argmax(labels, axis=0)
        #         if CLASS_NAMES[index] == 'successful_parries':
        #             parry_reward = 1
        #     except Exception as e:
        #         print(str(e))
        
        print('focus window')
        t1 = time.time()

        print('release keys')
        headers = {"Content-Type": "application/json"}
        requests.post(f"http://{self.agent_ip}:6000/action/release_keys", headers=headers)

        
        frame = self.grab_screen_shot()
        
        print('reward update')
        t2 = time.time()
        time_alive, percent_through, hp, self.death, dmg_reward, find_reward, time_since_boss_seen = self.rewardGen.update(frame)

        audio_buffer = self.audio_cap.get_audio()
        if not audio_buffer is None:
            audio_input = np.expand_dims(np.frombuffer(audio_buffer, dtype=np.int16), axis=-1)
            audio_input = np.expand_dims(audio_input, axis=0)
            audio_input = audio_input.astype(np.float32)
            #print(audio_input.shape)
            # FFT 결과 shape: (1, 2048, 1)
            fft = audio_to_fft(audio_input)

            # (1, 2048, 1) → (1, 2048)
            fft = tf.squeeze(fft, axis=-1)

            # (1, 2048) → (1, 512, 1)
            fft = tf.image.resize(tf.expand_dims(fft, -1), [512, 1])

            # (1, 512, 1) → (1, 512)
            fft = tf.reshape(fft, (1, 512))  # ✅ 이 줄로 shape을 정확히 명시
            print("Final FFT shape:", fft.shape)

            result = self.parry_detector.signatures["serving_default"](keras_tensor=tf.convert_to_tensor(fft, dtype=tf.float32))
            pred = result['output_0'].numpy()[0][0]

            pred = np.squeeze(pred)
            if pred > 0.99:
                parry_reward = 1
            # pred_idx = np.argmax(pred, axis=0)
            # print(pred_idx)
            # if CLASS_NAMES[int(pred_idx)] == 'successful_parries' and pred[int(pred_idx)] > 0.99:
            #     parry_reward = 1

        t3 = time.time()
        self.logger.add_scalar('time_alive', time_alive, self.iteration)
        self.logger.add_scalar('percent_through', percent_through, self.iteration)
        self.logger.add_scalar('hp', hp, self.iteration)
        self.logger.add_scalar('dmg_reward', dmg_reward, self.iteration)
        self.logger.add_scalar('find_reward', find_reward, self.iteration)
        self.logger.add_scalar('parry_reward', parry_reward, self.iteration)
        
        if hp > 0 and (time.time() - self.time_since_r) > 1.0:
            hp = 0

        self.reward = time_alive + percent_through + hp + dmg_reward + find_reward # + parry_reward

        print(f"[STEP STATUS] death: {self.death}, done: {self.done}, first_step: {self.first_step}, time_since_seen_boss: {self.rewardGen.time_since_seen_boss}, time limit: {time.time() - self.t_start} ")
        print(f"[STEP DECISION] using action: {action}")

        if not self.done:
            if self.death:
                print(f"[!] Agent died, but still processing this step for learning.")
            # Time limit for fighting Tree sentienel (600 seconds or 10 minutes)
            else:
               if (time.time() - self.t_start) > TOTAL_ACTIONABLE_TIME and self.rewardGen.time_since_seen_boss > 2.5:
                headers = {"Content-Type": "application/json"}
                requests.post(f"http://{self.agent_ip}:6000/action/custom/{4}", headers=headers)
                requests.post(f"http://{self.agent_ip}:6000/action/release_keys/{1}", headers=headers)
                time.sleep(1)
                requests.post(f"http://{self.agent_ip}:6000/action/return_to_grace", headers=headers)
                requests.post(f"http://{self.agent_ip}:6000/action/release_keys/{1}", headers=headers)
                self.done = True
                self.reward = -1
                self.rewardGen.time_since_death = time.time()
               else:
                if int(action) == 10:
                    self.time_since_r = time.time()
                headers = {"Content-Type": "application/json"}
                print(int(action))
                requests.post(f"http://{self.agent_ip}:6000/action/custom/{int(action)}", headers=headers)
                
                self.consecutive_deaths = 0
        else:
            headers = {"Content-Type": "application/json"}
            # if we load in and die with 5 seconds, restart game because we are frozen on a black screen
            if self.first_step:
                self.consecutive_deaths += 1
                if self.consecutive_deaths > 5:
                    self.consecutive_deaths = 0
                    headers = {"Content-Type": "application/json"}
                    requests.post(f"http://{self.agent_ip}:6000/action/stop_elden_ring", headers=headers)
                    time.sleep(2 * 60)
                    headers = {"Content-Type": "application/json"}
                    requests.post(f"http://{self.agent_ip}:6000/action/start_elden_ring", headers=headers)
                    time.sleep(100)

                    headers = {"Content-Type": "application/json"}
                    requests.post(f"http://{self.agent_ip}:6000/action/load_save", headers=headers)

                    headers = {"Content-Type": "application/json"}
                    requests.post(f"http://{self.agent_ip}:6000/action/return_to_grace", headers=headers)
            else:
                headers = {"Content-Type": "application/json"}
                #requests.post(f"http://{self.agent_ip}:6000/recording/stop", headers=headers)
                requests.post(f"http://{self.agent_ip}:6000/action/death_reset", headers=headers)
                for i in range(4):
                    requests.post(f"http://{self.agent_ip}:6000/action/custom/{4}", headers=headers)
                    requests.post(f"http://{self.agent_ip}:6000/action/release_keys/{1}", headers=headers)
                    time.sleep(1)

            self.done = True
        print('final steps')
        t4 = time.time()
        observation = cv2.resize(frame, (MODEL_WIDTH, MODEL_HEIGHT))
        info = {}
        self.iteration += 1

        if self.reward < -1:
            self.reward = -1
        if self.reward > 1:
            self.reward = 1

        if self.max_reward is None:
            self.max_reward = self.reward
        elif self.max_reward < self.reward:
            self.max_reward = self.reward

        self.logger.add_scalar('reward', self.reward, self.iteration)
        self.reward_history.append(self.reward)

        t_end = time.time()
        print("Iteration: {} took {:.2f} seconds".format(self.iteration, t_end - t0))
        print("t0-t1 took {:.5f} seconds".format(t1 - t0))
        print("t1-t2 took {:.5f} seconds".format(t2 - t1))
        print("t2-t3 took {:.5f} seconds".format(t3 - t2))
        print("t3-t4 took {:.5f} seconds".format(t4 - t3))
        print("t4-t_end took {:.5f} seconds".format(t_end - t4))
        self.last_fps.append(1 / (t_end - t0))
        desired_fps = (1 / 30) 
        time_to_sleep = desired_fps - (t_end - t0)
        #print(1 / (time.time() - t0))
        if time_to_sleep > 0:
            print(time_to_sleep)
            time.sleep(time_to_sleep)
        self.logger.add_scalar('step_time', (time.time() - t0), self.iteration)
        self.logger.add_scalar('FPS', 1 / (time.time() - t0), self.iteration)
        self.prev_step_end_ts = time.time()
        self.first_step = False
        if (self.iteration % 32768) == 0:
            json_message = {'text': 'Collecting rollout buffer'}
            headers = {"Content-Type": "application/json"}
            requests.post(f"http://{self.agent_ip}:6000/status/update", headers=headers, data=json.dumps(json_message))
        return observation, self.reward, self.done, info
    
    def reset(self):
        self.done = False
        #self.cap = ThreadedCamera('/dev/video0')
        headers = {"Content-Type": "application/json"}
        requests.post(f"http://{self.agent_ip}:6000/action/release_keys/{1}", headers=headers)
        requests.post(f"http://{self.agent_ip}:6000/action/custom/{4}", headers=headers)
        self.num_runs += 1
        self.logger.add_scalar('iteration_finder', self.iteration, self.num_runs)

        self.parry_dict['vod_duration'] = time.time() - self.t_start

        json_message = {'text': 'Check for frozen screen'}
        headers = {"Content-Type": "application/json"}
        requests.post(f"http://{self.agent_ip}:6000/status/update", headers=headers, data=json.dumps(json_message))

        avg_fps = 0
        for i in range(len(self.last_fps)):
            avg_fps += self.last_fps[i]
        if len(self.last_fps) > 0:
            avg_fps = avg_fps / len(self.last_fps)
        self.last_fps = []
        #requests.post(f"http://{self.agent_ip}:6000/action/death_reset", headers=headers)
        self.boss_hp_end_history.append(self.rewardGen.boss_hp)
        avg_boss_hp = 0
        if len(self.boss_hp_end_history) > 0:
            for i in range(len(self.boss_hp_end_history)):
                avg_boss_hp += self.boss_hp_end_history[i]
            avg_boss_hp /= len(self.boss_hp_end_history)
        json_message = {"death": self.death,
                        "reward": avg_fps,
                        "num_run": self.num_runs,
                        "lowest_boss_hp": avg_boss_hp}

        requests.post(f"http://{self.agent_ip}:6000/obs/log", headers=headers, data=json.dumps(json_message))

        frame = self.grab_screen_shot()
        # next_text_image = frame[1015:1040, 155:205]
        # next_text_image = cv2.resize(next_text_image, ((205-155)*3, (1040-1015)*3))
        # next_text = pytesseract.image_to_string(next_text_image,  lang='eng',config='--psm 6 --oem 3')
        # loading_screen = "Next" in next_text
        loading_screen_history = []
        max_loading_screen_len = 30 * 15
        time.sleep(2)
        requests.post(f"http://{self.agent_ip}:6000/action/release_keys", headers=headers)
        requests.post(f"http://{self.agent_ip}:6000/action/custom/{4}", headers=headers)
        t_check_frozen_start = time.time()
        t_since_seen_next = None
        while True:
            frame = self.grab_screen_shot()
            next_text_image = frame[1120:1150, 160:260]  # Y, X
            next_text_image = cv2.resize(next_text_image, ((260 - 160) * 3, (1150 - 1120) * 3))
            next_text = pytesseract.image_to_string(next_text_image,  lang='kor+eng',config='--psm 6 --oem 3')
            cv2.imwrite(f"debug_hp/boss_name_debug{self.debug_idx}.png", next_text_image)
            self.debug_idx += 1
            loading_screen = "다음" in next_text #next
            if loading_screen:
                t_since_seen_next = time.time()
            if not t_since_seen_next is None and ((time.time() - t_check_frozen_start) > 7.5) and (time.time() - t_since_seen_next) > 7.5:
                break
            elif not t_since_seen_next is None and  ((time.time() - t_check_frozen_start) > 30):
                break
            elif t_since_seen_next is None and ((time.time() - t_check_frozen_start) > 20):
                break
        self.logger.add_scalar('check_frozen_time', time.time() - t_check_frozen_start, self.num_runs)

        # This didnt work :(
        lost_connection_image = frame[475:550, 675:1250]
        lost_connection_image = cv2.resize(lost_connection_image, ((1250-675)*3, (550-475)*3))
        lost_connection_text = pytesseract.image_to_string(lost_connection_image,  lang='eng',config='--psm 6 --oem 3')
        lost_connection_words = ["connection", "game", "server", "lost"]
        lost_connection = False
        for word in lost_connection_words:
            if word in lost_connection_text:
                lost_connection = True
        
        headers = {"Content-Type": "application/json"}
        response = requests.post(f"http://{self.agent_ip}:6000/action/check_er", headers=headers)

        data = response.json()
        print(response.text)
        if ((time.time() - t_check_frozen_start) > 30) or lost_connection or (not data.get('ER', False)):
            print(f"Lost connection: {lost_connection}")
            print(f"Loading Screen Length: {len(loading_screen_history)}")
            print(f"Check ER: {not data.get('ER', False)}")

            headers = {"Content-Type": "application/json"}
            requests.post(f"http://{self.agent_ip}:6000/action/stop_elden_ring", headers=headers)
            time.sleep(2 * 60)
            headers = {"Content-Type": "application/json"}
            requests.post(f"http://{self.agent_ip}:6000/action/start_elden_ring", headers=headers)
            time.sleep(100)

            headers = {"Content-Type": "application/json"}
            requests.post(f"http://{self.agent_ip}:6000/action/load_save", headers=headers)

            headers = {"Content-Type": "application/json"}
            requests.post(f"http://{self.agent_ip}:6000/action/return_to_grace", headers=headers)


        #headers = {"Content-Type": "application/json"}
        #requests.post(f"http://{self.agent_ip}:6000/recording/tag_latest/{self.max_reward}/{self.num_runs}'", headers=headers, data=json.dumps(self.parry_dict))

        observation = cv2.resize(frame, (MODEL_WIDTH, MODEL_HEIGHT))
        self.done = False
        self.first_step = True
        self.locked_on = False
        self.rewardGen.curr_boss_hp = 3200
        self.max_reward = None
        self.rewardGen.seen_boss = False
        self.rewardGen.time_since_seen_boss = time.time()
        self.rewardGen.prev_hp = self.rewardGen.max_hp
        self.rewardGen.curr_hp = self.rewardGen.max_hp
        self.rewardGen.time_since_reset = time.time()
        self.rewardGen.boss_hp = 1
        if len(self.reward_history) > 0:
            total_r = 0
            for r in self.reward_history:
                total_r += r
            avg_r = total_r / len(self.reward_history)
            self.logger.add_scalar('average_reward_per_run', avg_r, self.num_runs)
        self.reward_history = []
        self.parry_dict = {'vod_duration':None,
                           'parries': []}

        self.t_start = time.time()
        headers = {"Content-Type": "application/json"}
        #requests.post(f"http://{self.agent_ip}:6000/recording/start", headers=headers)

        time.sleep(0.5)
        headers = {"Content-Type": "application/json"}
        requests.post(f"http://{self.agent_ip}:6000/action/release_keys", headers=headers)
        requests.post(f"http://{self.agent_ip}:6000/action/custom/{4}", headers=headers)
        requests.post(f"http://{self.agent_ip}:6000/action/init_fight", headers=headers)

        json_message = {'text': 'Step'}
        headers = {"Content-Type": "application/json"}
        requests.post(f"http://{self.agent_ip}:6000/status/update", headers=headers, data=json.dumps(json_message))

        return observation

    def render(self, mode='human'):
        pass

    def close (self):
        self.cap.release()





