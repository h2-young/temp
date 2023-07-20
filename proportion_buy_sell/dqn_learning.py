import src.env.env_bs as env
import src.utils.utils as utils
import pandas as pd

urls = './data/hyundai_train.csv'

DEFAULT_OPTION = {
    'initial_balance_coef': 10,  # 초기 자본금 계수, 초기 자본금 = 최대종가 * 비율정밀도 * 계수, 1일 때, 최대가격을 기준으로 모든 비율의 주식을 구매할 수 있도록 함
    'start_index': 0,  # 학습 시작 인덱스
    'end_index': 500,  # 학습 종료 인덱스
    'window_size': 20,  # 학습에 사용할 데이터 수, 최근 수치에 따라 얼마나 많은 데이터를 사용할지 결정
    'proportion_precision': 4,  # 비율 정밀도 (결정할 수 있는 비율의 수) 20이면 0.05 단위로 결정
    'commission': .0003,  # 수수료
    'selling_tax': .00015,  # 매도세
    'reward_threshold': 0.03,  # 보상 임계값 : 수익률이 이 값을 넘으면 보상을 1로 설정
}

# default_options = {
#     'initial_balance': 100_000_000,
#     'max_balance': 1_000_000_000,
#     'commission': .003,
#     'selling_tax': .00015,
# }

stock_info = utils.load_data(urls)

env = env.MyEnv(stock_info, option=DEFAULT_OPTION)

from stable_baselines3 import DQN

model = DQN('MultiInputPolicy', env, verbose=1, tensorboard_log="./logs/DQN/")

model.learn(total_timesteps=100000, log_interval=1, progress_bar=True)

model.save("./models/DQN_hyundai_model.zip")
