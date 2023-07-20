import optuna
import numpy as np
from stable_baselines3 import DQN
import src.env.env_bs as env
import src.utils.utils as utils

urls = './data/google_test.csv'

DEFAULT_OPTION = {
        'initial_balance_coef': 10,  # 초기 자본금 계수, 초기 자본금 = 최대종가 * 비율정밀도 * 계수, 1일 때, 최대가격을 기준으로 모든 비율의 주식을 구매할 수 있도록 함
        'start_index': 0,  # 학습 시작 인덱스
        'end_index': 200,  # 학습 종료 인덱스
        'window_size': 20,  # 학습에 사용할 데이터 수, 최근 수치에 따라 얼마나 많은 데이터를 사용할지 결정
        'proportion_precision': 4,  # 비율 정밀도 (결정할 수 있는 비율의 수) 20이면 0.05 단위로 결정
        'commission': .0003,  # 수수료
        'selling_tax': .00015,  # 매도세
        'reward_threshold': 0.03,  # 보상 임계값 : 수익률이 이 값을 넘으면 보상을 1로 설정
    }

stock_info = utils.load_data(urls)

env = env.MyEnv(stock_info, option=DEFAULT_OPTION)

param = {
        # 'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
        # 'buffer_size': trial.suggest_categorical('buffer_size', [100_000, 500_000, 1_000_000]),
        'batch_size' : [16, 32, 64, 128],
        'buffer_size' : [100_000, 500_000, 1_000_000]
}

# DQN 모델 생성 및 학습
model = DQN('MultiInputPolicy', env, verbose=1, **param)
model.learn(total_timesteps=10000)  # 적절한 학습 시간으로 조정

# def objective(trial):
#     # 하이퍼파라미터 탐색 공간 정의
#     param = {
#         # 'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
#         # 'buffer_size': trial.suggest_categorical('buffer_size', [100_000, 500_000, 1_000_000]),
#         'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
#         'buffer_size': trial.suggest_categorical('buffer_size', [100_000, 500_000, 1_000_000]),
#     }
#
#     # DQN 모델 생성 및 학습
#     model = DQN('MultiInputPolicy', env, verbose=1, **param)
#     model.learn(total_timesteps=10000)  # 적절한 학습 시간으로 조정
#
#     # 평가 지표로서 최대 보상 값을 반환
#     return model
#
# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=10)  # 적절한 시행 횟수로 조정
#
# print("Best Parameters: ", study.best_params)
# print("Best Reward: ", study.best_value)
