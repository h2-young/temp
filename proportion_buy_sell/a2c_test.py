import src.env.env_bs as env
import src.utils.utils as utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

from stable_baselines3 import A2C

model = A2C.load("./models/A2C_google_model.zip", env)
ITER_LIMIT = 365 * 3

obs = env.reset()

rewards = []
profits = []

print(obs)

for i in range(ITER_LIMIT):
    action, _ = model.predict(obs, deterministic=False)

    print(i + 1, "번째 action : ", action)

    obs, reward, done, info = env.step(action)

    # print(i + 1, "번째 observation : ", obs)
    print(i + 1, "번째 reward : ", reward)
    print(i + 1, "번째 info : ", info, obs['holding'])

    rewards.append(reward)
    profits.append(info['profit'])

    if i > 567: # 테스트용 주식 데이터가 567일
        done = True

    if done:
        obs = env.reset()
        print("Episode done!")
        print("final reward : ", reward)

        break

# reward 그래프 그리기
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Reward Graph')
plt.show()

plt.plot(profits)
plt.xlabel('Episode')
plt.ylabel('Profit')
plt.title('Profit Graph')
plt.show()

