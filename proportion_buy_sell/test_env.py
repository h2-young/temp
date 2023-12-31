import src.env.env_bs as env
import src.utils.utils as utils

urls = './data/google.csv'

stock_info = utils.load_data(urls)

env = env.MyEnv(stock_info)

env.reset()

while True:
    while env._done is False:
        user_input = input("Enter your action \"\": ")
        env.step(int(user_input))
        env.print_current_state()

    env.reset()