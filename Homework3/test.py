import argparse
import numpy as np
from environment import Environment

seed = 11037

def parse():
    parser = argparse.ArgumentParser(description="ADL HW3")
    parser.add_argument('--test_pg', action='store_true', help='whether test policy gradient')
    parser.add_argument('--test_dqn', action='store_true', help='whether test DQN')
    parser.add_argument('--video_dir', default=None, help='output video directory')
    parser.add_argument('--do_render', action='store_true', help='whether render environment')
    args = parser.parse_args()
    return args


def test(agent, env, total_episodes=30):
    rewards = []
    env.seed(seed)
    for i in range(total_episodes):
        state = env.reset()
        agent.init_game_setting()
        done = False
        episode_reward = 0.0

        # playing one game
        while not done:
            action = agent.make_action(state, test=True)
            state, reward, done, info = env.step(action)
            episode_reward += reward

        rewards.append(episode_reward)
    print('Run %d episodes'%(total_episodes))
    print('Mean:', np.mean(rewards))
    print('Median:', np.median(rewards))

def run(args):
    if args.test_pg:
        env = Environment('LunarLander-v2', args, test=True)
        from agent.agent_pg import AgentPG
        agent = AgentPG(env, args)
        test(agent, env)

    if args.test_dqn:
        env = Environment('MsPacmanNoFrameskip-v0', args, atari_wrapper=True, test=True)
        from agent.agent_dqn import AgentDQN
        agent = AgentDQN(env, args)
        test(agent, env, total_episodes=50)

if __name__ == '__main__':
    args = parse()
    run(args)

