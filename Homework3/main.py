import argparse
from test import test
from environment import Environment


def parse():
    parser = argparse.ArgumentParser(description="ADL HW3")
    parser.add_argument('--env_name', default=None, help='environment name')
    parser.add_argument('--train_pg', action='store_true', help='whether train policy gradient')
    parser.add_argument('--train_dqn', action='store_true', help='whether train DQN')
    parser.add_argument('--test_pg', action='store_true', help='whether test policy gradient')
    parser.add_argument('--test_dqn', action='store_true', help='whether test DQN')
    parser.add_argument('--video_dir', default=None, help='output video directory')
    parser.add_argument('--do_render', action='store_true', help='whether render environment')

    args = parser.parse_args()
    return args


def run(args):
    if args.train_pg:
        env_name = args.env_name or 'LunarLander-v2'
        env = Environment(env_name, args, atari_wrapper=False)
        from agent.agent_pg import AgentPG
        agent = AgentPG(env, args)
        agent.train()

    if args.train_dqn:
        env_name = args.env_name or 'MsPacmanNoFrameskip-v0'
        env = Environment(env_name, args, atari_wrapper=True)
        from agent.agent_dqn import AgentDQN
        agent = AgentDQN(env, args)
        agent.train()

    if args.test_pg:
        env = Environment('LunarLander-v2', args, test=True)
        from agent.agent_pg import AgentPG
        agent = AgentPG(env, args)
        test(agent, env, total_episodes=30)

    if args.test_dqn:
        env = Environment('MsPacmanNoFrameskip-v0', args, atari_wrapper=True, test=True)
        from agent.agent_dqn import AgentDQN
        agent = AgentDQN(env, args)
        test(agent, env, total_episodes=100)


if __name__ == '__main__':
    args = parse()
    run(args)

