from itertools import count
import numpy as np

def train(env, num_episodes, agent1, agent2):

    for i_episode in range(num_episodes):
        state, info = env.reset()
        parameters = dict()

        loss_mean = 0
        cum_reward = 0
        cum_discounted_reward = 0
        cum_gamma = agent1.gamma
        for t in count():
            agent1.step += 1
            agent2.step += 1

            action1 = agent1.select_action(state)
            action2 = agent2.select_action(state)
            action = action1 + 4 * action2

            observation, reward, terminated, truncated, _ = env.step(action)

            cum_reward += reward
            cum_discounted_reward += cum_gamma * reward
            cum_gamma *= agent1.gamma
            done = terminated or truncated
            next_state = None if terminated else observation

            # Store the transition in memory
            agent1.store_transition(state, action1, next_state, reward)
            agent2.store_transition(state, action2, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            agent1.optimize_model()
            loss_mean += (agent1.loss.item() - loss_mean) / (t + 1)
            agent1.update_network()

            agent2.optimize_model()
            loss_mean += (agent2.loss.item() - loss_mean) / (t + 1)
            agent2.update_network()

            if done:
                agent1.steps_done += 1
                agent2.steps_done += 1
                parameters = {
                    'loss': loss_mean,
                    'episode_len': t + 1,
                    'cum_reward': cum_reward,
                    'disc_reward': cum_discounted_reward,
                    'i_episode': i_episode
                    }
                
                agent1.output_writer(parameters)
                agent2.output_writer(parameters)

                # Saving and evaluating the model
                saved = agent1.save_model(i_episode, name='agent1')
                saved = agent2.save_model(i_episode, name='agent2')
                if saved or i_episode % 1000 == 0:
                    evaluate(50, i_episode, env, agent1, agent2)   
                break
    agent1.writer.close()
    agent2.writer.close()

def evaluate(num_episodes, episode_number, env, agent1, agent2):
    agent1.evaluate_flag = True
    agent2.evaluate_flag = True
    history_cum_reward = []
    history_cum_disc_reward = []
    for i_episode in range(num_episodes):
        state, info = env.reset()

        cum_reward = 0
        cum_discounted_reward = 0
        cum_gamma = agent1.gamma
        for t in count():
            action1 = agent1.select_action(state)
            action2 = agent2.select_action(state)
            action = action1 + 4 * action2

            observation, reward, terminated, truncated, _ = env.step(action)

            cum_reward += reward
            cum_discounted_reward += cum_gamma * reward
            cum_gamma *= agent1.gamma
            done = terminated or truncated

            next_state = None if terminated else observation 

            # Move to the next state
            state = next_state

            if done:
                history_cum_reward.append(cum_reward)
                history_cum_disc_reward.append(cum_discounted_reward)
                break
    agent1.writer.add_scalar('reward/evaluated', np.mean(history_cum_reward), episode_number)
    agent1.writer.add_scalar('disc_reward/evaluated', np.mean(cum_discounted_reward), episode_number)
    agent1.evaluate_flag = False

    agent2.writer.add_scalar('reward/evaluated', np.mean(history_cum_reward), episode_number)
    agent2.writer.add_scalar('disc_reward/evaluated', np.mean(cum_discounted_reward), episode_number)
    agent2.evaluate_flag = False