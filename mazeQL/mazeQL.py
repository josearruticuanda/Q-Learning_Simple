#  Import Libraries
import numpy as np
import pandas as pd
import random

# Rewards / Rows = Rooms / Columns = Actions possible in room
rewards = np.array([
    [-1, -1, -1, 0, -1, -1, -1, -1, -1],
    [-1, -1, 0, -1, 0, -1, -1, -1, -1],
    [-1, 0, -1, -1, -1, -1, -1, -1, -1],
    [0, -1, -1, -1, 0, -1, -1, -1, -1],
    [-1, 0, -1, 0, -1, -1, -1, 0, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, 100],
    [-1, -1, -1, -1, -1, -1, -1, 0, -1],
    [-1, -1, -1, -1, 0, -1, 0, -1, 100],
    [-1, -1, -1, -1, -1, 0, -1, 0, -1]
])

# Using pandas for display
# print(pd.DataFrame(rewards))

# Initialize Q Table
def initialize_q(m, n):
    return np.zeros((m, n))

q_table = initialize_q(9, 9)
# print(pd.DataFrame(q_table))

# Set initial state/room
def set_initial_state(rooms = 9):
    return np.random.randint(0, rooms)

def get_action(current_state, reward_table):
    valid_actions = []
    for action in enumerate(reward_table[current_state]):
        if action[1] != -1:
            valid_actions += [action[0]]

    return random.choice(valid_actions)

def take_action(current_state, reward_table, gamma, verbose = False):

    # Take a single action
    action = get_action(current_state, reward_table)
    sa_reward = reward_table[current_state, action] # Current state-action reward
    ns_reward = max(q_table[action, ]) # Next state-action reward
    print(ns_reward)
    q_current_state = sa_reward + (gamma * ns_reward)
    q_table[current_state, action] = q_current_state # Mutate q_table
    new_state = action

    if verbose:
        print(q_table)
        print(f"Old state: {current_state} | New state: {new_state}\n\n")
        if new_state == 8:
            print("Agent has reached it's goal!")

    return new_state

def initialize_episode(reward_table, initial_state, gamma, verbose = False):

    current_state = initial_state
    while True:
        current_state = take_action(current_state, reward_table, gamma, verbose)
        if current_state == 8:
            break

def train_agent(iterations, reward_table, gamma, verbose = False):
    for episode in range(iterations):
        initial_state = set_initial_state()
        initialize_episode(reward_table, initial_state, gamma, verbose)
    print("\nTraining complete.\n")

    return q_table

def normalize_table(q_table):
    normalized_q = q_table / max(q_table[q_table.nonzero()]) * 100
    return normalized_q.astype(int)

# Test run of full training
gamma = 0.8
initial_state = set_initial_state()
initialize_action = get_action(initial_state, rewards)

q_matrix = train_agent(2000, rewards, gamma, False)

print(pd.DataFrame(q_matrix))

def deploy_agent(init_state, q_matrix):
    print("\n\nStart: ", init_state)
    state = init_state
    steps = 0
    while True:
        steps += 1
        action = np.argmax(q_matrix[state,:])
        print(action)
        state = action
        if action == 8:
            print("Finished!")
            return steps

start_room = 0
steps = deploy_agent(start_room, q_matrix)
print("Number of rooms/actions: ", steps)
