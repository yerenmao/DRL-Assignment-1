# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym

import math

# global variables
first = True
stations = None
passenger = None
destination = None
agent_station = 0

# Q-Table
q_table = np.load("q-table.npy", allow_pickle=True).item()


# functions
def closest(curr, remaining):
    return min(
        remaining,
        key=lambda coord: math.sqrt(
            (coord[0] - curr[0]) ** 2 + (coord[1] - curr[1]) ** 2
        ),
    )


def greedy_sort_stations(curr, stations):
    sorted_stations = []
    remaining = stations[:]
    while remaining:
        next_station = closest(curr, remaining)
        sorted_stations.append(next_station)
        remaining.remove(next_station)
        curr = next_station
    return sorted_stations


def get_direction_vector(curr, target):
    dx, dy = target[0] - curr[0], target[1] - curr[1]
    dx = 1 if dx > 0 else (-1 if dx < 0 else 0)
    dy = 1 if dy > 0 else (-1 if dy < 0 else 0)
    return (dx, dy)


def get_station(agent_station):
    return (
        stations[agent_station]
        if 0 <= agent_station <= 3
        else random.choice[agent_station]
    )


def get_target():
    # 在各種不同情況下，選出相對應的 target
    if passenger is None:  # 不知道乘客在哪
        target = get_station(agent_station)
    else:  # 知道乘客在哪 (知道的那瞬間就應該 pickup)
        if destination is None:  # 不知道終點在哪
            target = get_station(agent_station)
        else:  # 知道終點在哪
            target = destination
    return target


# get_state 期望的是每次執行 action 前被呼叫，其他時候請不要呼叫
# vector[0], vector[1], obs⬆, obs_⬇, obs_r, obs_l, agent_status (0: normal, 1: PICKUP, 2: DROPOFF)
def get_state(obs):
    global first, stations, passenger, destination, agent_station

    curr = (obs[0], obs[1])
    if first:  # 記錄 s1, s2, s3, s4，並使用 greedy nearest-neighbor sorting
        stations = [(obs[i], obs[i + 1]) for i in range(2, 9, 2)]
        stations = greedy_sort_stations(curr, stations)
        first = False

    target = get_target()

    # 計算 target 的 direction vector
    vector = get_direction_vector(curr, target)

    # 如果 vector 為 (0, 0)，代表該位置有 passenger 或 destination：
    # 如果是 passenger，應該馬上進行 PICKUP
    # 如果是 destination：
    # 如果知道 passenger，應該要在車上，馬上進行 DROPOFF
    # 如果不知道 passenger，直接換到下一個 target
    # 否則當到達 (0, 0)，馬上給他下一個 station 的 target
    agent_status = 0
    if vector == (0, 0) and curr in stations:  # should happen together
        if obs[14] == 1 and passenger is None:  # 目前在有 passenger 的 station
            passenger = curr  # 代表已經經過 passenger 了
            agent_status = 1  # 強迫 agent PICKUP
            if agent_station < 3:
                agent_station += 1
        elif obs[15] == 1:  # 目前在 destination 的 station
            destination = curr  # 強迫 agent
            if passenger is not None:
                agent_status = 2  # 強迫 agent DROPOFF
            else:
                if agent_station < 3:
                    agent_station += 1
                    vector = get_direction_vector(curr, get_station(agent_station))
        else:
            if agent_station < 3:
                agent_station += 1
                vector = get_direction_vector(curr, get_station(agent_station))
    return (vector[0], vector[1], obs[10], obs[11], obs[12], obs[13], agent_status)


def get_action(obs):
    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys.
    #       Otherwise, even if your agent performs well in training, it may fail during testing.
    state = get_state(obs)
    action = np.argmax(q_table[state])
    return action  # , state
    # You can submit this random agent to evaluate the performance of a purely random strategy.


"""
    obs is a 16 tuple like this:
    (1, 1, 0, 0, 0, 4, 4, 0, 4, 4, 0, 0, 0, 0, 0, 0)

    texi_pos = (obs[0], obs[1])
    stations[0] = (obs[2], obs[3])
    stations[1] = (obs[4], obs[5])
    stations[2] = (obs[6], obs[7])
    stations[3] = (obs[8], obs[9])
    obstacle_north = obs[10]    上方有沒有障礙物
    obstacle_south = obs[11]    下方有沒有障礙物
    obstacle_east = obs[12]     右方有沒有障礙物
    obstacle_west = obs[13]     左方有沒有障礙物
    passenger_look              乘客是否在「上、下、左、右、中」其中之一
    destination_look            乘客是否在「上、下、左、右、中」其中之一
    """
