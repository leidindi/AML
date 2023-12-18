import itertools as it
from abc import ABC, abstractmethod
from enum import IntEnum
from typing import List, Tuple

import gymnasium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces
from IPython import display
from IPython.display import HTML
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from scipy import stats
from tqdm.notebook import tqdm  # For progress bars
import pickle 

# %matplotlib inline
# Constants defining the environment
GOAL = (140, 120)
CENTER = (132, 132)
AVG_MOVEMENT_SIZE = 24
ACCEPTABLE_DISTANCE_TO_GOAL = (AVG_MOVEMENT_SIZE // 2) + 1
RADIUS = 72
WINDOW_SIZE = 28
TIME_LIMIT = 200
TIMEOUT_REWARD = -100.0
MOVE_REWARD = -1
INVALID_MOVE_REWARD = -5


# This is for type inference
State = Tuple[int, int]


# Action space
class Actions(IntEnum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3


# Boundaries
class Boundary(IntEnum):
    WEST = CENTER[0] - RADIUS
    EAST = CENTER[0] + RADIUS
    NORTH = CENTER[1] - RADIUS
    SOUTH = CENTER[1] + RADIUS

# Augmented boundarie, not used in this assignment
class AugmentedArea(IntEnum):
    WEST = Boundary.WEST - (WINDOW_SIZE // 2)
    EAST = Boundary.EAST + (WINDOW_SIZE // 2)
    NORTH = Boundary.NORTH - (WINDOW_SIZE // 2)
    SOUTH = Boundary.SOUTH + (WINDOW_SIZE // 2)


# Image
ORIGINAL_IMAGE = plt.imread("UU_LOGO.png")
# Convert to one color channel (using only the red channel), with white background
IMAGE = ORIGINAL_IMAGE[:, :, 0] * ORIGINAL_IMAGE[:, :, 3] + (1.0 - ORIGINAL_IMAGE[:, :, 3])


# Get a "camera view" at the position indicated by state
# Use reshape=True to format the output as a data point for the neural network
def get_window(state: State, reshape=False) -> np.ndarray:
    # When indexing the image as an array, switch the coordinates: im[state[1], state[0]]
    window = IMAGE[(state[1] - 14):(state[1] + 14), (state[0] - 14):(state[0] + 14)]
    if reshape:
        return np.reshape(window, (1, 28, 28, 1))
    return window


# Is the state close enough to the goal to be considered a success?
# There is a margin for error, so that the agent can't jump over the goal
def is_goal_reached(state: State) -> bool:
    if np.amax(np.abs(np.asarray(state) - np.asarray(GOAL))) <= AVG_MOVEMENT_SIZE / 2 + 1:
        return True
    return False


# This is a helper function to render a run
def updatefig(j, images, imgplot, text_act_plot, text_reward_plot):
    # set the data in the axesimage object
    img, time_point, from_state, to_state, act, current_reward = images[min(len(images), j)]
    imgplot.set_data(img)
    text_act_plot.set_text(f"Time step: {time_point} - Action: {act}\nState: {from_state} -> {to_state}")
    text_reward_plot.set_text(f"Current total reward: {current_reward}")
    # return the artists set
    return [imgplot, text_act_plot]


# This will render a run of a full epoch
# The function needs a list of tuples containing an image array, a State, the performed action
def render_epoch(animation_data: List[Tuple[np.ndarray, State, Actions]], interval=100, blit=True, **kwargs):
    if not len(animation_data):
        return f"No images in the list"
    fig, ax = plt.subplots()
    imgplot = ax.imshow(np.zeros_like(animation_data[0][0]))
    text_act_plot = ax.set_title("", color="red", fontweight="extra bold", loc="left")
    text_reward_plot = ax.text(5, 255, "", color="red", fontweight="extra bold")
    params = [animation_data, imgplot, text_act_plot, text_reward_plot]
    ani = FuncAnimation(fig,
                        updatefig,
                        fargs=params,
                        frames=len(animation_data),
                        interval=interval,
                        blit=blit,
                        **kwargs)
    animation = HTML(ani.to_jshtml())
    plt.close()
    return display.display(animation)

# This function can be used to smooth obtained plots
def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    return np.convolve(y, box, mode='valid')

class grid_tilings():
    def __init__(self, number_of_grids = 1, offsets = np.array([[0.0, 0.0]])) -> None:       
        # Low value for each dimension, for each grid / tile
        lows = np.array([[Boundary.NORTH, Boundary.WEST]]*number_of_grids)
        # High value for each dimension, for each grid / tile
        highs = np.array([[Boundary.SOUTH, Boundary.EAST]]*number_of_grids)
        # Number of discrete bin for each each dimension, for each grid / tile
        bins = np.array([[9, 9]]*number_of_grids)
        # The offset is used to setup the overlap of grids
        # offsets = np.array([[0.0, 0.0]]) is one grid starting from lows to highs 
        # offsets = np.array([[20.0, 20.0]]) is one grid starting from lows+[20.0, 20.0] to highs+[20.0, 20.0]

        self.grids = []

        for l, h, b, o in zip(lows, highs, bins, offsets):
            grid = {}
            grid['size']  = b
            grid['low'] = l
            grid['offset'] = o
            grid['points'] = []
            grid['step'] = []
            for dim in range(len(b)):
                points, step = np.linspace(l[dim], h[dim], b[dim]+1, endpoint=False, retstep=True)
                points += o[dim]
                grid['points'].append(points)
                grid['step'].append(step)

            grid['step'] = np.array(grid['step'])
            grid['weights'] = np.zeros(grid['size'])
            self.grids.append(grid)

    # Get the sum of the weights for given continuous coordinates
    def get_weight(self, sample):
        encoded_sample = self.tile_encode(sample)
        w = 0.0
        for grid, (x,y) in zip(self.grids, encoded_sample):
            w += grid['weights'][x,y]
        return w

    # Set the weights for given continuous coordinates
    def set_weight(self, sample, target):
        encoded_sample = self.tile_encode(sample)
        for grid, (x,y) in zip(self.grids, encoded_sample):
            grid['weights'][x,y] = target/len(self.grids)

    # Return the discrete coordinates from continuous coordinates for all grids
    def tile_encode(self, sample):
        encoded_sample = []
        for grid in self.grids:
            encoded_sample.append(self.discretize(sample, grid))
        return encoded_sample    
    
    # Return the discrete coordinates from continuous coordinates for a given grid
    def discretize(self, sample, grid):
        sample = np.array(sample) - (grid['low'] + grid['offset'])
        sample = np.maximum(sample, np.array([0]*len(grid['size'])))
        index = sample // grid['step']
        index = np.minimum(index, grid['size']-1)
        
        return list(index.astype(int))

    # Plot the different grids
    def visualize_tilings(self):
        """Plot each tiling as a grid."""
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        linestyles = ['-', '--', ':']
        legend_lines = []
        
        for i, grid in enumerate(self.grids):
            for x in grid['points'][0]:
                l = plt.axvline(x=x, color=colors[i % len(colors)], linestyle=linestyles[i % len(linestyles)], label=i)
            for y in grid['points'][1]:
                l = plt.axhline(y=y, color=colors[i % len(colors)], linestyle=linestyles[i % len(linestyles)])
            legend_lines.append(l)
        plt.legend(legend_lines, ["Tiling #{}".format(t) for t in range(len(legend_lines))], facecolor='white', framealpha=0.9)
        plt.title("Tilings")


# default setting for 3 tilings:
offsets = np.array([[0.0, 0.0], [20.0, 20.0], [-20.0, 15.0]])
tilings = grid_tilings(3, offsets)

# example for one tiling
# tilings = grid_tilings()

plt.imshow(IMAGE[:, :], cmap='gray', vmin=0, vmax=1.0)
tilings.visualize_tilings()

# Test with some sample values
samples = np.random.rand(5, 2) * 264
sampls = np.random.rand(1,2) * 264
print("\nSamples:", samples, sep="\n")
encoded_samples = [tilings.tile_encode(sample) for sample in samples]
print("\nIndexes of samples:", *[s for s in encoded_samples], sep="\n")

class Environment(gymnasium.Env):
    metadata = {'render.modes': ['human', 'rgba_array']}
    bx = np.array([AugmentedArea.WEST, AugmentedArea.WEST, AugmentedArea.EAST, AugmentedArea.EAST, AugmentedArea.WEST])
    by = np.array([AugmentedArea.NORTH, AugmentedArea.SOUTH, AugmentedArea.SOUTH, AugmentedArea.NORTH, AugmentedArea.NORTH])

    def __init__(self):
        self.num_actions = Actions
        self.action_space = spaces.Discrete(len(self.num_actions))
        self.observation_space = spaces.Discrete(1)
        self.display = None
        self.img, self.img_container = Environment._init_visual_area(IMAGE)
        # self.time shoulde be initialized at 0 !!!!!!!!!!!!! ==================== !!!!!!!!!!!!!!!!!!
        self.time = 0

    def seed(self, seed=None) -> int:
        np.random.seed(seed)
        return seed

    def step(self, action: Actions):
        assert self.action_space.contains(action)
        (x, y), was_invalid = self._validate_state(self._move(self.state, action))

        self.state = (x, y)
        reward = MOVE_REWARD if not was_invalid else INVALID_MOVE_REWARD
        reward = TIMEOUT_REWARD if self.time >= TIME_LIMIT else reward
        reward = self.time * reward
        done = is_goal_reached(self.state) or self.time >= TIME_LIMIT
        self.time += 1
        return self.state, reward, done, {}

    def reset(self, state: State = None) -> State:
        self.state = self.starting_state() if not state else state
        # self.time shoulde be initialized at 0 !!!!!!!!!!!!! ==================== !!!!!!!!!!!!!!!!!!
        self.time = 0
        return self.state

    # returns the current environment situation
    def render(self, mode='rgba_array'):
        curr_img = np.array(self.img_container.get_array())
        x, y = self.state
        scaler = 4
        w, e, n, s = x - scaler, x + scaler, y - scaler, y + scaler
        curr_img[n:s, w:e, 0] = 255
        curr_img[n:s, w:e, 1] = 0
        curr_img[n:s, w:e, 2] = 255
        curr_img[n:s, w:e, 3] = 255
        cropped_img = curr_img  # Just for debugging purposes
        if mode == 'rgba_array':
            plt.close()
            return cropped_img  # return RGB frame suitable for video
        if mode == 'human':
            container = plt.imshow(curr_img)
            ax = container.axes
            ax.set_xlim(Boundary.WEST, Boundary.EAST, auto=None)
            ax.set_ylim(Boundary.SOUTH, Boundary.NORTH, auto=None)
            return container
        else:
            raise Exception(f"Please specify either 'rgba_array' or 'human' as mode parameter!")

    # Return a randomly chosen non-terminal state as starting state
    def starting_state(self) -> State:
        while True:
            state = (
                np.random.randint(Boundary.WEST, Boundary.EAST + 1),
                np.random.randint(Boundary.NORTH, Boundary.SOUTH + 1),
            )
            if not is_goal_reached(state):
                return state

    @staticmethod
    def _move(state: State, action: Actions) -> State:
        x, y = state
        if action == Actions.NORTH:
            y -= AVG_MOVEMENT_SIZE + np.random.randint(5) - 2
        elif action == Actions.SOUTH:
            y += AVG_MOVEMENT_SIZE + np.random.randint(5) - 2
        elif action == Actions.WEST:
            x -= AVG_MOVEMENT_SIZE + np.random.randint(5) - 2
        elif action == Actions.EAST:
            x += AVG_MOVEMENT_SIZE + np.random.randint(5) - 2
        return x, y

    @staticmethod
    def _validate_state(state: State) -> Tuple[State, bool]:
        x, y = state
        is_invalid = False
        if y < Boundary.NORTH:
            is_invalid = True
            y = int(Boundary.NORTH)
        if y > Boundary.SOUTH:
            is_invalid = True
            y = int(Boundary.SOUTH)
        if x < Boundary.WEST:
            is_invalid = True
            x = int(Boundary.WEST)
        if x > Boundary.EAST:
            is_invalid = True
            x = int(Boundary.EAST)
        return (x, y), is_invalid

    @staticmethod
    def _init_visual_area(img) -> np.ndarray:
        x, y = img.shape
        my_dpi = 80
        fig = Figure(figsize=(y / my_dpi, x / my_dpi), dpi=my_dpi)
        canvas = FigureCanvasAgg(fig)
        ax = fig.gca()

        ax.plot(GOAL[0], GOAL[1], 'ro', linewidth=5)
        ax.plot(Environment.bx, Environment.by, 'b-')
        img_container = ax.imshow(img[:, :], cmap='gray', vmin=0, vmax=1.0)
        ax.axis('off')
        fig.tight_layout()
        canvas.draw()  # draw the canvas, cache the renderer
        s, (width, height) = canvas.print_to_buffer()
        image = np.frombuffer(s, np.uint8).reshape((height, width, 4))
        img_container.set_data(image)
        plt.close()
        return image, img_container

class Agent(ABC):
    def __init__(self, alpha: float, gamma: float, epsilon: float):
        # set up the value of epsilon
        self.alpha = alpha  # learning rate or step size
        self.gamma = gamma  # discount rate
        self.epsilon = epsilon  # exploration rate
        self.hist = []

    # Choose action at state based on epsilon-greedy policy and valueFunction
    def select_action(self, state: Tuple[int, int], use_greedy_strategy: bool = True) -> int:
        val_state = self.values(state)

        # override epsilon-greedy w./ greedy
        if use_greedy_strategy:
            prob_greedy=1.
        else:
            prob_greedy = np.random.rand()

        if prob_greedy>=self.epsilon:
            best_actions = np.argmax(val_state)
        else: 
            best_actions = np.random.randint(0, 4)
            
        self.hist.append(val_state)
        return int(best_actions)

    # Return estimated action value of given state and action
    @abstractmethod
    def value(self, state: State, action: Actions) -> float:
        pass

    # Return vector of estimated action values of given state, for each action
    @abstractmethod
    def values(self, state: State) -> np.ndarray:
        pass

    # learn with given state, action and target
    # different between on-policy and off-policy
    @abstractmethod
    def learn(self, state: State, action: Actions, next_state: State, reward: float, done: bool = False) -> int:
        return None

    # Return estimated state value, based on the estimated action values
    def state_value(self, state):
        return np.max(self.values(state))

    # Plot the state value estimates. Use a larger stride for lower resolution.
    def plot_state_values(self, stride=1):
        self.v = np.zeros(
            ((Boundary.SOUTH - Boundary.NORTH + stride) // stride, (Boundary.EAST - Boundary.WEST + stride) // stride))
        for j, x in enumerate(range(Boundary.WEST, Boundary.EAST + 1, stride)):
            for i, y in enumerate(range(Boundary.NORTH, Boundary.SOUTH + 1, stride)):
                self.v[i, j] = self.state_value((x, y))

        plt.imshow(self.v, cmap='nipy_spectral')
        plt.colorbar()
        return plt.show()

    def plot_q_values(self, skip=1):
        return pd.DataFrame(self.hist[::skip]).plot()

# This class handles the function approximation, with several methods to query and update it. 
# A linear approximation function is used, making the computation much faster.
class QAgent(Agent):
    def __init__(self, alpha: float, gamma: float, epsilon: float):
        super().__init__(alpha, gamma, epsilon)
        # Use a tile coding
        self.tilings = [grid_tilings() for a in Actions]

    # Return estimated action value of given state and action
    def value(self, state, action):
        if is_goal_reached(state):
            return 0.0
        return self.tilings[action].get_weight(state)

    # Return vector of estimated action values of given state, for each action
    def values(self, state):
        if is_goal_reached(state):
            return np.zeros(4)
        return [self.tilings[action].get_weight(state) for action in Actions]

    # learn with given state, action and target
    # different between on-policy and off-policy
    def learn(self, state: State, action: Actions, next_state: State, reward: float, done: bool = False):
        # TO BE FILLED (1 point)
        R = reward
        q_est = self.value(state, action)
        w = self.tilings[action].get_weight(state)
        sample = state

        if done:
            target = w + self.alpha*(R - q_est)
            self.tilings[action].set_weight(sample=sample, target=target)
            return 0
        
        next_action = self.select_action(next_state, use_greedy_strategy=True)
        q_est_next = self.value(next_state, next_action)
        target =w + self.alpha * (R + self.gamma*q_est_next - q_est)
        self.tilings[action].set_weight(sample=sample, target=target)
        
        return next_action

       
    
class SARSAAgent(Agent):
    def __init__(self, alpha: float, gamma: float, epsilon: float):
        super().__init__(alpha, gamma, epsilon)

        # Use a tile coding
        self.tilings = [grid_tilings() for a in Actions]

    # Return estimated action value of given state and action
    def value(self, state, action):
        if is_goal_reached(state):
            return 0.0
        return self.tilings[action].get_weight(state)

    # Return vector of estimated action values of given state, for each action
    def values(self, state):
        if is_goal_reached(state):
            return np.zeros(4)
        return [self.tilings[action].get_weight(state) for action in Actions]

    # learn with given state, action and target
    # different between on-policy and off-policy
    def learn(self, state: State, action: Actions, next_state: State, reward: float, done: bool = False):
        # TO BE FILLED (1 point)
        R = reward
        q_est = self.value(state, action)
        w = self.tilings[action].get_weight(state)
        sample = state

        if done:
            target = w + self.alpha*(R - q_est)
            self.tilings[action].set_weight(sample=sample, target=target)
            return 0
        
        next_action = self.select_action(next_state, use_greedy_strategy=False)
        q_est_next = self.value(next_state, next_action)
        target =w + self.alpha * (R + self.gamma*q_est_next - q_est)
        self.tilings[action].set_weight(sample=sample, target=target)
        
        return next_action
# env: Environment in which the agent is supposed to run 
# agent: agent to learn
# initital state: Starting state for the environment
# is_learning: should the value function be updated during the simulation?
# is_rendering: should the run be recorded? (Takes some time to compute)


def run_episode(env: Environment,
                agent: Agent,
                initial_state: State,
                is_learning: bool = True,
                is_rendering: bool = False) -> Tuple[State, float]:
    # Initialize reward for episode
    total_reward = 0.0
    # Initialize the policy (if is_greedy=True then the agent follows its optimal policy, otherwise it will randomly select an action)
    is_greedy = not is_learning
    # Get initial action
    current_state = initial_state
    current_action = agent.select_action(initial_state, use_greedy_strategy=is_greedy)

    # Track the rendering
    animation_data = []
    animation_data.append((env.render(), env.time, None, current_state, None, 0))
    # Initialize variables
    next_state = None
    done = False
    next_action = None
    while not done:
        next_state, reward, done, _ = env.step(current_action)
        
        total_reward += reward
        if is_rendering:
            curr_img = env.render()
            animation_data.append((curr_img, env.time, current_state, next_state, current_action, total_reward))
        
        # Execute the learning and update the state and action
        # TO BE FILLED (1 point)
        if is_learning:
            next_action = agent.learn(state=current_state, action=current_action, next_state=next_state, reward=reward, done=done)
        else:
            next_action = agent.select_action(state=current_state, use_greedy_strategy=is_greedy)
        current_state = next_state
        current_action = next_action

    
        
    agent.last_action = None
    return current_state, total_reward, animation_data



filepath = 'C:/Users/lazar/OneDrive/Υπολογιστής/test/final_dict.pkl'
with open(filepath, 'rb') as file:
    data = pickle.load(file)


SARSAA_optimal_agent = data['SARSAAgent_a0.01_g0.95_v0'][0]
SARSAA_optimal_agent.plot_state_values()
SARSAA_optimal_agent.plot_q_values(skip=1)
Q_optimal_agent = data['QAgent_a0.01_g0.95_v0'][0]
Q_optimal_agent.plot_q_values(skip=1)
#random_agent = RAgent(alpha=1.,gamma=1.,epsilon=1.)

env = Environment()
start_state = (200,200)
start_state = env.reset(state=start_state)

env.state=start_state
end_state, total_reward, animation_data1 = run_episode(env, SARSAA_optimal_agent, start_state, is_learning=False, is_rendering=True)

start_state = env.reset(state=start_state)
end_state, total_reward, animation_data2 = run_episode(env, Q_optimal_agent, start_state, is_learning=False, is_rendering=True)

start_state = env.reset(state=start_state)
end_state, total_reward, animation_data3 = run_episode(env, random_agent, start_state, is_learning=False, is_rendering=True)
render_epoch(animation_data1, interval=100)
render_epoch(animation_data2, interval=100)
render_epoch(animation_data3, interval=100)






# Your code here for running the simulation and ploting the sum of immediate reward in each episode.

# Init the environment and run Q-Learning & SARSA

# Save and plot the results

# TO BE FILLED.

alpha = [0.001, 0.01, 0.1]
gamma = [0.75, 0.95, 0.85]
num_runs = 5
agents = [SARSAAgent, QAgent]
epsilon = 1e-2

data = {}
for a in alpha:
    for g in gamma:
        for agent in agents:
            for num in range(num_runs):
                if agent is SARSAAgent:
                    agent_name = "SARSAAgent"
                else:
                    agent_name = "QAgent"
                data[f'{agent_name}_a{a}_g{g}_v{num}'] = [agent(a, g, epsilon),[]]
                
num_episodes = 25000

# Specify the file name to save the dictionary
file_name = 'my_dict.pkl'

for agent_name, agent_object in data.items():
    print(agent_name)
    print(agent_object)
    for eps in range(1,num_episodes):
        if eps % 250 == 0:
            print('o',end="")
            if eps%1000==0 and not eps%3000==0:
                print("|",end="")
            elif eps%3000==0:
                print()
        env = Environment()
        start_state = env.reset()
        end_state, total_reward, animation_data = run_episode(env, agent_object[0], start_state, is_learning=True, is_rendering=True)
        agent_object[1].append(total_reward)
    
    # Open the file in binary write mode and save the dictionary using pickle.dump()
    with open(file_name, 'wb') as file:
        pickle.dump(data, file)

    print(f"Dictionary saved to '{file_name}'")
    agent_object[0].plot_state_values()
    agent_object[0].plot_q_values(skip=1)
    render_epoch(animation_data, interval=100)




