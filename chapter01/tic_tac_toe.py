import re
import random
import numpy as np
from itertools import product
from enum import Enum, unique
from abc import ABC, abstractmethod
from typing import Iterator, Dict, List

SIZE = 3  # board size (represented by numpy.array SIZExSIZE)

# for human player
KEYS = {'q': (0, 0), 'w': (0, 1), 'e': (0, 2),
        'a': (1, 0), 's': (1, 1), 'd': (1, 2),
        'z': (2, 0), 'x': (2, 1), 'c': (2, 2)}

# first player must be 1 and the second one -1
SYMBOLS = {"player1": 1, "player2": -1}

# first player: X, second player: O, empty: . (+ remove array brackets)
GUI = {'0': 'O', '1': '.', '2': 'X', '[': None, ']': None, ' ': None}


@unique
class Status(Enum):
    INVALID, NOTEND, PLAYER1, PLAYER2, DRAW = range(-1, 4)


def get_rcd_sums(board: np.array) -> tuple:
    """Return a tuple of sums of all rows, columns and diagonals."""
    return (*(sum(x) for x in board),                      # rows
            *(sum(x) for x in board.T),                    # columns
            np.trace(board), np.trace(np.flip(board, 0)))  # diagonals


def get_status(board: np.array) -> Status:
    """Return a status of a game for given board."""
    total = np.sum(board)  # sum of all fields

    if total not in (0, 1):  # because 1 - first player, -1 - second player
        return Status.INVALID

    rcd_sums = get_rcd_sums(board)  # sum of rows, cols, and diags

    # do players have 3 in row
    player1_has3, player2_has3 = 3 in rcd_sums, -3 in rcd_sums

    # player1 won, unless player2 moved afterwards (invalid)
    # or winning move was done after player2 won already (invalid)
    if player1_has3:
        return Status.INVALID if total == 0 or player2_has3 else Status.PLAYER1

    # player2 won, unless player1 moved afterwards (invalid)
    if player2_has3:
        return Status.INVALID if total == 1 else Status.PLAYER2

    # draw or the game is not done yet
    return Status.NOTEND if (board == 0).any() else Status.DRAW


def get_id(board: np.array) -> int:
    """Get unique id for a board."""
    board_id = 0

    # 3 possible state of a field -> the factor 3 ensure uniqueness
    # -1 -> 2 to avoid negative ids
    for field in board.flatten():
        board_id = 3 * board_id + (field if field >= 0 else 2)

    return board_id


def get_free_cells(board: np.array) -> tuple:
    """Return a tuple with (i, j) coordinates of available cells."""
    return np.argwhere(board == 0)


# all possible combinations of -1, 0, 1 on a SIZExSIZE board
# (including impossible tic tac toe states)
BOARD_GENERATOR = (np.array(state).reshape(SIZE, SIZE)
                   for state in product(range(-1, 2), repeat=SIZE**2))


# (board, state) generator
STATE_GENERATOR = ((board, get_status(board)) for board in BOARD_GENERATOR)


# dictionary of valid tic tac toe states -> id: board, status, free cells
STATES = {get_id(board):
          {'board': board, 'status': status, 'free': get_free_cells(board)}
          for board, status in STATE_GENERATOR if status != Status.INVALID}


def show_board(board: np.array):
    """Print given board."""
    # board + 1: 0 (player2), 1 (empty), 2 (player1) instead of -1, 0, 1
    print(np.array_str(board + 1).translate(str.maketrans(GUI)), end='\n\n')


def show_result(status: Status):
    """Print the game status."""
    print(status.name, "won!"
          if status == Status.PLAYER1 or status == Status.PLAYER2 else '')


class Player(ABC):
    """Abstract player class."""

    def __init__(self, symbol: str) -> None:
        """Assign a symbol - player1: 1, player2: -1"""
        self.symbol = SYMBOLS[symbol]

    @abstractmethod
    def make_move(self, state_id: int) -> tuple:
        """Return (i, j) coordinates for the next move."""
        pass

    def reset(self):
        pass

    def save(self, state_id: int):
        pass

    def update_estimates(self):
        pass


class Human_player(Player):

    def make_move(self, state_id: int) -> tuple:
        """Take input from keyboard until valid move is provided."""
        while True:
            key = input()

            if key not in KEYS.keys():
                print("Use qweasdzxc.\n")
                continue

            if KEYS[key] not in ((i, j) for i, j in STATES[state_id]['free']):
                print("Choose empty cell.\n")
                continue

            return KEYS[key]


class Agent(Player):
    """RL agent."""

    def __init__(self, symbol: str, step_size: float=0.1,
                 epsilon: float=0.1)-> None:
        """
        step_size -- the step size used in the temporal-difference rule
        epsilon   -- the probability of exploration
        """
        Player.__init__(self, symbol)

        self.step_size = step_size
        self.epsilon = epsilon

        self.V: Dict[int, float] = dict()  # estimates of state-value
        self.init_estimations()            # arbitrary initialized

        self.history: List[int] = []      # all "visited" states
        self.explore_ids: List[int] = []  # the indices of exploratory moves

    def reset(self) -> None:
        """Clear history etc."""
        self.history.clear()
        self.explore_ids.clear()

    def save(self, state_id: int) -> None:
        """Remember all visited state in current episode."""
        self.history.append(state_id)

    def init_estimations(self) -> None:
        """Initilize estimate V."""
        # symbol == 1 for player1
        win, lose = ((Status.PLAYER1, Status.PLAYER2) if self.symbol == 1
                     else (Status.PLAYER2, Status.PLAYER1))

        # generate estimates for all possible states
        for state_id, state in STATES.items():
            # win -> 1, lose -> 0, draw or game in progress -> 0.5
            reward = (1.0 if state['status'] == win else
                      0.0 if state['status'] == lose else 0.5)

            self.V[state_id] = reward

    def make_move(self, state_id: int) -> tuple:
        """Exploratory (random) move or based on current est. Q."""
        if random.random() < self.epsilon:
            # exploratory move
            self.explore_ids.append(len(self.history))
            return tuple(random.choice(STATES[state_id]['free']))

        values = []  # the list of tuple(estimation, (i, j) - next move)

        for i, j in STATES[state_id]['free']:
            # get board and add player's symbol on next free cell
            board = STATES[state_id]['board'].copy()
            board[i][j] = self.symbol
            # value + (i, j)
            values.append((self.V[get_id(board)], (i, j)))

        # return (i, j) corresponding to the highest current value
        return sorted(values, key=lambda x: x[0], reverse=True)[0][1]

    def update_estimates(self) -> None:
        """Update estimates according to last episode."""
        for i in reversed(range(len(self.history) - 1)):
            # skip exploratory moves
            if i in self.explore_ids:
                continue

            temp_diff = self.V[self.history[i+1]] - self.V[self.history[i]]
            self.V[self.history[i]] += self.step_size * temp_diff

    def stop_exploring(self) -> None:
        """Only greedy moves will be performed."""
        self.epsilon = 0.0


class Game:
    """A single tic tac toe game."""

    def __init__(self, player1: Player, player2: Player) -> None:
        """Note, that it is hardcoded that player1 = 1 and player2 = -1."""
        self.player1 = player1
        self.player2 = player2

    def queue(self) -> Iterator[Player]:
        """Next player generator."""
        while True:
            yield self.player1
            yield self.player2

    def play(self, show=False) -> Status:
        _queue = self.queue()
        self.player1.reset()
        self.player2.reset()
        self.state_id = 0  # empty board

        # play until state != NOTEND
        while STATES[self.state_id]['status'] == Status.NOTEND:
            player = next(_queue)  # current player

            # get current board and update according to player move
            board = STATES[self.state_id]['board'].copy()
            board[player.make_move(self.state_id)] = player.symbol

            not show or show_board(board)

            self.state_id = get_id(board)  # update board state

            # save current state in agent's history
            self.player1.save(self.state_id)
            self.player2.save(self.state_id)

        not show or show_result(STATES[self.state_id]['status'])

        # after an episode it is time to update V
        self.player1.update_estimates()
        self.player2.update_estimates()

        return STATES[self.state_id]['status']

if __name__ == "__main__":

    agent = Agent("player1")
    game = Game(agent, Agent("player2"))

    nof_episodes = 10000

    for i in range(nof_episodes):
        game.play()
        print("Agent's training... [{:>7.2%}]\r".format(i/nof_episodes), end='')

    print("\n\nPlay with the agent using qweasdzxc:\n\n")

    agent.stop_exploring()
    game = Game(agent, Human_player("player2"))

    while True:
        game.play(True)
