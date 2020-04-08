import numpy as np
import copy

''' The game is to join 4 points in a grid using the fewest number of cells.
    Example:
             [' ' ' ' ' ' ' ' ' ' ' ' ' ']
             [' ' 'O' ' ' ' ' ' ' ' ' ' ']
             [' ' 'x' 'x' 'x' 'x' 'O' ' ']
             [' ' ' ' ' ' 'x' ' ' ' ' ' ']
             [' ' ' ' ' ' 'x' ' ' ' ' ' ']
             [' ' ' ' ' ' 'x' ' ' ' ' ' ']
             [' ' ' ' ' ' 'O' ' ' ' ' ' ']
             [' ' 'O' 'x' 'x' ' ' ' ' ' ']
             [' ' ' ' ' ' ' ' ' ' ' ' ' ']

    where:
    'O': points
    'x': cells used to join the points
    ' ': celss not used

    Game source: https://twitter.com/TeoremaPi/status/1246159918832418816

    How is it solved:

    Short answer: applying reinforcement learning (specifically Value Iteration
    algorithm).
    
    Long answer: modeling the game as a MDP (Markov Decision Process) and then 
    applying a specific RL algorithm to this model.

    MDP Space state is represented by two variables:
    - agent position (x,y) in the grid
    - number of red points not visited yet

    MDP Action space contains four actions

    Note: designed MDP does not model exactly the game requirements. 
          It focuses in minimizing the number of steps required to join all 4 
          points sequentially (its purpose is not to reuse cells to go and come
          back from a point in order to minimize the number of cells required
          to join the 4 points).

    TODO: Reformulate MDP in order to take into account the number of cells
          used in the solution.
          

    Once said that, the algorithm is able to find many optimal solutions from 
    the point of view of the MDP perspective (the fewest steps to go from any
    point to the other 3 ones).
    Then, all solutions using the minimum number of cells are filetered and 
    prompted to console.

    Note: code compatible with python2 and python3
'''

# Grid size
M,N = 7,9

# Points
game_points = [[1,1],[5,2],[3,6],[1,7]]
target_points = game_points[1:] # Let's assume we start at the first game point
K = len(target_points)

# MDP state vars (MDP size = 7*9*8 = 504 estados)
points_map = (1<<K) - 1  # 0b111
pos = [0,0]

# Auxiliary vars
moves = np.array([[1,0],[-1,0],[0,1],[0,-1]])

def reset(x,y,bitmap):
    ''' Reset MDP state vars '''
    global pos
    global points_map

    pos = [x,y]
    points_map = bitmap
    

def step(action):
    ''' Take an action and return this info: (observation, reward, done)
        Actions: rigth(0), left(1), down(2), up(3)
    '''
    global pos
    global points_map

    if points_map == 0:
        # Goal met
        reward = 0
        done = True
    else:
        done = False

        x,y = pos + moves[action]
        if x>=M or x<0 or y>=N or y<0:
            # invalid move
            reward = -10
        else:
            pos = [x,y]
            for i,p in enumerate(target_points):
                if pos == p:
                    if points_map & (1<<i) != 0x00:
                        points_map &= ~(1<<i) 
                        break

            if points_map == 0:
                done = True
                reward = 20
            else:
                reward = -1


    state = [pos, points_map]

    return state, reward, done

def value_iteration(theta = 0.01, discount_factor=1):
    ''' Apply value iteration algorithm over this MDP with state represented by:
        - x,y current pos
        - points_map (number of not reached points).
    '''

    # Compute Optimal Value Function
    V = np.zeros((M,N,1<<K))

    done = False
    while not done:
        V_old = np.copy(V)

        done = True
        for x in range(M):            # 0-6
            for y in range(N):        # 0-8
                for z in range(1<<K): # 0-7
                    nA = 4            # number of actions
                    VA = np.zeros(nA)
                    for a in range(nA):
                        reset(x,y,z)
                        state, reward, _ = step(a)

                        (x_n,y_n),z_n = state
                        VA[a] = reward + discount_factor*V_old[x_n][y_n][z_n]
                    V[x][y][z] = np.max(VA)

                    if np.abs(V[x][y][z] - V_old[x][y][z]) > theta:
                        done = False

    # Compute all optimal policies
    policy = np.zeros((M,N,1<<K), dtype=np.int8)
    policy = [[[0 for _ in range(1<<K)] for _ in range(N)] for _ in range(M)]
    for x in range(M):
        for y in range(N):
            for z in range(1<<K):
                nA = 4
                VA = np.zeros(nA)
                for a in range(nA):
                    reset(x,y,z)
                    state, reward, _ = step(a)
                    (x_n,y_n),z_n = state
                    VA[a] = reward + discount_factor*V[x_n][y_n][z_n]
                # The single optimal Value function, may result in one or several optimal policies
                # Let's keep track of all them
                policy[x][y][z] = np.argwhere(VA == np.amax(VA)).flatten().tolist()

    return V, policy

def showPath(path):
    ''' Show game solution '''
    grid = [[' ' for _ in range(M)] for _ in range(N)]
    for x,y in path:
        grid[y][x] = 'x'

    for x,y in game_points:
        grid[y][x] = 'O'

    print('\n')
    for line in grid:
        print(np.array(line))
              
def nextPathStep(path, local_pos, local_points_map, done, action):
    ''' Find trajectories generated from all optimal policies '''
    global min_len

    if done:
        # Count all non-repeated elements
        length = len(set([x*N+y for x,y in path]))
        min_len = min(min_len, length)
        if show_len == length:
            showPath(path)

    else:
        path.append(pos)
        x,y = local_pos
        z   = local_points_map
        for action in policy[x][y][z]:
            reset(x,y,z)
            state, _, done = step(action)
            new_pos, new_points_map = state
            nextPathStep(copy.copy(path), copy.copy(new_pos), new_points_map, done, action)


if __name__ == "__main__":

    # Find all optimal policies
    _, policy = value_iteration()

    # Remind that optimal policy, in this case, means taking the minimum number
    # of steps to go from a point to the other three, regardless of whether the 
    # number of different cells visited is minimum or not.
    # Let's check all optimal policies and choose the ones minimizing this extra
    # constraint

    # Find and show solutions. Run it twice: first to find the minimum cells used
    # over all solutions, and second to show only those meeting this constraing.

    show_len = 0
    for _ in range(2):
        # Init state
        points_map = 0x7
        pos = game_points[0]
        path = []

        min_len = 1000
        nextPathStep(copy.copy(path), copy.copy(pos), points_map, False, None)
        show_len = min_len

