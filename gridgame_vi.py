import numpy as np
import copy

''' This code uses the MDP framework to model the problem. Once this is done,
    any available RL algorithm will do the task (find the optimal solution).
    Specifically, this code implements 'Value Iteration' algorithm.

    Space state is represented by two variables:
    - agent position (x,y) in the grid
    - number of red points not visited yet

    Action space contais four actions

    TODO: I don't know how to model inside the MPD the constraint of visiting
          the minimum grid cells. 
          This algorithm, applied to this problem, finds many optimal solutions
          (optimal = minimum steps to visit all red pointso, which is not the same
          as minimum number of grid cells visited).
          The workaround is to review all optimal solutions found and then filter 
          those visiting the minimum grid cells (this is why we need to find 
          trajectories twice: first to find the minimun cells visited, and then to
          filter out all longer solutions)

    The output consists of all optimal solutions to the problem

    Note: It works under both python2 and python3
'''

# Grid size
M,N = 7,9

# Puntos (rojos)
red_points = [[1,1],[5,2],[3,6],[1,7]]
target_points = red_points[1:] # Let's assume we already are in the first point
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

    # Compute optimal policy
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
                # Let's keep track of all possible policies
                policy[x][y][z] = np.argwhere(VA == np.amax(VA)).flatten().tolist()

    return V, policy

def showPath(path):
    grid = [[' ' for _ in range(M)] for _ in range(N)]
    for x,y in path:
        grid[y][x] = 'x'

    for x,y in red_points:
        grid[y][x] = 'O'

    for _ in range(2): print()
    for line in grid:
        print(np.array(line))
              
def nextPathStep(path, local_pos, local_points_map, done, actionPath, action):
    global min_len

    if done:
        # Count all non-repeated elements
        length = len(set([x*N+y for x,y in path]))
        min_len = min(min_len, length)
        if show_len == length:
            showPath(path)

    else:
        path.append(pos)
        actionPath.append(action)
        x,y = local_pos
        z   = local_points_map
        for action in policy[x][y][z]:
            reset(x,y,z)
            state, _, done = step(action)
            new_pos, new_points_map = state
            nextPathStep(copy.copy(path), copy.copy(new_pos), new_points_map, done, copy.copy(actionPath), action)


if __name__ == "__main__":
    _, policy = value_iteration()

    # Optimal policy for our algorithm means taking the minimum number of steps,
    # regardless of whether the number of cells used is minimum or not.
    # Let's check all optimal policies and choose the one minimizing this extra
    # constraint
    # TODO: figure out how to add this constraint to the MDP itself

    # First run to compute all min length paths
    # Second run to show them

    show_len = 0
    for _ in range(2):
        # Init state
        points_map = 0x7
        pos = red_points[0]
        path = []
        actionPath = []

        min_len = 1000
        nextPathStep(copy.copy(path), copy.copy(pos), points_map, False, copy.copy(actionPath), None)
        show_len = min_len

