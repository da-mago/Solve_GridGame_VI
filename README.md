# Introduction

The game is to join 4 points in a grid using the fewest number of cells. Example:

             [' ' ' ' ' ' ' ' ' ' ' ' ' ']
             [' ' 'O' ' ' ' ' ' ' ' ' ' ']
             [' ' 'x' 'x' 'x' 'x' 'O' ' ']
             [' ' ' ' ' ' 'x' ' ' ' ' ' ']
             [' ' ' ' ' ' 'x' ' ' ' ' ' ']
             [' ' ' ' ' ' 'x' ' ' ' ' ' ']
             [' ' ' ' ' ' 'O' ' ' ' ' ' ']
             [' ' 'O' 'x' 'x' ' ' ' ' ' ']
             [' ' ' ' ' ' ' ' ' ' ' ' ' ']

Game challenge source: https://twitter.com/TeoremaPi/status/1246159918832418816

This repository solves this problem by using a well known reinforcement learning algorithm
called 'Value Iteration'. Being purists, I would say that it falls under the umbrella of 
Planning algorithms instead of RL ones (although it is the first topic on any RL course).

Any RL algorithm expects a problem expressed in the form of a MDP (Markov Decision Process)
model. Once the MDP is defined, the nature of the problem (a game in this case) is ignored.
RL algorithms only see numbers...

MDP definition

State space:
- agent position (x,y) in the grid
- number of points not visited yet

Action space:
- Four actions: right(0), left(1), down(2), up(3)

Reward
- Any valid move: -1
- Any invalid move: -10
- Goal met: +20

This MDP is modeled as an agent moving inside the grid and reaching all four points. It is 
similar to the game requirements, but not exactly the same.
The agent starts at point 0 and then moves to find the other ones. To encourage the agent 
to find all points as fast as possible, each movement is penalized (negative reward). When 
the goal is met, a big reward is received.

MDP solutions contain all paths starting from point 0 and meeting the goal of finding all 
points, regardless the number of different cells visited (which is the game requirement).
Example:

             [' ' ' ' ' ' ' ' ' ' ' ' ' ']       [' ' ' ' ' ' ' ' ' ' ' ' ' ']
             [' ' 'O' 'x' 'x' 'x' 'x' ' ']       [' ' 'O' 'x' 'x' ' ' ' ' ' ']
             [' ' ' ' ' ' 'x' 'x' 'O' ' ']       [' ' ' ' ' ' 'x' 'x' 'O' ' ']
             [' ' ' ' ' ' 'x' ' ' ' ' ' ']       [' ' ' ' ' ' 'x' ' ' ' ' ' ']
             [' ' ' ' ' ' 'x' ' ' ' ' ' ']       [' ' ' ' ' ' 'x' ' ' ' ' ' ']
             [' ' ' ' ' ' 'x' ' ' ' ' ' ']       [' ' ' ' ' ' 'x' ' ' ' ' ' ']
             [' ' ' ' ' ' 'O' ' ' ' ' ' ']       [' ' ' ' ' ' 'O' ' ' ' ' ' ']
             [' ' 'O' 'x' 'x' ' ' ' ' ' ']       [' ' 'O' 'x' 'x' ' ' ' ' ' ']
             [' ' ' ' ' ' ' ' ' ' ' ' ' ']       [' ' ' ' ' ' ' ' ' ' ' ' ' ']
                         (a)                                  (b)

Both 'a' and 'b' are valid MDP solutions, but only 'b' is valid solution to the game.

At the end, all MDP solutions are checked to filter the ones using the fewer number
of cells and dump them to console.
