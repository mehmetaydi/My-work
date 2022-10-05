# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 17:15:34 2021

@author: mehmet
"""



import math
import numpy as npy
import sys
import copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    if board == initial_state():
        return X

    numpy_board = npy.array(board)

    Xno = npy.count_nonzero(numpy_board == X)

    Ono = npy.count_nonzero(numpy_board == O)

    if Xno > Ono:

        return O

    else:

        return X

def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    Result = set()

    for k in range(3):

        for l in range(3):

            if board[k][l] == EMPTY:

                Result.add((k, l))

    return Result


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    i = action[0]
    j = action[1]

    if board[i][j] != EMPTY:
        raise Exception("Invalid Action")

    new_player = player(board)

    new_board = copy.deepcopy(board)

    new_board[i][j] = new_player

    return new_board

def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    for i in range(3):

        if (board[i][0] == board[i][1] == board[i][2] and board[i][0] != EMPTY):

            return board[i][0]

        if (board[0][0] == board[1][1] == board[2][2] or (board[0][2] == board[1][1] == board[2][0]) and board[1][1] != EMPTY):

             return board[1][1]
        if (board[0][i] == board[1][i] == board[2][i] and board[0][i] != EMPTY):

             return board[1][i]
        # else:

        #      return None
def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board) != None:

        return True;

    numpy_board = npy.array(board)

    empty_no = npy.count_nonzero(numpy_board == EMPTY)

    if (empty_no == 0):

        return True
    else:

        return False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    win_player = winner(board)

    if (win_player == X):

        return 1

    elif (win_player == O):

        return -1

    else:

        return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None

    currentPlayer = player(board)
    if currentPlayer == X:
        return max_value(board)[1]
    else:
        return min_value(board)[1]


def max_value(board):

    if terminal(board):

        return (utility(board), None)

    value = -sys.maxsize-1

    optimalAction = None
    for action in actions(board):

        possibleResult = min_value(result(board, action))

        if possibleResult[0] > value:

            value = possibleResult[0]

            optimalAction = action


        if value == 1:
            break

    return (value, optimalAction)


def min_value(board):

    if terminal(board):

        return (utility(board), None)

    value = sys.maxsize

    optimalAction = None

    for action in actions(board):

        possibleResult = max_value(result(board, action))

        if possibleResult[0] < value:

            value = possibleResult[0]

            optimalAction = action


        if value == -1:
            break

    return (value, optimalAction)