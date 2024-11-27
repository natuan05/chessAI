# utils.py

import os
import numpy as np
import chess
import torch


def get_filename_without_extension(file_path):
    """
    Helper function to get filename without extension.
    """
    return os.path.splitext(os.path.basename(file_path))[0]


def get_loss_from_model_name(file_name):
    """
    Extract loss value from the model filename.
    """
    try:
        loss_str = file_name.split('_')[-1]
        return float(loss_str)
    except ValueError:
        raise ValueError(f"Filename '{file_name}' does not contain a valid loss value.")


def board_to_tensor(board):
    """
    Converts a chess.Board object to a tensor representation.
    """
    board_tensor = np.zeros((13, 8, 8), dtype=np.float32)

    # Define piece mappings to tensor indices
    piece_map = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }

    # Place pieces on the tensor
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        row, col = divmod(square, 8)
        if piece:
            board_tensor[piece_map[piece.symbol()], row, col] = 1
        else:
            board_tensor[12, row, col] = 1  # Empty squares

    # Flip the board if it is Black's turn to move
    if board.turn == chess.BLACK:
        board_tensor = np.flip(board_tensor, axis=(1, 2)).copy()

    return torch.tensor(board_tensor)


def fen_to_tensor(fen):
    """Converts FEN to a tensor representation."""
    board = chess.Board(fen)
    board_tensor = np.zeros((13, 8, 8), dtype=np.float32)

    piece_map = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
                 'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11}

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        row, col = divmod(square, 8)
        if piece:
            board_tensor[piece_map[piece.symbol()], row, col] = 1
        else:
            board_tensor[12, row, col] = 1  # Empty squares

    if board.turn == chess.BLACK:
        board_tensor = np.flip(board_tensor, axis=(1, 2)).copy()

    return torch.tensor(board_tensor)
