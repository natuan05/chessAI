import os
import re

import chess
import chess.engine
import chess.pgn

# Path to Stockfish engine (update this to the correct path on your system)
STOCKFISH_PATH = "./stockfish/stockfish-macos-m1-apple-silicon"
engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

# Set the path to your PGN file
PGN_PATH = "data/raw_data/ficsgamesdb_202401_chess_nomovetimes_401277.pgn"  # Path to PGN file with chess games

# Ensure output directory exists
output_directory = "data/prepared_data"
os.makedirs(output_directory, exist_ok=True)


def _get_stockfish_evaluation(board):
    """
    Given a chess.Board object, returns the Stockfish evaluation score.
    """
    info = engine.analyse(board, chess.engine.Limit(time=0.1))
    score = info["score"].relative.score(mate_score=100_000) / 100.0  # Convert to centipawns
    if score > 100 or score < -100:
        return None
    return score


def _get_file_count(pgn_file: str, output_dir: str) -> int:
    # Define regex pattern to match filenames like "pgn_file_1_1000.txt"
    pattern = re.compile(rf"^{re.escape(pgn_file)}_(\d+)_\d+\.txt$")

    # Initialize the max file count found
    max_count = 0

    # List all files in the output directory
    for filename in os.listdir(output_dir):
        # Check if the filename matches the pattern
        match = pattern.match(filename)
        if match:
            # Extract the file count number and update max_count
            file_count = int(match.group(1))
            max_count = max(max_count, file_count)

    # Return the next file count
    return max_count + 1


def _get_filename_without_extension(file_path: str) -> str:
    # Extract the filename with extension
    filename_with_extension = os.path.basename(file_path)
    # Remove the file extension
    filename_without_extension = os.path.splitext(filename_with_extension)[0]
    return filename_without_extension


def parse_and_evaluate_positions(pgn_file, output_dir, max_positions_per_file=1000):
    """
    Parses a PGN file, extracts FEN positions, evaluates them with Stockfish,
    and saves the FEN and evaluation to files in data/raw_data/pos_eval_{num}.txt.
    """
    file_name = _get_filename_without_extension(pgn_file)
    file_count = _get_file_count(file_name, output_dir)
    positions = []
    evaluations = []
    position_count = 0

    with open(pgn_file, 'r') as pgn:
        game = chess.pgn.read_game(pgn)

        while game:
            board = game.board()
            # Iterate over moves and extract positions
            for move in game.mainline_moves():
                board.push(move)
                fen = board.fen()
                eval_score = _get_stockfish_evaluation(board)

                # Skip if checkmate
                if not eval_score:
                    continue

                positions.append(fen)
                evaluations.append(eval_score)
                position_count += 1

                # When max positions reached, save to new file
                if position_count >= max_positions_per_file:
                    file_name = _get_filename_without_extension(pgn_file)
                    output_file = os.path.join(output_dir, f"{file_name}_{file_count}_{max_positions_per_file}.txt")
                    with open(output_file, "w") as f:
                        for fen, eval_score in zip(positions, evaluations):
                            f.write(f"{fen},{eval_score}\n")
                    print(f"Saved {position_count} positions to {output_file}")
                    # Reset for the next file
                    positions = []
                    evaluations = []
                    position_count = 0
                    file_count += 1

            game = chess.pgn.read_game(pgn)

    # Save any remaining positions in the last file
    if positions:
        output_file = os.path.join(output_dir, f"pos_eval_{file_count}.txt")
        with open(output_file, "w") as f:
            for fen, eval_score in zip(positions, evaluations):
                f.write(f"{fen},{eval_score}\n")
        print(f"Saved {position_count} positions to {output_file}")


if __name__ == '__main__':
    # Example usage
    parse_and_evaluate_positions(PGN_PATH, output_directory, max_positions_per_file=100)

    # Close Stockfish engine after processing
    engine.close()
