import os
import threading

import chess
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from glob import glob

# Training parameters
NUM_EPOCHS = 500  # Number of training epochs
BATCH_SIZE = 256  # Batch size
LEARNING_RATE = 0.001  # Learning rate for optimizer
MODEL_SAVE_PATH = "data/saved_models/model_with_loss_.pth"  # Model save path
MODEL_PATH = "data/saved_models"
MAX_FILES = 1000  # Max number of files to load for training
MODEL_EXTENSION = "pth"

# To stop the loop
stop_training = False

# Ensure the model save directory exists
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)


# Define custom Dataset class
class ChessDataset(Dataset):
    def __init__(self, data_files):
        self.positions = []
        self.evaluations = []

        for file in data_files:
            with open(file, "r") as f:
                for line in f:
                    fen, eval_score = line.strip().split(',')
                    self.positions.append(self.fen_to_tensor(fen))
                    self.evaluations.append(float(eval_score))

        self.evaluations = torch.tensor(self.evaluations, dtype=torch.float32)

    def __len__(self):
        return len(self.evaluations)

    def __getitem__(self, idx):
        return self.positions[idx], self.evaluations[idx]

    @staticmethod
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


# CNN Model Definition
class ChessCNN(nn.Module):
    def __init__(self):
        super(ChessCNN, self).__init__()
        self.conv1 = nn.Conv2d(13, 32, kernel_size=3, padding=1)  # Input channels = 13, output = 32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Further channels
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # Flatten then fully connected
        self.fc2 = nn.Linear(128, 1)  # Single output for evaluation

    def forward(self, x):
        x = torch.relu(self.conv1(x))  # First convolution with ReLU activation
        x = torch.relu(self.conv2(x))  # Second convolution
        x = x.view(x.size(0), -1)  # Flatten for fully connected layer
        x = torch.relu(self.fc1(x))  # First FC layer
        x = self.fc2(x)  # Final output layer
        return x


def load_data(test_size=0.2):
    """Loads and splits data from text files into training and test sets."""
    data_files = glob("data/prepared_data/*.txt")[:MAX_FILES]
    print(f"Loading data from {len(data_files)} files")

    train_files, test_files = train_test_split(data_files, test_size=test_size, random_state=42)

    train_dataset = ChessDataset(train_files)
    test_dataset = ChessDataset(test_files)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader


def check_input():
    global stop_training
    while not stop_training:
        user_input = input()
        if user_input == "stop":
            stop_training = True


def train():
    # Load data
    train_loader, test_loader = load_data()

    # Initialize model, loss function, and optimizer
    train_model = ChessCNN()
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = optim.Adam(train_model.parameters(), lr=LEARNING_RATE)

    # Start the input-checking thread
    input_thread = threading.Thread(target=check_input)
    input_thread.start()

    # Training loop
    for epoch in range(NUM_EPOCHS):

        if stop_training:
            print("Training loop stopped by input \"stop\".")
            break

        train_model.train()  # Set model to training mode
        total_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()  # Clear gradients
            outputs = train_model(inputs)  # Forward pass
            loss = criterion(outputs, targets.unsqueeze(1))  # Calculate loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            total_loss += loss.item()  # Accumulate loss for reporting

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}")

    # Save the trained model
    torch.save(train_model.state_dict(), MODEL_SAVE_PATH.split(".")[0] +
               f"{evaluate(test_loader, train_model, criterion)}." + MODEL_EXTENSION)
    print(f"Model saved to {MODEL_SAVE_PATH}")


def _board_to_tensor(board):
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


def model_eval(model, board):
    """
    Evaluates the board position using the model.
    """
    model.eval()
    with torch.no_grad():
        position_tensor = _board_to_tensor(board).unsqueeze(0)
        return model(position_tensor)


def evaluate(test_loader, model, criterion) -> float:
    """Evaluate the model on the test dataset and print the average test loss."""
    model.eval()  # Set model to evaluation mode
    total_test_loss = 0
    with torch.no_grad():  # Disable gradient calculation
        for inputs, targets in test_loader:
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, targets.unsqueeze(1))  # Calculate test loss
            total_test_loss += loss.item()  # Accumulate loss for reporting

    avg_test_loss = total_test_loss / len(test_loader)
    print(f"Average Test Loss: {avg_test_loss:.4f}")
    return avg_test_loss


def _test_model(model, fen):
    """Tests the trained model on a single FEN position."""
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        position_tensor = ChessDataset.fen_to_tensor(fen).unsqueeze(0)  # Add batch dimension
        prediction = model(position_tensor)
        print(f"Predicted evaluation for position {fen}: {prediction.item()} centipawns")


def get_best_model():
    """Find the model with the lowest recorded loss and return its file path."""
    best_loss = float('inf')
    best_model_path = None

    files = glob(os.path.join(MODEL_PATH, "*"))
    if not files:
        raise FileNotFoundError(
            f"No model files found in {MODEL_PATH}. Ensure models are saved in this directory.")

    for f in files:
        file_name = _get_filename_without_extension(f)
        cur_loss = _get_loss_from_model_name(file_name)

        if cur_loss < best_loss:
            best_loss = cur_loss
            best_model_path = f

    if best_model_path is None:
        raise ValueError("No valid model files with loss information found in filenames.")

    return best_model_path


def _get_filename_without_extension(file_path):
    """Helper function to get filename without extension."""
    return os.path.splitext(os.path.basename(file_path))[0]


def _get_loss_from_model_name(file_name):
    """Extract loss value from the model filename."""
    try:
        loss_str = file_name.split('_')[-1]
        return float(loss_str)
    except ValueError:
        raise ValueError(f"Filename '{file_name}' does not contain a valid loss value.")


if __name__ == "__main__":
    train()

    # Example FEN position for testing
    test_fen = "rnb1kbnr/ppp1pppp/8/8/2q5/8/PPPP1PPP/RNBQK1NR b KQkq - 0 1"  # Some position
    model = ChessCNN()
    model.load_state_dict(torch.load(get_best_model(), weights_only=True))  # Load the saved model with weights only
    _test_model(model, test_fen)  # Test the model on the example FEN
