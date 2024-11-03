import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from glob import glob

# Training parameters
NUM_EPOCHS = 500  # Number of training epochs
BATCH_SIZE = 128  # Batch size
LEARNING_RATE = 0.001  # Learning rate for optimizer
MODEL_SAVE_PATH = "data/saved_models/model_with_loss_.pth"  # Model save path
MODEL_PATH = "data/saved_models"
MODEL_EXTENSION = "pth"
MAX_FILES = 100  # Max number of files to load for training (for manageability)

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

        # Convert evaluations to tensors
        self.evaluations = torch.tensor(self.evaluations, dtype=torch.float32)

    def __len__(self):
        return len(self.evaluations)

    def __getitem__(self, idx):
        return self.positions[idx], self.evaluations[idx]

    @staticmethod
    def fen_to_tensor(fen):
        """
        Converts FEN to a tensor representation.
        """
        import chess
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

        # Reverse if turn black
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
    """
    Loads and splits data from text files in data/raw_data into training and test sets.
    """
    # Load and shuffle file list
    data_files = glob("data/prepared_data/*.txt")[:MAX_FILES]  # Get limited number of files
    print(f"Loading data from {len(data_files)} files")

    # Split into train and test files
    train_files, test_files = train_test_split(data_files, test_size=test_size, random_state=42)

    # Create dataset and dataloader for train and test sets
    train_dataset = ChessDataset(train_files)
    test_dataset = ChessDataset(test_files)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader


def train():
    # Load data
    train_loader, test_loader = load_data()

    # Initialize model, loss function, and optimizer
    train_model = ChessCNN()
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = optim.Adam(train_model.parameters(), lr=LEARNING_RATE)

    # Training loop
    for epoch in range(NUM_EPOCHS):
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


def evaluate(test_loader, model, criterion) -> float:
    """
    Evaluate the model on the test dataset and print the average test loss.
    """
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


def test_model(model, fen):
    """
    Tests the trained model on a single FEN position.
    """
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        # Convert FEN to tensor
        position_tensor = ChessDataset.fen_to_tensor(fen).unsqueeze(0)  # Add batch dimension
        prediction = model(position_tensor)
        print(f"Predicted evaluation for position {fen}: {prediction.item()} centipawns")


def get_best_model():
    """
    Find the model with the lowest recorded loss in the specified directory and return its file path.
    """
    # Initialize loss to a high value to ensure any actual loss is lower
    best_loss = float('inf')
    best_model_path = None

    # Get all model files in the directory
    files = glob(os.path.join(MODEL_PATH, "*"))
    if not files:
        raise FileNotFoundError(
            f"No model files found in {MODEL_PATH}. Ensure models are saved in this directory.")

    # Iterate over all model files to find the one with the lowest loss in its filename
    for f in files:
        file_name = _get_filename_without_extension(f)
        cur_loss = _get_loss_from_model_name(file_name)

        # Check if this model has the lowest loss so far
        if cur_loss < best_loss:
            best_loss = cur_loss
            best_model_path = f

    if best_model_path is None:
        raise ValueError("No valid model files with loss information found in filenames.")

    return best_model_path


def _get_filename_without_extension(file_path):
    """ Helper function to get filename without extension. """
    return os.path.splitext(os.path.basename(file_path))[0]


def _get_loss_from_model_name(file_name):
    """
    Extract loss value from the model filename.
    Assumes filename format like 'model_loss_{loss_value}.pth'.
    """
    try:
        loss_str = file_name.split('_')[-1]
        return float(loss_str)
    except ValueError:
        raise ValueError(f"Filename '{file_name}' does not contain a valid loss value.")


if __name__ == "__main__":
    train()

    # Example FEN position for testing
    test_fen = "4r3/p4k1p/8/1p1R1b1N/1b4P1/5P1P/3K4/r7 w - - 0 39"  # Standard starting position
    model = ChessCNN()
    model.load_state_dict(torch.load(get_best_model(), weights_only=True))  # Load the saved model with weights only
    test_model(model, test_fen)  # Test the model on the example FEN
