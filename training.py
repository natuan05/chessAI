import os
import threading
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from glob import glob
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import get_loss_from_model_name, get_filename_without_extension, board_to_tensor, fen_to_tensor


def get_loss(model, test_loader, criterion) -> float:
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


class Trainer:
    def __init__(self, num_epochs: int, learning_rate: float, model_save_path: str = None, batch_size: int = 64,
                 max_files: int = 1000, model_extension: str = "pth"):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = learning_rate
        self.model_save_path = model_save_path
        self.model_path = os.path.dirname(self.model_save_path)
        self.max_files = max_files
        self.model_extension = model_extension

        self.stop_training = False  # To stop the loop

        # Ensure the model save directory exists
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)

        self.train()

    def load_data(self, test_size=0.2):
        """Loads and splits data from text files into training and test sets."""
        data_files = glob("data/prepared_data/*.txt")[:self.max_files]
        print(f"Loading data from {len(data_files)} files")

        train_files, test_files = train_test_split(data_files, test_size=test_size, random_state=42)

        train_dataset = ChessDataset(train_files)
        test_dataset = ChessDataset(test_files)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader

    def check_input(self):
        while not self.stop_training:
            user_input = input()
            if user_input == "stop":
                self.stop_training = True

    def train(self):
        # Load data
        train_loader, test_loader = self.load_data()

        # Initialize model, loss function, and optimizer
        train_model = ChessCNN()
        criterion = nn.MSELoss()  # Mean Squared Error for regression
        optimizer = optim.AdamW(train_model.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

        # Start the input-checking thread
        input_thread = threading.Thread(target=self.check_input)
        input_thread.start()

        try:
            # Training loop
            for epoch in range(self.num_epochs):
                if self.stop_training:
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
                print(f"Epoch [{epoch + 1}/{self.num_epochs}], Loss: {avg_loss:.4f}")
                scheduler.step(avg_loss)

            # Save the trained model
            path = self.model_save_path.split(".")[0] + f"{get_loss(train_model, test_loader, criterion)}." + self.model_extension
            torch.save(train_model.state_dict(), path)
            print(f"Model saved to {path}")
        finally:
            # Ensure the input-checking thread is stopped
            self.stop_training = True
            input_thread.join()  # Wait for the thread to finish
            print("Training finished and resources cleaned up.")


def model_eval(model, board):
    """
    Evaluates the board position using the model.
    """
    model.eval()
    with torch.no_grad():
        position_tensor = board_to_tensor(board).unsqueeze(0)
        return model(position_tensor)


class ChessDataset(Dataset):
    def __init__(self, data_files):
        self.positions = []
        self.evaluations = []

        for file in data_files:
            with open(file, "r") as f:
                for line in f:
                    fen, eval_score = line.strip().split(',')
                    self.positions.append(fen_to_tensor(fen))
                    self.evaluations.append(float(eval_score))

        self.evaluations = torch.tensor(self.evaluations, dtype=torch.float32)

    def __len__(self):
        return len(self.evaluations)

    def __getitem__(self, idx):
        return self.positions[idx], self.evaluations[idx]


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual  # Residual connection
        return torch.relu(x)


class ChessCNN(nn.Module):
    def __init__(self):
        super(ChessCNN, self).__init__()
        self.layer1 = ResidualBlock(13, 32)
        self.layer2 = ResidualBlock(32, 64)
        self.layer3 = ResidualBlock(64, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    trainer = Trainer(num_epochs=100, learning_rate=0.001, model_save_path="data/saved_models/chess_model.pth")
