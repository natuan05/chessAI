import os
import engine as ce
import chess as ch


class Main:
    def __init__(self, board=ch.Board()):
        self.board = board

    def playHumanMove(self):
        while True:
            try:
                print(self.board.legal_moves)
                print('To undo your last move, type "undo".')
                move = input("Your move: ")

                if move == "undo":
                    if len(self.board.move_stack) > 1:
                        self.board.pop()
                        self.board.pop()
                        print("Last move undone.")
                    else:
                        print("No moves to undo.")
                    continue

                # Apply the move if it's legal
                self.board.push_san(move)
                break
            except ValueError:
                print("Invalid move. Please try again.")

    def playEngineMove(self, model, max_time, color):
        engine = ce.Engine(model, self.board, max_time, color)
        best_move = engine.get_best_move()
        self.board.push(best_move)

    def startGame(self):
        """Starts a new game and handles player inputs and engine moves."""
        # Get human player's color
        color = None
        while color not in ["b", "w"]:
            color = input('Play as (type "b" or "w"): ').lower()

        # Get the model
        model = ""
        try:
            while not os.path.exists(model):
                model = input('Type in the model path: ').lower()
                if not os.path.exists(model):
                    print(f"Model '{model}' not found. Please try again.")
        except Exception as e:
            print(f"An error occurred: {e}")
            return

        # Get thinking time
        while True:
            try:
                max_time = int(input("Choose thinking time (in seconds): "))
                break
            except ValueError:
                print("Invalid input. Please enter an integer.")

        # Game loop
        #if self.board.is_checkmate():
        #    if self.
        #    print()
        if color == "b":
            while not self.board.is_checkmate():
                print("The engine is thinking...")
                self.playEngineMove(model, max_time, ch.WHITE)
                print(self.board)
                self.playHumanMove()
                print(self.board)
        elif color == "w":
            while not self.board.is_checkmate():
                print(self.board)
                self.playHumanMove()
                print(self.board)
                print("The engine is thinking...")
                self.playEngineMove(model, max_time, ch.BLACK)

        self.printGameOutcome()
        # Reset the board and start a new game for next move
        self.board.reset()
        self.startGame()

    def printGameOutcome(self):
        print(self.board)
        print(f"Game Over: {self.board.outcome()}")


if __name__ == '__main__':
    newBoard = ch.Board()
    game = Main(newBoard)
    game.startGame()
