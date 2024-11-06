import engine as ce
import chess as ch


class Main:

    def __init__(self, board=ch.Board):
        self.board = board

    # play human move
    def playHumanMove(self):
        try:
            print(self.board.legal_moves)
            print("""To undo your last move, type "undo".""")
            # get human move
            play = input("Your move: ")
            if play == "undo":
                self.board.pop()
                self.board.pop()
                self.playHumanMove()
                return
            self.board.push_san(play)
        except:
            self.playHumanMove()

    # play engine move
    def playEngineMove(self, max_time, color):
        engine = ce.Engine(self.board, max_time, color)
        self.board.push(engine.getBestMove())

    # start a game
    def startGame(self):
        # get human player's color
        color = None
        while color != "b" and color != "w":
            color = input("""Play as (type "b" or "w"): """)
        max_time = None
        while not isinstance(max_time, int):
            try:
                max_time = int(input("""Choose thinking time: """))
            except ValueError:
                print("Invalid input. Please enter an integer.")

        if color == "b":
            while not self.board.is_checkmate():
                print("The engine is thinking...")
                self.playEngineMove(max_time, ch.WHITE)
                print(self.board)
                self.playHumanMove()
                print(self.board)
            print(self.board)
            print(self.board.outcome())
        elif color == "w":
            while not self.board.is_checkmate():
                print(self.board)
                self.playHumanMove()
                print(self.board)
                print("The engine is thinking...")
                self.playEngineMove(max_time, ch.BLACK, max_depth)
            print(self.board)
            print(self.board.outcome())
        # reset the board
        self.board.reset()
        # start another game
        self.startGame()


if __name__ == '__main__':
    newBoard = ch.Board()
    game = Main(newBoard)
    game.startGame()
