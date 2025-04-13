import re

# Đường dẫn file gốc và file lọc
INPUT_FILE = "D:/AI/chessAI/data/lichess_db_standard_rated_2013-01.pgn"
OUTPUT_FILE = "D:/AI/chessAI/data/filtered_elo_1000_1500.pgn"

MIN_ELO = 1000
MAX_ELO = 1500

def get_elo(line):
    match = re.search(r'"(\d+)"', line)
    return int(match.group(1)) if match else None

with open(INPUT_FILE, "r", encoding="utf-8") as fin, open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
    buffer = []
    white_elo = None
    black_elo = None
    for line in fin:
        buffer.append(line)

        if line.startswith('[WhiteElo'):
            white_elo = get_elo(line)
        elif line.startswith('[BlackElo'):
            black_elo = get_elo(line)

        # Khi kết thúc 1 game
        if line.strip() == "":
            if white_elo and black_elo and MIN_ELO <= white_elo <= MAX_ELO and MIN_ELO <= black_elo <= MAX_ELO:
                fout.writelines(buffer)
            buffer = []
            white_elo = None
            black_elo = None

print("Complete, New file:", OUTPUT_FILE)
