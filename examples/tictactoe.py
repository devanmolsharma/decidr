"""Tic-tac-toe against an LLM playing purely by decidr decisions -- no
generation, no chain of thought, just one forward pass per move.

Every move is a single decide() call: the board plus the empty squares as
options. Whichever square scores highest gets played. This is a decent real
test of the technique because tic-tac-toe needs simple but *correct* spatial
reasoning under a fixed, enumerable option set -- exactly the shape decidr
targets, and unforgiving about it: there's no partial credit for "close."

    python examples/tictactoe.py [model] [host]
"""

import sys

from decidr import Client, DecisionError

LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
NAMES = ["top-left", "top-mid", "top-right", "mid-left", "center", "mid-right", "bottom-left", "bottom-mid", "bottom-right"]


def render(board: list[str]) -> str:
    rows = [" | ".join(c or "." for c in board[r * 3:r * 3 + 3]) for r in range(3)]
    return "\n".join(rows)


def winner(board: list[str]) -> str | None:
    for a, b, c in LINES:
        if board[a] and board[a] == board[b] == board[c]:
            return board[a]
    return None


def llm_move(client: Client, board: list[str], me: str) -> int:
    them = "O" if me == "X" else "X"
    empty = [i for i, c in enumerate(board) if not c]
    row = {
        "id": "move",
        "state": {
            "game": "tic-tac-toe on a 3x3 grid, indices 0-8 left-to-right top-to-bottom",
            "you_play": me,
            "opponent_plays": them,
            "board": render(board),
        },
        "question": f"You play {me}. Which empty square should you take? Win immediately if you "
                    f"can complete three in a row. Otherwise block the opponent from completing "
                    f"three in a row. Otherwise take the strongest remaining square.",
        "options": [{"id": str(i), "description": f"the {NAMES[i]} square"} for i in empty],
    }
    d = client.decide(row)
    return int(d.choice), d


def main() -> int:
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen3.5:4b"
    host = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:11434"
    client = Client(model=model, host=host)

    board = [""] * 9
    turn = "X"  # LLM plays X and moves first; a scripted "opponent" plays O
    move_no = 0

    print(f"model={model}\n")
    while True:
        move_no += 1
        if turn == "X":
            try:
                sq, d = llm_move(client, board, "X")
            except DecisionError as e:
                print(f"LLM error: {e}")
                return 1
            if board[sq]:
                print(f"move {move_no}: LLM chose {NAMES[sq]}, which is already taken. Bug or bad pick.")
                return 1
            top = sorted(d.probabilities.items(), key=lambda kv: -kv[1])[:3]
            print(f"move {move_no}: X -> {NAMES[sq]}  ({d.confidence:.1%}, mode={d.mode})"
                  + ("" if d.is_reliable() else f"  [unscored: {d.unscored}]"))
            board[sq] = "X"
        else:
            # Scripted opponent: win if possible, else block, else center, else first empty.
            empty = [i for i, c in enumerate(board) if not c]
            sq = None
            for i in empty:
                board[i] = "O"
                if winner(board) == "O":
                    sq = i
                board[i] = ""
                if sq is not None:
                    break
            if sq is None:
                for i in empty:
                    board[i] = "X"
                    if winner(board) == "X":
                        sq = i
                    board[i] = ""
                    if sq is not None:
                        break
            if sq is None:
                sq = 4 if 4 in empty else empty[0]
            board[sq] = "O"
            print(f"move {move_no}: O -> {NAMES[sq]}  (scripted)")

        print(render(board), "\n")

        if (w := winner(board)):
            print(f"{w} wins.")
            return 0
        if all(board):
            print("draw.")
            return 0
        turn = "O" if turn == "X" else "X"


if __name__ == "__main__":
    raise SystemExit(main())
