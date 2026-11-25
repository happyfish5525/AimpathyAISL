"""A tiny text editor model: cursor, insert, delete, word-wrap.

Pure Python (no OpenCV) so it can be tested on its own. The wrapping works on
index ranges into the real string, so no character (including trailing
spaces) is ever dropped and the cursor always has an exact line/column.
"""


class TextEditor:
    def __init__(self):
        self.text = ""
        self.cursor = 0

    # ---- editing ---------------------------------------------------------
    def insert(self, s: str) -> None:
        self.text = self.text[:self.cursor] + s + self.text[self.cursor:]
        self.cursor += len(s)

    def add_word(self, word: str) -> None:
        """Insert a recognised word, e.g. THANK_YOU -> 'THANK YOU ' (with spacing)."""
        word = word.replace("_", " ")
        before = self.text[self.cursor - 1] if self.cursor > 0 else " "
        self.insert(("" if before in " \n" else " ") + word + " ")

    def backspace(self) -> None:
        if self.cursor > 0:
            self.text = self.text[:self.cursor - 1] + self.text[self.cursor:]
            self.cursor -= 1

    def delete(self) -> None:
        if self.cursor < len(self.text):
            self.text = self.text[:self.cursor] + self.text[self.cursor + 1:]

    def clear(self) -> None:
        self.text, self.cursor = "", 0

    # ---- movement --------------------------------------------------------
    def left(self) -> None:
        self.cursor = max(0, self.cursor - 1)

    def right(self) -> None:
        self.cursor = min(len(self.text), self.cursor + 1)

    def home(self, width: int) -> None:
        line, _ = self.cursor_line_col(width)
        self.cursor = self.layout(width)[line][0]

    def end(self, width: int) -> None:
        lines = self.layout(width)
        line, _ = self.cursor_line_col(width)
        self.cursor = self._line_max_cursor(lines, line)

    def up(self, width: int) -> None:
        self._move_vertical(width, -1)

    def down(self, width: int) -> None:
        self._move_vertical(width, +1)

    # ---- layout ----------------------------------------------------------
    def layout(self, width: int):
        """Return a list of (start, end) index ranges, one per visual line."""
        lines, pos = [], 0
        for para in self.text.split("\n"):
            p_end = pos + len(para)
            s = pos
            if s == p_end:
                lines.append((s, s))  # empty line
            while s < p_end:
                e = min(s + width, p_end)
                if e < p_end:  # break at the last space if there is one
                    space = self.text.rfind(" ", s, e)
                    if space > s:
                        e = space + 1
                lines.append((s, e))
                s = e
            pos = p_end + 1  # skip the newline character
        return lines

    def cursor_line_col(self, width: int):
        lines = self.layout(width)
        for i, (s, e) in enumerate(lines):
            nxt = lines[i + 1][0] if i + 1 < len(lines) else None
            if s <= self.cursor < e or (self.cursor == e and nxt != e):
                return i, self.cursor - s
        return len(lines) - 1, 0

    # ---- helpers ---------------------------------------------------------
    @staticmethod
    def _line_max_cursor(lines, i: int) -> int:
        """Right-most cursor index that still displays on line i."""
        s, e = lines[i]
        nxt = lines[i + 1][0] if i + 1 < len(lines) else None
        return e - 1 if (nxt == e and e > s) else e

    def _move_vertical(self, width: int, step: int) -> None:
        lines = self.layout(width)
        line, col = self.cursor_line_col(width)
        target = line + step
        if 0 <= target < len(lines):
            s = lines[target][0]
            self.cursor = min(s + col, self._line_max_cursor(lines, target))
