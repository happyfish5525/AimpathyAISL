"""Turns the CTC model's per-frame output into text while the camera runs.

The model is run every ~0.1-0.2 s on the latest 4 seconds of hand frames. Letters
are "tentative" while recent and "committed" (typed for good) once older than
DELAY_SEC. WHICH letters appear and whether a repeat is two letters are decided
by the network; the constants only stop one letter being typed twice while the
window slides.
"""
from seq_data import T, WINDOW_SEC, collapse_path, tokens_to_text

DELAY_SEC = 0.5
EDGE_SEC = 0.3
MIN_GAP_SEC = 0.10
SAME_GAP_SEC = 0.20

_FRAME_STEP = WINDOW_SEC / (T - 1)


class StreamingDecoder:
    def __init__(self):
        self.committed = []      # [(absolute_time, token_index)]
        self.tentative = []

    def reset(self):
        self.committed, self.tentative = [], []

    @property
    def text(self) -> str:
        return tokens_to_text([tok for _, tok in self.committed])

    @property
    def tentative_text(self) -> str:
        return tokens_to_text(self.tentative)

    def update(self, path, now: float, final: bool = False):
        start = now - WINDOW_SEC
        new, tentative = [], []
        for centre, tok in collapse_path(path):
            t = start + centre * _FRAME_STEP
            if t < start + EDGE_SEC:
                continue
            if self.committed:
                last_t, last_tok = self.committed[-1]
                gap = SAME_GAP_SEC if tok == last_tok else MIN_GAP_SEC
                if t <= last_t + gap:
                    continue
            if final or t <= now - DELAY_SEC:
                self.committed.append((t, tok))
                new.append(tok)
            else:
                tentative.append(tok)
        self.tentative = tentative
        return new
