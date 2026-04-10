from __future__ import annotations

from pydantic import BaseModel


class CodeChunk(BaseModel):
    chunk_id: str
    file_path: str
    start_line: int
    end_line: int
    content: str

    symbol_name: str | None
    chunk_type: str
    language: str

    lexical_score: float
    symbol_score: float
    final_score: float

    content_hash: str

    def to_model_view(self) -> str:
        """Compact string for inclusion in model packets.

        Returns file path, line range, and content only — no scores
        or internal metadata.
        """
        header = f"# {self.file_path} L{self.start_line}-{self.end_line}"
        return f"{header}\n{self.content}"
