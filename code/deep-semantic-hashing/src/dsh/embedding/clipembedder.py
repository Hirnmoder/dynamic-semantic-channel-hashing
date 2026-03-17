import open_clip as clip

from dsh.embedding.embedder import Embedder
from dsh.utils.types import TextRawInput


# Technically this is not an embedder, but rather a tokenizer
# However, CLIP models require tokens as input and handle the embedding internally
class CLIPEmbedder(Embedder):
    def __init__(self, tokenizer: clip.tokenizer.SimpleTokenizer | clip.tokenizer.HFTokenizer | clip.tokenizer.SigLipTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, text: str | list[str]) -> TextRawInput:
        if isinstance(text, str):
            text = [text]
        tokens = self.tokenizer(text)
        # now reshape from (len(text), seq_length) to (len(text), seq_length, emb_dim=1)
        # but if len(text)==1 then squeeze first dimension
        tokens = tokens.squeeze(dim=0).unsqueeze(dim=-1)
        return tokens
