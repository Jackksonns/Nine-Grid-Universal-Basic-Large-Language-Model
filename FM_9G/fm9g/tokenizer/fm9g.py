import io
import json
from typing import Dict
from typing import IO
from typing import List

import pkg_resources
from pytrie import StringTrie


def load_vocab(fp: IO[bytes]) -> Dict[str, int]:
    """Loads a vocabulary file into a dictionary."""
    vocab: Dict[str, int] = {}

    reader = io.TextIOWrapper(fp, encoding="utf-8")
    for token in reader.readlines():
        token = token.strip()
        if len(token) == 0:
            continue
        token = json.loads(token)
        vocab[token] = len(vocab)
    return vocab


class FM9GTokenizer(object):
    def __init__(self, path=None):
        self.unk_token = "<unk>"
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self.byte_list = ["<0x0{}>".format(hex(i).upper()[2:]) for i in range(0x10)] + [
            "<0x{}>".format(hex(i).upper()[2:]) for i in range(0x10, 0x100)
        ]

        self._special_token_set = set([self.unk_token, self.bos_token, self.eos_token] + self.byte_list)

        if path:
            all_tokens = load_vocab(io.FileIO(path, "rb"))
        else:
            all_tokens = load_vocab(pkg_resources.resource_stream("fm9g", "/fm9g/vocabs/fm9g.txt"))

        self.encoder: Dict[str, int] = {}
        self._special_encoder: Dict[str, int] = {}
        for token, token_id in all_tokens.items():
            if token in self._special_token_set:
                self._special_encoder[token] = token_id
            else:
                self.encoder[token] = token_id

        self.decoder = {v: k for k, v in self.encoder.items()}
        self._byte_decoder = {self._special_encoder[token]: i for i, token in enumerate(self.byte_list)}

        self._max_word_len = max([len(x) for x in self.encoder.keys()])

        self._len_word_first = {}
        for x in self.encoder.keys():
            if not x[0] in self._len_word_first:
                self._len_word_first[x[0]] = 1
            if len(x) > self._len_word_first[x[0]]:
                self._len_word_first[x[0]] = len(x)
        self.tencoder = StringTrie(self.encoder)

    def get_piece(self, text: str) -> str:
        if text[0] in self._len_word_first:
            text = text[: self._len_word_first[text[0]]]
            len_text = len(text)
            for i in range(len(text)):
                sub = text[: len_text - i]
                if sub in self.encoder:
                    return sub
        return text[0]

    @property
    def vocab_size(self):
        return len(self)

    @property
    def eos_id(self):
        return self._special_encoder[self.eos_token]

    @property
    def bos_id(self):
        return self._special_encoder[self.bos_token]

    @property
    def unk_id(self):
        return self._special_encoder[self.unk_token]

    def __len__(self):
        return len(self.encoder) + len(self._special_encoder)

    def tokenize(self, text: str) -> List[str]:
        output_tokens: List[str] = []
        st = 0
        while st < len(text):
            piece = self.get_piece(text[st:])
            output_tokens.append(piece)
            st += len(piece)
        return output_tokens

    @staticmethod
    def escape(text: str) -> str:
        return text

    @staticmethod
    def unescape(text: str) -> str:
        return text

    def encode(self, text: str) -> List[int]:
        #if len(text) > 20480:
        #    return [0 for _ in range(20480)]
        ret = []
        for x in self.tokenize(text):
            if x in self.encoder:
                ret.append(self.encoder[x])
            else:
                ret.extend(self._encode_unicode(x))
        return ret

    def decode(self, tokens: List[int]):
        """Decode ids into a string."""
        ret = []
        st = 0

        while st < len(tokens):
            if tokens[st] in self.decoder:
                ret.append(self.decoder[tokens[st]])
                st += 1
            elif tokens[st] in self._byte_decoder:
                if (
                    st + 3 < len(tokens)
                    and tokens[st + 1] in self._byte_decoder
                    and tokens[st + 2] in self._byte_decoder
                    and tokens[st + 3] in self._byte_decoder
                ):
                    first_id = self._byte_decoder[tokens[st]]
                    plane_id = self._byte_decoder[tokens[st + 1]]
                    row_id = self._byte_decoder[tokens[st + 2]]
                    cell_id = self._byte_decoder[tokens[st + 3]]
                    int_bytes = int.to_bytes(first_id << 24 | plane_id << 16 | row_id << 8 | cell_id, 4, "big")
                    try:
                        decoded_str = int_bytes.decode("utf-8", errors="replace")
                        ret.append(decoded_str)
                        #print(decoded_str)
                    except UnicodeDecodeError as e:
                        print(f"UnicodeDecodeError: {e}")
 
                    st += 4
                elif (
                    st + 2 < len(tokens)
                    and tokens[st + 1] in self._byte_decoder
                    and tokens[st + 2] in self._byte_decoder
                ):
                    plane_id = self._byte_decoder[tokens[st]]
                    row_id = self._byte_decoder[tokens[st + 1]]
                    cell_id = self._byte_decoder[tokens[st + 2]]
                    int_bytes = int.to_bytes(plane_id << 16 | row_id << 8 | cell_id, 3, "big")
                    try:
                        decoded_str = int_bytes.decode("utf-8", errors="replace")
                        ret.append(decoded_str)
                    except UnicodeDecodeError as e:
                        print(f"UnicodeDecodeError: {e}")
                    st += 3
                elif st + 1 < len(tokens) and tokens[st + 1] in self._byte_decoder:
                    row_id = self._byte_decoder[tokens[st]]
                    cell_id = self._byte_decoder[tokens[st + 1]]
                    int_bytes = int.to_bytes(row_id << 8 | cell_id, 2, "big")
                    try:
                        decoded_str = int_bytes.decode("utf-8", errors="replace")
                        ret.append(decoded_str)
                    except UnicodeDecodeError as e:
                        print(f"UnicodeDecodeError: {e}")
                    #ret.append(int.to_bytes(row_id << 8 | cell_id, 2, "big").decode("utf-8"))
                    st += 2
                else:
                    cell_id = self._byte_decoder[tokens[st]]
                    int_bytes = int.to_bytes(cell_id, 1, "big")
                    try:
                        decoded_str = int_bytes.decode("utf-8", errors="replace")
                        ret.append(decoded_str)
                    except UnicodeDecodeError as e:
                        print(f"UnicodeDecodeError: {e}")
                    #ret.append(int.to_bytes(cell_id, 1, "big").decode("utf-8"))
                    st += 1
            elif tokens[st] == self.eos_id:
                ret.append(self.eos_token)
                st += 1
            elif tokens[st] == self.bos_id:
                ret.append(self.bos_token)
                st += 1
            else:
                ret.append(self.unk_token)
                st += 1
        return "".join(ret)

    def _encode_unicode(self, token):
        # wrap unicode encoding into a helper function
        ids = []
        utf8_id = token.encode("utf-8")
        for _id in utf8_id:
            ids.append(self._special_encoder[self.byte_list[_id]])
        return ids

    def next_token(self, text):
        # fast next token matching
        token, token_id = self.tencoder.longest_prefix_item(text, (None, None))
        if token is None:
            token = text[0]
            token_ids = self._encode_unicode(token)
        else:
            token_ids = [token_id]
        return token, token_ids
