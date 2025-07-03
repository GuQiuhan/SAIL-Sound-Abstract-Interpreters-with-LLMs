import os

from antlr4 import CommonTokenStream, InputStream
from miniDSL.miniDSLLexer import miniDSLLexer


def parse_tokens_file(filepath: str):
    token_id_to_name = {}
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if "=" not in line or line.startswith("'"):
                continue
            name, id_str = line.split("=")
            if id_str.isdigit():
                token_id_to_name[int(id_str)] = name
    return token_id_to_name


TOKENS_FILE = "./miniDSL/miniDSLLexer.tokens"
token_map = parse_tokens_file(TOKENS_FILE)


dsl = """
transformer deeppoly{
   HardSigmoid -> -2.50

}
"""

input_stream = InputStream(dsl)
lexer = miniDSLLexer(input_stream)
token_stream = CommonTokenStream(lexer)
token_stream.fill()


for token in token_stream.tokens:
    name = token_map.get(token.type, "UNKNOWN")
    print(f"{token.text:<20} -> {token.type:<4} ({name})")
