import os

import antlr4
from antlr4 import CommonTokenStream, InputStream
from dslLexer import dslLexer

inputfile = "test.txt"

lexer = dslLexer(antlr4.FileStream(inputfile))
token_stream = CommonTokenStream(lexer)

token_stream.fill()


for token in token_stream.tokens:
    print(
        f"{token.text:<20} -> {token.type:<4} ({dslLexer.symbolicNames[token.type] if token.type < len(dslLexer.symbolicNames) else 'UNKNOWN'})"
    )
