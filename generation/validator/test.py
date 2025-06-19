import os

import miniDSL.miniDSLLexer
from antlr4 import CommonTokenStream, InputStream
from miniDSL.miniDSLLexer import miniDSLLexer

dsl = """
transformer deeppoly{
   HardSigmoid
}
"""

input_stream = InputStream(dsl)
lexer = miniDSLLexer(input_stream)
token_stream = CommonTokenStream(lexer)
token_stream.fill()

for token in token_stream.tokens:
    print(
        f"{token.text:<20} -> {token.type:<4} ({miniDSLLexer.symbolicNames[token.type] if token.type < len(miniDSLLexer.symbolicNames) else 'UNKNOWN'})"
    )
