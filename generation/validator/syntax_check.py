import re

from antlr4 import *
from antlr4 import CommonTokenStream, InputStream
from antlr4.error.ErrorListener import ConsoleErrorListener, ErrorListener
from antlr4.error.ErrorStrategy import BailErrorStrategy
from miniDSL.miniastBuilder import ASTBuilder
from miniDSL.miniDSLLexer import miniDSLLexer
from miniDSL.miniDSLParser import miniDSLParser


class SilentErrorListener(ErrorListener):
    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        pass


class SyntaxChecker:
    """
    - Only supports DSLs with one transformer and one op_stmt.
    """

    def __init__(self):
        pass

    def check(self, dsl: str) -> bool:
        try:

            input_stream = InputStream(dsl)
            lexer = miniDSLLexer(input_stream)
            token_stream = CommonTokenStream(lexer)

            token_stream.fill()
            for token in token_stream.tokens:
                print(f"{token.text} -> {miniDSLLexer.symbolicNames[token.type]}")

            parser = miniDSLParser(token_stream)
            parser.removeErrorListeners()
            parser.addErrorListener(SilentErrorListener())
            parser._errHandler = BailErrorStrategy()

            tree = parser.transformer()  # check syntax

            return dsl
        except:
            if re.search(r"\.\s*(sum|avg|len)\b", dsl):
                print(
                    "❗ Detected invalid attribute call like '.sum', attempting to fix."
                )
                dsl = self.fix_invalid_attribute_calls(dsl)

            elif not self.check_brackets(dsl):
                print(
                    "❗ Detected unbalanced brackets. Please fix bracket nesting first."
                )
                dsl = self.fix_brackets(code=dsl)

            elif self.has_negative_floats(dsl):
                print("❗ Detected negative floats.")
                dsl = self.fix_negative_floats(code=dsl)

            elif "&&" in dsl or "||" in dsl:
                print(
                    "❗ Detected illegal operators (&&, ||). Auto-fixing to 'and', 'or'"
                )
                dsl = self.check_and_fix_illegal_operators(dsl)

            else:
                print("here")

            return dsl

    def check_brackets(self, code: str) -> bool:
        """
        Checks whether round brackets `()` and curly braces `{}` are balanced in the given DSL code.

        1. Uses ANTLR's token stream to precisely track the position of each bracket.
        2. Maintains two independent stacks: one for parentheses `()` and one for braces `{}`.

        Returns: True/False
        """
        # @qiuhan: TODO: add one more check, need at least a pair of () after ->

        input_stream = InputStream(code)
        lexer = miniDSLLexer(input_stream)
        token_stream = CommonTokenStream(lexer)
        token_stream.fill()

        matched = True

        stack_map = {
            "(": [],
            "{": [],
        }
        open_tokens = {
            miniDSLLexer.LPAREN: "(",
            miniDSLLexer.LBRACE: "{",
        }
        close_tokens = {
            miniDSLLexer.RPAREN: ")",
            miniDSLLexer.RBRACE: "}",
        }
        bracket_pairs = {")": "(", "}": "{"}

        for token in token_stream.tokens:
            if token.type in open_tokens:
                symbol = open_tokens[token.type]
                stack_map[symbol].append((token.line, token.column))

            elif token.type in close_tokens:
                closing = close_tokens[token.type]
                opening = bracket_pairs[closing]
                if stack_map[opening]:
                    stack_map[opening].pop()
                else:
                    print(
                        f"❌ Unmatched closing '{closing}' at line {token.line}, column {token.column}"
                    )
                    matched = False

        for opening, stack in stack_map.items():
            for line, col in stack:
                print(f"❌ Unmatched opening '{opening}' at line {line}, column {col}")
                matched = False

        return matched

    def fix_brackets(self, code: str) -> str:
        """
        1. Separately locates and removes the last `;` and `}` tokens (in that order)
        2. Balances round parentheses
        3. Appends exactly one `;}` at the end
        """
        code = code.rstrip()

        semi_idx = code.rfind(";")
        if semi_idx != -1:
            code = code[:semi_idx] + code[semi_idx + 1 :]

        brace_idx = code.rfind("}")
        if brace_idx != -1:
            code = code[:brace_idx] + code[brace_idx + 1 :]

        # Step 4: Fix unmatched parentheses
        open_paren = code.count("(")
        close_paren = code.count(")")
        if open_paren > close_paren:
            code += ")" * (open_paren - close_paren)
        elif close_paren > open_paren:
            # Insert missing '(' right after each `->`
            deficit = close_paren - open_paren

            def insert_parens(match):
                return f"{match.group(0)}" + "(" * deficit

            code = re.sub(r"->\s*", insert_parens, code)

        # Step 5: Ensure it ends with `;}`
        return code.rstrip() + ";}"

    def has_negative_floats(self, code: str) -> bool:
        input_stream = InputStream(code)
        lexer = miniDSLLexer(input_stream)

        token_stream = CommonTokenStream(lexer)
        token_stream.fill()
        tokens = token_stream.tokens

        for i in range(len(tokens) - 1):
            if (
                tokens[i].type == miniDSLLexer.MINUS
                and tokens[i + 1].type == miniDSLLexer.FloatConst
            ):
                return True
        return False

    def fix_negative_floats(self, code: str) -> str:
        input_stream = InputStream(code)
        lexer = miniDSLLexer(input_stream)
        token_stream = CommonTokenStream(lexer)
        token_stream.fill()
        tokens = token_stream.tokens

        new_code_parts = []
        i = 0
        while i < len(tokens):
            if tokens[i].type == miniDSLLexer.MINUS and i + 1 < len(tokens):
                next_token = tokens[i + 1]
                if next_token.type == miniDSLLexer.FloatConst:
                    new_code_parts.append(f"(0 - {next_token.text})")
                    i += 2
                    continue
            new_code_parts.append(tokens[i].text)
            i += 1

        return "".join(new_code_parts)

    def fix_invalid_attribute_calls(self, code: str) -> str:
        """
        Fixes `expr.sum`, `expr.avg`, `expr.len` → `sum(expr)` etc.
        """
        pattern = re.compile(r"(\w+\s*\[[^\]]+\])\s*\.(sum|avg|len)\b")

        def replace(match):
            inner = match.group(1)
            func = match.group(2)
            print(f"❌ [Fix] Rewriting '{inner}.{func}' → '{func}({inner})'")
            return f"{func}({inner})"

        return pattern.sub(replace, code)

    def check_and_fix_illegal_operators(self, code: str) -> str:
        """
        Replace illegal operators like '&&' with legal 'and', and '||' with 'or'.
        """
        fixed = code.replace("&&", "and").replace("||", "or")
        return fixed


if __name__ == "__main__":
    dsl = """
    transformer T {
        Avgpool -> ((1.0 / curr[size]) * (prev[l].sum));
    }
    """

    fixed_code = SyntaxChecker().check(dsl)
    print(fixed_code)
