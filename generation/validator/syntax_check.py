import re
from typing import Optional, Tuple

from antlr4 import *
from antlr4 import CommonTokenStream, InputStream
from antlr4.error.ErrorListener import ConsoleErrorListener, ErrorListener
from antlr4.error.ErrorStrategy import BailErrorStrategy

from generation.validator.miniDSL.miniastBuilder import ASTBuilder
from generation.validator.miniDSL.miniDSLLexer import miniDSLLexer
from generation.validator.miniDSL.miniDSLParser import miniDSLParser


class SilentErrorListener(ErrorListener):
    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        pass


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


class SyntaxChecker:
    """
    - Only supports DSLs with one transformer and one op_stmt.
    Hard Coding.

    Returns: (success: bool, fixed_code: str, err: Optional[str])
    """

    def __init__(self):
        # self.token_map = parse_tokens_file(TOKENS_FILE)
        self.MAX_RETRIES = 5
        self.metadata = ["WEIGHT", "BIAS", "EQUATIONS", "LAYER"]

    def check(self, dsl: str) -> tuple[bool, str, Optional[str]]:
        last_attempt = None

        for attempt in range(self.MAX_RETRIES):
            try:
                input_stream = InputStream(dsl)
                lexer = miniDSLLexer(input_stream)
                token_stream = CommonTokenStream(lexer)

                parser = miniDSLParser(token_stream)
                parser.removeErrorListeners()
                parser.addErrorListener(SilentErrorListener())
                parser._errHandler = BailErrorStrategy()

                parser.transformer()
                print("✅ Syntax correct.")
                return True, dsl, None
            except:
                if re.search(r"\.\s*(sum|avg|len)\b", dsl):
                    print(
                        "❗ Detected invalid attribute call like '.sum', attempting to fix."
                    )
                    dsl = self.fix_invalid_attribute_calls(dsl)
                    last_attempt = "Issue Type: Invalid attribute call like '.sum'."

                elif not self.check_brackets(dsl):
                    print("❗ Detected unbalanced brackets. Attempting to fix.")
                    dsl = self.fix_brackets(code=dsl)
                    last_attempt = "Issue Type: Unbalanced brackets."

                elif "&&" in dsl or "||" in dsl or "&" in dsl:
                    print("❗ Detected illegal operators. Attempting to fix.")
                    dsl = self.check_and_fix_illegal_operators(dsl)
                    last_attempt = (
                        "Issue Type: Illegal logical operators like '&&', '||', '&'."
                    )

                elif "DOT" in dsl or any(kw in dsl for kw in self.metadata):
                    print(
                        "❗ Detected DOT or uppercase metadata usage. Attempting to fix."
                    )
                    dsl = self.fix_dot_and_metadata(dsl)
                    last_attempt = "Issue Type: DOT or uppercase metadata usage."

                elif re.search(r'\[\s*"(?i)(layer|weight|bias|equations)"\s*\]', dsl):
                    print("❗ Detected metadata in string form. Attempting to fix.")
                    dsl = self.fix_metadata_quotes(dsl)
                    last_attempt = (
                        'Issue Type: Metadata used as quoted string (e.g., "layer").'
                    )

                else:
                    print("❗ Syntax error but unknown cause.")
                    last_attempt = "Unknown syntax error."

                """
                elif self.has_negative_floats(dsl):
                    print("❗ Detected negative floats. Attempting to fix.")  # TODO
                    dsl = self.fix_negative_floats(code=dsl, token_map=self.token_map)
                    last_attempt = "Issue Type: Negative float constant."
                """

        return False, dsl, last_attempt

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
                        f"❌ Unmatched closing '{closing}' at line {token.line}, column {token.column}, attempting to fix."
                    )
                    matched = False

        for opening, stack in stack_map.items():
            for line, col in stack:
                print(
                    f"❌ Unmatched opening '{opening}' at line {line}, column {col}, attempting to fix."
                )
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

        print(f"⚠️ [Fixed]")

        # Step 5: Ensure it ends with `;}`
        return code.rstrip() + ";}"

    # about the negative float, need to modify constraintflow instead of check it later
    def has_negative_floats(self, code: str) -> bool:

        """
        input_stream = InputStream(code)
        lexer = miniDSLLexer(input_stream)
        token_stream = CommonTokenStream(lexer)
        token_stream.fill()
        tokens = token_stream.tokens

        for token in token_stream.tokens:
            name = self.token_map.get(token.type, "UNKNOWN")
            print(
                f"{token.text:<20} -> {token.type:<4} ({name})"
            )

        for i in range(len(tokens) - 1):
            if self.token_map.get(token[i].type, "UNKNOWN") == 'MINUS' and self.token_map.get(token[i+1].type, "UNKNOWN") == 'FloatConst':
                return True
        return False
        """
        return False

    def fix_negative_floats(self, code: str, token_map: dict) -> str:
        input_stream = InputStream(code)
        lexer = miniDSLLexer(input_stream)
        token_stream = CommonTokenStream(lexer)
        token_stream.fill()
        tokens = token_stream.tokens

        new_code_parts = []
        i = 0
        while i < len(tokens):
            if tokens[i].type == token_map["MINUS"] and i + 1 < len(tokens):
                if tokens[i + 1].type == token_map["FloatConst"]:
                    new_code_parts.append(f"(0 - {tokens[i + 1].text})")
                    i += 2
                    continue
            new_code_parts.append(tokens[i].text)
            i += 1

        print(f"⚠️ [Fixed]")

        return "".join(new_code_parts)

    def fix_invalid_attribute_calls(self, code: str) -> str:
        """
        Fixes `expr.sum`, `expr.avg`, `expr.len` → `sum(expr)` etc.
        """
        pattern = re.compile(r"(\w+\s*\[[^\]]+\])\s*\.(sum|avg|len)\b")

        def replace(match):
            inner = match.group(1)
            func = match.group(2)
            print(f"⚠️ [Fixed] Rewriting '{inner}.{func}' → '{func}({inner})'")
            return f"{func}({inner})"

        return pattern.sub(replace, code)

    def check_and_fix_illegal_operators(self, code: str) -> str:
        """
        Replace illegal operators like '&&' with legal 'and', and '||' with 'or'.
        """
        fixed = code.replace("&&", "and").replace("||", "or").replace("&", "and")
        return fixed

    def fix_dot_and_metadata(self, code: str) -> str:
        """
        Fix `DOT` to `.` and convert all metadata tokens (like WEIGHT, BIAS) to lowercase.
        """

        # Replace DOT with .
        code = re.sub(r"\bDOT\b", ".", code)

        # Replace uppercase metadata with lowercase
        for keyword in self.metadata:
            code = re.sub(rf"\b{keyword}\b", keyword.lower(), code)

        return code

    def fix_metadata_quotes(self, code: str) -> str:
        """
        Fixes incorrect metadata usage like curr["layer"] -> curr[layer]
        """
        pattern = re.compile(r'\[\s*"(layer|weight|bias|equations)"\s*\]')

        def replace(match):
            metadata = match.group(1)
            print(f"⚠️ [Fixed] Rewriting '\"{metadata}\"' → {metadata}")
            return f"[{metadata}]"

        return pattern.sub(replace, code)


if __name__ == "__main__":
    dsl = """
transformer deeppoly {
    Avgpool -> (
        (curr["layer"]),
       0,0,0
    );
}
    """

    result, fixed_code, err = SyntaxChecker().check(dsl)
    print(result)

    print(fixed_code)

    print(err)
