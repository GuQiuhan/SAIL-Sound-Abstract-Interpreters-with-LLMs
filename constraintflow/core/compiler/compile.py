import antlr4 as antlr

from constraintflow.core.ast_cflow import astBuilder, astTC, dslLexer, dslParser
from constraintflow.core.compiler import codeGen
from constraintflow.core.compiler import convertToIr as c2r
from constraintflow.core.compiler import representations
from constraintflow.core.compiler.optimizations import (
    copyPropagation,
    cse,
    dce,
    loopInvariantCodeMotion,
    polyOpt,
    rewrite,
    symexpCount,
)

optimizations_rewrite = [
    cse.cse,
    copyPropagation.copy_proagate,
    cse.cse,
    copyPropagation.copy_proagate,
    polyOpt.poly_opt,
    cse.cse,
    dce.dce,
    rewrite.rewrite,
    cse.cse,
    copyPropagation.copy_proagate,
    dce.dce,
    dce.dce,
    dce.dce,
    loopInvariantCodeMotion.licm,
    cse.cse,
    copyPropagation.copy_proagate,
    cse.cse,
    copyPropagation.copy_proagate,
    cse.cse,
    copyPropagation.copy_proagate,
    cse.cse,
    symexpCount.correct_symexp_size,
    copyPropagation.copy_proagate,
]


def compile(inputfile, output_path):
    lexer = dslLexer.dslLexer(antlr.FileStream(inputfile))
    tokens = antlr.CommonTokenStream(lexer)
    parser = dslParser.dslParser(tokens)
    tree = parser.prog()

    ast = astBuilder.ASTBuilder().visit(tree)
    astTC.ASTTC().visit(ast)

    ir = c2r.ConvertToIr().visit(ast)
    representations.ssa(ir)

    optimizations = optimizations_rewrite

    for opt in optimizations:
        opt(ir)
    representations.remove_phi(ir)
    codeGen.CodeGen(output_path).visit(ir)

    return True
