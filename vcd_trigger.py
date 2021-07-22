#!/usr/bin/env python3

from sly import Lexer, Parser


# data, addr : sig1 == 2 and sig1 < 3 [ : head 5 ] 
class CalcLexer(Lexer):
    tokens = {
        NAME,
        NUMBER,
        HEAD, 
        TAIL, 

        LPAREN, RPAREN,
        AND, OR, EQ, NEQ
    }
    ignore = ' \t'

    literals = {'{', '}', '[', ']', ',', ':'}

    # Tokens
    NAME = r'[a-zA-Z_][a-zA-Z0-9_]*'
    NUMBER = r'\d+'

    AND = r'&'
    OR = r'\|'
    EQ = r'=='
    NEQ = r'!='
    LPAREN = r'\('
    RPAREN = r'\)'

    HEAD = r'HEAD'
    TAIL = r'TAIL'

    # Ignored pattern
    ignore_newline = r'\n+'

    # Extra action for newlines
    def ignore_newline(self, t):
        self.lineno += t.value.count('\n')

    def error(self, t):
        print("Illegal character '%s'" % t.value[0])
        self.index += 1

class CalcParser(Parser):
    tokens = CalcLexer.tokens

    precedence = (
        ('left', OR, AND),
        ('left', EQ, NEQ),
    )

    def __init__(self):
        self.names = {}

        self.selected = {}
        self.condition = {}

    @_('select ":" expr')
    def top(self, p):
        self.selected = set(p.select)
        self.condition = p.expr
        return self.selected, self.condition

    @_('NAME "," select')
    def select(self, p):
        return [p.NAME] + p.select

    @_("NAME")
    def select(self, p):
        return [p.NAME]

    @_('expr AND expr')
    def expr(self, p):
        return p.expr0 + "&&" + p.expr1

    @_('expr OR expr')
    def expr(self, p):
        return p.expr0 + "||" + p.expr1

    @_('expr EQ expr')
    def expr(self, p):
        return p.expr0  + "==" + p.expr1

    @_('expr NEQ expr')
    def expr(self, p):
        return p.expr0  + "!=" + p.expr1

    @_('LPAREN expr RPAREN')
    def expr(self, p):
        return p.expr

    @_('NUMBER')
    def expr(self, p):
        return p.NUMBER # int(p.NUMBER)

    @_('NAME')
    def expr(self, p):
        return p.NAME


if __name__ == '__main__':
    lexer = CalcLexer()
    parser = CalcParser()
    input = "sig1, sig2 : aa == 2 "
    print(input)
    select, where = parser.parse(lexer.tokenize(input))
    print(f"sel: {select}, cond: {where}")
    # while True:
    #     try:
    #         text = input('calc > ')
    #     except EOFError:
    #         break
    #     if text:
    #         parser.parse(lexer.tokenize(text))