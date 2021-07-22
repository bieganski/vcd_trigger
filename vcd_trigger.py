#!/usr/bin/env python3

from typing import Dict, List
from sly import Lexer, Parser
import inspect
from Verilog_VCD import parse_vcd
from pathlib import Path
import pandasql as ps 
import sys

fst = lambda x: x[0]
snd = lambda x: x[1]
pt = lambda x : print(type(x))

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
        self.where_idents = set()

    @_('select ":" expr')
    def top(self, p):
        self.selected = set(p.select)
        self.condition = p.expr
        return self.selected, self.condition, self.where_idents

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
        self.where_idents.add(p.NAME)
        return p.NAME

def preprocess(vcd):
    # TODO
    # it drops fst_states as they seems not to have 'tv' key. aliases?
    return {v['nets'][0]['name']: v for _, v in vcd.items() if 'tv' in v}


def get_timepoints(vcd):
    tvs = set()
    for sig in vcd.values():
        tvs.update(map(fst, sig['tv']))
    return sorted(tvs)

# returns VCD as a pandas DataFrame.
def gen_table(vcd):
    import numpy as np
    import pandas as pd
    all_ts = get_timepoints(vcd)
    # print(f"ALL: {all_ts}")
    table = []
    for _, sig in vcd.items():
        timevals = sig['tv'] # list of pairs [time, val] 
        ts, vs = list(zip(*timevals)) # unzip
        assert ts[0] == 0
        # print("===============================")
        # print(f"ts: {ts}")
        # print(f"vs: {vs}")
        counts = np.searchsorted(all_ts, ts, side="right")
        counts = np.diff(counts)
        counts = np.append(counts, [len(all_ts) - sum(counts)])
        # print(f"counts: {counts}")
        res = np.repeat(vs, counts)
        # print(f"res: {res}")
        table.append(res)
    table = np.vstack(table)
    names = vcd.keys()
    table = pd.DataFrame(table,index=names).T
    return table


def var_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    all = [var_name for var_name, var_val in callers_local_vars if var_val is var]
    assert(len(all) == 1)
    return all[0]


from typing import List, Optional, Dict
def gen_query(table_varname : str, select, where: str):
    select = ",".join(select)
    q = f"select {select} from {table_varname}"
    if where:
        q += f" where {where}"
    return q

def parse_query(input):
    lexer = CalcLexer()
    parser = CalcParser()
    select, where, where_idents = parser.parse(lexer.tokenize(input))
    select.update(where_idents)
    q = gen_query("df", select, where)
    return q

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("ERROR: no input specified")
        exit(1)
    q = parse_query(sys.argv[1])
        
    # vcd_path = Path("/home/mateusz/github/mig/mod.vcd")
    vcd_path = Path("/home/mateusz/github/mtkcpu/jtag.vcd")
    vcd = parse_vcd(vcd_path)
    vcd = preprocess(vcd)
    df = gen_table(vcd)
    print(df)

    res = ps.sqldf(q, globals())
    print(q)
    print(res)

