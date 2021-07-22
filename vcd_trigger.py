#!/usr/bin/env python3

from os import name
from typing import Dict, List, Tuple
from sly import Lexer, Parser
import inspect
from Verilog_VCD import parse_vcd
from pathlib import Path
import pandasql as ps 
import sys
from argparse import ArgumentParser


fst = lambda x: x[0]
snd = lambda x: x[1]
pt = lambda x : print(type(x))

# data, addr : sig1 == 2 and sig1 < 3 [ : head 5 ] 
class CalcLexer(Lexer):
    tokens = {
        NAME,
        NUMBER,
        STAR,
        
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
    STAR = r'\*'

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

    @_('STAR')
    def select(self, p):
        return "*"

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

def preprocess(vcd, select_names=['*']):
    # TODO
    # it drops fst_states as they seems not to have 'tv' key. aliases?
    def gen_name(v):
        name = v['nets'][0]['name'] 
        hier = v['nets'][0]['hier']
        if hier == 'top':
            return name
        name = ".".join(hier.split('.')[1:]) + "." + name
        return name
    res = {gen_name(v): v for _, v in vcd.items() if 'tv' in v}
    if '*' in select_names:
        return res
    return {k: v for k, v in res.items() if k in select_names}


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

def parse_query(input) -> Tuple[str, List[str]]:
    lexer = CalcLexer()
    parser = CalcParser()
    select, where, where_idents = parser.parse(lexer.tokenize(input))
    if '*' not in select: 
        select.update(where_idents)
    q = gen_query("df", select, where)
    return q, select

def do_find(vcd_path, list):
    vcd = parse_vcd(vcd_path)
    vcd = preprocess(vcd)
    res = [k for k, v in vcd.items() if list in k]
    from pprint import pprint
    pprint(res)
        
def do_query(vcd_path, query):
    q, select = parse_query(query)        
    vcd = parse_vcd(vcd_path)
    vcd = preprocess(vcd, select)
    df = gen_table(vcd)
    sanity_check(df, select)
    print(df)
    res = ps.sqldf(q, locals())
    print(q)
    print(res)

def sanity_check(df, select):
    all = list(df.keys()) 
    for sig in select:
        if sig not in all:
            print(f"ERROR: There is no '{sig}' signal in given waveform!")
            exit(1)

def main(query, list):
    # vcd_path = Path("/home/mateusz/github/mig/mod.vcd")
    vcd_path = Path("/home/mateusz/github/mtkcpu/jtag.vcd")

    if query:
        do_query(vcd_path, query)
    elif list:
        do_find(vcd_path, list)
    else:
        raise ValueError("Either --list or --query must be specified!")


if __name__ == '__main__':
    parser = ArgumentParser(description="Trigger VCD")
    parser.add_argument("--query", required=False)
    parser.add_argument("--list", required=False)
    main(**vars(parser.parse_args()))
    