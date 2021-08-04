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
from difflib import get_close_matches
from typing import List, Optional, Dict
from pprint import pprint


fst = lambda x: x[0]
snd = lambda x: x[1]
pt = lambda x : print(type(x))

# data, addr : sig1 == 2 and sig1 < 3 [ : head 5 ] 
class CalcLexer(Lexer):
    tokens = {
        NAME,
        NUMBER,
        STAR,
        
        # HEAD, 
        # TAIL, 

        LPAREN, RPAREN,
        AND, OR, EQ, NEQ
    }
    ignore = ' \t'

    literals = {'{', '}', '[', ']', ',', ':'}

    # Tokens
    NAME = r'[a-zA-Z_][a-zA-Z0-9_\.\$]*'
    # NUMBER = r'\d+'
    NUMBER = r'(0[bx])?[0-9a-fA-F]+' # catch also hex and bin
    STAR = r'\*'

    AND = r'&'
    OR = r'\|'
    EQ = r'=='
    NEQ = r'!='
    LPAREN = r'\('
    RPAREN = r'\)'

    # HEAD = r'HEAD'
    # TAIL = r'TAIL'

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

    @_('select')
    def top(self, p):
        self.selected = set(p.select)
        return self.selected, None, None

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
        n = p.NUMBER
        # base = 10
        # if len(n) >= 2:
        #     base = {
        #         'b': 2,
        #         'x': 16,
        #     }.get(n[1], 10)
        return n # str(int(n, base=base))

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
        sep = '_'
        name = sep.join(hier.split('.')[1:]) + sep + name
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
        def gen_val(v):
            base_map = {
                'b': 2,
                'x': 16,
            }
            base = base_map.get(v[0], 10)
            if v[0] < '0' or v[0] > '9':
                v = v[1:]
            return int(v, base=base)
        vs = list(map(gen_val, vs))
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
        if where_idents:
            select.update(where_idents)
    q = gen_query("df", select, where)
    return q, select

def do_find(vcd_path, list, verbose=False):
    vcd = parse_vcd(vcd_path)
    vcd = preprocess(vcd)
    res = [k for k, _ in vcd.items() if list in k]
    if verbose:
        pprint(res)
    return res
        
def do_query(vcd_path, query):
    q, select = parse_query(query)        
    vcd = parse_vcd(vcd_path)
    pprint(vcd)
    vcd = preprocess(vcd, select)
    all = do_find(vcd_path, "", verbose=False)
    sanity_check(all, select)
    df = gen_table(vcd)
    print(f"=== =========================================")
    pprint(vcd)
    print(f"=== df len: {len(df)}")
    print(df.head())
    res = ps.sqldf(q, locals())
    print(q)

    # TODO make it configurable, e.g.
    # select clk as hex, addr
    print(res.applymap(hex))

def sanity_check(all, select):
    if "*" in select:
        return
    for sig in select:
        if sig not in all:
            print(f"ERROR: There is no '{sig}' signal in given waveform!")
            maybe = get_close_matches(sig, all, n=10)
            print(f"Maybe you meant one of following?:")
            pprint(maybe)
            exit(1)

def main(query, list, path):
    if query:
        do_query(path, query)
    elif list:
        do_find(path, list, verbose=True)
    else:
        raise ValueError("Either --list or --query must be specified!")

if __name__ == '__main__':
    parser = ArgumentParser(description="Trigger VCD")
    parser.add_argument("--query", required=False)
    parser.add_argument("--list", required=False)
    parser.add_argument("--path", required=False, default="example_vcd/mod.vcd") # "/home/mateusz/github/mtkcpu/jtag.vcd")
    main(**vars(parser.parse_args()))
    