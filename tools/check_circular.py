#!/usr/bin/env python

import os
import sys
import networkx as nx

def build_graph(path):
    graph = nx.DiGraph()
    for f in os.listdir(path):
        graph.add_node(f)
        cnt = 0
        with open(path+"/"+f) as fp:
            line = fp.readline()
            enter_include = False
            exit_include = False
            while line and not exit_include:
                if line.startswith('#include "'):
                    enter_include = True
                    node = line.split()[1].replace('"', '')
                    graph.add_node(node)
                    graph.add_edge(f, node)
                elif enter_include:
                    exit_include = True
                line = fp.readline()
    return graph

def main():
    graph = build_graph("../include/xtensor")
    cycle = list(nx.simple_cycles(graph))
    for x in cycle:
        print(x)
    exception_message = ' - '.join([str(y) for y in cycle])
    if len(cycle) != 0:
        raise Exception('CircularInclude', exception_message)

main()
