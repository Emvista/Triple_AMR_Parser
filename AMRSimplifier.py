"""
This script is used to simplify AMR graphs by removing variables and wiki links.
"""
from AMR_utils import noop_penman_decode
from typing import Tuple, List, Dict
from penman.graph import Graph
import logging
from argparse import ArgumentParser
from AMR_utils import triples_to_str
from postprocess.postprocess_utils import concat_triples_to_str

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)
def is_same_order(ref_list, mod_list):
    # sanity check to make sure the order of the triples is preserved after simplification
    # this is not a perfect test since it does not take into account coreferences
    # but it is a good enough test for now
    original_index = 0

    for i, element in enumerate(mod_list):
        index = ref_list.index(element, original_index)
        if original_index <= index:
            original_index = index
        else:
            return False

    return True

def is_relation_triple(triple: Tuple[str]) -> bool:
    # instance triples: (x, :instance, y)
    # relation triples: non instance triples
    return triple[1] != ":instance"


class AMRSimplifier:
    def __init__(self, graph:Graph):
        self.graph = graph
        self.var_to_instance = self.map_var_to_instance(self.graph.triples)

    @staticmethod
    def map_var_to_instance(triples) -> Dict[str, str]:
        # return a dict of {variable: instance}
        mapping = {}
        for triple in triples:
            if not is_relation_triple(triple):
                mapping.update({triple[0]: triple[2]})
        return mapping

    @staticmethod
    def filter_instance_triples(triples: List[Tuple[str]]) -> List[Tuple[str]]:
        relation_triples = []

        for triple in triples:
            if is_relation_triple(triple):
                relation_triples.append(triple)

        return relation_triples

    def simplify(self)-> str:
        orig_triples = self.graph.triples
        triples = self.filter_instance_triples(orig_triples)

        if len(triples) == 0: # if there are no relation triples, return the original triples as it is
            triples = orig_triples

        triples = self.remove_variables(triples)
        triples = self.remove_wiki(triples)

        orig_triples = self.remove_variables(self.graph.triples)
        is_ok = is_same_order(orig_triples, triples)
        if not is_ok:
            logging.error(f"Order of triples is not preserved after simplification: {self.graph.triples} -> {triples}")

        return concat_triples_to_str(triples)

    def remove_variables(self, triples: List[Tuple[str]]) -> List[Tuple[str]]:
        # replace variables with their instances
        simple_triples = []
        for src, rel, tgt in triples:
            if src in self.var_to_instance:
                src = self.var_to_instance[src]
            if tgt in self.var_to_instance:
                tgt = self.var_to_instance[tgt]
            simple_triples.append((src, rel, tgt))

        return simple_triples
    def remove_wiki(self, triples) -> List[Tuple[str]]:
        # remove wiki links
        simple_triples = []
        for src, rel, tgt in triples:
            if rel != ":wiki":
                simple_triples.append((src, rel, tgt))

        return simple_triples

