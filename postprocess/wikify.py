from typing import List, Tuple
import wikipedia
import re

def insert_wiki_triples(wiki_triples, triples_without_wiki) -> List[Tuple[str]]:
    wiki_triples = list(reversed(wiki_triples))

    i = 0
    while wiki_triples:

        # if the current ne category (country, name, person) matches the current wiki triple category
        if triples_without_wiki[i][0] == wiki_triples[-1][0] and triples_without_wiki[i][1] == ":name":
            triples_without_wiki.insert(i, wiki_triples.pop())

            i += 1 # increment the index to insert the next wiki triple
        i += 1
    return triples_without_wiki

# def wikify(triples: List[Tuple[str]]) -> List[Tuple[str]]:
#
#     named_entities = extract_named_entities(triples)
#     wiki_triples = []
#
#     if len(named_entities) == 0:
#         return triples # no named entities to wikify in the input
#
#     else:
#         for parent_node, named_entity in named_entities:
#             try: # may an exception raised from API call
#                 wiki_ref = get_wiki_ref(named_entity)
#                 wiki_triples.append((parent_node, ":wiki", wiki_ref))
#             except Exception as e:
#                 print(f"Error: {e}")
#                 print(f"Could not get wiki reference for {named_entity}")
#                 wiki_triples.append((parent_node, ":wiki", "-"))
#
#         # insert the wiki triples in the right place
#         new_triples = insert_wiki_triples(wiki_triples, triples)
#
#         return new_triples

def wikify_simple(triples: List[Tuple[str]]) -> List[Tuple[str]]:
    '''
    Restore the named entities in the triples with their corresponding wiki references
    Add them to the triples list
    '''
    wiki_triples = []

    for triple in triples:
        src, rel, tgt = triple

        # if tgt is wrapped in "" and the relation type is name
        if re.match(r'".*"', tgt) and rel.strip() == ":name":
            named_entity = tgt.replace('"', "")
            named_entity = named_entity.replace("_", " ")

            try: # may an exception raised from API call
                wiki_ref = get_wiki_ref(named_entity)
                wiki_triples.append((src, ":wiki", wiki_ref))

            except Exception as e:
                print(f"Error: {e}")
                print(f"Could not get wiki reference for {named_entity}")
                wiki_triples.append((src, ":wiki", wiki_ref)) # TODO: make sure it does not introduce errors during post processing

    # insert the wiki triples in the right place
    new_triples = insert_wiki_triples(wiki_triples, triples)
    return new_triples

def fill_space(s: str) -> str:
    return s.replace(" ", "_")

def get_wiki_ref(instance: str) -> str:
    none_wiki = "-"
    search_results = wikipedia.search(instance)

    if len(search_results) > 0:
        search_result = fill_space(search_results[0])
        return '"' + search_result + '"'  # the most relevant result
    else:
        return none_wiki

# def extract_named_entities(triples: List[Tuple[str]]) -> List[Tuple[str]]:
#
#     boi = "outside" # beginning, inside, outside the instance
#     named_entities = []
#     curr_named_entity = ""
#     curr_ne_category = ""
#     name_concept = re.compile(r'(name|n[0-9]*)') # either 'name' concept or 'n' followed by a number (e.g. n, n1, n2, ...)
#
#     for triple in triples:
#         src, rel, tgt = triple
#         # if the relation is :name and the current wiki instance is empty
#         if rel == ":name" and curr_named_entity == "":
#             curr_ne_category = src
#             boi = "beginning"
#
#         # if the relation is :opX and the current wiki instance is empty
#         elif re.fullmatch(name_concept, src) and rel.startswith(":op") and boi == "beginning":
#             curr_named_entity = tgt.replace('"', "")
#             boi = "inside"
#
#         # if the relation is :opX and the current wiki instance is not empty
#         elif re.fullmatch(name_concept, src) and rel.startswith(":op") and boi == "inside":
#             curr_named_entity += "_" + tgt.replace('"', "")
#
#         # if the relation is not :opX and the current wiki instance is not empty
#         else:
#             if curr_named_entity != "":
#                 named_entities.append((curr_ne_category, '"' + curr_named_entity + '"'))
#                 curr_named_entity = ""
#                 boi = "outside"
#
#     if curr_named_entity != "":
#         named_entities.append((curr_ne_category, '"' + curr_named_entity + '"')) # add the last named entity if it is not empty
#
#     return named_entities


if __name__ == '__main__':
    from AMR_utils import to_single_line_amr
    from AMR_utils import noop_penman_decode
    from AMRSimplifier import AMRSimplifier
    from postprocess.postprocess_utils import amr_str_to_list
    # from postprocess.str_2_triple import amr_str_to_list, restore_vars


    amr1 = """
    (t / terrify-01
      :ARG0 (c / column
            :ord (o / ordinal-entity :value 5))
      :ARG1 (p / person
            :ARG1-of (i / interest-01
                  :ARG2 (p2 / political-party :wiki "Communist_Party_of_China" :name (n / name :op1 "Communist" :op2 "Party"))
                  :ARG2-of (v / vest-01)))
      :mod (t2 / thing
            :ARG1-of (g / good-02)))
    """
    amr2 = """
(p / prohibit-01
      :ARG1 (c / country
            :ARG2-of (i / include-01
                  :ARG1 (a / and
                        :op1 (c2 / country :wiki "India"
                              :name (n / name :op1 "India"))
                        :op2 (c3 / country :wiki "Israel"
                              :name (n2 / name :op1 "Israel"))
                        :op3 (c4 / country :wiki "Pakistan"
                              :name (n3 / name :op1 "Pakistan"))))
            :ARG0-of (s / sign-01 :polarity -
                  :ARG1 (t / treaty :wiki "Treaty_on_the_Non-Proliferation_of_Nuclear_Weapons"
                        :name (n4 / name :op1 "Nuclear" :op2 "Non-Proliferation" :op3 "Treaty"))))
      :ARG2 (p2 / participate-01
            :ARG0 c
            :ARG1 (t2 / trade-01
                  :mod (n5 / nucleus)
                  :mod (i2 / international)
                  :ARG2-of (i3 / include-01
                        :ARG1 (p3 / purchase-01
                              :ARG1 (o / or
                                    :op1 (r / reactor)
                                    :op2 (f / fuel
                                          :mod (u / uranium))
                                    :op3 (y / yellowcake)))))))

    """

    amr = to_single_line_amr(amr2)
    g = noop_penman_decode(amr)
    simplifier = AMRSimplifier(g)
    amr = simplifier.simplify()
    amr = amr_str_to_list(amr)
    amr = wikify(amr)
    print("done")









