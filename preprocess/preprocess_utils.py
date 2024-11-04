import re

def remove_emptyspace(amr:str) -> str:
    # remove extra spaces
    # removing all spaces causes unwanted results
    # e.g. :modeimperative => cannot be restored to :mode imperative

    new_string = amr.replace("\n", " ")
    new_string = new_string.replace("  ", " ")
    new_string = new_string.replace(" )", ")")
    new_string = new_string.replace("( ", "(")
    new_string = new_string.replace(" )", ")")
    new_string = new_string.replace(" )", ")")
    new_string = new_string.replace(" )", ")")
    return new_string

def apply_regex_recursively(pattern, replacement, string):
    while True:
        new_string = re.sub(pattern, replacement, string)
        # Check if the substitution changed the string
        if new_string == string:
            break
        string = new_string
    return new_string

def add_emptyspace(amr:str) -> str:
    # IMPORTANT: The order of the following regex is important
    # Do not change the order of the following regex
    # Add spac ebefore and after '/' when it is variable / concept
    new_string = re.sub(r"\(([\S]+?)/([\S]+?)", r"( \1 / \2", amr)
    # Add space before '"named entity"' if not present already
    new_string = re.sub(r'([\S])"([\S]*?)"', r'\1 "\2"', new_string)
    # Add space before ")" if not present already
    new_string = apply_regex_recursively(r"([\S])\)", r"\1 )", new_string)
    # Add space after ")" if not present already
    new_string = apply_regex_recursively(r"\)([\S])", r") \1", new_string)
    # Add space after "(" if not present already
    new_string = apply_regex_recursively(r"\(([\S])", r"( \1", new_string)
    # Add space before "(" if not present already
    new_string = apply_regex_recursively(r"([\S])\(", r"\1 (", new_string)
    # Add space before ":" if not present already
    new_string = apply_regex_recursively(r"([\S]):", r"\1 :", new_string)
    # remove space when it is url
    new_string = re.sub(r"(https?)\s:", r"\1:", new_string)
    return new_string

def add_space_between_quotes(amr:str) -> str:
    # Add space before and after quotes if not present already
    new_string = re.sub(r'\"([^\"\s]+?)\"', r'" \1 "', amr)
    return new_string

def ne_recategorize(input_string:str, with_variables=True) -> str:
    #TODO: test for with variable and without variable!
    '''
    Recategorize named entities to simplify the AMR string
    before recategorization: (w3 / world-region :name (n2 / name :op1 "North" :op2 "Africa"))
    after recategorization: (w3 / world-region :name "North_Africa"))
    '''

    original_input_string = input_string
    input_string = remove_emptyspace(input_string)
    if with_variables:
        # patterns to find sth like  ':name (v2 / name :op1 "North" :op2 "Africa" :op3 14)'
        named_entity_pattern = r':\s?name\s?(\([a-z]\d* \/ name(?: :op\d+ (?:\"[^\"]*\"|\d+))+\))'
    else:
        # pattern to find sth like ':name (name :op1 " 1st " :op2 " Amendment ")'
        named_entity_pattern = r':\s?name\s?(\(name(?: :op\d+ (?:\"[^\"]*\"))+\s?\))'

    # patterns to find ["North", "Africa", 14] from (:op1 "North" :op2 "Africa" :op3 14)
    word_pattern = r':op\d+\s+("[^"]+"|\d+)|:op\d+\s+'
    # Find all matches of name patterns in the input string
    named_entities = re.findall(named_entity_pattern, input_string)

    for named_entity in named_entities:
        # Replace the name pattern with the name pattern with the correct NER category
        words = re.findall(word_pattern, named_entity)
        words = [word.strip('"') for word in words]
        words = [word.strip() for word in words]
        concat_words = '_'.join(words)
        # name_string = name_string.replace('"','') # remove quotes
        input_string = input_string.replace(named_entity, f'" {concat_words} "')

    # to avoid an error case e.g. (n / name :op1 "DeSwiss") -> seems to be an annotation error in the data
    if len(input_string) > 0 and  "(" not in input_string:
        print("## Error in recategorizing named entities, returning original string ##")
        return original_input_string

    return add_emptyspace(input_string)




if __name__ == '__main__':
    # add empty space to van noord linearized file
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=Path, help="input file path")
    args = parser.parse_args()

    output = args.input_file.with_suffix(".vnd")
    results = []
    with open(args.input_file, 'r') as f:
        lines = [line.rstrip() for line in f.readlines()]
        for line in lines:
            recat = ne_recategorize(line, with_variables=False)
            with_space = add_emptyspace(recat)
            with_space = add_space_between_quotes(with_space)
            results.append(with_space)

    with open(output, 'w') as f:
        f.write("\n".join(results))
        f.write("\n")

