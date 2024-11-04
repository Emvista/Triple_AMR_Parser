# original code from https://github.com/RikVN/AMR/blob/master/reformat_single_amrs.py
# minimal modification to handle single amr line (original code handles a file with multiple amr lines)

def reverse_tokenize(new_line):
    while ' )' in new_line or '( ' in new_line:                 #restore tokenizing
        new_line = new_line.replace(' )',')').replace('( ','(')

    return new_line

def variable_match(spl, idx, no_var_list):
    '''Function that matches entities that are variables occurring for the second time'''
    # Beginning or end are never variables
    if idx >= len(spl) or idx == 0:
        return False
    return spl[idx-1] != '/' and any(char.isalpha() for char in spl[idx]) and spl[idx] not in no_var_list and not spl[idx].startswith(':') and len([x for x in spl[idx] if x.isalpha() or x.isdigit()]) == len(spl[idx]) and (len(spl[idx]) == 1 or (len(spl[idx]) > 1 and spl[idx][-1].isdigit()))


def tokenize_line(line):
    new_l = line.replace('(', ' ( ').replace(')',' ) ')
    # We want to make sure that we do Wiki links correctly
    # They always look like this :wiki "link_(information)"
    new_l = new_l.replace('_ (', '_(').replace(') "', ')"')
    return " ".join(new_l.split())

def reformat_to_multiline(single_line_amr:str) -> str:
    # receive singline penman and return multiline penman (from penman to penman)

    tokenized_line = tokenize_line(single_line_amr).split()
    num_tabs = 0
    amr_string = []

    # Loop over parts of tokenized line
    for count, part in enumerate(tokenized_line):
        if part == '(':
            num_tabs += 1
            amr_string.append(part)
        elif part == ')':
            num_tabs -= 1
            amr_string.append(part)
        elif part.startswith(':'):
            try:
                # Variable coming up
                if tokenized_line[count + 3] == '/':
                    amr_string.append('\n' + num_tabs * '\t' + part)
                # Variable coming, add newline here
                elif variable_match(tokenized_line[count + 1]):
                    amr_string.append('\n' + num_tabs * '\t' + part)
                else:
                    amr_string.append(part)
            except:
                amr_string.append(part)
        else:
            amr_string.append(part)

    original_line = reverse_tokenize(" ".join(amr_string))
    original_line = original_line.replace('_ (', '_(').replace(') "', ')"')
    return original_line


if __name__ == '__main__':
    from amr_utils import get_default_amr
    singline = get_default_amr(return_type="pm")
    multiline = reformat_to_multiline(singline)
    print(multiline)