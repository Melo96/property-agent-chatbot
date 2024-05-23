import json
from pathlib import Path
from semantic_router import Route

def output_parser(output, split_string='', mode='split'):
    if mode=='split' and split_string:
        if split_string in output:
            return output.split(split_string)[1]
        else:
            print('LLM output is not in desired format')
            return output
    else:
        return output

def get_routes(routes_path='routes'):
    all_routes = Path(routes_path).glob('*.json')
    result = []
    for r in all_routes:
        with open(r, 'r') as f:
            config = json.load(f)
        result.append(Route(**config))
    return result