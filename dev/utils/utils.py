import json
from pathlib import Path
from semantic_router import Route

def output_parser(output, split_string='', mode='split'):
    if split_string:
        if split_string in output:
            intermediate = output.split(split_string)[1].strip()
        else:
            print(f'LLM output is not in desired format: {output}')
            return output
    
    if mode=='split':
        return intermediate
    elif mode=='json':
        return json.loads(intermediate)
    return output

def get_routes(routes_path='routes'):
    all_routes = Path(routes_path).glob('*.json')
    result = []
    for r in all_routes:
        with open(r, 'r') as f:
            config = json.load(f)
        result.append(Route(**config))
    return result