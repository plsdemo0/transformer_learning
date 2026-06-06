import json

path = r'D:\OpenCodeNLLM\OpenCode\Transformer_Notebooks\transformer_learning\notebooks\15_final_end_to_end.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
for i, cell in enumerate(cells):
    print(f'--- Cell {i} [{cell["cell_type"]}] ---')
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        lines = src.split('\n')
        for j, l in enumerate(lines):
            print(f'{j:3d}: {l}')
    print()
