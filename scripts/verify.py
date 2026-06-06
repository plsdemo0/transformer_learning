import json, glob

for name in ['13', '14', '15']:
    pattern = f'D:/OpenCodeNLLM/OpenCode/Transformer_Notebooks/transformer_learning/notebooks/{name}_*.ipynb'
    for p in glob.glob(pattern):
        with open(p, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        code_cells = [(i, c) for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code']
        total_comment_lines = 0
        has_logging_import = False
        has_log_call = False
        for i, c in code_cells:
            src = ''.join(c['source'])
            lines = src.split('\n')
            comments = sum(1 for l in lines if l.strip().startswith('#'))
            total_comment_lines += comments
            if 'import logging' in src:
                has_logging_import = True
            if 'log.info' in src or 'log.debug' in src:
                has_log_call = True
        short_name = p.split('/')[-1].split('\\')[-1]
        print(f'{short_name}: {len(code_cells)} code cells, {total_comment_lines} comment lines, logging_import={has_logging_import}, log_calls={has_log_call}')
