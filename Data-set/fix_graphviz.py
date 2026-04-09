import json

# Read notebook
with open('predict-diabetes-from-medical-records.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find and fix the graphviz cells
for cell in nb['cells']:
    if cell.get('cell_type') == 'code':
        source = cell.get('source', [])
        new_source = []
        for line in source:
            if 'graphviz.Source' in line:
                new_source.append('    # graphviz.Source disabled - graphviz executable not found')
            elif 'import graphviz' in line:
                new_source.append('    # import graphviz  # disabled - graphviz executable not found')
            else:
                new_source.append(line)
        cell['source'] = new_source

# Save
with open('predict-diabetes-from-medical-records.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Fixed graphviz in notebook!')