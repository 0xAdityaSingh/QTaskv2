"""
Module to parse Mermaid class diagrams and output a parser-friendly component list for code generation.
"""
import re
from typing import List, Dict

def infer_path(name: str, type_: str) -> str:
    """
    Infer the file path for a component based on its name and type.
    Only used if the LLM did not provide a Path.
    """
    name = name.strip('`"\' ')
    if type_ == 'interface' or (name.startswith('I') and name[1:2].isupper()):
        folder = 'Abstractions'
    elif name.endswith('Service'):
        folder = 'Services'
    elif name.endswith('Factory'):
        folder = 'Infrastructure'
    elif name.endswith('Controller'):
        folder = 'Controllers'
    elif name.endswith('Repository'):
        folder = 'Repositories'
    elif name.endswith('Dto') or name.endswith('Model'):
        folder = 'Models'
    else:
        folder = 'Models'
    return f"{folder}/{name}.cs"

def mermaid_to_component_list(mermaid_code: str) -> str:
    """
    Parse a Mermaid class diagram and output a parser-friendly component list.
    Args:
        mermaid_code (str): The Mermaid class diagram code as a string.
    Returns:
        str: The component list in plain-text format for code generation.
    """
    lines = [l.strip() for l in mermaid_code.splitlines() if l.strip() and not l.strip().startswith('classDiagram') and not l.strip().startswith('direction')]
    components: Dict[str, Dict] = {}
    relationships = []
    current_class = None
    in_class_block = False
    for line in lines:
        m = re.match(r'class\s+([A-Za-z0-9_]+)\s*\{', line)
        if m:
            current_class = m.group(1)
            components[current_class] = {'name': current_class, 'fields': [], 'methods': [], 'type': 'class', 'constructor': '', 'dependencies': []}
            in_class_block = True
            continue
        if in_class_block and line == '}':
            current_class = None
            in_class_block = False
            continue
        m = re.match(r'class\s+([A-Za-z0-9_]+)\s*\{?\s*<<([A-Za-z0-9_ ]+)>>', line)
        if m:
            current_class = m.group(1)
            stereotype = m.group(2).strip().lower()
            components[current_class] = {'name': current_class, 'fields': [], 'methods': [], 'type': 'interface' if 'interface' in stereotype else 'class', 'constructor': '', 'dependencies': []}
            in_class_block = True
            continue
        m = re.match(r'class\s+([A-Za-z0-9_]+)$', line)
        if m:
            cname = m.group(1)
            if cname not in components:
                components[cname] = {'name': cname, 'fields': [], 'methods': [], 'type': 'class', 'constructor': '', 'dependencies': []}
            continue
        if in_class_block and current_class:
            m = re.match(r'[+\-#]?\s*([A-Za-z0-9_<>]+)\s+([A-Za-z0-9_]+)\(([^)]*)\)', line)
            if m:
                ret_type = m.group(1)
                method_name = m.group(2)
                params = m.group(3)
                components[current_class]['methods'].append(f"{ret_type} {method_name}({params})")
                if method_name.lower() == current_class.lower():
                    components[current_class]['constructor'] = f"{method_name}({params})"
                continue
            m = re.match(r'[+\-#]?\s*([A-Za-z0-9_<>]+)\s+([A-Za-z0-9_]+)$', line)
            if m:
                ftype = m.group(1)
                fname = m.group(2)
                components[current_class]['fields'].append(f"{ftype} {fname}")
                continue
        m = re.match(r'([A-Za-z0-9_]+)\s*([<|.\-]+)\s*([A-Za-z0-9_]+)\s*:?(.*)', line)
        if m:
            src, rel, dst, label = m.groups()
            relationships.append((src, rel, dst, label.strip()))
            if rel.strip().endswith('>'):
                if src in components and dst not in components[src]['dependencies']:
                    components[src]['dependencies'].append(dst)
            continue
    output = []
    for cname, comp in components.items():
        type_ = comp.get('type', 'class')
        # Do not infer path here; let extraction function handle it if needed
        output.append(f"Component: {cname}")
        output.append(f"Type: {type_}")
        # Path intentionally omitted here
        if comp['dependencies']:
            output.append(f"Dependencies: {', '.join(comp['dependencies'])}")
        if comp['fields']:
            output.append(f"Fields: {', '.join(comp['fields'])}")
        if comp['constructor']:
            output.append(f"Constructor: {comp['constructor']}")
        if comp['methods']:
            output.append(f"Methods: {', '.join(comp['methods'])}")
        output.append("")
    return '\n'.join(output).strip()

def extract_components_from_lld(lld_text: str) -> str:
    """
    Extract all component definitions from an LLD (Mermaid, Markdown, plain-text),
    merge them, and output a parser-friendly component list for code generation.
    Args:
        lld_text (str): The full LLD markdown text.
    Returns:
        str: The merged component list in plain-text format.
    """
    # 1. Extract Mermaid class diagram(s)
    mermaid_blocks = re.findall(r'```mermaid(.*?)```', lld_text, re.DOTALL)
    mermaid_components = {}
    for block in mermaid_blocks:
        comp_list = mermaid_to_component_list(block)
        current = {}
        for line in comp_list.splitlines():
            if line.startswith('Component:'):
                if current.get('name'):
                    mermaid_components[current['name']] = current
                current = {'name': line.split(':',1)[1].strip()}
            elif ':' in line:
                k, v = line.split(':',1)
                current[k.strip().lower()] = v.strip('`"\' ')
        if current.get('name'):
            mermaid_components[current['name']] = current

    # 2. Extract Markdown component lists (#### Component: ...)
    md_components = {}
    md_blocks = re.split(r'####\s+Component:', lld_text)
    for block in md_blocks[1:]:
        lines = block.split('\n')
        name = lines[0].strip('`"\' ')
        comp = {'name': name}
        for line in lines[1:]:
            m = re.match(r'- \*\*([A-Za-z ]+)\*\*:\s*(.*)', line)
            if m:
                key = m.group(1).strip().lower()
                val = m.group(2).strip('`"\' ')
                if key == 'file':
                    comp['path'] = val
                else:
                    comp[key] = val
            m2 = re.match(r'\s*- `([^`]+)`', line)
            if m2:
                if 'methods' in comp:
                    comp['methods'] += ', ' + m2.group(1).strip()
                else:
                    comp['methods'] = m2.group(1).strip()
        # Only infer path if not present
        if 'path' not in comp:
            comp['path'] = infer_path(name, comp.get('type', 'class'))
        md_components[comp['name']] = comp

    # 3. Extract plain-text component lists (Component: ... Type: ...)
    pt_components = {}
    pt_blocks = re.split(r'\nComponent:', lld_text)
    for block in pt_blocks[1:]:
        lines = ('Component:' + block).split('\n')
        comp = {}
        for line in lines:
            if ':' in line:
                k, v = line.split(':',1)
                comp[k.strip().lower()] = v.strip('`"\' ')
        if 'component' in comp:
            comp['name'] = comp['component']
            del comp['component']
        if 'name' in comp:
            if 'path' not in comp:
                comp['path'] = infer_path(comp['name'], comp.get('type', 'class'))
            pt_components[comp['name']] = comp

    # 4. Merge all components by name, preferring Markdown > plain-text > Mermaid
    all_names = set(mermaid_components) | set(md_components) | set(pt_components)
    merged = {}
    for name in all_names:
        merged[name] = {}
        if name in mermaid_components:
            merged[name].update(mermaid_components[name])
        if name in pt_components:
            merged[name].update(pt_components[name])
        if name in md_components:
            merged[name].update(md_components[name])
    # 5. Output in parser-friendly format
    output = []
    for name, comp in merged.items():
        output.append(f"Component: {comp.get('name', name)}")
        if 'type' in comp:
            output.append(f"Type: {comp['type']}")
        if 'path' in comp:
            output.append(f"Path: {comp['path']}")
        if 'dependencies' in comp:
            output.append(f"Dependencies: {comp['dependencies']}")
        if 'fields' in comp:
            output.append(f"Fields: {comp['fields']}")
        if 'constructor' in comp:
            output.append(f"Constructor: {comp['constructor']}")
        if 'methods' in comp:
            output.append(f"Methods: {comp['methods']}")
        output.append("")
    return '\n'.join(output).strip() 