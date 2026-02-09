import sys
import re
import os
from pprint import pprint
from typing import Any, List, Dict, Optional

class RefResolver:
    def __init__(self):
        self.cache: Dict[str, Any] = {}

    def get_file_data(self, filename: str) -> Any:
        if filename not in self.cache:
            path = filename if filename.endswith('.ynfo') else f"{filename}.ynfo"
            if not os.path.exists(path):
                return None
            with open(path, "r") as f:
                lines = f.readlines()
            self.cache[filename] = parse_lines(lines)
        return self.cache[filename]

    def resolve_path(self, data: Any, path: str) -> Any:
        parts = re.split(r'\.|\[|\]', path)
        parts = [p for p in parts if p]
        
        current = data
        for part in parts:
            try:
                if isinstance(current, list) and part.isdigit():
                    current = current[int(part)]
                elif isinstance(current, dict):
                    current = current.get(part)
                else:
                    return None
            except (IndexError, KeyError, TypeError):
                return None
        return current

    def process(self, data: Any, current_file: str) -> Any:
        if isinstance(data, dict):
            return {k: self.process(v, current_file) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.process(i, current_file) for i in data]
        elif isinstance(data, str):
            match = re.match(r"^([\w\-]+)\.([\w\.\[\]\-]+)$", data)
            if match:
                prefix, path = match.groups()
                target_filename = current_file if prefix == "self" else prefix
                target_data = self.get_file_data(target_filename)
                
                if target_data is not None:
                    resolved = self.resolve_path(target_data, path)
                    return resolved
        return data

def parse_lines(lines: List[str], indent: int = 0) -> Any:
    result = {}
    while lines:
        line = lines[0]
        clean_line = strip_inline_comment(line)
        current_indent = len(line) - len(line.lstrip(' '))
        if current_indent < indent:
            break
        if not clean_line.strip():
            lines.pop(0)
            continue
        if clean_line.lstrip().startswith('.'):
            field_line = lines.pop(0)
            clean_field_line = strip_inline_comment(field_line)
            field_indent = len(field_line) - len(field_line.lstrip(' '))
            field = clean_field_line.split(':')[0].strip()[1:]
            after_colon = clean_field_line.split(':', 1)[1].strip()
            if after_colon:
                value = parse_value(after_colon)
                result[field] = value
            else:
                if lines and (len(lines[0]) - len(lines[0].lstrip(' '))) > field_indent:
                    next_clean = strip_inline_comment(lines[0])
                    if next_clean.lstrip().startswith('.'):
                        value = parse_lines(lines, indent=field_indent + 1)
                        result[field] = value
                    else:
                        value = parse_list(lines, indent=field_indent + 1)
                        result[field] = value
                else:
                    result[field] = ""
        else:
            if not result:
                values = parse_inline_values(lines, indent=current_indent)
                if len(values) == 1:
                    return values[0]
                return values
            break
    return result

def parse_list(lines: List[str], indent: int) -> List[Any]:
    items = []
    buffer = []
    while lines:
        line = lines[0]
        clean_line = strip_inline_comment(line)
        current_indent = len(line) - len(line.lstrip(' '))
        if current_indent < indent or not clean_line.strip():
            if not clean_line.strip():
                lines.pop(0)
            else:
                break
            continue
        if clean_line.lstrip().startswith('.'):
            item = parse_lines(lines, indent=current_indent)
            items.append(item)
        else:
            while lines:
                next_line = lines[0]
                next_clean = strip_inline_comment(next_line)
                if (len(next_line) - len(next_line.lstrip(' '))) < indent:
                    break
                if next_clean.lstrip().startswith('.') or not next_clean.strip():
                    break
                buffer.append(next_clean.strip())
                lines.pop(0)
            joined = ' '.join(buffer)
            for val in tokenize_values(joined):
                items.append(parse_value(val))
            buffer = []
    return items

def parse_value(value: str) -> Any:
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if re.match(r"^[0-9]+\.[0-9]+$", value):
        return float(value)
    elif re.match(r"^[0-9]+$", value):
        return int(value)
    elif value == "true":
        return True
    elif value == "false":
        return False
    else:
        return value

def parse_inline_values(lines: List[str], indent: int) -> List[Any]:
    items = []
    while lines:
        line = lines[0]
        clean_line = strip_inline_comment(line)
        current_indent = len(line) - len(line.lstrip(' '))
        if current_indent < indent or not clean_line.strip():
            if not clean_line.strip():
                lines.pop(0)
            else:
                break
            continue
        if clean_line.lstrip().startswith('.'):
            break
        tokens = tokenize_values(clean_line.strip())
        lines.pop(0)
        for token in tokens:
            items.append(parse_value(token))
    return items

def strip_inline_comment(line: str) -> str:
    if '<' not in line:
        return line
    out = []
    i = 0
    in_comment = False
    while i < len(line):
        ch = line[i]
        if not in_comment and ch == '<':
            in_comment = True
            i += 1
            continue
        if in_comment and ch == '>':
            in_comment = False
            i += 1
            continue
        if not in_comment:
            out.append(ch)
        i += 1
    return ''.join(out)

def tokenize_values(text: str) -> List[str]:
    tokens = []
    buf = []
    in_quotes = False
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == '"':
            in_quotes = not in_quotes
            buf.append(ch)
            i += 1
            continue
        if ch.isspace() and not in_quotes:
            if buf:
                tokens.append(''.join(buf))
                buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    if buf:
        tokens.append(''.join(buf))
    return tokens

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parser.py <filename.ynfo>")
        sys.exit(1)

    input_file = sys.argv[1]
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    
    resolver = RefResolver()
    
    raw_data = resolver.get_file_data(base_name)
    
    final_data = resolver.process(raw_data, base_name)
    
    pprint(final_data)
