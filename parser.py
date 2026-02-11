import sys
import re
import os
from pprint import pprint
from typing import Any, List, Dict, Optional

MISSING = object()

class RefResolver:
    def __init__(self):
        self.cache: Dict[str, Any] = {}

    def get_file_data(self, filename: str) -> Any:
        if re.match(r"^[0-9]", filename):
            raise ValueError(f"Invalid filename '{filename}': filenames cannot start with a number.")
        if filename not in self.cache:
            path = filename if filename.endswith('.ynfo') else f"{filename}.ynfo"
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")
            with open(path, "r") as f:
                content = f.read()
            lines = content.replace('\r\n', '\n').replace('\r', '\n').split('\n')
            self.cache[filename] = parse_lines(lines, allow_top_level_scalar=True)
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
                    if part in current:
                        current = current[part]
                    else:
                        return MISSING
                else:
                    return MISSING
            except (IndexError, KeyError, TypeError):
                return MISSING
        return current

    def process(self, data: Any, current_file: str) -> Any:
        if isinstance(data, dict):
            return {k: self.process(v, current_file) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.process(i, current_file) for i in data]
        elif isinstance(data, str):
            if is_ip(data):
                return data
            match = re.match(r"^([\w\-]+)\.([\w\.\[\]\-]+)$", data)
            if match:
                prefix, path = match.groups()
                if re.match(r"^[0-9]", prefix):
                    raise ValueError(f"Invalid filename '{prefix}': filenames cannot start with a number.")
                target_filename = current_file if prefix == "self" else prefix
                target_data = self.get_file_data(target_filename)

                resolved = self.resolve_path(target_data, path)
                if resolved is MISSING:
                    raise ValueError(f"Reference not found: {prefix}.{path}")
                return resolved
        return data


def parse_lines(
    lines: List[str],
    indent: int = 0,
    allow_top_level_scalar: bool = False,
) -> Any:
    entries = []
    seen_fields = set()
    seen_unnamed = False
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        line_without_tabs = line.replace('\t', '    ')
        current_indent = len(line_without_tabs) - len(line_without_tabs.lstrip(' '))
        
        if current_indent < indent:
            break
            
        if not line.strip():
            i += 1
            continue
        
        clean_line = strip_inline_comment(line.rstrip())
        if not clean_line.strip():
            i += 1
            continue
        
        if clean_line.lstrip().startswith('.'):
            field_content = clean_line.lstrip()[1:].lstrip()

            if ':' not in field_content:
                raise SyntaxError(f"Missing ':' in field declaration: {clean_line.strip()}")

            if ':' in field_content:
                
                colon_pos = -1
                in_quotes = False
                for idx, ch in enumerate(field_content):
                    if ch == '"':
                        in_quotes = not in_quotes
                    elif ch == ':' and not in_quotes:
                        colon_pos = idx
                        break
                
                if colon_pos >= 0:
                    field_schema_part = field_content[:colon_pos].strip()
                    value_part = field_content[colon_pos + 1:].strip()
                else:
                    raise SyntaxError(f"Missing ':' in field declaration: {clean_line.strip()}")
            else:
                field_name = field_content.strip()
                value_part = ""

            i += 1

            if ':' not in field_content:
                field_schema_part = field_content.strip()

            field_name, schema_tokens = parse_field_schema(field_schema_part)

            if not field_name:
                raise SyntaxError(f"Missing field name: {clean_line.strip()}")

            if field_name in seen_fields:
                raise ValueError(f"Duplicate key: {field_name}")
            seen_fields.add(field_name)
            
            
            nested_lines = []
            
            
            while i < len(lines):
                next_line = lines[i]
                next_without_tabs = next_line.replace('\t', '    ')
                next_indent = len(next_without_tabs) - len(next_without_tabs.lstrip(' '))
                
                
                if next_indent <= current_indent:
                    break
                
                nested_lines.append(next_line)
                i += 1
            
            if value_part:

                value = parse_value_or_list(value_part)
                validate_schema(schema_tokens, value, field_name)
                entries.append(("field", field_name, value))

                if nested_lines:
                    pass
            else:
                if nested_lines:
                    first_nested_clean = strip_inline_comment(nested_lines[0].rstrip())

                    if first_nested_clean.lstrip().startswith('.'):
                        value = parse_lines(
                            nested_lines,
                            indent=current_indent + 1,
                            allow_top_level_scalar=False,
                        )
                    elif first_nested_clean.lstrip().startswith(':'):
                        value = parse_list(nested_lines, indent=current_indent + 1)
                    elif first_nested_clean.lstrip().startswith('-'):
                        value = parse_list(nested_lines, indent=current_indent + 1)
                    else:
                        value = parse_list(nested_lines, indent=current_indent + 1)
                else:
                    value = ""
                validate_schema(schema_tokens, value, field_name)
                entries.append(("field", field_name, value))
        elif clean_line.lstrip().startswith(':') or clean_line.lstrip().startswith('['):
            # Unnamed field/list item at this level
            if clean_line.lstrip().startswith(':'):
                item_content = clean_line.lstrip()[1:].strip()
                schema_tokens, item_content = parse_unnamed_schema(item_content)
            else:
                schema_tokens, rest = parse_schema_prefix(clean_line.lstrip())
                if not rest.startswith(':'):
                    raise SyntaxError(f"Missing ':' after schema: {clean_line.strip()}")
                item_content = rest[1:].lstrip()

            nested_lines = []
            i += 1

            while i < len(lines):
                next_line = lines[i]
                next_without_tabs = next_line.replace('\t', '    ')
                next_indent = len(next_without_tabs) - len(next_without_tabs.lstrip(' '))

                if next_indent <= current_indent:
                    break

                nested_lines.append(next_line)
                i += 1

            if item_content:
                item_value = parse_value_or_list(item_content)
                validate_schema(schema_tokens, item_value, "<unnamed>")
                if nested_lines:
                    first_nested = strip_inline_comment(nested_lines[0].rstrip())
                    if first_nested.lstrip().startswith('.'):
                        nested_obj = parse_lines(
                            nested_lines,
                            indent=current_indent + 1,
                            allow_top_level_scalar=False,
                        )
                        if isinstance(item_value, dict):
                            item_value.update(nested_obj)
                            entries.append(("unnamed", item_value))
                        else:
                            entries.append(("unnamed", {'value': item_value, **nested_obj}))
                    else:
                        nested_list = parse_list(nested_lines, indent=current_indent + 1)
                        entries.append(("unnamed", {'value': item_value, 'items': nested_list}))
                else:
                    entries.append(("unnamed", item_value))
            else:
                if nested_lines:
                    first_nested = strip_inline_comment(nested_lines[0].rstrip())
                    if first_nested.lstrip().startswith('.'):
                        unnamed_value = parse_lines(
                            nested_lines,
                            indent=current_indent + 1,
                            allow_top_level_scalar=False,
                        )
                        validate_schema(schema_tokens, unnamed_value, "<unnamed>")
                        entries.append(("unnamed", unnamed_value))
                    else:
                        unnamed_value = parse_list(nested_lines, indent=current_indent + 1)
                        validate_schema(schema_tokens, unnamed_value, "<unnamed>")
                        entries.append(("unnamed", unnamed_value))
                else:
                    validate_schema(schema_tokens, [], "<unnamed>")
                    entries.append(("unnamed", []))
            seen_unnamed = True
        else:
            if not entries and allow_top_level_scalar:
                if clean_line.lstrip().startswith('-'):
                    return parse_list(lines[i:], indent=current_indent)
                values = []
                while i < len(lines):
                    current_line = lines[i]
                    line_without_tabs = current_line.replace('\t', '    ')
                    line_indent = len(line_without_tabs) - len(line_without_tabs.lstrip(' '))
                    
                    if line_indent != current_indent:
                        break
                    
                    clean_current = strip_inline_comment(current_line.rstrip())
                    if clean_current.strip():
                        if clean_current.lstrip().startswith('-'):
                            # List item
                            item_content = clean_current.lstrip()[1:].strip()
                            if item_content:
                                values.append(parse_value_or_list(item_content))
                        elif clean_current.lstrip().startswith(':'):
                            item_content = clean_current.lstrip()[1:].strip()
                            if item_content:
                                values.append(parse_value_or_list(item_content))
                            else:
                                values.append([])
                        else:
                            tokens = tokenize_values(clean_current.strip())
                            for token in tokens:
                                values.append(parse_value(token))

                    i += 1

                if len(values) == 1:
                    return values[0]
                return values
            else:
                raise SyntaxError(f"Unexpected line (missing '.' or ':'): {clean_line.strip()}")

    if not entries:
        return {}
    if not seen_unnamed:
        return {name: value for _, name, value in entries}

    list_out = []
    for entry in entries:
        if entry[0] == "field":
            _, name, value = entry
            list_out.append({name: value})
        else:
            _, value = entry
            list_out.append(value)

    if len(list_out) == 1 and entries[0][0] == "unnamed":
        return list_out[0]
    return list_out


def parse_list(lines: List[str], indent: int) -> List[Any]:
    items = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        line_without_tabs = line.replace('\t', '    ')
        current_indent = len(line_without_tabs) - len(line_without_tabs.lstrip(' '))
        
        if current_indent < indent:
            break
            
        if not line.strip():
            i += 1
            continue
            
        clean_line = strip_inline_comment(line.rstrip())
        if not clean_line.strip():
            i += 1
            continue

        if clean_line.lstrip().startswith('-'):
            item_content = clean_line.lstrip()[1:].strip()
            
            nested_lines = []
            i += 1
            
            while i < len(lines):
                next_line = lines[i]
                next_without_tabs = next_line.replace('\t', '    ')
                next_indent = len(next_without_tabs) - len(next_without_tabs.lstrip(' '))
                
                if next_indent <= current_indent:
                    break
                
                nested_lines.append(next_line)
                i += 1
            
            if item_content:
                item_value = parse_value_or_list(item_content)
                
                if nested_lines:
                    first_nested = strip_inline_comment(nested_lines[0].rstrip())
                    
                    if first_nested.lstrip().startswith('.'):
                        nested_obj = parse_lines(
                            nested_lines,
                            indent=current_indent + 1,
                            allow_top_level_scalar=False,
                        )
                        if isinstance(item_value, dict):
                            item_value.update(nested_obj)
                            items.append(item_value)
                        elif item_value == "":
                            items.append(nested_obj)
                        else:
                            items.append({'value': item_value, **nested_obj})
                    else:
                        nested_list = parse_list(nested_lines, indent=current_indent + 1)
                        if item_value == "":
                            items.append(nested_list)
                        else:
                            items.append({'value': item_value, 'items': nested_list})
                else:
                    items.append(item_value)
            else:
                if nested_lines:
                    first_nested = strip_inline_comment(nested_lines[0].rstrip())
                    
                    if first_nested.lstrip().startswith('.'):
                        items.append(
                            parse_lines(
                                nested_lines,
                                indent=current_indent + 1,
                                allow_top_level_scalar=False,
                            )
                        )
                    else:
                        items.append(parse_list(nested_lines, indent=current_indent + 1))
                else:
                    items.append("")
        elif clean_line.lstrip().startswith(':') or clean_line.lstrip().startswith('['):
            if clean_line.lstrip().startswith(':'):
                item_content = clean_line.lstrip()[1:].strip()
                schema_tokens, item_content = parse_unnamed_schema(item_content)
            else:
                schema_tokens, rest = parse_schema_prefix(clean_line.lstrip())
                if not rest.startswith(':'):
                    raise SyntaxError(f"Missing ':' after schema: {clean_line.strip()}")
                item_content = rest[1:].lstrip()

            nested_lines = []
            i += 1

            while i < len(lines):
                next_line = lines[i]
                next_without_tabs = next_line.replace('\t', '    ')
                next_indent = len(next_without_tabs) - len(next_without_tabs.lstrip(' '))

                if next_indent <= current_indent:
                    break

                nested_lines.append(next_line)
                i += 1

            if item_content:
                item_value = parse_value_or_list(item_content)
                validate_schema(schema_tokens, item_value, "<unnamed>")
                if nested_lines:
                    first_nested = strip_inline_comment(nested_lines[0].rstrip())
                    if first_nested.lstrip().startswith('.'):
                        nested_obj = parse_lines(nested_lines, indent=current_indent + 1, allow_top_level_scalar=False)
                        if isinstance(item_value, dict):
                            item_value.update(nested_obj)
                            items.append(item_value)
                        else:
                            items.append({'value': item_value, **nested_obj})
                    else:
                        nested_list = parse_list(nested_lines, indent=current_indent + 1)
                        items.append({'value': item_value, 'items': nested_list})
                else:
                    items.append(item_value)
            else:
                if nested_lines:
                    first_nested = strip_inline_comment(nested_lines[0].rstrip())
                    if first_nested.lstrip().startswith('.'):
                        unnamed_value = parse_lines(nested_lines, indent=current_indent + 1, allow_top_level_scalar=False)
                        validate_schema(schema_tokens, unnamed_value, "<unnamed>")
                        items.append(unnamed_value)
                    else:
                        unnamed_value = parse_list(nested_lines, indent=current_indent + 1)
                        validate_schema(schema_tokens, unnamed_value, "<unnamed>")
                        items.append(unnamed_value)
                else:
                    validate_schema(schema_tokens, [], "<unnamed>")
                    items.append([])
        elif clean_line.lstrip().startswith('.'):
            nested_lines = [line]
            i += 1
            
            while i < len(lines):
                next_line = lines[i]
                next_without_tabs = next_line.replace('\t', '    ')
                next_indent = len(next_without_tabs) - len(next_without_tabs.lstrip(' '))
                
                if next_indent < current_indent:
                    break
                
                nested_lines.append(next_line)
                i += 1
            
            items.append(
                parse_lines(
                    nested_lines,
                    indent=current_indent,
                    allow_top_level_scalar=False,
                )
            )
        else:
            tokens = tokenize_values(clean_line.strip())
            for token in tokens:
                items.append(parse_value(token))
            i += 1
    
    return items


def parse_value(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""

    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]

    if is_ip(value):
        return IP(value)

    if re.match(r"^[+-]?[0-9]+\.[0-9]+$", value):
        return float(value)
    elif re.match(r"^[+-]?[0-9]+$", value):
        return int(value)

    if value.lower() == "true":
        return True
    elif value.lower() == "false":
        return False
    elif value.lower() == "null":
        return None

    if re.match(r"^[\w\-]+\.[\w\.\[\]\-]+$", value):
        return value

    raise SyntaxError(f"Unquoted or invalid value: {value}")


def parse_value_or_list(text: str) -> Any:
    if not text.strip():
        return ""

    tokens = tokenize_values(text)
    if len(tokens) == 1:
        return parse_value(tokens[0])
    
    return [parse_value(t) for t in tokens]


def parse_field_schema(text: str) -> tuple[str, List[str]]:
    if '[' not in text:
        return text.strip(), []
    name_part, schema_part = text.split('[', 1)
    name_part = name_part.strip()
    schema_part = '[' + schema_part
    tokens = parse_schema_tokens(schema_part)
    return name_part, tokens


def parse_unnamed_schema(text: str) -> tuple[List[str], str]:
    tokens, rest = parse_schema_prefix(text)
    if rest.startswith(':'):
        rest = rest[1:].lstrip()
    return tokens, rest


def parse_schema_prefix(text: str) -> tuple[List[str], str]:
    tokens = []
    i = 0
    text_len = len(text)
    while i < text_len:
        while i < text_len and text[i].isspace():
            i += 1
        if i >= text_len or text[i] != '[':
            break
        end = text.find(']', i + 1)
        if end == -1:
            raise SyntaxError(f"Unterminated schema token in: {text}")
        token = text[i + 1:end].strip()
        if not token:
            raise SyntaxError(f"Empty schema token in: {text}")
        tokens.append(token)
        i = end + 1
    rest = text[i:].lstrip()
    if tokens:
        validate_schema_tokens(tokens)
    return tokens, rest


def parse_schema_tokens(text: str) -> List[str]:
    tokens, rest = parse_schema_prefix(text)
    if rest:
        raise SyntaxError(f"Invalid schema syntax: {text}")
    return tokens


def validate_schema_tokens(tokens: List[str]) -> None:
    allowed = {"String", "Float", "Int", "Bool", "Any", "List", "Ip"}
    for t in tokens:
        if t not in allowed:
            raise SyntaxError(f"Unknown schema type: {t}")
    if tokens and tokens[0] != "List" and len(tokens) > 1:
        raise SyntaxError("Only List can chain multiple types.")


def build_schema(tokens: List[str]) -> Dict[str, Any]:
    if not tokens:
        return {"kind": "any"}
    first = tokens[0]
    if first != "List":
        return {"kind": "type", "name": first}
    if len(tokens) == 1:
        return {"kind": "list", "element": {"kind": "any"}}
    if tokens[1] == "List":
        return {"kind": "list", "element": build_schema(tokens[1:])}
    element_types = []
    for t in tokens[1:]:
        if t == "List":
            raise SyntaxError("List must be the first type in a list schema.")
        element_types.append({"kind": "type", "name": t})
    return {"kind": "list", "element": {"kind": "union", "types": element_types}}


def validate_schema(tokens: List[str], value: Any, context: str) -> None:
    if not tokens:
        return
    schema = build_schema(tokens)
    if not validate_value(schema, value):
        raise ValueError(f"Schema mismatch for {context}: expected {tokens}, got {type(value).__name__}")


def validate_value(schema: Dict[str, Any], value: Any) -> bool:
    kind = schema["kind"]
    if kind == "any":
        return True
    if kind == "type":
        name = schema["name"]
        if name == "String":
            return isinstance(value, str) and not isinstance(value, IP)
        if name == "Ip":
            return isinstance(value, IP)
        if name == "Float":
            return isinstance(value, float)
        if name == "Int":
            return isinstance(value, int) and not isinstance(value, bool)
        if name == "Bool":
            return isinstance(value, bool)
        if name == "Any":
            return True
        if name == "List":
            return isinstance(value, list)
        return False
    if kind == "union":
        return any(validate_value(t, value) for t in schema["types"])
    if kind == "list":
        if not isinstance(value, list):
            return False
        element_schema = schema["element"]
        return all(validate_value(element_schema, v) for v in value)
    return False


def parse_inline_values(lines: List[str], indent: int) -> List[Any]:
    items = []
    
    for line in lines:
        clean_line = strip_inline_comment(line.rstrip())
        if clean_line.strip():
            tokens = tokenize_values(clean_line.strip())
            for token in tokens:
                items.append(parse_value(token))
    
    return items


def strip_inline_comment(line: str) -> str:
    if '<' not in line:
        return line

    out = []
    i = 0
    in_comment = False
    in_quotes = False

    while i < len(line):
        ch = line[i]
        if ch == '"':
            in_quotes = not in_quotes
            out.append(ch)
            i += 1
            continue
        if not in_comment and not in_quotes and ch == '<':
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

    if in_quotes:
        raise SyntaxError(f"Unterminated quote in: {text}")

    return tokens


class IP(str):
    pass


def is_ip(value: str) -> bool:
    parts = value.split('.')
    if len(parts) != 4:
        return False
    for part in parts:
        if not part.isdigit():
            return False
        num = int(part)
        if num < 0 or num > 255:
            return False
    return True




if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parser.py <filename.ynfo>")
        sys.exit(1)

    input_file = sys.argv[1]
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    if re.match(r"^[0-9]", base_name):
        raise ValueError(f"Invalid filename '{base_name}': filenames cannot start with a number.")

    resolver = RefResolver()
    raw_data = resolver.get_file_data(base_name)
    final_data = resolver.process(raw_data, base_name)

    pprint(final_data)
