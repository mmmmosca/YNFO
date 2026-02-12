import sys
import re
import os
import copy
from pprint import pprint
from typing import Any, List, Dict, Optional

MISSING = object()

class QuotedString(str):
    pass


def read_text_file(path: str) -> List[str]:
    with open(path, "r") as f:
        content = f.read()
    return content.replace('\r\n', '\n').replace('\r', '\n').split('\n')


def normalize_input_path(path: str, required_ext: str) -> str:
    if os.path.exists(path):
        return path
    if not path.endswith(required_ext):
        candidate = f"{path}{required_ext}"
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"File not found: {path}")


def parse_numeric_bound(raw: str) -> float:
    bound = raw.strip()
    if re.match(r"^[+-]?[0-9]+$", bound):
        return float(int(bound))
    if re.match(r"^[+-]?[0-9]+\.[0-9]+$", bound):
        return float(bound)
    raise SyntaxError(f"Invalid numeric bound in schema constraint: {raw}")


def parse_range_constraint(text: str) -> Dict[str, float]:
    body = text[1:-1].strip()
    parts = [p.strip() for p in body.split(",")]
    if len(parts) != 2:
        raise SyntaxError(f"Range constraint must be '(min,max)': {text}")

    min_v = parse_numeric_bound(parts[0])
    max_v = parse_numeric_bound(parts[1])
    if min_v > max_v:
        raise SyntaxError(f"Invalid range constraint (min > max): {text}")
    return {"min": min_v, "max": max_v}


def constraint_target_type(tokens: List[str]) -> Optional[str]:
    if not tokens:
        return None

    idx = 0
    while idx < len(tokens) and tokens[idx] == "List":
        idx += 1

    if idx != len(tokens) - 1:
        return None

    terminal = tokens[idx]
    if terminal in ("Int", "Float"):
        return terminal
    return None


def normalize_schema_default_value(value: Any) -> Any:
    if isinstance(value, QuotedString):
        return str(value)
    if isinstance(value, list):
        return [normalize_schema_default_value(v) for v in value]
    if isinstance(value, dict):
        return {k: normalize_schema_default_value(v) for k, v in value.items()}
    return value


def split_schema_declaration(
    text: str
) -> tuple[str, List[str], bool, Optional[Dict[str, float]], Any]:
    parts = text.split(None, 1)
    field_name = parts[0].strip() if parts else ""
    rest = parts[1].strip() if len(parts) > 1 else ""

    tokens = []
    optional = False
    constraint: Optional[Dict[str, float]] = None
    default_value = MISSING

    if rest:
        if rest == ":":
            return field_name, tokens, optional, constraint, default_value
        tokens, leftover = parse_schema_prefix(rest)
        leftover = leftover.strip()
        if leftover.startswith('?'):
            optional = True
            leftover = leftover[1:].strip()
        if leftover.startswith('('):
            end_idx = leftover.find(')')
            if end_idx == -1:
                raise SyntaxError(f"Invalid schema declaration: {text}")
            constraint_text = leftover[:end_idx + 1]
            constraint = parse_range_constraint(constraint_text)
            if constraint_target_type(tokens) is None:
                raise SyntaxError(f"Range constraints require Int/Float schema: {text}")
            leftover = leftover[end_idx + 1:].strip()
        if leftover.startswith(':'):
            default_text = leftover[1:].strip()
            if default_text:
                default_value = normalize_schema_default_value(parse_value_or_list(default_text))
            leftover = ""
        if leftover:
            raise SyntaxError(f"Invalid schema declaration: {text}")

    if default_value is not MISSING:
        if tokens:
            validate_schema(tokens, default_value, f"schema default for {field_name}")
        if constraint is not None:
            validate_numeric_range_constraint(default_value, constraint, f"schema default for {field_name}")

    return field_name, tokens, optional, constraint, default_value


def parse_schema_block(lines: List[str], start: int, indent: int) -> tuple[Dict[str, Any], int]:
    schema_fields: Dict[str, Any] = {}
    i = start

    while i < len(lines):
        line = lines[i]
        line_without_tabs = line.replace('\t', '    ')
        current_indent = len(line_without_tabs) - len(line_without_tabs.lstrip(' '))

        clean_line = strip_inline_comment(line.rstrip())
        if not clean_line.strip():
            i += 1
            continue

        if current_indent < indent:
            break
        if current_indent > indent:
            raise SyntaxError(f"Unexpected indentation in schema: {clean_line.strip()}")

        stripped = clean_line.lstrip()
        if not stripped.startswith('.'):
            raise SyntaxError(f"Schema lines must start with '.': {clean_line.strip()}")

        declaration = stripped[1:].strip()
        field_name, schema_tokens, optional, constraint, default_value = split_schema_declaration(declaration)
        if not field_name:
            raise SyntaxError(f"Missing field name in schema: {clean_line.strip()}")
        if field_name in schema_fields:
            raise ValueError(f"Duplicate schema key: {field_name}")

        i += 1
        child_lines: List[str] = []

        while i < len(lines):
            next_line = lines[i]
            next_without_tabs = next_line.replace('\t', '    ')
            next_indent = len(next_without_tabs) - len(next_without_tabs.lstrip(' '))

            clean_next = strip_inline_comment(next_line.rstrip())
            if not clean_next.strip():
                child_lines.append(next_line)
                i += 1
                continue
            if next_indent <= current_indent:
                break

            child_lines.append(next_line)
            i += 1

        children = None
        if child_lines:
            first_indent = None
            for child in child_lines:
                child_clean = strip_inline_comment(child.rstrip())
                if not child_clean.strip():
                    continue
                child_no_tabs = child.replace('\t', '    ')
                first_indent = len(child_no_tabs) - len(child_no_tabs.lstrip(' '))
                break
            if first_indent is not None:
                children, _ = parse_schema_block(child_lines, 0, first_indent)

        schema_fields[field_name] = {
            "tokens": schema_tokens,
            "optional": optional,
            "constraint": constraint,
            "default": default_value,
            "children": children,
        }

    return schema_fields, i


def load_yns_schema(schema_file: str) -> Dict[str, Any]:
    path = normalize_input_path(schema_file, '.yns')
    lines = read_text_file(path)

    first_indent = None
    for line in lines:
        clean = strip_inline_comment(line.rstrip())
        if not clean.strip():
            continue
        line_no_tabs = line.replace('\t', '    ')
        first_indent = len(line_no_tabs) - len(line_no_tabs.lstrip(' '))
        break

    if first_indent is None:
        return {}

    schema_fields, _ = parse_schema_block(lines, 0, first_indent)
    return schema_fields


def validate_against_yns_schema(data: Any, schema_fields: Dict[str, Any], context: str = "root") -> None:
    if not isinstance(data, dict):
        raise ValueError(f"Schema mismatch for {context}: expected object, got {type(data).__name__}")

    apply_schema_defaults(data, schema_fields)

    expected = set(schema_fields.keys())
    actual = set(data.keys())

    missing_required = [
        key for key, field in schema_fields.items()
        if not field["optional"] and key not in data
    ]
    if missing_required:
        raise ValueError(f"Schema mismatch for {context}: missing required fields {missing_required}")

    extra = sorted(actual - expected)
    if extra:
        raise ValueError(f"Schema mismatch for {context}: unexpected fields {extra}")

    for key, field in schema_fields.items():
        if key not in data:
            continue
        value = data[key]
        field_context = f"{context}.{key}"

        if field["children"] is not None:
            validate_against_yns_schema(value, field["children"], field_context)

        tokens = field["tokens"]
        if tokens:
            validate_schema(tokens, value, field_context)
        constraint = field["constraint"]
        if constraint is not None:
            validate_numeric_range_constraint(value, constraint, field_context)


def apply_schema_defaults(data: Dict[str, Any], schema_fields: Dict[str, Any]) -> None:
    for key, field in schema_fields.items():
        if key not in data and field["default"] is not MISSING:
            data[key] = copy.deepcopy(field["default"])

    for key, field in schema_fields.items():
        if key not in data:
            continue
        if field["children"] is not None and isinstance(data[key], dict):
            apply_schema_defaults(data[key], field["children"])


def validate_numeric_range_constraint(value: Any, constraint: Dict[str, float], context: str) -> None:
    if isinstance(value, list):
        for idx, item in enumerate(value):
            validate_numeric_range_constraint(item, constraint, f"{context}[{idx}]")
        return

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Schema mismatch for {context}: range constraint requires numeric value")

    min_v = constraint["min"]
    max_v = constraint["max"]
    numeric_value = float(value)
    if numeric_value < min_v or numeric_value > max_v:
        raise ValueError(
            f"Schema mismatch for {context}: value {value} out of allowed range [{min_v:g}, {max_v:g}]"
        )


def parse_cli_args(argv: List[str]) -> tuple[str, Optional[str]]:
    input_file: Optional[str] = None
    schema_file: Optional[str] = None

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in ("-s", "--schema"):
            if i + 1 >= len(argv):
                raise ValueError("Missing schema file after -s/--schema")
            schema_file = argv[i + 1]
            i += 2
            continue
        if arg.startswith("--schema="):
            schema_file = arg.split("=", 1)[1]
            i += 1
            continue
        if arg.startswith("-"):
            raise ValueError(f"Unknown flag: {arg}")
        if input_file is not None:
            raise ValueError("Only one input .ynfo file can be provided")
        input_file = arg
        i += 1

    if input_file is None:
        raise ValueError("Missing input .ynfo file")

    return input_file, schema_file

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
            lines = read_text_file(path)
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
        elif isinstance(data, QuotedString):
            return str(data)
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
                
                
                clean_next = strip_inline_comment(next_line.rstrip())
                if not clean_next.strip():
                    nested_lines.append(next_line)
                    i += 1
                    continue
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

                clean_next = strip_inline_comment(next_line.rstrip())
                if not clean_next.strip():
                    nested_lines.append(next_line)
                    i += 1
                    continue
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
                
                clean_next = strip_inline_comment(next_line.rstrip())
                if not clean_next.strip():
                    nested_lines.append(next_line)
                    i += 1
                    continue
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

                clean_next = strip_inline_comment(next_line.rstrip())
                if not clean_next.strip():
                    nested_lines.append(next_line)
                    i += 1
                    continue
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
        return QuotedString(value[1:-1])

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
        print("Usage: python parser.py [-s schema.yns|--schema schema.yns] <filename.ynfo>")
        sys.exit(1)

    input_file_arg, schema_file_arg = parse_cli_args(sys.argv[1:])
    input_path = normalize_input_path(input_file_arg, '.ynfo')
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    if re.match(r"^[0-9]", base_name):
        raise ValueError(f"Invalid filename '{base_name}': filenames cannot start with a number.")

    resolver = RefResolver()
    raw_lines = read_text_file(input_path)
    raw_data = parse_lines(raw_lines, allow_top_level_scalar=True)
    resolver.cache[base_name] = raw_data
    final_data = resolver.process(raw_data, base_name)

    if schema_file_arg:
        schema_fields = load_yns_schema(schema_file_arg)
        validate_against_yns_schema(final_data, schema_fields)

    pprint(final_data)
