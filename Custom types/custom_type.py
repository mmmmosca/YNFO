import re
import parser

url_re = re.compile(r"") #Insert here the regex

parser.register_custom_type(
    name= "Type here the name of your type",
    validator=lambda v: isinstance(v, str) and bool(url_re.match(v)),
    parser=lambda s: s if url_re.match(s) else parser.PARSER_NO_MATCH,
    prepend_parser=True,
)
