#!/usr/bin/env python

from unidecode import unidecode
import re

def pre_process( column ):
    '''
    Some rudimentary field cleaning prior to analysis
    '''
    if column == '':
        return column

    column = unidecode( unicode(column) )
    column = re.sub(r'http://www\.', '', column)
    column = re.sub(r'www\.', ' ', column)
    column = re.sub(r'^http:', '', column)
    column = re.sub(r'\.com', ' ', column)
    column = re.sub(r'\.net', ' ', column)
    column = re.sub(r'/', ' ', column)
    column = re.sub(r'\.', ' ', column)
    column = re.sub(r'&', r'and', column)
    column = re.sub(r'photography ?by', r'pb', column)
    column = re.sub(r'photography', r'p', column)
    column = re.sub(r'photographer', r'pr', column)
    column = re.sub(r'  +', ' ', column)
    column = re.sub(r'\n', ' ', column)
    column = column.strip().strip('"').strip("'").lower().strip()
    if column == 'none' or column == 'null':
        column = ''
    return column

def translate_state( value ):
    state_to_code = { "vermont": "vt", "georgia": "ga", "iowa": "ia", "armed forces pacific": "ap", "guam": "gu", 
                      "kansas": "ks", "florida": "fl", "american samoa": "as", "north carolina": "nc", "hawaii": "hi", 
                      "new york": "ny", "california": "ca", "alabama": "al", "idaho": "id", "federated states of micronesia": "fm", 
                      "armed forces americas": "aa", "delaware": "de", "alaska": "ak", "illinois": "il", "armed forces africa": "ae", 
                      "south dakota": "sd", "connecticut": "ct", "montana": "mt", "massachusetts": "ma", "puerto rico": "pr", 
                      "armed forces canada": "ae", "new hampshire": "nh", "maryland": "md", "new mexico": "nm", "mississippi": "ms", 
                      "tennessee": "tn", "palau": "pw", "colorado": "co", "armed forces middle east": "ae", "new jersey": "nj", 
                      "utah": "ut", "michigan": "mi", "west virginia": "wv", "washington": "wa", "minnesota": "mn", "oregon": "or", 
                      "virginia": "va", "virgin islands": "vi", "marshall islands": "mh", "wyoming": "wy", "ohio": "oh", "south carolina": "sc", 
                      "indiana": "in", "nevada": "nv", "louisiana": "la", "northern mariana islands": "mp", "nebraska": "ne", 
                      "arizona": "az", "wisconsin": "wi", "north dakota": "nd", "armed forces europe": "ae", "pennsylvania": "pa", 
                      "oklahoma": "ok", "kentucky": "ky", "rhode island": "ri", "district of columbia": "dc", "arkansas": "ar", 
                      "missouri": "mo", "texas": "tx", "maine": "me" }
    if value.lower() in state_to_code.keys():
        return state_to_code[value.lower()]
    else:
        return value
