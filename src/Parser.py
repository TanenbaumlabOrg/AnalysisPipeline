import re
import jdcal

def parse_metadata_string(metadata_string):
    # Initialize the result dictionary
    metadata_dict = {}

    # Split the metadata string into sections based on section headers
    sections = re.split(r'(\w+ Settings:)', metadata_string)
    
    # Metadata section
    metadata_section = sections[0].strip().replace('Metadata:\r\n', '')
    metadata_dict.update(parse_key_value_pairs(metadata_section))
    
    # Camera and Microscope settings
    for i in range(1, len(sections), 2):
        section_name = sections[i].strip().replace(':', '')
        section_content = sections[i + 1].strip()
        metadata_dict[section_name] = parse_key_value_pairs(section_content)
        
    return metadata_dict

def parse_key_value_pairs(section_content):
    pairs_dict = {}
    lines = section_content.split('\r\n')
    
    current_key = None
    current_value = []
    
    for line in lines:
        if ':' in line:
            if current_key:
                pairs_dict[current_key] = ' '.join(current_value).strip()
            key, value = map(str.strip, line.split(':', 1))
            pairs_dict[key] = value
            current_key = key
            current_value = [value]
        else:
            current_value.append(line.strip())
    
    if current_key:
        pairs_dict[current_key] = ' '.join(current_value).strip()
    
    return pairs_dict

def julian_day_to_gregorian(julian_day_number):
    # jdcal.gcal2jd() returns the Gregorian date corresponding to the Julian Day Number
    # gcal2jd returns (year, month, day, fraction_of_day)
    year, month, day, fraction_of_day = jdcal.jd2gcal(julian_day_number, 0.5)
    return year, month, day
