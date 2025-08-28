def get_nested_attribute(obj, attr_path):
    attrs = attr_path.split('/')
    for attr in attrs:
        obj = getattr(obj, attr)
    return obj