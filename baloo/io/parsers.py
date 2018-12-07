from .csv import read_csv

# TODO: could implement more io by just delegating to pandas, but do we really want that (now)?
# TODO: parsers could be lazy, i.e. only read header and metadata, however this only defers parsing to later
