import sys

from reader.labels_reader.labels_reader_txt import LabelsReaderTxt
from utils.fileutils import debug


def create_reader(filename, info_format, conf, batches_id, language_scheme = None, alt = False):

    #read labels with txt format
    if info_format == "txt": return LabelsReaderTxt(filename, conf, batches_id, language_scheme, alt)

    raise ValueError('label format must be "txt"')

