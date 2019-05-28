import os
import datetime as dt
import glob
import fnmatch
import struct
import pandas as pd
from tensorboardX.proto import event_pb2


def get_event_files(directory, recursive=True):
    """ Return the full path to all files in directory matching the specified
    pattern.
    Arguments:
        directory (str): Directory path in which to look
        recursive (bool): Searches recursively if True
    Returns:
        A list of matching file paths
    """

    pattern = "events.*"
    matches = list()
    if recursive is False:
        it = glob.iglob(os.path.join(directory, pattern))
        for filename in it:
            matches.append(filename)
        return matches

    # If we want to recurse, use os.walk instead
    for root, dirnames, filenames in os.walk(directory):
        extend = [os.path.join(root, ss) for ss in
                  fnmatch.filter(filenames, pattern)]
        matches.extend(extend)
    return matches


class EventReader(object):
    def __init__(self, filename=None, mode="rb"):
        self.filename = filename
        self.mode = mode
        self._fh = None
        self._buf = None
        self._index = 0
        
    def __enter__(self):
        self._fh = open(self.filename, self.mode)
        self._buf = self._fh.read()
        self._index = 0
        return self
    
    def _read_event(self):       
        # Read the header which tells the length of the event string
        header_str = self._buf[self._index: self._index + 8]
        header = struct.unpack('Q', header_str)
        self._index += 12
        
        # Read the event string
        header_len = int(header[0])
        event_str = self._buf[self._index: self._index + header_len]        
        self._index += (header_len + 4)
        
        # Parse event string
        ev = event_pb2.Event()
        try:
            ev.ParseFromString(event_str)
        except:
            raise
        
        return ev

    def read(self):
        events = []
        while self._index < len(self._buf):
            event = self._read_event()
            if event is not None:
                events.append(event)
        return events
    
    def __exit__(self, *args):
        self._fh.close()


def get_summary_events(event_file):
    with EventReader(event_file, "rb") as fh:
        events = fh.read()
    for event in events:
        if event.HasField("summary"):
            yield event


def get_valid_event_files(event_files):
    valid = []
    for ef in event_files:
        try:
            it = get_summary_events(ef)
            ev = next(iter(it))
        except:
            continue
        valid.append(ef)
    return valid


def get_scalar_dataframe(event_file, maxlen=200, load=True, store=True):
    df_fname = os.path.join(os.path.dirname(event_file), "extracted.csv")
    if load and os.path.isfile(df_fname):
        df = pd.read_csv(df_fname)
        return df
    
    values = dict()
    for event in get_summary_events(event_file):
        for v in event.summary.value:
            if v.HasField("simple_value"):
                tag = v.tag
                values.setdefault(tag, list()).append((dt.datetime.fromtimestamp(event.wall_time), 
                                                       event.step, v.simple_value))
    df = None
    for nm, value in values.items():
        if len(value) > maxlen:
            continue
        if df is None:
            df = pd.DataFrame(value, columns=["wall_time", "step", nm]).set_index("step")
            #df["relative_time"] = df["wall_time"] - df.loc[0]["wall_time"]
        else:
            df = df.join(pd.DataFrame(value, columns=["wall_time", "step", nm]).set_index("step")[nm], how="outer")
    if store and (df is not None):
        df.to_csv(df_fname)
    return df
