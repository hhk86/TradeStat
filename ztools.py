# -*- coding: utf-8 -*-
'''
Created on 2018.11.06

@author: DoubleZ
'''
from collections import defaultdict
import copy
import re
import datetime
from collections import UserDict
from itertools import zip_longest as zip_longest

from dateutil.parser import parse
import numpy as np
import pandas as pd
from jinja2 import Template

#from ipdb import set_trace
"""
This file provides frequently-used functions.
"""

__all__ = ['zlist',  # similar as list(), different when argument is string
           'zset',  # similar as set(), different when argument is string
           'znested_dict',  # nested_dict, dict with infinity levels
           'zmatdict',  # another version of dict, 2-dimensional dict

           'zcodeparse',  # parse stock code format
           'ztimeparse',  # parse and convert date/time format
           'zparse',  # improved version of dateutil.parser.parse

           'intraday_timeseries',  # intraday timeseries at specified freq
                                   # on some day

           'with_function_style',  # decorate class with functional style,
                                   # directly call "class_(args)" is OK,
                                   # class_ must have __call__ func

           'zsqueeze'  # compress pd.df to series,
                       # and drop unnecessary index levels
           ]


to_be_check = ['zlistofdicts_to_nesteddict']  # dict/list of dicts into nested dict, like a tree


def to_iter(raw, type_='list'):
    """
    Convert raw into iterable type.

    Arguments
    ---------
    raw: string\list\dict\set\tuple, and so on
        The data to be converted.
    type_: string, 'list' or 'set'
        The iterable type of returned value.

    Returns
    -------
    Iterable varible with type 'list' (default) or 'set'
    """
    if type_ == 'list':
        return [[raw], list(raw)][hasattr(raw, '__iter__')]
    elif type_ == 'set':
        return [[raw], set(raw)][hasattr(raw, '__iter__')]
    else:
        raise Exception('to_iter: Type ' + type_ + " hasn't been implemented")


def zlist(raw):
    """
    zlist provide similar function as list(), except for string.
    list("abc") returns ["a", "b", "c"] while zlist return ["abc"].
    """
    if hasattr(raw, '__iter__'):
        if isinstance(raw, list):
            return raw
        elif isinstance(raw, (bytes, str)):
            return [raw]
        else:
            return list(raw)
    else:
        return [raw]


def zset(raw):
    """
    zlist provide similar function as set(), except for string.
    list("abc") returns {"a", "b", "c"} while zlist return {"abc"}.
    """
    return to_iter(raw, type_='set')


def znested_dict():
    return defaultdict(znested_dict)


def zlistofdicts_to_nesteddict(dicts, leveled_keys, nest=None, drop=False):
    '''
    Convert dict/list of dicts into nested dict representing a tree structure.

    It could be optimized for the piece of code
    converting list of dicts into nested dict.

    I have ever figured out two ideas:

    1、    This is ugly and will be improved soon.

        keys = [map(lambda d: d.pop(key), list_of_dicts) for key in key_levels]
        for key in reversed(keys):
            list_of_dicts = [{k: v} for k, v in zip(key, list_of_dicts)]

    2、    Another idea with taking advantage of pd.MultiIndex, very ugly

    df = pd.DataFrame(list_of_dicts).set_index(leveled_keys)
    df = df.groupby(leveled_keys).apply(lambda df: df.to_dict(orient='records')[0])

    leveled_keys.pop()
    while leveled_keys:
        df = df.groupby(leveled_keys).apply(lambda df: df.to_dict(orient='records')[0])
        leveled_keys.pop()
    '''
    list_of_dicts = copy.deepcopy(zlist(dicts))
    if drop:
        keys = [map(lambda d: d.pop(key), list_of_dicts) for key in leveled_keys]
    else:
        keys = [map(lambda d: d[key], list_of_dicts) for key in leveled_keys]
    keys = zip(*keys)

    if nest is None:
        nest = znested_dict()

    expr = Template("nest{%- for k in tuplekey -%}['{{k}}']{%- endfor -%}")

    for tuplekey, val in zip(keys, list_of_dicts):
        exec(expr.render(tuplekey=tuplekey) + "=val" in globals(), locals())

    return nest


def intraday_timeseries(date, freq,
                        tradetime=[['9:15', '11:30'], ['13:00', '15:00']],
                        closed='right'):
    """
    Intraday timeseries at given date for specified freq.

    Arguments
    ---------
    date: str
    freq: str
    tradetime: list
    closed: str
        return left-end or right-end of time interval

    Returns
    -------
    numpy.array, each time tick could be Timestamp or str
    """
    from pandas import date_range

    # time series to be returned
    timeseries = np.array([])

    # make freq format suitable for pd.date_range requirement
    pattern = '^([1-9][0-9]*)([HTMShtms])$'
    g = re.match(pattern, freq)
    if g:
        n, s = g.group(1), g.group(2)
        freq = n + [s, 'T'][s in ['M', 'm']]
    else:
        raise Exception("Incorrect freq")

    # generate timeseries at given freq with pd.date_range
    for start, end in tradetime:
        start = date + ' ' + start
        end = date + ' ' + end
        nts = date_range(start=start, end=end, freq=freq, closed=closed)
        # nts = nts.strftime('%Y%m%d %H:%T:%S').values
        timeseries = np.concatenate([timeseries, nts.to_pydatetime()])

    # return
    return timeseries


def ztimeparse(s, date_format, *args, **kwargs):
    """
    Align datetime string (or list/array of datetime string)
    into the same datetime format.
    """
    if isinstance(s, pd.Series):
        s = s.values
    data = pd.to_datetime(s, *args, **kwargs).strftime(date_format)
    if isinstance(data, pd.Index):
        data = data.tolist()
    return data


def zparse(timestr):
    """
    Improved version of dateutil.parser.parse

    If timestr is string, parse it and return datetime.date.
    If timestr is datetime.date or datetime.datetime, just return.
    Else, raise Exception
    """
    try:
        datetime_ = parse(timestr)
        if datetime_.time() == datetime.time.min:
            return datetime_.date()
        else:
            return datetime_
    except TypeError:
        if isinstance(timestr, (datetime.date, datetime.datetime)):
            return timestr
        else:
            raise TypeError("Unknown input in zparse.")


def with_function_style(class_):
    """
    This function decorate class with functional style.
    Instead of: "obj = class_(); obj(args);",
    now you can directly call "class_(args)" if class_ definition
    is already decorated by this function.
    """
    obj = class_()

    def wrapper(*args, **kwargs):
        return obj(*args, **kwargs)
    return wrapper


@with_function_style
class zcodeparse(object):
    """
    This class convert input codes (list-like) into
    codes with specified format.

    Example:
    # convert stk codes into output with format "600000.SH"
    v = ["600891","SZ000001","600001.SH",('0','600019')]
    print ParseStkCode(v,"600000.SH")

    Inside ParseStkCode, stk code are stored with format:("0","600000")
    """

    _instance = None

    # Stk Code Format
    # When adding new format, append new format in the end, and
    # add handle code for new format in __parse method
    _mode = "|".join(["^(SH|SZ|sh|sz){1}(00\d{4}|30\d{4}|15\d{4}|60\d{4}|51\d{4})",
                      "(00\d{4}|30\d{4}|15\d{4}|60\d{4}|51\d{4}).(SH|SZ|sh|sz){1}",
                      "(00\d{4}|30\d{4}|15\d{4}|60\d{4}|51\d{4})$"])

    _mapping_exch_mkt = {"SH": "0", "sh": "0", "SZ": "1", "sz": "1",
                         "0": "0", "1": "1"}

    def __new__(cls):
        "This class work in form of singleton"
        if cls._instance is None:
            cls._mode = re.compile(cls._mode)
            cls._instance = object.__new__(cls)
        return cls._instance

    def __init__(self):
        pass

    def __parse(self, code):
        """
        convert input code(str) into internal format: ("0","600000")
        """
        if isinstance(code, tuple):
            assert len(code) == 2
            exch_mkt, code_num = code
            assert exch_mkt in ['0', '1']
            if re.search("^(00\d{4}|30\d{4}|15\d{4}|60\d{4}|51\d{4})$", code_num):
                return exch_mkt, code_num
            else:
                raise Exception("Unkown Code Num: {}".format(code_num))

        elif isinstance(code, (str, bytes)):
            parsed_result = self._mode.search(code).groups()
            valid_parsed_result, valid_position =\
                self._filter_None(parsed_result)
            if valid_position == (0, 1):
                exch_mkt, code_num = valid_parsed_result
            elif valid_position == (2, 3):
                code_num, exch_mkt = valid_parsed_result
            elif valid_position == (4,):
                code_num, = valid_parsed_result
                exch_mkt = self._infer_exchmkt_bycode(code_num)
            else:
                raise Exception("Unknown Code Format.")
            exch_mkt = self._mapping_exch_mkt[exch_mkt]
            return exch_mkt, code_num

        else:
            raise Exception("Code type should be Tuple or Str")

    def __vparse(self, codes):
        """
        convert input codes(list-like) into internal format: ("0","600000")
        """
        if isinstance(codes, (tuple, str, bytes)):
            return self.__parse(codes)
        else:
            return list(map(lambda v: self.__parse(v), codes))

    def __format_reconstructcode(self, formatlike):
        """
        According to specified format, make corresponding function
        that converts internal codes (with format: ("0","600000"))
        into specified format.
        """
        if isinstance(formatlike, tuple):
            assert len(formatlike) == 2
            exch_mkt, code_num = formatlike
            assert exch_mkt in ['0', '1']
            if re.search("^(00\d{4}|30\d{4}|15\d{4}|60\d{4}|51\d{4})$", code_num):
                return lambda code: code
            else:
                raise Exception("Unkown Code Num: {}".format(code_num))
        elif isinstance(formatlike, (str, bytes)):
            parsed_result = self._mode.search(formatlike).groups()
            _, valid_position = self._filter_None(parsed_result)
            if valid_position == (0, 1):
                return lambda code: {"0": "SH", "1": "SZ"}[code[0]]+code[1]
            elif valid_position == (2, 3):
                return lambda code: code[1]+"."+{"0": "SH", "1": "SZ"}[code[0]]
            elif valid_position == (4,):
                return lambda code: code[1]

    def _infer_exchmkt_bycode(self, code_num):
        code_num = str(code_num)
        if code_num[:2] in ["00", "30", "15"]:
            return "SZ"
        elif code_num[:2] in ["60", "51"]:
            return "SH"
        else:
            raise Exception("Unknown Code Number.")

    def _filter_None(self, list_like):
        """
        This function filter all None value of input argument-list_like.
        And assert only One value is valid and return.
        """
        result = filter(lambda v: v[1] is not None, enumerate(list_like))
        result = list(zip(*result))
        return result[1], result[0]

    def __call__(self, codes, formatlike):
        # parse code, convert into internal format: ("0","600000")
        if isinstance(codes, (tuple, str, bytes)):
            codes = [codes]
        codes = self.__vparse(zlist(codes))
        # Convert internal codes into specified format
        format_func = self.__format_reconstructcode(formatlike)
        codes = list(map(format_func, codes))
        return codes[0] if len(codes) == 1 else codes


class zmatdict(UserDict):
    """
    """
    def __init__(self, index_columns_pairlist, values=None):
        """
        default arrange: column first
        """
        try:
            keys_values = zip_longest(index_columns_pairlist,
                                      values, fillvalue=None)
        except Exception:
            keys_values = zip_longest(index_columns_pairlist,
                                      [None], fillvalue=None)
        self.index_ = [pair[0] for pair in index_columns_pairlist]
        self.columns_ = [pair[1] for pair in index_columns_pairlist]
        self.data = UserDict(keys_values)

    @property
    def index(self):
        return self.index_

    @property
    def columns(self):
        return self.columns_


def zsqueeze(s_or_df):
    s_or_df = s_or_df.squeeze()
    if isinstance(s_or_df.index, pd.MultiIndex):
        s_or_df.index = s_or_df.index.remove_unused_levels()
        len_eq_1 = list(map(lambda i: len(i) == 1, s_or_df.index.levels))
        droped_level = [i for i, b in enumerate(len_eq_1) if b]
        s_or_df.index = s_or_df.index.droplevel(droped_level)
    return s_or_df
