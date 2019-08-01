# -*- coding: utf-8 -*-
'''
Created on 2019.01.10
@author: DoubleZ
'''
import sys
import numpy as np
import pandas as pd

from jinja2 import Template
#from ipdb import set_trace
from ztools import zparse, zsqueeze

try:
    sys.path.append("C:\\Programs\\Tinysoft\\Analyse.NET")
    import TSLPy3 as tsl3
except ImportError:
    sys.path.append("D:/tinysoft/Analyse.NET")
    import TSLPy3 as tsl3

from ztools import ztimeparse, zcodeparse, with_function_style, zlist


__all__ = ['ts']


# TinySoft Server Infomation
TinySoft_Server = "tsl.tinysoft.com.cn", 443
User_Pwd = "fzzqjyb", "123456"


# _TsPy metaclass, indeed same as type until now
class TsPyMeta(type):
    def __new__(cls, *args, **kwargs):
        return super(TsPyMeta, cls).__new__(cls, *args, **kwargs)

    def __init__(cls, *args, **kwargs):
        super(TsPyMeta, cls).__init__(*args, **kwargs)


class TsPy(object):
    """
    TsInterface provide an uniform interface to Tinysoft server.
    The tinysoft server return values
    with format "[{...},{...},...]" and "gbk" coding.
    In this class, the list return values are converted
    into pandas DataFrame, with unicode coding.
    """
    __metaclass__ = TsPyMeta
    _instance = None

    def __new__(cls):
        '''
        This class is singleton mode.
        '''
        if cls._instance is None:
            obj = super(TsPy, cls).__new__(cls)
            cls._instance = obj
        return cls._instance

    def __init__(self):
        '''
        connect tinysoft server by calling self.start()
        '''
        super(TsPy, self).__init__()
        self.isconnected = False
        self.ts = tsl3
        self.start()
        self.isconnected = True

    def __del__(self):
        if self.isconnected:
            self.stop()

    def start(self):
        '''
        connect tinysoft server.
        '''
        if not self.isconnected:
            fail, _, _ = self.ts.RemoteExecute('return 1;', {})
            if fail:
                self.ts.ConnectServer(*TinySoft_Server)
                dl = self.ts.LoginServer(*User_Pwd)
                if dl[0] != 0:
                    raise Exception("TS server Reloginning Refused!")
                self.isconnected = self.ts.Logined()
            print("connect to Ts server")

    def stop(self):
        '''
        Disconnect tinysoft server.
        '''
        self.ts.Disconnect()
        self.isconnected = False
        print('disconnect with Ts server.')

    def calltsl(self, tsl, columns=None, index_names=None):
        raw_data = self._fetch(tsl)
        if raw_data:
            return _ts_rawdata_to_dataframe(raw_data, columns, index_names)
        else:
            return None

    def _fetch(self, tsl):
        "Get data through tinysoft interface."
        fail, data, _ = self.ts.RemoteExecute(tsl, {})
        if not fail:
            return data
        else:
            raise Exception("Error when execute tsl")

    def callfunc(self, func, args):
        fail, data, _ = self.ts.RemoteCallFunc(func, args, {})
        if not fail:
            data["ticker"] = zcodeparse(data["ticker"], "600000.SH")
            return pd.DataFrame(data).set_index(["date", "ticker"])
        else:
            raise Exception("Error when execute callfunc.")

    def statement(self, table_name, entries, codes, start_date, end_date):
        codes = zlist(zcodeparse(codes, formatlike='SH600000'))
        start_date = ztimeparse(start_date, '%Y-%m-%d')
        end_date = ztimeparse(end_date, '%Y-%m-%d')
        table_num = {'balance': '44', 'income': '46', 'cash': '48'}
        tsltemp = Template('''
                    SetSysParam('ReportMode',-1);
                    Return select ['StockID'] as 'ticker',
                                  ['StockName'] as 'name',
                                  ['截止日'] as 'report_period',
                                  ['公布日'] as 'ann_date'
                                  {%- for tsk, myk in entries.items() -%}
                                      ,\n['{{tsk}}'] as '{{myk}}'
                                  {%- endfor %}
                           from infotable {{tnum}} of array('{{codes}}')
                           where ['截止日']>=DateToInt(strtodate('{{startdate}}'))
                               and ['公布日']<=DateToInt(strtodate('{{enddate}}'))
                    end;
                    ''')
        tsl = tsltemp.render(entries=entries,
                             codes="','".join(codes),
                             startdate=start_date,
                             enddate=end_date,
                             tnum=table_num[table_name])
        return self.calltsl(tsl, None, ['ann_date', 'ticker', 'report_period'])

    def shares(self, codes, start_date, end_date):
        codes = zlist(zcodeparse(codes, formatlike='SH600000'))
        start_date = ztimeparse(start_date, '%Y-%m-%d')
        end_date = ztimeparse(end_date, '%Y-%m-%d')
        tsltemp = Template('''
                  stockarray:=array('{{codes}}');
                  begt:=strtodate('{{startdate}}');
                  endt:=strtodate('{{enddate}}');
                  total:=array();
                  dates:=MarketTradeDayQk(begt,endt);
                  for i:=0 to length(stockarray)-1 do
                  begin
                      for j:=0 to length(dates)-1 do
                      begin
                          total union= ``array(
                                         'ticker':stockarray[i],
                                         'date':datetostr(dates[j]),
                                         'totalshares':Spec(StockTotalShares(dates[j]),stockarray[i]),
                                         'floatshares':Spec(StockNegotiableShares(dates[j]),stockarray[i]));
                      end
                  end
                  return total;
                  ''')
        tsl = tsltemp.render(codes="','".join(codes),
                             startdate=start_date,
                             enddate=end_date
                             )
        return self.calltsl(tsl, None, ['date', 'ticker'])

    def price(self, codes, start_date, end_date, params={}):
        codes = zlist(zcodeparse(codes, formatlike='SH600000'))
        start_date = ztimeparse(start_date, '%Y-%m-%d')
        end_date = ztimeparse(end_date, '%Y-%m-%d')
        pn_rate = params.setdefault('pn_rate', 0)
        tsltemp = Template('''
                  stockarray:=array('{{codes}}');
                  begt:=strtodate('{{startdate}}');
                  endt:=strtodate('{{enddate}}');
                  datearray:= MarketTradeDayQk(begt, endt);
                  total:=array();

                  for i:=0 to length(stockarray)-1 do
                  begin
                      for j:=0 to length(datearray)-1 do
                      begin
                          setsysparam(Pn_stock(), stockarray[i]);
                          setsysparam(Pn_rate(), {{pn_rate}});
                          setsysparam(pn_date(), datearray[j]);
                          total union= ``array('ticker': stockarray[i],
                                        'date': datetostr(datearray[j]),
                                        'open': open(),
                                        'high': high(),
                                        'low': low(),
                                        'prevclose': StockPrevClose3(),
                                        'close': close(),
                                        'ret': stockzf3(),
                                        'vol': vol(),
                                        'amount': amount());
                      end
                  end
                  return total;
                  ''')
        tsl = tsltemp.render(codes="','".join(codes),
                             startdate=start_date,
                             enddate=end_date,
                             pn_rate=pn_rate
                             )
        return self.calltsl(tsl, None, ['date', 'ticker'])

    def tsfactor(self, codes, start_date, end_date, factor_tsql):
        codes = zlist(zcodeparse(codes, formatlike='SH600000'))
        start_date = ztimeparse(start_date, '%Y-%m-%d')
        end_date = ztimeparse(end_date, '%Y-%m-%d')
        tsltemp = Template('''
                  stockarray:=array('{{codes}}');
                  begt:=strtodate('{{startdate}}');
                  endt:=strtodate('{{enddate}}');
                  datearray:= MarketTradeDayQk(begt, endt);
                  total:=array();
                  SetSysParam(PN_Precision(),6);
                  setsysparam(Pn_rate(), 0);
                  for i:=0 to length(stockarray)-1 do
                  begin
                      for j:=0 to length(datearray)-1 do
                      begin
                          setsysparam(Pn_stock(), stockarray[i]);
                          setsysparam(pn_date(), datearray[j]);
                          rdate := NewReportDateOfEndT2(datearray[j]);
                          total union= ``array('ticker': stockarray[i],
                                        'date': datetostr(datearray[j]),
                                        'value': {{factor_tsql}});
                      end
                  end
                  return total;
                  ''')
        tsl = tsltemp.render(codes="','".join(codes),
                             startdate=start_date,
                             enddate=end_date,
                             factor_tsql=factor_tsql
                             )
        return self.calltsl(tsl, None, ['date', 'ticker']).squeeze()

    def fsdata_ttm(self, entry, codes, start_date, end_date):
        codes = zlist(zcodeparse(codes, formatlike='SH600000'))
        start_date = ztimeparse(start_date, '%Y-%m-%d')
        end_date = ztimeparse(end_date, '%Y-%m-%d')
        entries = {'revenue': 46002, 'net_income': 46033,
                   'net_income2major': 46078,
                   'cost': 46005,
                   'net_value2major': 44140, 'total_asset': 44059,
                   'total_debt': 44097,
                   'float_asset': 44028, 'float_debt': 44083,
                   'inventory': 44019,
                   'operating_cashflow': 48018}
        tsltemp = Template('''
                  stockarray:=array('{{codes}}');
                  begt:=strtodate('{{startdate}}');
                  endt:=strtodate('{{enddate}}');
                  total:=array();
                  dates:=MarketTradeDayQk(begt,endt);
                  for i:=0 to length(stockarray)-1 do
                  begin
                      for j:=0 to length(dates)-1 do
                      begin
                          setsysparam(pn_stock(),stockarray[i]);
                          setsysparam(pn_date(), dates[j]);
                          RDate:=NewReportDateOfEndT2(dates[j]);
                          v:=Last12MData(RDate,{{entrynum}});
                          total union= ``array('ticker':stockarray[i],
                                               'date':datetostr(dates[j]),
                                               '{{entry}}':v);
                      end
                  end
                  return total;
                  ''')
        tsl = tsltemp.render(codes="','".join(codes),
                             startdate=start_date,
                             enddate=end_date,
                             entry=entry, entrynum=str(entries[entry]))
        return self.calltsl(tsl, None, ['date', 'ticker']).squeeze()

    def fsdata_raw(self, entry, codes, start_date, end_date):
        codes = zlist(zcodeparse(codes, formatlike='SH600000'))
        start_date = ztimeparse(start_date, '%Y-%m-%d')
        end_date = ztimeparse(end_date, '%Y-%m-%d')
        entries = {'net_value2major': 44140, 'total_asset': 44059,
                   'inventory': 44019, 'receivable': 44009,
                   'revenue': 46002, 'operating_profit': 46015,
                   'cost': 46005,
                   'net_income2major': 46078}
        tsltemp = Template('''
                  stockarray:=array('{{codes}}');
                  begt:=strtodate('{{startdate}}');
                  endt:=strtodate('{{enddate}}');
                  datearray:= MarketTradeDayQk(begt, endt);
                  total:=array();
                  for i:=0 to length(stockarray)-1 do
                  begin
                      for j:=0 to length(datearray)-1 do
                      begin
                          setsysparam(pn_stock(),stockarray[i]);
                          setsysparam(pn_date(), datearray[j]);

                          RDate:=NewReportDateOfEndT2(datearray[j]);
                          RtDate:=PreviousReportDate(RDate);
                          Rt2Date:=PreviousReportDate(RtDate);
                          RyDate:=strtoint(FormatDateTime('yyyy',inttodate(RtDate))+'1231');
                          Ry2Date:=strtoint(FormatDateTime('yyyy',inttodate(Rt2Date))+'1231');

                          RpDate:=PreviousReportDateOfQuarter(RDate,1);
                          RptDate:=PreviousReportDate(RpDate);
                          RpyDate:=strtoint(FormatDateTime('yyyy',inttodate(RptDate))+'1231');

                          v1:=ReportOfAll({{entrynum}},RDate);
                          v2:=ReportOfAll({{entrynum}},RyDate);
                          v3:=ReportOfAll({{entrynum}},RtDate);

                          v4:=ReportOfAll({{entrynum}},RpDate);
                          v5:=ReportOfAll({{entrynum}},RpyDate);
                          v6:=ReportOfAll({{entrynum}},RptDate);

                          v7:=ReportOfAll({{entrynum}},Rt2Date);
                          v8:=ReportOfAll({{entrynum}},Ry2Date);

                          total union= ``array('ticker':stockarray[i],
                                               'date':datetostr(datearray[j]),
                                               'latest_report':datetostr(inttodate(RDate)),
                                               'latest_data':v1,
                                               'latest_annual_report':datetostr(inttodate(RyDate)),
                                               'latest_annual_data': v2,
                                               'latest_lastyear_report':datetostr(inttodate(RtDate)),
                                               'latest_lastyear_data': v3,
                                               'latest_annual2_report': Ry2Date,
                                               'latest_annual2_data': v8,
                                               'latest_last2year_report': Rt2Date,
                                               'latest_last2year_data': v7,
                                               'latest_prev_report':datetostr(inttodate(RpDate)),
                                               'latest_prev_data': v4,
                                               'latest_prev_annual_report':datetostr(inttodate(RpyDate)),
                                               'latest_prev_annual_data': v5,
                                               'latest_prev_lastyear_report':datetostr(inttodate(RptDate)),
                                               'latest_prev_lastyear_data': v6 );
                      end
                  end
                  return  total;
                  ''')
        tsl = tsltemp.render(codes="','".join(codes),
                             startdate=start_date,
                             enddate=end_date,
                             entrynum=entries[entry])
        return self.calltsl(tsl, None, ['date', 'ticker'])

    def tsbeta(self, index, codes, start_date, end_date, params):
        codes = zlist(zcodeparse(codes, formatlike='SH600000'))
        index = zcodeparse(index, formatlike='SH600000')
        start_date = ztimeparse(start_date, '%Y-%m-%d')
        end_date = ztimeparse(end_date, '%Y-%m-%d')
        ns = {'1M': 21, '3M': 63, '6M': 126, '12M': 252}[params['window']]
        tsltemp = Template('''
            stockarray:=array('{{codes}}');
            begt:=strtodate('{{startdate}}');
            endt:=strtodate('{{enddate}}');
            datearray:= MarketTradeDayQk(begt, endt);
            total:=array();
            SetSysParam(PN_Precision(),6);
            for j:=0 to length(datearray)-1 do
            begin
                SetSysParam(pn_cycle(), cy_day());
                t1 := datearray[j];
                setsysparam(pn_date(), t1);
                t0 := ref(sp_time(), {{ns}});
                for i:=0 to length(stockarray)-1 do
                begin
                    setsysparam(Pn_stock(), stockarray[i]);
                    total union= ``array('ticker': stockarray[i],
                                    'date': datetostr(datearray[j]),
                                    'value': StockBeta('{{index}}', t0, t1));
                end
            end
            return total;
                  ''')
        tsl = tsltemp.render(codes="','".join(codes),
                             startdate=start_date,
                             enddate=end_date,
                             index=index,
                             ns=ns)
        return self.calltsl(tsl, None, ['date', 'ticker'])

    def tsmomentum(self, codes, start_date, end_date, params):
        codes = zlist(zcodeparse(codes, formatlike='SH600000'))
        start_date = ztimeparse(start_date, '%Y-%m-%d')
        end_date = ztimeparse(end_date, '%Y-%m-%d')
        ns = {'1M': 21, '3M': 63, '6M': 126, '12M': 252}[params['window']]
        tsltemp = Template('''
                  stockarray:=array('{{codes}}');
                  begt:=strtodate('{{startdate}}');
                  endt:=strtodate('{{enddate}}');
                  datearray:= MarketTradeDayQk(begt, endt);
                  total:=array();
                  setsysparam(Pn_precision(), 6);

                  t0 := array();
                  t1 := datearray;
                  setsysparam(pn_stock(), 'SH000300');
                  for j:=0 to length(t1)-1 do
                  begin
                      setsysparam(pn_date(), t1[j]);
                      t0 union= array(nday({{ns}}+1, 't0',
                                strtodate(datetimetostr(sp_time())))[0]);
                  end
                  t0 := sselect ['t0'] from t0 end;

                  for i:=0 to length(stockarray)-1 do
                  begin
                      for j:=0 to length(t1)-1 do
                      begin
                          setsysparam(Pn_stock(), stockarray[i]);
                          setsysparam(Pn_rate(), 1);
                          total union= ``array('ticker': stockarray[i],
                                        'date': datetostr(t1[j]),
                                        'momentum': stockzf(t0[j],t1[j]));
                      end
                  end
                  return total;
                  ''')
        tsl = tsltemp.render(codes="','".join(codes),
                             startdate=start_date,
                             enddate=end_date,
                             ns=ns
                             )
        return self.calltsl(tsl, None, ['date', 'ticker'])

    def tsvolatility(self, codes, start_date, end_date, params):
        codes = zlist(zcodeparse(codes, formatlike='SH600000'))
        start_date = ztimeparse(start_date, '%Y-%m-%d')
        end_date = ztimeparse(end_date, '%Y-%m-%d')
        ns = {'1M': 21, '3M': 63, '6M': 126, '12M': 252}[params['window']]

        tsql = '''
            datearray:= MarketTradeDayQk(strtodate('2005-01-01'),
                                         strtodate('{0}'));
            results := array();
            for i:=0 to length(datearray)-1 do
            begin
                results[i] := datetostr(datearray[i]);
            end
            return results;
        '''.format(start_date)
        calendar = zsqueeze(ts.calltsl(tsql))
        calendar = calendar.apply(lambda t: t.decode('utf-8')).values
        calendar = np.sort(calendar)

        data_start_date = calendar[calendar < start_date][-ns]

        pri = self.price(codes, data_start_date, end_date, {'pn_rate': 1})
        pri.loc[pri['amount'] < 1e-6, 'ret'] = None

        ret = pri['ret'].sort_index(level=['ticker', 'date'])

        def _std(s):
            v = s[-ns:].dropna().values
            if v.shape[0] < 2:
                return None
            return np.std(v, ddof=1)

        results = ret.groupby(level='ticker').apply(_std)
        '''
        results = pri.groupby(level='ticker', as_index=False)['ret']\
                     .rolling(ns, min_periods=2)\
                     .apply(_std, raw=False)
        results.index = results.index.droplevel(level=0)
        '''
        return results

    def tsvolatility2(self, codes, start_date, end_date, params):
        codes = zlist(zcodeparse(codes, formatlike='SH600000'))
        start_date = ztimeparse(start_date, '%Y-%m-%d')
        end_date = ztimeparse(end_date, '%Y-%m-%d')
        ns = {'1M': 21, '3M': 63, '6M': 126, '12M': 252}[params['window']]

        tsql = '''
            datearray:= MarketTradeDayQk(strtodate('2005-01-01'),
                                         strtodate('{0}'));
            results := array();
            for i:=0 to length(datearray)-1 do
            begin
                results[i] := datetostr(datearray[i]);
            end
            return results;
        '''.format(start_date)
        calendar = zsqueeze(ts.calltsl(tsql))
        calendar = calendar.apply(lambda t: t.decode('utf-8')).values
        calendar = np.sort(calendar)

        data_start_date = calendar[calendar < start_date][-ns]

        pri = self.price(codes, data_start_date, end_date, {'pn_rate': 1})
        pri.loc[pri['amount'] < 1e-6, 'ret'] = None

        ret2 = pri['ret']**2

        def _std(s):
            v = s.dropna().values
            if v.shape[0] < 2:
                return None
            return np.sqrt(v.sum()/(len(v)-1))

        results = ret2.groupby(level='ticker').apply(_std)
        return results


def _ts_rawdata_to_dataframe(raw_data, columns, index_names):
    """
    This function convert raw data returned by Tinysoft
    into pd.DataFrame. The string of 'gbk' format is also
    converted into unicode.
    """
    array_data = np.asanyarray(raw_data)
    if isinstance(array_data[0], dict):
        array_data = _dict_to_series_with_decode('gbk', array_data)
        df_data = pd.concat(array_data, axis=1).transpose()
    elif isinstance(array_data[0], (str, bytes)):
        df_data = pd.DataFrame(array_data)
    elif isinstance(array_data[0], np.ndarray):
        df_data = pd.DataFrame.from_records(array_data)
    else:
        raise NotImplementedError()
    # reindex columns
    if columns:
        df_data.columns = columns
    # set index
    if index_names:
        df_data.set_index(index_names, inplace=True)
    # return result
    return df_data.sort_index()


@np.vectorize
def _dict_to_series_with_decode(coding, dict_):
    """
    This function convert dict of raw value into pd.Series.
    Any key/value with 'gbk' format will be converted into unicode.
    """
    new_dict = {}
    for key, val in dict_.items():
        # decode any key with string format
        if isinstance(key, (str, bytes)):
            key = key.decode(coding)
        # decode any value with string format
        if isinstance(val, (str, bytes)):
            val = val.decode(coding)
        # construct new dict
        new_dict[key] = val
    # convert new dict into series and return
    return pd.Series(new_dict)


ts = TsPy()


if __name__ == '__main__':
    tsport = TsPy()

    print(tsport.statement(table_name='income',
                           codes=['601186.SH'],
                           entries={'营业总收入': 'revenue1', '主营业务收入': 'rev'},
                           start_date='2018-05-01', end_date='2018-10-30'))

    '''
    print(tsport.shares(codes=['600000.SH', '000002.SZ', '601318.SH'],
                        start_date='20180801',
                        end_date='20180831'))
    '''
    '''
    print(tsport.price(codes=['600000.SH', '002236.SZ'],
                       start_date='20160415',
                       end_date='20160430'))
    '''
    #t1 = (tsport.fsdata_raw(codes=['601628.SH'],
    #                        start_date='2018-08-28', end_date='2018-08-31'))
    #t2 = (tsport.fsdata_ttm(codes=['600000.SH'],
    #                        start_date='2018-08-28', end_date='2018-08-31'))
