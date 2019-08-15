# 将records_analyst复制到空白的tradeStat.py，开始制作自己的分析程序。现测试303账户在7月31日交易的一篮子股票
# 但prices不能使用本地数据，要从天软获取。目前做到这一步


# -*- coding: utf-8 -*-
'''
Created on 2019.2.19, updated on 2019.08.01

@author: DoubleZ and Hongkai Huang
'''

import numpy as np
import pandas as pd
from ztools import zparse, zcodeparse
from collections import defaultdict, OrderedDict
import itertools
import datetime
import sys
import pprint


pp = pprint.PrettyPrinter(indent=4)


date = '2019-07-31'
acct = '303'
filename = 'order_20190731.xlsx'
# date = '2019-02-14'
# acct = '119'
# filename = '119帐户所有委托记录.xlsx'




# 1. order relationship


def nested_dict():
    # 无线深度的dictionary
    return defaultdict(nested_dict)


def saveData(data, data_name):
    # 数据存储
    store = pd.HDFStore('tradeanalyst_data2.h5')
    store[data_name] = data
    store.close()


class Analysis(object):
    def __init__(self, filename):
        # 所有委托记录数据
        order_records = pd.read_excel(filename)
        order_records.columns = order_records.loc[3, :].values
        order_records = order_records.loc[4:, ['合同序号', '资金帐号', '下单时间', '合法标志',
                                               '交易类型', '证券类别', '证券代码', '证券名称',
                                               '委托方向', '撤单标志', '委托价格', '委托数量',
                                               '委托撤单数量', '委托状态']]
        # 仅获取合法委托
        order_records = order_records[order_records['合法标志'] == '合法']
        order_records.index = range(order_records.shape[0])
        order_records['日期'] = order_records['下单时间'].apply(
            lambda s: zparse(s).strftime('%Y-%m-%d'))
        self.allorders = order_records

        # 所有成交记录数据
        knock_records = pd.read_excel('trade_20190731.xlsx')
        knock_records.columns = knock_records.loc[3, :].values
        knock_records = knock_records.loc[4:, ['成交时间', '清算时间', '资金帐号', '合同序号',
                                               '证券代码', '买卖方向', '成交价格', '成交数量',
                                               '成交金额', '证券类别', '一级过户费', '证管费',
                                               '经手费', '交易类型']]
        knock_records.index = range(knock_records.shape[0])
        knock_records = knock_records.iloc[:-1]
        knock_records['日期'] = knock_records['成交时间'].apply(
            lambda s: zparse(s).strftime('%Y-%m-%d'))
        self.allknocks = knock_records

    def getTYData(self, code_list):
        # prices needed
        from tspy import ts
        prices = pd.DataFrame([])
        codes = list(set(zcodeparse(code_list, 'SH600000')))
        for group_idx, group in itertools.groupby(enumerate(codes), lambda it: it[0] // 10):
            print('group: ' + str(group_idx))
            _code = list(map(lambda it: it[1], group))
            tsql = '''
                Setsysparam(pn_cycle(),cy_1s());
                Return Select datetimetostr(["date"]) as "time",
                       ['StockID'] as 'ticker', ["price"] as "price",
                       ["buy1"] as "buy1", ["sale1"] as "sale1"
                from MarketTable
                DateKey strtodate('{0}') to strtodate('{0}')+0.99
                Of "{1}"
                end;
            '''.format(date, ",".join(_code))
            _price = ts.calltsl(tsql, None, ['time', 'ticker']).squeeze()
            prices = pd.concat([prices, _price])
        prices.sort_index(level=['time', 'ticker'], inplace=True)
        prices = prices.loc[prices.index.get_level_values('time') < (date + ' 14:57:00')]
        for col in prices:
            prices.loc[(prices[col] < 1e-6).values, col] = None
        prices.fillna(method='ffill', inplace=True)
        prices.index.set_levels(zcodeparse(prices.index.levels[1], '600000.SH'),
                                level='ticker', inplace=True)

        # 存储prices
        saveData(prices, 'prices')

        return prices

    def analyze(self, date, acct, params={'assetclass': '股票', 'action': '买卖'}):
        # data to be analyzed
        ords = self.allorders[((self.allorders['资金帐号'] == acct.zfill(12))
                               & (self.allorders['日期'] == date)).values]
        knos = self.allknocks[((self.allknocks['资金帐号'] == acct.zfill(12))
                               & (self.allknocks['日期'] == date)).values]
        if params['assetclass'] == '股票':
            ords = ords[ords['证券类别'] == '股票']
            knos = knos[knos['证券类别'] == '股票']
        if params['action'] == '买卖':
            ords = ords[ords['交易类型'] == '正常买卖']
            knos = knos[knos['交易类型'] == '正常买卖']
        ords.index = range(ords.shape[0])
        knos.index = range(knos.shape[0])
        ords.loc[:, '证券代码'] = zcodeparse(ords['证券代码'], '600000.SH')
        knos.loc[:, '证券代码'] = zcodeparse(knos['证券代码'], '600000.SH')
        ords['消息类别'] = 'Order'
        knos['消息类别'] = 'Knock'

        # 天软获取证券市场价格
        # prices = self.getTYData(ords['证券代码'])
        store = pd.HDFStore('tradeanalyst_data2.h5')
        prices = store['prices']
        store.close()
        # 读取已存储的市场价格数据
        # store = pd.HDFStore('tradeanalyst_data.h5')
        # prices = store['prices']
        # store.close()

        # analyze
        records = nested_dict()
        for _, order in ords.iterrows():
            # 合并orders
            ticker, bs = order['证券代码'], order['委托方向']
            if bs in {'买入', '卖出'}:
                records[ticker, bs][order['下单时间']] = order.squeeze()
            elif bs == '撤单':
                bs = ords.loc[((ords['委托方向'].isin({'买入', '卖出'}))
                               & (ords['合同序号'] == order['合同序号'])).values,
                              '委托方向'].squeeze()
                records[ticker, bs][order['下单时间']] = order.squeeze()
            else:
                raise Exception("unknown ord['委托方向']")

        for _, kno in knos.iterrows():
            # 合并knocks
            ticker, bs = kno['证券代码'], kno['买卖方向']
            records[ticker, bs][kno['成交时间']] = kno.squeeze()

        trade_intents = nested_dict()
        for ticker, bs in records:
            # 将委托及其成交、撤单信息合并
            bs_astk = OrderedDict(sorted(records[ticker, bs].items(), key=lambda s: s[0]))
            for time in bs_astk:
                if bs_astk[time]['消息类别'] == 'Order' and bs_astk[time][
                    '撤单标志'] != '撤单':  # 撤单时委托方向也可能是买入或卖出，为实现同合同号数据合并，以撤单标志判断该笔记录
                    trade_intents[ticker, bs][bs_astk[time]['合同序号']] = {'ticker': bs_astk[time]['证券代码'],
                                                                        'bs': bs_astk[time]['委托方向'],
                                                                        'date_time': time,
                                                                        'pit_price': prices.loc[
                                                                            (time, ticker), 'price'],
                                                                        'pit_b1': prices.loc[(time, ticker), 'buy1'],
                                                                        'pit_s1': prices.loc[(time, ticker), 'sale1'],
                                                                        'order_price': bs_astk[time]['委托价格'],
                                                                        'order_qty': bs_astk[time]['委托数量'],
                                                                        'status': bs_astk[time]['委托状态'],
                                                                        'finished_qty': 0,  # 成交量
                                                                        'finished_amt': 0  # 成交金额
                                                                        }
                elif bs_astk[time]['消息类别'] == 'Knock':
                    # 此合同号下完成成交量
                    trade_intents[ticker, bs][bs_astk[time]['合同序号']]['finished_qty'] = bs_astk[time]['成交数量']
                    # 此合同号下完成成交金额
                    trade_intents[ticker, bs][bs_astk[time]['合同序号']]['finished_amt'] = bs_astk[time]['成交金额']

        # 合并报单数据
        trade_result = defaultdict(dict)
        for ticker, bs in trade_intents:
            # 将相关委托合并
            _qty = 0  # 剩余量
            _bs = ''  # 委托方向
            _ckey = ''  # 合同号
            for key, value in trade_intents[ticker, bs].items():
                if value['order_qty'] > _qty + 100 or value['bs'] != _bs:
                    # 此次需求结束，开始下一次交易
                    _qty = value['order_qty'] - value['finished_qty']
                    _bs = value['bs']
                    _ckey = key
                    trade_result[ticker, bs, _ckey] = {'ticker': value['ticker'],
                                                       'bs': value['bs'],
                                                       'con_list': [key],  # 合同号序列
                                                       'pit_price_list': [value['pit_price']],  # 最新价序列
                                                       'pit_b1_list': [value['pit_b1']],  # 买一价序列
                                                       'pit_s1_list': [value['pit_s1']],  # 卖一价序列
                                                       'order_price_list': [value['order_price']],  # 委托价序列
                                                       'order_qty_list': [value['order_qty']],  # 委托数量序列
                                                       'status_list': [value['status']],  # 委托状态序列
                                                       'finished_qty_list': [value['finished_qty']],  # 成交量序列
                                                       'finished_amt_list': [value['finished_amt']],  # 成交金额序列
                                                       're_qty': _qty,  # 未成交量
                                                       'time': [value['date_time']],  # 订单发出时间
                                                       'num_order': 0,  # 撤补次数
                                                       'time_fee': 0,  # 花费时间
                                                       'ave_price': 0  # 成交均价
                                                       }

                else:
                    # 继续完成此次委托
                    trade_result[ticker, bs, _ckey]['con_list'].append(key)
                    trade_result[ticker, bs, _ckey]['pit_price_list'].append(value['pit_price'])
                    trade_result[ticker, bs, _ckey]['pit_b1_list'].append(value['pit_b1'])
                    trade_result[ticker, bs, _ckey]['pit_s1_list'].append(value['pit_s1'])
                    trade_result[ticker, bs, _ckey]['order_price_list'].append(value['order_price'])
                    trade_result[ticker, bs, _ckey]['order_qty_list'].append(value['order_qty'])
                    trade_result[ticker, bs, _ckey]['status_list'].append(value['status'])
                    trade_result[ticker, bs, _ckey]['finished_qty_list'].append(value['finished_qty'])
                    trade_result[ticker, bs, _ckey]['finished_amt_list'].append(value['finished_amt'])
                    trade_result[ticker, bs, _ckey]['time'].append(value['date_time'])
                    # 剩余未成交数量
                    _qty = _qty - value['finished_qty']
                    trade_result[ticker, bs, _ckey]['re_qty'] = _qty

            # 成本核算：时间、交易、撤补量
            # try:
            #     print(value["re_qty"])
            # except:
            #     print(pp.pprint(value))
            #     print(_qty)
            #     sys.exit()

            for key, value in trade_result.items():
                if value['re_qty'] < 100:
                    # 撤补次数
                    trade_result[key]['num_order'] = len(value['con_list']) - 1
                    # 花费时间
                    trade_result[key]['time_fee'] = 0 if len(value['con_list']) == 1 \
                        else (datetime.datetime.strptime(value['time'][-1], "%Y-%m-%d %H:%M:%S") - \
                              datetime.datetime.strptime(value['time'][0], "%Y-%m-%d %H:%M:%S")).seconds / 60
                    # 成交均价
                    trade_result[key]['ave_price'] = np.matmul(value['finished_qty_list'],
                                                               value['order_price_list']) / np.sum(
                        value['finished_qty_list'])
                else:
                    trade_result[key]['time_fee'] = '未完成'
                    trade_result[key]['num_order'] = len(value['con_list']) - 1
                    trade_result[key]['ave_price'] = '未成交' if np.sum(value['finished_qty_list']) == 0 else np.matmul(
                        value['finished_qty_list'], value['order_price_list']) / np.sum(value['finished_qty_list'])
            # for key, value in trade_result.items():
            #     pass

        return trade_result


trade_result = Analysis(filename).analyze(date, acct)
for kw, order in trade_result.items():
    print("\n---------------------------------------------------")
    for key, value in order.items():
        print(key, ':', value)


