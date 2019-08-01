import numpy as np
import pandas as pd
import cx_Oracle
from zcodeparse import *


pd.options.display.max_columns = None
pd.options.display.max_rows = None


class OracleSql(object):
    '''
    Query data from database
    '''

    def __init__(self, pt=False):
        '''
        Initialize database
        '''
        self.host, self.oracle_port = '18.210.64.72', '1521'
        self.db, self.current_schema = 'tdb', 'wind'
        self.user, self.pwd = 'reader', 'reader'
        self.pt = pt

    def __enter__(self):
        '''
        Connect to database
        :return: self
        '''
        self.conn = self.__connect_to_oracle()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.close()

    def __connect_to_oracle(self):
        '''
        Connect to database
        :return: connection
        '''
        dsn = self.host + ':' + self.oracle_port + '/' + self.db
        try:
            connection = cx_Oracle.connect(self.user, self.pwd, dsn, encoding="UTF-8", nencoding="UTF-8")
            connection.current_schema = self.current_schema
            if self.pt is True:
                print('Connected to Oracle database successful!')
        except Exception:
            print('Failed on connecting to Oracle database!')
            connection = None
        return connection

    def query(self, sql: str) -> pd.DataFrame:
        '''
        Query data
        '''
        return pd.read_sql(sql, self.conn)

    def execute(self, sql: str):
        '''
        Execute SQL scripts, including inserting and updating

        '''
        self.conn.cursor().execute(sql)
        self.conn.commit()


# def calStockPNL() -> float:
#     PNL1 = calBalancePNL()
#     PNL2 = calStockTradingPNL()
#     return PNL1 + PNL2


def calStockTradingPNL() -> float:
    trading_data = pd.read_csv("302_0726.csv", encoding="gbk", converters={"证券代码": str})
    trading_data = trading_data[["证券代码", "证券名称", "成交结果", "成交价格", "成交数量", "交易费用"]]
    trading_data.columns = ["code", "stock_name", "direction", "price", "quantity", "fee"]
    sql = \
        '''
        SELECT
        ''' + '''
        S_INFO_WINDCODE,
        TRADE_DT,
        S_DQ_CLOSE
    FROM
        AShareEODPrices 
    WHERE
        TRADE_DT = 20190726
    '''
    with OracleSql() as oracle:
        stock_price = oracle.query(sql)
    trading_data = pd.merge(trading_data, stock_price, left_on="code", right_on="S_INFO_WINDCODE", how="left")
    trading_data["direction"] = trading_data.direction.apply(side)
    trading_data.ix[trading_data["code"] == "511880.SH", "S_DQ_CLOSE"] = 101.600
    trading_data.ix[trading_data["code"] == "510500.SH", "S_DQ_CLOSE"] = 5.285
    # trading_data["trading_fee"] = trading_data.quantity.mul(trading_data.price) * 0.0010887
    # trading_data.ix[trading_data["code"] == "511880.SH", "trading_fee"] *= 0.0000000 / 0.0010887
    # trading_data.ix[trading_data["code"] == "510500.SH", "trading_fee"] *= 0.000045 / 0.0010887
    trading_data["PNL"] = trading_data.S_DQ_CLOSE.sub(trading_data.price).mul(trading_data.quantity).mul(
        trading_data.direction).sub(trading_data.fee)
    # print(trading_data)
    # trading_data.to_csv("trading_data_debug.csv", encoding ="gbk")
    return trading_data.PNL.sum()


def side(s: str) -> int:
    if s == "卖出成交":
        return -1
    elif s == "买入成交":
        return 1
    else:
        return 0
        # print(s)
        # raise ValueError("Wrong data in column: direction")


# def calBalancePNL() -> float:
#     balance = pd.read_csv("302_balance.csv", encoding="gbk", converters={"证券代码": str})
#     balance = balance[["证券代码", "证券名称", "股份余额"]]
#     balance.columns = ["code", "stock_name", "balance"]
#     balance.code = ['0' * (6 - len(str(code))) + code for code in list(balance.code)]
#     balance.code = zcodeparse(list(balance.code), formatlike="600000.SH")
#     sql_1 = \
#         '''
#         SELECT
#         ''' + '''
#         S_INFO_WINDCODE,
#         TRADE_DT,
#         S_DQ_CLOSE
#     FROM
#         AShareEODPrices
#     WHERE
#         TRADE_DT = 20190724
#     '''
#     with OracleSql() as oracle:
#         P0_df = oracle.query(sql_1)
#     balance = pd.merge(balance, P0_df, left_on="code", right_on="S_INFO_WINDCODE")
#     sql_2 = \
#         '''
#         SELECT
#         ''' + '''
#         S_INFO_WINDCODE,
#         TRADE_DT,
#         S_DQ_CLOSE
#     FROM
#         AShareEODPrices
#     WHERE
#         TRADE_DT = 20190725
#     '''
#     with OracleSql() as oracle:
#         P1_df = oracle.query(sql_2)
#     balance = pd.merge(balance, P1_df, left_on="code", right_on="S_INFO_WINDCODE")
#     balance["PNL"] = balance.S_DQ_CLOSE_y.sub(balance.S_DQ_CLOSE_x).mul(balance.balance)
#     return balance.PNL.sum()


def calFutureTradingPNL() -> float:
    future_data = pd.read_csv("0726_future.csv", encoding="gbk")
    future_data = future_data[["证券代码", "证券名称", "成交数量", "成交均价", "最新价", "委托方向"]]
    future_data.columns = ["code", "contract_name", "quantity", "price", "new_price", "direction"]
    future_data["direction"] = future_data.direction.apply(futureSide)
    future_data["PNL"] = future_data.new_price.sub(future_data.price).mul(future_data.quantity) * 200
    return future_data.PNL.sum()


def futureSide(s: str) -> int:
    if s == "卖出平仓" or s == "卖出开仓":
        return -1
    elif s == "买入平仓" or s == "买入开仓":
        return 1
    else:
        raise ValueError("Wrong data in column: direction")


def calPNL() -> float:
    print("2019年7月26日交易盈亏（元）：")
    return calStockTradingPNL() + calFutureTradingPNL()


print(calPNL())
print(calStockTradingPNL())
# print(calFutureTradingPNL())

