import asyncio
import websockets
import json
import pandas as pd
import datetime as dt
import pprint


async def call_api(msg):
   async with websockets.connect('wss://test.deribit.com/ws/api/v2') as websocket:
       await websocket.send(msg)
       while websocket.open:
           response = await websocket.recv()
           json_par = json.loads(response)
           pprint.pprint(json_par)
           return response


def async_loop(api, message):
    return asyncio.get_event_loop().run_until_complete(api(message))


def retrieve_historic_data(start, end, instrument, timeframe):
    msg = \
        {
            "jsonrpc": "2.0",
            "id": 833,
            "method": "public/get_tradingview_chart_data",
            "params": {
                "instrument_name": instrument,
                "start_timestamp": start,
                "end_timestamp": end,
                "resolution": timeframe
            }
        }
    # msg = {
    #     "jsonrpc" : "2.0",
    #     "id" : 8106,
    #     "method" : "public/ticker",
    #     "params" : {
    #         "instrument_name" : instrument,
    #         "start_timestamp": start,
    #         "end_timestamp": end,
    #         "resolution": timeframe
    #     }
    # }
    
    resp = async_loop(call_api, json.dumps(msg))

    return resp


def json_to_dataframe(json_resp):
    res = json.loads(json_resp)
    df = pd.DataFrame(res['result'])
    df['ticks'] = df.ticks / 1000
    df['timestamp'] = [dt.datetime.utcfromtimestamp(date) for date in df.ticks]

    return df

def get_data(date1, instrument, tf='1'):
    n_days = (dt.datetime.now() - date1).days
    df_master = pd.DataFrame()

    d1 = date1
    for _ in range(n_days):
        d2 = d1 + dt.timedelta(days=1)

        t1 = dt.datetime.timestamp(d1)*1000
        t2 = dt.datetime.timestamp(d2)*1000

        json_resp = retrieve_historic_data(t1, t2, instrument, tf)

        temp_df = json_to_dataframe(json_resp)
        print(len(temp_df));

        df_master = pd.concat([df_master, temp_df], axis=0)

        print(f'collected data for dates: {d1.isoformat()} to {d2.isoformat()}')
        print(f'length of df_master: {len(df_master)}')

        d1 = d2

    return df_master

if __name__ == '__main__':
    #change this to two years prior to the day you are using this script
    start = dt.datetime(2019, 3, 13, 0, 0)
    instrument = "ETH-PERPETUAL"
    tf = "60"

    df_master = get_data(start, instrument, tf)
    #save the file to your data folder, I named mine btc_master.csv
    #df_master.to_csv('data/btc_master.csv')
    df_master.to_csv('eth-perp_final.csv')