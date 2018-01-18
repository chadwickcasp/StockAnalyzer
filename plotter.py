from Robinhood import Robinhood
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import pandas as pd

def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')

t = Robinhood()
t.login(username="chadwickcasp", password="Orange$7179")
# print t.investment_profile()
# print t.get_historical_quotes("MBLY", "5minute", "week", "regular")
symbol = raw_input('Symbol of interest: ')

week_vals = t.get_historical_quotes(symbol, "5minute", "week",
    "regular")['results'][0]['historicals']

_week_vals = t.get_historical_quotes(symbol, "5minute", "week",
    "regular")
print(_week_vals)

# print week_vals

five_min_interval_data = []
days_data = []
time_data = []
times = []

for i in range(len(week_vals)):
    # print week_vals[i]
    val = week_vals[i]['open_price']
    time = week_vals[i]['begins_at']

    times.append(time)
    five_min_interval_data.append(val)
    days_data.append('{:.10}'.format(time))
    time_data.append(time[-9:-1])

df = pd.DataFrame(data=np.array([days_data, time_data, five_min_interval_data]).T,
                  columns=['Date', 'Time', 'Five Minute Interval Data'])

fig, ax = plt.subplots()
datefmt = '%Y-%m-%dT%H:%M:%SZ'
dt_x = [dt.datetime.strptime(t, datefmt) for t in times]
print(dt_x[0].tzinfo)
print_full(dt_x)
x = dt_x
x = [mdates.date2num(i) for i in dt_x]
formatter = mdates.DateFormatter('%m/%d/%y %H:%M')
ax.xaxis.set_major_formatter(formatter)
plt.scatter(x, five_min_interval_data, marker='>')
fig.autofmt_xdate()
# plt.show()
print_full(df)
mov_avg_data = np.round(df['Five Minute Interval Data'].rolling(window = 12, center = False).mean(), 2)
# mov_avg_data = pd.rolling_mean(df['Five Minute Interval Data'], 12)
print mov_avg_data
plt.scatter(x, mov_avg_data, marker=(5,2))

fig2, ax2 = plt.subplots()
datefmt = '%Y-%m-%dT%H:%M:%SZ'
dt_x = [dt.datetime.strptime(t, datefmt) for t in times]
print(dt_x[0].tzinfo)
print_full(dt_x)
x = dt_x
x = [mdates.date2num(i) for i in dt_x]
formatter = mdates.DateFormatter('%m/%d/%y %H:%M')
ax.xaxis.set_major_formatter(formatter)
plt.plot(five_min_interval_data, marker='>')
fig.autofmt_xdate()
# plt.show()
print_full(df)
mov_avg_data = np.round(df['Five Minute Interval Data'].rolling(window = 12, center = False).mean(), 2)
# mov_avg_data = pd.rolling_mean(df['Five Minute Interval Data'], 12)
print mov_avg_data
plt.plot(mov_avg_data, marker=(5,2))

plt.show()

