############################################################################
# Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    #
#                                                                          #
# Distributed under the terms of the BSD 3-Clause License.                 #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

import random
import sys
import graphitesend
from time import sleep

import pandas as pd
from datetime import datetime

import socket
hostname = socket.gethostname()


SERVER = "78.47.109.88"
LOCAL = '172.17.0.2'
g = graphitesend.init(graphite_server=SERVER, prefix=hostname, system_name='')

print("Upload results? [Y/N]")
yes = {'yes','y', 'ye', ''}
no = {'no','n'}

choice = input().lower()
if choice not in yes:
    sys.exit(0)

with open("./results.csv") as fi:
  for idx, line in enumerate(fi):
    if line.startswith('name,iterations,real_time'):
      skip_index = idx
      break
    elif line.startswith("20"):
      benchdate = datetime.strptime(line.strip(), '%Y-%m-%d %H:%M:%S')

print("The date is: ", benchdate)

df = pd.read_csv("./results.csv", skiprows=skip_index)

print("The results are: ")
print(df[['name', 'cpu_time']])
print("\n\n")

print("╒══════════════════════════════════════════════════════════════════════╕")
print("│░░░░{:^62s}░░░░│".format("Beginning Upload for: " + hostname))
print("╘══════════════════════════════════════════════════════════════════════╛\n")

for index, row in df.iterrows():
    # g.send(row['name'], row.cpu_time)
    name = row['name']
    idx = name.find('_')
    if idx:
      name[idx] = '.'
    g.send(row['name'], row['cpu_time'], timestamp=benchdate.timestamp())
    print("Uploading: {}, {}, timestamp: {}".format(row['name'], row.cpu_time, benchdate.timestamp()))
