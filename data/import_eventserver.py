"""
Import sample data for classification engine
"""

import predictionio
import argparse

def import_events(client, file):
  f = open(file, 'r')
  count = 0
  print "Importing data..."
  for line in f:
    data = line.rstrip('\r\n').split(",")
    plan = data[0]
    attr = data[1].split(" ")
    client.create_event(
      event="$set",
      entity_type="user",
      entity_id=str(count), # use the count num as user ID
      properties= {
        "attr0" : double(attr[0]),
        "attr1" : double(attr[1]),
        "attr2" : double(attr[2]),
        "attr3" : double(attr[3]),
        "attr4" : double(attr[4]),
        "attr5" : double(attr[5]),
        "attr6" : double(attr[6]),
        "attr7" : double(attr[7]),
        "attr8" : double(attr[8]),
        "attr9" : double(attr[9]),
        "attr10" : double(attr[10]),
        "attr11" : double(attr[11]),
        "attr12" : double(attr[12]),
        "attr13" : double(attr[13]),
        "attr14" : double(attr[14]),
        "attr15" : double(attr[15]),
        "attr16" : double(attr[16]),
        "attr17" : double(attr[17]),
        "attr18" : double(attr[18]),
        "attr19" : double(attr[19]),
        "attr20" : double(attr[20]),
        "attr21" : double(attr[21]),
        "attr22" : double(attr[22]),
        "attr23" : double(attr[23]),
        "attr24" : double(attr[24]),
        "plan" : double(plan)
      }
    )
    count += 1
  f.close()
  print "%s events are imported." % count

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description="Import sample data for classification engine")
  parser.add_argument('--access_key', default='invald_access_key')
  parser.add_argument('--url', default="http://localhost:7070")
  parser.add_argument('--file', default="./data/data.txt")

  args = parser.parse_args()
  print args

  client = predictionio.EventClient(
    access_key=args.access_key,
    url=args.url,
    threads=5,
    qsize=500)
  import_events(client, args.file)
