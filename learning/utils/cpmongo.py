#!/usr/bin/env python
'''
cpmongo.py:  copy a mongo collection from one db on one host to another
'''
import argparse
import pymongo
from progress import console_progress

class ParseException(Exception):
    '''
    ParseException
    '''
    def __init__( self, msg ):
        Exception.__init__(self, msg)

def parse_mongo_spec( path_desc ):
    '''
    split a mongo collection path description into a 4-tuple (host,port,db,collection)
    '''
    valid_host = r'^((?=.{1,255}$)[0-9A-Za-z](?:(?:[0-9A-Za-z]|-){0,61}[0-9A-Za-z])?(?:\.[0-9A-Za-z](?:(?:[0-9A-Za-z]|-){0,61}[0-9A-Za-z])?)*\.?)?'
    import re
    pattern = re.compile( valid_host + r'(:[0-9]+)?/(\w+)\.(\w+)' )
    match = pattern.match( path_desc )
    if not match:
        raise ParseException( "Invalid mongo collection path specifier" )
    group = match.groups()
    host = 'localhost'
    port = 27017
    if group[0]:
        host = group[0]
    if group[1]:
        port = int(group[1].lstrip(':'))

    return ( host, port, group[2], group[3] )

def _main():
    '''
    the main
    '''

    parser = argparse.ArgumentParser( description='cpmongo:  copy mongo collection from one host to another',
                                      usage='./cpmongo.py "<source>:<port>/<database>.<collection>" "<destination>:<port>/<database>.<collection>"\n\t\t(host:port default to localhost:27017)' )
    parser.add_argument( 'source' )
    parser.add_argument( 'destination' )

    args = parser.parse_args()

    try:
        (host1, port1, db1, collection1) = parse_mongo_spec( args.source )
        (host2, port2, db2, collection2) = parse_mongo_spec( args.destination )
    except ParseException, ex:
        print( 'ParseException: %s' % ex )
        parser.print_usage()
        return 1

    connection1 = pymongo.MongoClient( host1, port1 )
    connection2 = pymongo.MongoClient( host2, port2 )

    total = connection1[db1][collection1].count()
    if total == 0:
        print( "warning: %s.%s is empty in source host, did nothing" % (db1, collection1) )
        return 0

    print( "copying %s:%d/%s.%s -> %s:%d/%s.%s" % (host1, port1, db1, collection1, host2, port2, db2, collection2) )
    for (idx, cursor) in enumerate( connection1[db1][collection1].find() ):
        console_progress( idx+1, total )
        connection2[db2][collection2].insert(cursor)

    print( "\nComplete!" )

    connection1.fsync()
    connection2.fsync()

    return 0



if __name__ == '__main__':
    import sys
    sys.exit( _main() )
