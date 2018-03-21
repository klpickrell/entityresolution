#!/usr/bin/env python
'''progress
--------------
Print various progress bars:

console_progress( count, maximum ):
      - used to print progress bar to console  e.g. progress.console_progress( 30, 100 ) prints 30%,
        while progress.console_progress( 30, 200 ) prints 15%
'''
from __future__ import division
import time

def console_progress( count, maximum, print_count=False, on_increment=None, next_marker=0.0 ):
    import sys
    if maximum <= 0:
        return (True, next_marker)

    on_increment = max(on_increment,0.01)
    current_ratio = float(count)/maximum
    if on_increment is None:
        strval = "\r%s%3d%%" % (('='*int(current_ratio*100))+('-'*int(100-(current_ratio*100))),current_ratio*100)
        if print_count:
            strval += " %d/%d" % (count,maximum)
        sys.stdout.write( strval )
        sys.stdout.flush()
    else:
        if current_ratio > next_marker:
            last_percentage = (int((current_ratio*100)/(on_increment*100))*int(on_increment*100))
            next_percentage = last_percentage/100.0 + on_increment
            strval = "%3d%%" % last_percentage
            if print_count:
                strval += " %d/%d" % (count,maximum)
            sys.stdout.write( strval+'\n' )
            sys.stdout.flush()
            next_marker = next_percentage

    return (count >= maximum, next_marker)

def _main( argv, argc ):
    import time
    print( 'style 1' )
    for i in range(100+1):
        time.sleep( 0.05 )
        if console_progress( i, 100 )[0]:
            print( "\nComplete!" )

    print( 'style 2' )
    for i in range(100+1):
        time.sleep( 0.05 )
        if console_progress( i, 100, print_count=True )[0]:
            print( "\nComplete!" )

    print( 'style 3' )
    last_marker = 0.0
    for i in range(100+1):
        time.sleep( 0.05 )
        (result, marker) = console_progress( i, 100, on_increment=0.1, next_marker=last_marker )
        if result:
            print( "\nComplete!" )
        else:
            last_marker = marker

    print( 'style 4' )
    last_marker = 0.0
    for i in range(100+1):
        time.sleep( 0.05 )
        (result, marker) = console_progress( i, 100, on_increment=0.1, next_marker=last_marker, print_count=True )
        if result:
            print( "\nComplete!" )
        else:
            last_marker = marker

if __name__ == '__main__':
    import sys
    sys.exit( _main( sys.argv, len(sys.argv) ) )
