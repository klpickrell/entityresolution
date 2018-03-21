#!/usr/bin/env cython

def jaccard_index( A, B ):
    # jaccard index is ( AnB ) / (AuB)
    return ( len(set(A).intersection( set(B) )) / max( 0.00000001, float(len(set(A).union(set(B)))) ) )
    

def main( argc, argv ):
    import argparse
    parser = argparse.ArgumentParser( description='simple jaccard index calculator', usage='jaccard.py <A> <B>' )
    parser.add_argument( "A", action='store' )
    parser.add_argument( "B", action='store' )

    args = parser.parse_args()
    print( "jaccard index: %f" % (jaccard_index(args.A,args.B)) )
    return 0


if __name__ == '__main__':
    import sys
    sys.exit( main( len(sys.argv), sys.argv ) )
