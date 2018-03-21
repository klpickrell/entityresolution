from learning.utils.json_util2 import get_value
from learning.utils.progress import console_progress
from string import maketrans

def write_report( output_file, clusters, records, header=None ):
    if not clusters or not records:
        print( 'no clusters or records' )
        return

    if not header:
        header = records[0].keys()
    total = len(clusters)
    last_marker = 0.0
    with open( output_file, 'wb' ) as csvfile:
        writer = csv.writer( csvfile, dialect='excel', quoting=csv.QUOTE_MINIMAL )
        head = ['cluster']
        head.extend(header)
        writer.writerow( head )

        for idx, cluster in enumerate(clusters):
            cluster_recs = [ records[i] for i in cluster ]
            for record in cluster_recs:
                row = [ idx ]
                record_data = [ get_value( record, key ).translate( maketrans('\r\n', '  ') ) for key in header ]
                row.extend(record_data)
                writer.writerow( row )

            writer.writerow([])

            (_, last_marker) = console_progress( idx+1, total, True, on_increment=0.01, next_marker=last_marker )
        console_progress( total, total, True )
        print( 'done! report written to {}'.format( output_file ) )


#simple clustering example
import cluster
import csv
stuff = [ item for item in csv.DictReader( open( '/var/tmp/target_jan_products.csv', 'rU' ), delimiter='^' ) ]
things = [ { k : ''.join(v.split()) for k,v in item.iteritems() if k == 'name' } for item in stuff ]
train_threshold = 600
run_threshold = 400
thresholds = [ train_threshold, run_threshold ]
all_clusters = cluster.cluster(things, thresholds)
write_report( '/var/tmp/prod_clusters.csv', all_clusters[run_threshold], stuff, header=None )

