# Motivating example
Consider a csv file of various duplicated products with distinct identifiers only distinguishable as a duplicate by name.  We would like to group products by some measure of similarity which defines particular similarities between strings as referring to the same entity.  This is what this library does:

```import cluster
import csv
products = [ item for item in csv.DictReader( open( '/var/tmp/products.csv', 'rU' ), delimiter='^' ) ]
things = [ { k : ''.join(v.split()) for k,v in item.iteritems() if k == 'name' } for item in products ]
train_threshold = 600
run_threshold = 400
thresholds = [ train_threshold, run_threshold ]
all_clusters = cluster.cluster(things, thresholds)
write_report( '/var/tmp/product_clusters.csv', all_clusters[run_threshold], products, header=None )```

