#!/usr/bin/env python
from unidecode import unidecode
from collections import OrderedDict

def get_value( json_doc, fieldname, default='' ):
    return stringify( get_raw_value( json_doc, fieldname, default ) )

def get_raw_value( json_doc, fieldname, default='' ):
    keys = fieldname.split('.')
    result = None
    last = json_doc
    for key in keys:
        if key.isdigit():
            if type(last) is list or type(last) is tuple:
                
                key_idx = int(key)
                if key_idx < len(last):
                    result = last[ key_idx ]
                else:
                    result = None
                    break

            elif getattr(last,'__getitem__',None) and last.has_key(key):
                result = last[ key ]
            else:
                result = None
                break
        elif getattr(last,'__getitem__',None):
            result = last.get( key, None )
            if result is None:
                break
        else:
            result = None
            break

        last = result

    if result is None and default is not None:
        return default
    else:
        return last

def stringify( value ):
    if value.__class__.__dict__.has_key('encode'):
        result = unidecode( value )
    else:
        result = str(value)
    return result.replace('\x00','')


def mongo_sanitize( json_doc ):

    if type(json_doc) is list:
        for i in xrange(len(json_doc)):
            mongo_sanitize(json_doc[i])
        return
    elif type(json_doc) is not dict:
        return

    for (key, value) in json_doc.items():
        if not key.__class__.__dict__.has_key( 'replace' ):
            continue

        if key == '_id':
            del json_doc['_id']
            continue

        new_key = key
        if '$' in key or '.' in key:
            new_key = key.replace( '$', '' ).replace( '.', '_' )
            del json_doc[key]
            json_doc[new_key] = value

        if type(value) is dict or type(value) is list:
            mongo_sanitize(json_doc[new_key])


def _main():
    import json
    import copy
    from pprint import pprint as pp
    json_doc = '''{
            "records" : [
            {
            "_id":  {
                "$oid":  "52543274da69274e87aacb6f"
            },
            "attributes":  {
                "claimed":  [
                    true
                ],
                "dead":  [
                    false
                ],
                "endDate":  {
                    "$date":  1305781200000
                },
                "market":  [
                    "NORTHERN CALIFORNIA BIG $"
                ],
                "marketCode":  [
                    "012"
                ],
                "paid":  [
                    false
                ],
                "services":  [
                    "Wedding Planners"
                ],
                "servicesCode":  [
                    "WCS"
                ],
                "startDate":  {
                    "$date":  1274158800000
                }
            },
            "description":  "At Jess Flood Event Design, we use a collaborative approach and a wealth of experience and creativity to help our clients create inspired events that unfold with ease and grace.",
            "features":  {
                "bing_listing":  {
                    "address":  "",
                    "city":  "",
                    "name":  "",
                    "phone":  "",
                    "site":  "bing",
                    "state":  "",
                    "zip":  ""
                },
                "citygrid_listing":  {
                    "address":  "",
                    "city":  "",
                    "email":  "",
                    "name":  "",
                    "phone":  "",
                    "state":  "",
                    "zip":  ""
                },
                "citysearch_listing":  {
                    "address":  "PO Box 530",
                    "city":  "Barrington",
                    "email":  "",
                    "name":  "Jim Livermore Designs Llc, .making.big.$",
                    "phone":  "(603) 742-5230",
                    "site":  "citysearch",
                    "state":  "NH",
                    "zip":  "03825"
                },
                "foursquare_listing":  {
                    "address":  "",
                    "city":  "",
                    "name":  "",
                    "phone":  "",
                    "site":  "foursquare",
                    "state":  "",
                    "zip":  ""
                },
                "local_listing":  {
                    "address":  "PO Box 729",
                    "city":  "Bodega Bay",
                    "email":  "",
                    "name":  "Cathy Beck Custom Designs",
                    "phone":  "7078759613",
                    "site":  "local",
                    "state":  "CA",
                    "zip":  "94923"
                },
                "mapquest_listing":  {
                    "address":  "",
                    "city":  "",
                    "name":  "",
                    "phone":  "",
                    "site":  "mapquest",
                    "state":  "",
                    "zip":  ""
                },
                "merchantcircle_listing":  {
                    "address":  "",
                    "city":  "",
                    "name":  "",
                    "phone":  "",
                    "site":  "merchantcircle",
                    "state":  "",
                    "zip":  ""
                },
                "scores":  {
                    "BinaryFeature":  {
                        "addressAbbreviation":  0,
                        "addressActual":  1,
                        "addressPopulated":  1,
                        "addressQualification":  0,
                        "ypZip1":  0.0,
                        "ypZip2":  0.0,
                        "ypZip4":  0.0,
                        "ypZip6":  0.0,
                        "ypZip8":  0.0
                    },
                    "DiscreteFeature":  {
                        "addressLength":  10.0,
                        "addressTokenCount":  3.0,
                        "cityLength":  11.0,
                        "cityTokenCount":  2.0,
                        "countryLength":  0.0,
                        "countryTokenCount":  0.0,
                        "postalCodeLength":  5.0,
                        "postalCodeTokenCount":  1.0,
                        "stateLength":  2.0,
                        "stateTokenCount":  1.0,
                        "urlPrefix1":  0.0,
                        "urlPrefix2":  1.0,
                        "urlsLength":  17.0,
                        "urlsTokenCount":  5.0
                    },
                    "RealValuedFeature":  {
                        "bingCity":  0,
                        "bingName":  0,
                        "bingPhone":  0,
                        "bingSearchName":  0.6,
                        "bingState":  0,
                        "bingStreet":  0,
                        "bingZip":  0,
                        "citygridCity":  0,
                        "citygridName":  0,
                        "citygridPhone":  0,
                        "citygridState":  0,
                        "citygridStreet":  0,
                        "citygridZip":  0,
                        "citysearchCity":  0.46060606060606063,
                        "citysearchName":  0.2927083015441895,
                        "citysearchPhone":  0.625,
                        "citysearchState":  0.0,
                        "citysearchStreet":  0.48,
                        "citysearchZip":  0.125,
                        "entropy":  4.101381529092886,
                        "foursquareCity":  0,
                        "foursquareName":  0,
                        "foursquarePhone":  0,
                        "foursquareState":  0,
                        "foursquareStreet":  0,
                        "foursquareZip":  0,
                        "localCity":  0.4212121212121212,
                        "localName":  0.2994791984558105,
                        "localPhone":  0.5555555555555556,
                        "localState":  1.0,
                        "localStreet":  0.46,
                        "localZip":  0.6,
                        "mapquestCity":  0,
                        "mapquestName":  0,
                        "mapquestPhone":  0,
                        "mapquestState":  0,
                        "mapquestStreet":  0,
                        "mapquestZip":  0,
                        "merchantcircleCity":  0,
                        "merchantcircleName":  0,
                        "merchantcirclePhone":  0,
                        "merchantcircleState":  0,
                        "merchantcircleStreet":  0,
                        "merchantcircleZip":  0,
                        "superpagesCity":  0,
                        "superpagesName":  0,
                        "superpagesPhone":  0,
                        "superpagesState":  0,
                        "superpagesStreet":  0,
                        "superpagesZip":  0,
                        "whitepagesCity":  0,
                        "whitepagesName":  0,
                        "whitepagesPhone":  0,
                        "whitepagesState":  0,
                        "whitepagesStreet":  0,
                        "whitepagesZip":  0,
                        "yahooCity":  0,
                        "yahooName":  0,
                        "yahooPhone":  0,
                        "yahooState":  0,
                        "yahooStreet":  0,
                        "yahooZip":  0,
                        "yelpCity":  0.5463869463869464,
                        "yelpEmail":  0,
                        "yelpName":  0.3543859481811523,
                        "yelpPhone":  0.3333333333333333,
                        "yelpState":  1.0,
                        "yelpStreet":  0.0,
                        "yelpZip":  0.8,
                        "ypCity":  0,
                        "ypEmail":  0,
                        "ypName":  0.47249999046325686,
                        "ypPhone":  0.375,
                        "ypState":  0,
                        "ypStreet":  0,
                        "ypZip":  0
                    }
                },
                "superpages_listing":  {
                    "address":  "",
                    "city":  "",
                    "name":  "",
                    "phone":  "",
                    "site":  "superpages",
                    "state":  "",
                    "zip":  ""
                },
                "whitepages_listing":  {
                    "address":  "",
                    "city":  "",
                    "name":  "",
                    "phone":  "",
                    "site":  "whitepages",
                    "state":  "",
                    "zip":  ""
                },
                "yahoo_listing":  {
                    "address":  "",
                    "city":  "",
                    "name":  "",
                    "phone":  "",
                    "site":  "yahoo",
                    "state":  "",
                    "zip":  ""
                },
                "yellow_pages_listing":  {
                    "address":  "",
                    "city":  "",
                    "email":  "",
                    "name":  "Service Co-Emergency Basement Flood Water Damage Clean Up",
                    "phone":  "(800) 704-7380",
                    "state":  "",
                    "zip":  ""
                },
                "yelp_listing":  {
                    "address":  "West Portal",
                    "city":  "San Francisco",
                    "email":  "",
                    "name":  "Dreams On A Dime Events & Weddings",
                    "phone":  "4152642764",
                    "state":  "CA",
                    "zip":  "94127"
                }
            },
            "locations":  [
                {
                    "address1":  "po box 309",
                    "city":  "valley ford",
                    "countryCode":  "",
                    "email":  "",
                    "fax":  "",
                    "phone":  "707 592 7973",
                    "postalCode":  "94972",
                    "state":  "ca"
                }
            ],
            "name":  "Jess Flood Event Design",
            "source":  {
                "applicationId":  10,
                "applicationName":  "Local",
                "id":  "3c16a454-5cdb-4682-a305-cafe881067ec",
                "objectType":  "InternetProfile",
                "urls":  [
                    "http://local.weddingchannel.com/Wedding-Vendors/Jess-Flood-Event-Design-profile?ProfileId=332525"
                ]
            },
            "timestamp":  1375247954047,
            "website":  "www.jessflood.com"
            }
       ],
            "selections" : 
            {
                "locations.0.city":  "9b08ebdb8058ec94",
                "locations.0.countryCode":  "9b08ebdb8058ec94",
                "locations.0.phone":  "9b08ebdb8058ec94",
                "website":  "9b08ebdb8058ec94"
            },
            "$other.key" : {
                       "$something" : [ "$howdy", ".bro" ]
            }
      }'''

    record = json.loads(json_doc)
    original = copy.deepcopy( record )
    mongo_sanitize( record )

    print( "original:" )
    pp( original )
    print( "sanitized:" )
    pp( record )

    json_doc2 = '''{ "records": [ {
                "_id":  {
                    "$oid":  "52543431da69274e87ad5863"
                },
                "attributes":  {
                    "category_ids":  [
                        144.0
                    ],
                    "category_labels":  [
                        "Retail",
                        "Fashion",
                        "Jewelry and Watches"
                    ],
                    "status":  1
                },
                "locations":  [
                    {
                        "address1":  "534 central st \\"# b",
                        "city":  "hudson",
                        "countryCode":  "usa",
                        "email":  "",
                        "fax":  "",
                        "phone":  "(828) 726-1009",
                        "postalCode":  "28638",
                        "state":  "nc"
                    }
                ],
                "name":  "Gold Mine Fine Jewelry and Gifts",
                "source":  {
                    "applicationId":  100,
                    "applicationName":  "Factual",
                    "id":  "4fc4071d-966d-4b05-b059-54b295c4f28a",
                    "objectType":  "place"
                },
                "timestamp":  1373605200000,
                "website":  "http://www.hudsongoldmine.com"
            },
            {
                "_id":  {
                    "$oid":  "5254343cda69274e87ad658e"
                },
                "attributes":  {
                    "category_ids":  [
                        144.0
                    ],
                    "category_labels":  [
                        "Retail",
                        "Fashion",
                        "Jewelry and Watches"
                    ],
                    "status":  1
                },
                "locations":  [
                    {
                        "address1":  "545 main st",
                        "city":  "hudson",
                        "countryCode":  "usa",
                        "email":  "",
                        "fax":  "(828) 726-8803",
                        "phone":  "(828) 726-1009",
                        "postalCode":  "28638",
                        "state":  "nc"
                    }
                ],
                "name":  "Gold Mine Fine Jewelry and Gifts",
                "source":  {
                    "applicationId":  100,
                    "applicationName":  "Factual",
                    "id":  "6f7531aa-166a-4a18-9c3e-b04e948d582b",
                    "objectType":  "place"
                },
                "timestamp":  1373605200000,
                "website":  "http://www.the-goldmine.com"
            }],
            "selections" : 
            {
                "locations.0.city":  "9b08ebdb8058ec94",
                "locations.0.countryCode":  "9b08ebdb8058ec94",
                "locations.0.phone":  "9b08ebdb8058ec94",
                "website":  "9b08ebdb8058ec94"
            }
          }'''

    record = json.loads(json_doc2)
    original = copy.deepcopy( record )
    mongo_sanitize( record )

    print( "original:" )
    pp( original )
    print( "sanitized:" )
    pp( record )

#    with open( '/tmp/document.pre.json' ) as fil:
#        json_doc3 = fil.read()
#        record = json.loads(json_doc3)
#        original = copy.deepcopy( record )
#        mongo_sanitize( record )

#    print( "original:" )
#    pp( original )
#    print( "sanitized:" )
#    pp( record )

    rec = {}
    rec['quality_score'] = {u'website': {u'score': 0.0}, u'locations_0_city': {u'score': 5.0}, u'name': {u'score': 5.0}, u'locations_0_countryCode': {u'score': 5.0}, u'locations_0_phone': {u'score': 4.5}, u'locations_0_fax': {u'score': 0.0}, u'locations_0_postalCode': {u'score': 5.0}, u'locations_0_address1': {u'score': 1.0}, u'locations_0_state': {u'score': 5.0}, u'locations_0_email': {u'score': 0.0}}
    result = get_value( rec, 'quality_score.website.score' )

    return 0

if __name__ == "__main__":
    import sys
    sys.exit( _main() )
