from .imports_and_configs import *

with open('db_config.json') as f:
        db_config = json.load(f)
teamcenter = db_config['teamcenter']
teamcenter_url = db_config['teamcenter_url']

# configure TeamCenter connection
es = Elasticsearch(teamcenter)


def tc_global_search(query):
    """
    Get dictionary of a Team Center global search
    :param query: str
    :return: dictionary
    """
    results = es.search(index='teamcenter', body={
        'query': {
            'query_string': {
                'query': query
            }
        }
    })
    print(str(results['hits']['total']) + ' results found')
    return results


def tc_part_search(query):
    """
    Get dictionary of a Team Center part search
    :param query: str
    :return: dictionary
    """
    results = es.search(index='teamcenter', doc_type='parts', body={
        'query': {
            'query_string': {
                'query': query
            }
        }
    })
    print(str(results['hits']['total']) + ' results found')
    return results


def get_tc_schema():
    """
    Get the Team Center schema as a json object
    :return: json object
    """
    results_bytes = requests.get(teamcenter_url).content
    results_string = str(results_bytes, encoding='UTF-8')
    parsed = json.loads(results_string)
    return json.dumps(parsed, indent=4, sort_keys=True)
