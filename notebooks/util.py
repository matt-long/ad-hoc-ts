import os
from glob import glob
from itertools import product

import yaml
import json
import intake

import pandas as pd

import dask
from dask_jobqueue import PBSCluster
from dask.distributed import Client

PBS_PROJECT = 'NCGD0011'
cache_catalog_dir = './assets'

cache_format = 'zarr'

def get_ClusterClient(memory='25GB'):
    """get cluster and client"""
    USER = os.environ['USER']    
    cluster = PBSCluster(
        cores=1,
        memory=memory,
        processes=1,
        queue='casper',
        local_directory=f'/glade/scratch/{USER}/dask-workers',
        log_directory=f'/glade/scratch/{USER}/dask-workers',
        resource_spec='select=1:ncpus=1:mem=25GB',
        project=PBS_PROJECT,
        walltime='06:00:00',
        interface='ib0',
    )

    dask.config.set({
        'distributed.dashboard.link':
        'https://jupyterhub.hpc.ucar.edu/stable/user/{USER}/proxy/{port}/status'
    })
    client = Client(cluster)
    return cluster, client


def intake_esm_get_keys_info(cat):
    """return a dictionary with the values of components of the keys in an
    intake catalog

    Example:
      key_info = {
        'experiment': '20C',
        'component': 'ocn',
        'stream': 'pop.h',
        'member_id': 1,
      }
    """

    # generate a list of lists with all possible values of each groupby_attr
    iterables = [cat.unique(columns=key)[key]['values'] for key in cat.groupby_attrs]

    # generate a dictionary of keys with the values of its attributes
    key_info = {}
    for values in product(*iterables):
        key = cat.sep.join([str(v) for v in values])
        if key in cat.keys():
            key_info[key] = {k: values[i] for i, k in enumerate(cat.groupby_attrs)}
    return key_info


def to_intake_esm(agg_member_id=False):
    """generate an intake-esm data catalog from funnel collections"""

    catalog_csv_file = f'{cache_catalog_dir}/collection-summary.csv.gz'
    catalog_json_file = f'{cache_catalog_dir}/collection-summary-agg_member_id-{agg_member_id}.json'

    files = sorted(glob(f'{cache_catalog_dir}/*.yml'))
    data = {}
    for f in files:
        with open(f) as fid:
            data[f] = yaml.safe_load(fid)

    # assume that there is a *single* esm_collection
    # this could be extended, but supporting multiple collections
    # raises all sorts of questions about validation
    esm_collection = [v['catalog_file'] for v in data.values()]
    assert len(set(esm_collection)) == 1
    catalog = intake.open_esm_datastore(esm_collection[0])

    # generate a dictionary of the key info dictionaries for each catalog
    groupby_attrs_values = intake_esm_get_keys_info(catalog)
    first_key = list(groupby_attrs_values.keys())[0]
    #columns = list(groupby_attrs_values[first_key].keys()) + ['variable', 'name', 'path']

    lines = []
    for f in files:
        key = data[f]['key']
        column_data = dict(**groupby_attrs_values[key])
        column_data.update(
            {k: v for k, v in data[f].items() if k not in column_data}
        )        
        lines.append(column_data)

    df = pd.DataFrame(lines)
    #assert set(df.columns) == set(columns), 'mismatch in expected columns'
    columns = df.columns
    
    # modify the json
    with open(esm_collection[0]) as fid:
        catalog_def = json.load(fid)

    catalog_def['catalog_file'] = catalog_csv_file
    catalog_def['attributes'] = [{'column_name': k, 'vocabulary': ''} for k in columns]
    catalog_def['assets'] = {'column_name': 'path', 'format': cache_format}

    # ensure that all `groupby_attrs` are in the columns of the DataFrame
    assert all(
        [c in columns for c in catalog_def['aggregation_control']['groupby_attrs']]
    ), 'not all groupby attrs found in columns'

    # add `name` to `groupby_attrs`
    catalog_def['aggregation_control']['groupby_attrs'] += ['name']

    # filter the aggregations rules to ensure only existing columns are included
    catalog_def['aggregation_control']['aggregations'] = [
        d
        for d in catalog_def['aggregation_control']['aggregations']
        if d['attribute_name'] in columns
    ]

    if agg_member_id and 'member_id' in catalog_def['aggregation_control']['groupby_attrs']:
        groupby_attrs = catalog_def['aggregation_control']['groupby_attrs']
        groupby_attrs = [v for v in groupby_attrs if v not in ['member_id']]

        catalog_def['aggregation_control']['groupby_attrs'] = groupby_attrs

        catalog_def['aggregation_control']['aggregations'].append(
            dict(type='join_new', attribute_name='member_id')
        )

    # persist
    df.to_csv(catalog_csv_file, index=False)

    with open(catalog_json_file, 'w') as fid:
        json.dump(catalog_def, fid)

    return intake.open_esm_datastore(catalog_json_file)
