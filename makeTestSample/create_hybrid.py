"""Make a hybrid of two files"""

from argparse import ArgumentParser
import h5py
import numpy as np
import sys
import math
import os

_njet_help = 'total number of jets in output'
_ttbar_help = 'space-separated list of ttbar files (can use shell pattern \
               matching)'
_Zprime_help = 'space-separated list of Z\' files (can use shell pattern \
                matching)'
_ttbarcomp_help = 'output composition fraction for ttbar jets, from 0 to 1'
_ptcut_help = 'pT cut for hybrid creation. Default 250 GeV.'
dataset_name = 'jets'




def get_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('-t', '--ttbar', nargs='+', help=_ttbar_help,
                        required=True)
    parser.add_argument('-Z', '--Zprime', nargs='+', help=_Zprime_help,
                        required=True)
    parser.add_argument('-o', '--output-file', default='hybrid.h5')
    parser.add_argument('-n', '--n-jets', default=1000, help=_njet_help,
                        type=int)
    parser.add_argument('-c', '--ttbarcomp', default=0.5, type=float,
                        help=_ttbarcomp_help)
    parser.add_argument('-p', '--ptcut', default=250000, type=float,
                        help=_ptcut_help)
    parser.add_argument('--no_cut', action='store_true')
    parser.add_argument('--write_tracks', action='store_true')
    split = parser.add_mutually_exclusive_group()
    split.add_argument('--even', action='store_true')
    split.add_argument('--odd', action='store_true')
    category = parser.add_mutually_exclusive_group()
    category.add_argument('--bjets', action='store_true')
    category.add_argument('--cjets', action='store_true')
    category.add_argument('--ujets', action='store_true')
    return parser.parse_args()


def get_jets(filename, sample_type, n_jets, eventNumber_parity=None):
    args = get_args()
    b_pdgid = 5
    pt_cut = args.ptcut

    print('Opening file', filename)
    data_set = h5py.File(filename, 'r')
    jets = data_set[dataset_name]
    if args.write_tracks:
        tracks = data_set['tracks']
    print('Total number of jets in file:', jets.size)

    if eventNumber_parity == 'even':
        parity_rejection = (jets['eventNumber'] % 2) == 1
    elif eventNumber_parity == 'odd':
        parity_rejection = (jets['eventNumber'] % 2) == 0
    elif eventNumber_parity is None:
        parity_rejection = False
    else:
        print("Unknown parity option:", eventNumber_parity)
        sys.exit(1)

    if args.bjets is True:
        category_rejection = jets['HadronConeExclTruthLabelID'] != b_pdgid
    elif args.cjets is True:
        category_rejection = jets['HadronConeExclTruthLabelID'] != 4
    elif args.ujets is True:
        category_rejection = jets['HadronConeExclTruthLabelID'] != 0
    else:
        category_rejection = False

    if (sample_type == 'ttbar'):
        if args.no_cut:
            indices_to_remove = np.where(parity_rejection |
                                         category_rejection)[0]
        else:
            indices_to_remove = np.where(
                parity_rejection |
                category_rejection |
                ((abs(jets['HadronConeExclTruthLabelID']) == b_pdgid) &
                 (jets['GhostBHadronsFinalPt'] > pt_cut)) |
                ((abs(jets['HadronConeExclTruthLabelID']) < b_pdgid) &
                 (jets['pt_uncalib'] > pt_cut))
            )[0]
    elif (sample_type == 'Zprime'):
        if args.no_cut:
            indices_to_remove = np.where(parity_rejection |
                                         category_rejection)[0]
        else:
            indices_to_remove = np.where(
                parity_rejection |
                category_rejection |
                ((abs(jets['HadronConeExclTruthLabelID']) == b_pdgid) &
                 (jets['GhostBHadronsFinalPt'] < pt_cut)) |
                ((abs(jets['HadronConeExclTruthLabelID']) < b_pdgid) &
                 (jets['pt_uncalib'] < pt_cut))
            )[0]
    else:
        print("Unknown sample type:", sample_type)
        sys.exit(1)
    del parity_rejection
    jets = np.delete(jets, indices_to_remove)[:n_jets]
    jets = jets[:n_jets]
    if args.write_tracks:
        tracks = np.delete(tracks, indices_to_remove, axis=0)[:n_jets]
        tracks = tracks[:n_jets]
        return jets, tracks
    else:
        return jets, None


def run():
    args = get_args()
    if args.even:
        index_parity = 'even'
    elif args.odd:
        index_parity = 'odd'
    else:
        index_parity = None

    if args.ttbarcomp < 0 or args.ttbarcomp > 1:
        print("Invalid ttbar composition fraction", args.ttbarcomp)
        sys.exit(1)

    output_n_ttbar = int(math.ceil(args.n_jets * args.ttbarcomp))
    ttbar_jets = None
    print("Loading ttbar files")
    for ttbar_filename in args.ttbar:
        if ttbar_jets is None:
            ttbar_jets, ttbar_tracks = get_jets(ttbar_filename, 'ttbar',
                                                output_n_ttbar, index_parity)
        else:
            jets, tracks = get_jets(ttbar_filename, 'ttbar', n_ttbar_to_get,
                                    index_parity)
            ttbar_jets = np.concatenate([ttbar_jets, jets])
            if tracks is not None:
                ttbar_tracks = np.concatenate([ttbar_tracks, tracks])
        print(ttbar_jets.size, "selected ttbar jets loaded so far")
        n_ttbar_to_get = output_n_ttbar - ttbar_jets.size
        print(n_ttbar_to_get)
        if n_ttbar_to_get <= 0:
            break
        print("Need", n_ttbar_to_get, "more ttbar jets")
    if n_ttbar_to_get > 0:
        print("Not enough selected jets from ttbar files, only",
              ttbar_jets.size)
        sys.exit(1)

    output_n_Zprime = args.n_jets - output_n_ttbar
    Zprime_jets = None
    print("Loading Z' files")
    for Zprime_filename in args.Zprime:
        if Zprime_jets is None:
            Zprime_jets, Zprime_tracks = get_jets(Zprime_filename, 'Zprime',
                                                  output_n_Zprime,
                                                  index_parity)
        else:
            jets, tracks = get_jets(Zprime_filename, 'Zprime', n_Zprime_to_get,
                                    index_parity)
            Zprime_jets = np.concatenate([Zprime_jets, jets])
            if tracks is not None:
                Zprime_tracks = np.concatenate([Zprime_tracks, tracks])
        print(Zprime_jets.size, "selected Z' jets loaded so far")
        n_Zprime_to_get = output_n_Zprime - Zprime_jets.size
        if n_Zprime_to_get <= 0:
            break
        print("Need", n_Zprime_to_get, "more Z' jets")
    if n_Zprime_to_get > 0:
        print("Not enough selected jets from Z' files, only", Zprime_jets.size)
        sys.exit(1)

    print("Concatenating ttbar and Z' arrays")
    cat = np.concatenate([ttbar_jets, Zprime_jets])
    if args.write_tracks:
        cat_tracks = np.concatenate([ttbar_tracks, Zprime_tracks])
    del ttbar_jets
    del Zprime_jets
    del ttbar_tracks
    del Zprime_tracks
    print("Shuffling array")
    rng_state = np.random.get_state()
    np.random.shuffle(cat)
    if args.write_tracks:
        np.random.set_state(rng_state)
        np.random.shuffle(cat_tracks)
    print("Writing output file")
    with h5py.File(args.output_file, 'w') as out_file:
        out_file.create_dataset(dataset_name, data=cat, compression='gzip')
        if args.write_tracks:
            out_file.create_dataset('tracks', data=cat_tracks,
                                    compression='gzip')


if __name__ == '__main__':
    run()
