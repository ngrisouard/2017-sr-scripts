import pickle
import getpass
# import os

if getpass.getuser() == 'NicoG':
    base = '/Users/NicoG/FS_in-house/outputs/'
elif getpass.getuser() == 'nicolas':
    base = '/data/nicolas/mbf/'
elif getpass.getuser() == 'mbfox':
    base = '/data/mbfox/'
else:
    raise NameError(
        'This is a new user or computer: what is the base directory?')

for ii in ['01']:  # 13:16: # experiment index
    for iN in ['68']:
        for iF in ['10', '12', '14']:
            for iR in ['10', '12', '14']:

                setup = 'N' + iN + 'F' + iF + 'r' + iR + '-' + ii + '/'

                root = base + setup

                pkl_file = open('{}spec_params.pkl'.format(root), 'rb')
                pp = pickle.load(pkl_file)
                # um = pickle.load(pkl_file)
                pkl_file.close()
