import numpy as np
import yt
import glob as glob
import sys,os
import time

yt.enable_parallelism()
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank
nprocs = comm.size

def make_pfs_allsnaps(folder,folder2, codetp, base_manual = None, checkredshift = True,index=0):
    """
    This function creates the file that contains the directory to all the simulation snapshots (sorted by redshift).


    Parameters
    ----------
    folder : str
        The directory to the folder containing the simulation snapshots.
    codetp : str
        The code type of the simulation (e.g. 'ENZO', 'GADGET3', 'AREPO', 'RAMSES', 'GIZMO', 'GEAR', 'ART', 'CHANGA').
        If the snapshot base is not in the specified format, set codetp to 'manual'
    base_manual : str, optional
        The base of the snapshot file name. The default is None. Only set if code type is 'manual'.
    checkredshift : bool, optional
        If True, the function will load the snapshhots to check the redshifts and sort the snapshots accordingly.
        If False, the function will use the index of the snapshot file name to sort the snapshots. The default is False.

    Returns
    -------
    None.

    """

    #Obtaining the directory to all the snapshot configuration files
    snapshot_files = []

    if codetp == 'ENZO' or codetp == 'AGORA_ENZO':
        bases = [["DD", "output_"],
                 ["DD", "output"],
                 ["DD", "data"],
                 ["DD", "DD"],
                 ["RD", "RedshiftOutput"],
                 ["RD", "RD"],
                 ["RS", "restart"]]
        for b in bases:
            snapshot_files += glob.glob("%s/%s????/%s????" % (folder,b[0],b[1]))
    elif codetp == 'GADGET3' or codetp == 'GADGET4':
        bases = [["snapshot_", "snapshot_"],
                 ["snapdir_", "snapshot_"]]
        for b in bases:
            snapshot_files += glob.glob("%s/%s???/%s???.0.hdf5" % (folder,b[0], b[1]))
    elif codetp == 'AREPO':
        bases = ["snap_", "snapshot_"]
        for b in bases:
            snapshot_files += glob.glob("%s/%s???.hdf5" % (folder,b))
    elif codetp == 'GIZMO':
        bases = "snapshot_"
        snapshot_files += glob.glob("%s/%s???.hdf5" % (folder,bases))
    elif codetp == 'GEAR':
        bases = "snapshot_"
        snapshot_files += glob.glob("%s/%s????.hdf5" % (folder,bases))
    elif codetp == 'RAMSES':
        bases = ["output_", "info_"]
        snapshot_files += glob.glob("%s/%s?????/%s?????.txt" % (folder,bases[0], bases[1]))
    elif codetp == 'ART':
        bases = '10MpcBox_csf512_'
        snapshot_files += glob.glob('%s/%s?????.d' % (folder,bases))
    elif codetp == 'CHANGA':
        bases = 'ncal-IV.'
        snapshot_files += glob.glob('%s/%s??????' % (folder,bases))
    elif codetp == 'manual':
        snapshot_files += glob.glob(base_manual)

    if checkredshift == False: #sort by the index of the snapshot file name
        snapshot_files.sort()
    else: #check the redshift of each snapshot
        #Loop through each output. The scale factor is the number after the '#a =' string in the
        #output text file
        # if codetp == 'ENZO' or codetp == 'AGORA_ENZO':
        #     #Create a list to store the redshifts of all the snapshots
        #     snapshot_redshifts = np.array([])
        #     for file_dir in snapshot_files:
        #         with open(file_dir, 'r') as file:
        #             for line in file:
        #                 if line.startswith('CosmologyCurrentRedshift'):
        #                     redshift = float(line.split()[-1])
        #                     break
        #         snapshot_redshifts = np.append(snapshot_redshifts, redshift)
        # else:
            my_storage = {}
            for sto, file_dir in yt.parallel_objects(snapshot_files, nprocs-1, storage = my_storage):
                ds = yt.load(file_dir)
                if codetp == 'AREPO':
                    try:
                        ds.derived_field_list
                    except:
                        pass
                redshift = ds.current_redshift
                current_time = ds.current_time.in_units('Myr').v
                sto.result = {}
                sto.result[1] = redshift
                sto.result[2] = current_time
            snapshot_redshifts = np.array([])
            current_time = np.array([])
            for c, vals in sorted(my_storage.items()):
                snapshot_redshifts = np.append(snapshot_redshifts, vals[1])
                current_time = np.append(current_time, vals[2])
            arg_z = np.argsort(-snapshot_redshifts)
            snapshot_files = np.array(snapshot_files)[arg_z]#sort in descending order
            snapshot_redshifts = snapshot_redshifts[arg_z]
            current_time = current_time[arg_z]
    #Write out a text file
    if yt.is_root():
        print('%s/pfs_allsnaps_%s.txt' % (folder2,index))
        final_file = []
        for i in range(len(snapshot_files)):
            final_file.append([snapshot_files[i],snapshot_redshifts[i],current_time[i]])
        np.savetxt('%s/pfs_allsnaps_%s.txt' % (folder2,index),final_file,fmt='%s')
