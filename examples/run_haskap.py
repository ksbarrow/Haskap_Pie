from haskap.haskap import Evolve_Tree, ensure_dir, minmass_calc
from haskap.make_pfs_allsnaps import make_pfs_allsnaps
import sys, os
import yt
import gc
import numpy as np


yt.enable_parallelism()
gc.enable()


from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank
nprocs = comm.size

if len(sys.argv) !=5:
        if rank==0:
            print(sys.argv)
            print('Need simulation file path and code type to run as well as whether to pre-run refined region, ex: python shinbad.py simulation/folder/path ENZO save/path skip_number')
else:
    string = sys.argv[1]
    code = sys.argv[2]
    savestring = sys.argv[3]
    skip = int(sys.argv[4])
    fake = False
    if rank==0:
        print(string,code,savestring,skip)
    fldn = 2013
    if fake:
        fldn = 1948
    organize_files = False
    Numlength = 1000
    non_sphere = True
    ensure_dir(savestring)
    ensure_dir(savestring+'/Refined')
    fld_list = None
    if not os.path.exists(savestring +  '/pfs_allsnaps_%s.txt' % fldn):
            fld_list = make_pfs_allsnaps(string,savestring,code,index=fldn)
            fld_list = comm.bcast(fld_list,root=0)
    fld_list = np.loadtxt(savestring + '/pfs_allsnaps_%s.txt' % fldn,dtype=str)[:,0]
    time_list_1 = np.loadtxt(savestring + '/pfs_allsnaps_%s.txt' % fldn,dtype=str)[:,2].astype(float)
    red_list_1 = np.loadtxt(savestring + '/pfs_allsnaps_%s.txt' % fldn,dtype=str)[:,1].astype(float)
    u = yt.units
    oden_list = [80,100,150,200,250,300,500,700,1000]
    oden1 = False
    oden2 = False
    offset = 0
    #minmass = 1e11
    make_tree = True
    END = False
    path = string
    find_dm = True
    find_stars = True
    resave = False
    last_timestep = len(fld_list) - 1
    if fake:
        find_stars = False
        resave = True
        minmass,refined = 21232, True
    else:
        minmass,refined = minmass_calc(code)
    if rank==0:
        print('Save iteration: ',fldn)
        #minmass_calc(code)
    #refined = False
    if code == 'GEAR' or code == 'GIZMO' or code == 'CHANGA' or code == 'GADGET3':# or code == 'ENZO':
            resave = True
    # if resave:
    #     resave_particles()
    Evolve_Tree(plot=False,codetp=code,skip_large=False,verbose=False,\
        from_tree=False,last_timestep=last_timestep,multitree=True,refined=refined,video=True,trackbig=False,tracktime=True)
    if organize_files and rank==0:
        tmp_folder = savestring+'/tmp_files/'
        os.system('rm -r %s' % tmp_folder)

