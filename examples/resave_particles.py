# THIS CODE IS A WORKAROUND IF THE RESAVE_PARTICLE() FUNCTION IN HASKAP.PY FREEZES WHEN RUNNING.
# RUN THIS CODE SEPARATELY BEFORE RUNNING HASKAP.PY 
# THE RESAVE_PARTICLE() FUNCTION IS MAINLY FOR PARTICLE-BASED CODES (GADGET3, GADGET4, GEAR, CHANGA, GIZMO), AS YT MAY HAVE ISSUES CREATING REGIONS IN THESE CODES. 

import yt
import numpy as np
import sys, os
from yt.data_objects.particle_filters import add_particle_filter
from yt.data_objects.unions import ParticleUnion
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank
nprocs = comm.size

def open_ds(timestep,codetp):
    # The conversion factor from code_length to physical unit is not correct in AGORA's GADGET3 and AREPO
        if codetp == 'AREPO':
            ds = yt.load(fld_list[timestep], unit_base = {"length": (1.0, "Mpccm/h")})
        elif codetp == 'GADGET3' or codetp == 'GADGET4':
            ds = yt.load(fld_list[timestep], unit_base = {"length": (1.0, "Mpccm/h"),"UnitMass_in_g": 1.989e43})
        else:
            ds = yt.load(fld_list[timestep])
        return ds
    
def pickup_particles(reg,codetp,find_dm, find_stars):
    # 'nbody'
    if codetp == 'ENZO':
        mass_all = reg[('all','particle_mass')].in_units('Msun')
        pos_all = reg[('all','particle_position')].in_units('m')
        vel_all = reg[('all','particle_velocity')].in_units('m/s')
        ids_all = reg[('all','particle_index')].v.astype(int)
        type_all = reg[('all','particle_type')].v.astype(int)
    if find_dm:  
        dm_name_dict = {'GEAR': ["PartType5","PartType2"],\
                'GADGET3': ["PartType5","PartType1"],'GADGET4': ["PartType5","PartType1"], 'AREPO': ["PartType2","PartType1"],\
                'GIZMO': ["PartType2","PartType1"], 'RAMSES': 'DM',\
                'ART': 'darkmatter', 'CHANGA': 'DarkMatter'}
        if codetp == 'ENZO':
            dm_bool = np.logical_and(np.logical_or(type_all == 1, type_all==4), mass_all > 1)
            mass = mass_all[dm_bool]
            pos = pos_all[dm_bool]
            vel = vel_all[dm_bool]
            ids = ids_all[dm_bool]
        elif codetp == 'GEAR' or codetp == 'GADGET3' or codetp == 'AREPO' or codetp == 'GIZMO' or codetp == 'GADGET4':
            for type_i in range(len(dm_name_dict[codetp])):
                if type_i == 0:
                    mass = reg[(dm_name_dict[codetp][type_i],'particle_mass')].in_units('Msun')
                    pos = reg[(dm_name_dict[codetp][type_i],'particle_position')].in_units('m')
                    vel = reg[(dm_name_dict[codetp][type_i],'particle_velocity')].in_units('m/s')
                    ids = reg[(dm_name_dict[codetp][type_i],'particle_index')].v.astype(int)
                else:
                    mass = np.append(mass, reg[(dm_name_dict[codetp][type_i],'particle_mass')].in_units('Msun'))
                    pos = np.append(pos, reg[(dm_name_dict[codetp][type_i],'particle_position')].in_units('m'), axis=0)
                    vel = np.append(vel, reg[(dm_name_dict[codetp][type_i],'particle_velocity')].in_units('m/s'), axis=0)
                    ids = np.append(ids, reg[(dm_name_dict[codetp][type_i],'particle_index')].v.astype(int))
        elif codetp == 'RAMSES' or codetp == 'ART':
            mass = reg[(dm_name_dict[codetp],'particle_mass')].in_units('Msun')
            pos = reg[(dm_name_dict[codetp],'particle_position')].in_units('m')
            vel = reg[(dm_name_dict[codetp],'particle_velocity')].in_units('m/s')
            ids = reg[(dm_name_dict[codetp],'particle_index')].v.astype(int)
        elif codetp == 'CHANGA':
            mass = reg[(dm_name_dict[codetp],'particle_mass')].in_units('Msun')
            pos = reg[(dm_name_dict[codetp],'particle_position')].in_units('m')
            vel = reg[(dm_name_dict[codetp],'particle_velocity')].in_units('m/s')
            ids = np.arange(len(mass)).astype(int)
    #Adding stars
    if find_stars:
        star_name_dict = {'GADGET3':'PartType4','GADGET4':'PartType4','GEAR':'PartType1','AREPO':'PartType4','GIZMO':'PartType4','RAMSES':'star','ART':'stars','CHANGA':'Stars'}
        if codetp == 'ENZO':
            star_bool = np.logical_and(np.logical_or.reduce((type_all == 2, type_all == 5, type_all == 7)), mass_all > 1)
            spos = pos_all[star_bool]
            svel = vel_all[star_bool]
            sids = ids_all[star_bool]
        else:
            try:
                spos = reg[(star_name_dict[codetp],'particle_position')].in_units('m')
                if codetp == 'CHANGA':
                    sids = np.arange(len(spos)) + 15368024 #only applicable when loading the whole box, issue with AGORA CHANGA data
                else:
                    sids = reg[(star_name_dict[codetp],'particle_index')].astype(int)
                svel = reg[(star_name_dict[codetp],'particle_velocity')].in_units('m/s')
            except:
                spos,svel,sids = np.reshape(np.array([]), (0,3)),np.reshape(np.array([]), (0,3)),np.array([])
    if find_dm and find_stars:
        return mass.in_units('kg').v,pos,vel,ids,spos,svel, sids
    elif find_stars and not find_dm:
        return spos,svel,sids
    elif find_dm and not find_stars:
        return mass.in_units('kg').v,pos,vel,ids


def job_scheduler(out_list,ranklim=1e99):
    '''
    Function to schedule jobs for each rank. This is the implementation of MPI to run parallel loops. Works with any given list.
    Parameters:
        out_list (list): List of jobs to be done
    Returns:
        tuple: Dictionary of jobs for each rank, and a dictionary to store the results
    '''
    ranks = np.arange(min(nprocs,ranklim)).astype(int)
    jobs = {i.item(): [] for i in ranks}
    sto = {t: {} for t in out_list}
    if rank == 0:
        count = 0
        while count < len(out_list):
            out_list_2 = np.copy(ranks)
            np.random.shuffle(out_list_2)
            for o in ranks:
                if count + out_list_2[o] < len(out_list):
                    i = count + out_list_2[o].item()
                    jobs[o].append(out_list[i])
            count += len(ranks)
        for o in jobs:
            np.random.shuffle(jobs[o])
    jobs = comm.bcast(jobs, root=0)
    return jobs, sto

def inteval_timelist(skip,fld_list):
    timelist = np.arange(len(fld_list))[::-1][::skip]
    interval = int(len(timelist)/10)
    if interval <=7:
        interval = len(timelist)-1
    return interval,timelist

def ensure_dir(f):
    if rank==0:
        if not os.path.exists(f):
            os.makedirs(f)

def resave_particles(ranklim=30):
    interval,timelist = inteval_timelist(skip,fld_list)
    ranks = np.arange(min(nprocs,ranklim))
    jobs,sto = job_scheduler(timelist,ranklim=ranklim)
    refined_times = np.array([])
    save_part = savestring+'/particle_save'
    ensure_dir(save_part)
    refined_times = np.array([])
    for times in timelist:
        len_now = len(timelist[timelist>=times])
        if (len_now%interval ==0 or times == last_timestep) and times >0:
            refined_times = np.append(refined_times,times)
    ll_all,ur_all = np.array([1e89,1e89,1e89]),-1*np.array([1e89,1e89,1e89])
    for times in refined_times:
        ll_o,ur_o = np.load(savestring + '/' + 'Refined/' + 'refined_region_%s.npy' % (int(times)),allow_pickle=True).tolist()
        ll_all = np.minimum(ll_all,ll_o)
        ur_all = np.maximum(ur_all,ur_o)
    buffer = (np.array(ur_all) - np.array(ll_all))*0.05
    ll_all,ur_all = np.array(ll_all)-buffer,np.array(ur_all)+buffer
    if os.path.exists(save_part+'/part_dict.npy'):
        part_dict = np.load(save_part+'/part_dict.npy',allow_pickle= True).tolist()
    else:
        part_dict = {}
    jobs = comm.bcast(jobs,root=0)
    for rank_now in ranks:
        if rank == rank_now:
            for t in jobs[rank]:
                if t not in part_dict:
                    numsegs = max(int(1+nprocs**(1/3)),3)
                    xx,yy,zz = np.meshgrid(np.linspace(ll_all[0],ur_all[0],numsegs),\
                                np.linspace(ll_all[1],ur_all[1],numsegs),np.linspace(ll_all[2],ur_all[2],numsegs))
                    ll = np.concatenate((xx[:-1,:-1,:-1,np.newaxis],yy[:-1,:-1,:-1,np.newaxis],zz[:-1,:-1,:-1,np.newaxis]),axis=3) #ll is lowerleft
                    ur = np.concatenate((xx[1:,1:,1:,np.newaxis],yy[1:,1:,1:,np.newaxis],zz[1:,1:,1:,np.newaxis]),axis=3) #ur is upperright
                    ll = np.reshape(ll,(ll.shape[0]*ll.shape[1]*ll.shape[2],3))
                    ur = np.reshape(ur,(ur.shape[0]*ur.shape[1]*ur.shape[2],3))
                    ds = open_ds(t,code)
                    meter = ds.length_unit.in_units('m')
                    reg = ds.all_data()
                    if find_dm == True and find_stars == False:
                        mass,pos,vel,ids = pickup_particles(reg,code, find_dm, find_stars)
                        bool_reg = (np.sum(pos >= ll_all*meter,axis=1) ==3)*(np.sum(pos < ur_all*meter,axis=1) ==3)
                        mass,pos,vel,ids = mass[bool_reg],pos[bool_reg],vel[bool_reg],ids[bool_reg]
                    elif find_dm == False and find_stars == True:
                        spos, svel, sids = pickup_particles(reg, code, find_dm, find_stars)
                        sbool_reg = (np.sum(spos >= ll_all*meter,axis=1) ==3)*(np.sum(spos < ur_all*meter,axis=1) ==3)
                        spos,svel,sids = spos[sbool_reg],svel[sbool_reg],sids[sbool_reg]
                    elif find_dm == True and find_stars == True:
                        mass,pos,vel,ids,spos,svel,sids = pickup_particles(reg, code, find_dm, find_stars)
                        bool_reg = (np.sum(pos >= ll_all*meter,axis=1) ==3)*(np.sum(pos < ur_all*meter,axis=1) ==3)
                        mass,pos,vel,ids = mass[bool_reg],pos[bool_reg],vel[bool_reg],ids[bool_reg]
                        sbool_reg = (np.sum(spos >= ll_all*meter,axis=1) ==3)*(np.sum(spos < ur_all*meter,axis=1) ==3)
                        spos,svel,sids = spos[sbool_reg],svel[sbool_reg],sids[sbool_reg]
                    sto[t]['ll'] = ll
                    sto[t]['ur'] = ur
                    for v in range(len(ll)):
                        if find_dm == True:
                            part = {}
                            bool_in = (np.sum(pos >= ll[v]*meter,axis=1) ==3)*(np.sum(pos < ur[v]*meter,axis=1) ==3)
                            part['pos'],part['mass'],part['vel'],part['ids'] = pos[bool_in],mass[bool_in],vel[bool_in],ids[bool_in]
                            if find_stars == True:
                                sbool_in = (np.sum(spos >= ll[v]*meter,axis=1) ==3)*(np.sum(spos < ur[v]*meter,axis=1) ==3)
                                part['spos'],part['svel'],part['sids'] = spos[sbool_in],svel[sbool_in],sids[sbool_in]
                            np.save(save_part+'/part_%s_%s.npy' % (t,v),part)
                        elif find_dm == False and find_stars == True:
                            part = np.load(save_part+'/part_%s_%s.npy' % (t,v),allow_pickle=True).tolist()
                            sbool_in = (np.sum(spos >= ll[v]*meter,axis=1) ==3)*(np.sum(spos < ur[v]*meter,axis=1) ==3)
                            part['spos'],part['svel'],part['sids'] = spos[sbool_in],svel[sbool_in],sids[sbool_in]
                            np.save(save_part+'/part_%s_%s.npy' % (t,v),part)
                    if find_dm == True:
                        mass,pos,vel,ids = 0,0,0,0
                    reg = 0
                    ds = 0
    part_dict = comm.bcast(part_dict,root=0)
    for rank_now in ranks:
        for t in jobs[rank_now]:
                sto[t] = comm.bcast(sto[t], root=rank_now)
    jobs = comm.bcast(jobs,root=0)
    if rank==0 and len(part_dict)==0:
        np.save(save_part+'/part_dict.npy',sto)

fldn = 2019 #make sure that this matches with the version in haskap.py
skip = 1
code = sys.argv[1]
savestring = sys.argv[2]
find_stars = False #enable this if you want the position, velocity, and ids of stars to be exported (along with the DM metadata)
find_dm = True #only disable this if the DM partsave files are already available, and the star metadata are just appended to the partsave files
fld_list = np.loadtxt(savestring + '/pfs_allsnaps_%s.txt' % fldn,dtype=str)[:,0]
last_timestep = len(fld_list) - 1
#print(savestring + '/pfs_allsnaps_%s.txt' % fldn)
resave_particles()
