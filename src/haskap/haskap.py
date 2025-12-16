### Potential Energy ---------------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import yt
import yt.units as ytunit
import random
from sklearn.cluster import KMeans,DBSCAN,MiniBatchKMeans
from scipy.spatial.distance import cdist
from scipy import stats
from scipy.spatial import ConvexHull
import time as time_sys
import os
import itertools
import matplotlib
import tracemalloc
import linecache
import sys
import gc
import cv2
import healpy as hp
import glob as glob
import numbers
import matplotlib.colors as mcolors
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 19})
import matplotlib.pyplot as plt
from natsort import natsorted
from haskap.Find_Refined import Find_Refined
from haskap.make_pfs_allsnaps import make_pfs_allsnaps
if int((yt.__version__).split('.')[0]) >= 4 and int((yt.__version__).split('.')[1]) >= 2: #ParticleUnion is only available in yt 4.2 and later
    from yt.data_objects.unions import ParticleUnion
else:
    from yt.data_objects.unions import Union as ParticleUnion
from yt.data_objects.particle_filters import add_particle_filter


yt.enable_parallelism()
gc.enable()
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank
nprocs = comm.size




class Evolve_Tree():
    """docstring for ."""
    def __init__(self,plot=False,codetp='ENZO',skip_large=False,verbose=False,from_tree=False,\
                last_timestep=None,video=False,multitree=False,refined=True,\
                reset=None,trackbig=None,tracktime=False,track_mem=False):
        # Galthers node list from a timestep
        if trackbig == False:
            trackbig = None
        if from_tree:
            halotree = np.load(string + "/halotree.npy", allow_pickle = True, encoding = "latin1").tolist()
            timestep = list(halotree[halo].keys())[0]
        else:
            timestep = last_timestep
        self.tracktime = tracktime
        self.optimization = False
        self.plot_width = 4
        self.all_times = np.arange(timestep+1)[::-1][::skip]
        self.track_mem = track_mem
        self.traceback = False
        # print(self.all_times)
        # red_int = np.arange(len(fld_list))
        # red_int = red_int[red_int <= last_timestep]
        self.interval = int(len(self.all_times)/10)
        if len(self.all_times) <=7:
            self.interval = len(self.all_times) -1
        if rank==0:
             print(len(self.all_times),len(self.all_times)/10,self.interval)
        self.codetp = codetp
        if self.codetp == 'AREPO':
            self.all_times = self.all_times[self.all_times>5]
        if self.codetp == 'ART':
            self.all_times = self.all_times[self.all_times>14]
        self.all_times = self.all_times.tolist()
        self.nprocs = nprocs
        #self.open_all_ds(self.codetp)
        self.reset = reset
        self.skip_large = skip_large
        self.verbose = verbose
        self.timestep = timestep
        self.time = {}
        self.rvir_0 = {}
        self.fof = {}
        self.dens = {}
        self.G = 6.6743e-11
        self.taken = []
        partf = 'particle_plots_%s' % fldn
        self.savestg = savestring +  '/' + partf
        ensure_dir(savestring)
        ensure_dir(self.savestg)
        self.new_halo = 0
        self.minmass = 1e20
        self.ll_all,self.ur_all = {},{}
        self.maxids = 4000
        self.compare = False
        self.time0 = time_sys.time()
        self.progs = True
        self.make_force = True
        self.plot_list = np.arange(3).astype(str)
        # Finds the initial halo list at the last timestep, inlcuding using a halo-finder if from_tree is "False"
        if trackbig != None:
            track = 'First'
        else:
            track = None
        self.done = False
        self.final_reverse = False
        self.idl = {}
        self.halotree = {}
        self.snaps = []
        if os.path.exists(savestring +  '/' + 'halotree_%s_final.npy' % (fldn)) and self.reset is None:
             self.done = True
             if rank ==0:
                 self.halotree = np.load(savestring + '/halotree_%s.npy' % (fldn),allow_pickle=True).tolist()
                 self.idl = np.load(savestring + '/idl_%s.npy' % (fldn),allow_pickle=True).tolist()
                 self.hullv = np.load(savestring + '/hullv_%s.npy' % (fldn),allow_pickle=True).tolist()
             if os.path.exists(savestring +  '/' + 'star_id_%s.npy' % (fldn)):
                 if rank ==min(nprocs-1,1) and find_stars:
                     self.star_id = np.load(savestring +  '/' + 'star_id_%s.npy' % (fldn),allow_pickle=True).tolist()
             min_time_0 = 0
        if os.path.exists(savestring +  '/' + 'max_time_%s.txt' % (fldn)):
            self.final_reverse = True
        #     self.halotree = np.load(savestring + '/halotree_%s_final.npy' % (fldn),allow_pickle=True).tolist()
        #     self.idl = np.load(savestring + '/idl_%s_final.npy' % (fldn),allow_pickle=True).tolist()
        if not self.done:
            if not os.path.exists(savestring + '/halotree_%s.npy' % (fldn)):
                if rank==0:
                  if not self.traceback and self.track_mem:
                      tracemalloc.start(50)
                      self.traceback = True
                self.ds,self.meter = open_ds(self.timestep,self.codetp,direction=1)
                self.dens[self.timestep] = Initial_Halo_Tree(codetp=self.codetp,skip_large=skip_large,ds=self.ds,lu=self.meter, \
                    from_tree=from_tree,last_timestep=timestep,verbose=self.verbose,refined=refined,trackbig=track)
                self.list_halos_from_nodes(self.dens[self.timestep].final_nodes)
                ll_all,ur_all = self.dens[self.timestep].ll_all,self.dens[self.timestep].ur_all
                if rank==0 and self.tracktime:
                  print('Initial FoF',time_sys.time()-self.time0)
                  self.time0 = time_sys.time()
            else:
                if refined:
                    fr = np.load(savestring + '/' + 'Refined/' + 'refined_region_%s.npy' % (self.timestep),allow_pickle=True).tolist()
                    ll_all,ur_all = fr
                    # if self.codetp == 'ENZO':
                    #     ll_all = np.array([0.4375,0.45078125,0.4375])
                    #     ur_all = np.array([0.5625,0.55703125,0.5625])
                    # elif self.codetp == "AGORA_ENZO":
                    #     ll_all = np.array([0.442765, 0.53123 , 0.46141 ])
                    #     ur_all = np.array([0.50466 , 0.5725 , 0.557315])
                    if rank==0 and self.verbose:
                        print('Refined region found :',ll_all,ur_all)
                # elif refined:
                #          ll_all = np.array([0.442765, 0.53123 , 0.46141 ])
                #          ur_all = np.array([0.50466 , 0.5725 , 0.557315])
                if rank ==0:
                    self.halotree = np.load(savestring + '/halotree_%s.npy' % (fldn),allow_pickle=True).tolist()
                    self.idl = np.load(savestring + '/idl_%s.npy' % (fldn),allow_pickle=True).tolist()
                self.hullv = np.load(savestring + '/hullv_%s.npy' % (fldn),allow_pickle=True).tolist()
                if os.path.exists(savestring +  '/' + 'star_id_%s.npy' % (fldn)):
                    if rank ==min(nprocs-1,1) and find_stars:
                        self.star_id = np.load(savestring +  '/' + 'star_id_%s.npy' % (fldn),allow_pickle=True).tolist()
                elif rank==min(nprocs-1,1):
                    self.star_id = {}
                    self.star_id['ids'] = {}
                    self.star_id['energies'] = {}
            if rank ==0:
                for v in self.halotree:
                    self.new_halo = max(int(v.split('_')[0]),self.new_halo)
                    if self.timestep in self.halotree[v]:
                        self.minmass = min(self.minmass,self.halotree[v][self.timestep]['Halo_Mass'])
            self.new_halo = comm.bcast(self.new_halo,root=0)
            self.minmass = comm.bcast(self.minmass,root=0)
            self.ds,self.meter = open_ds(self.timestep,self.codetp,direction=1,skip=True)
            if not refined:
                ll_all,ur_all = self.ds.domain_left_edge.in_units('m').v/self.meter,self.ds.domain_right_edge.in_units('m').v/self.meter
            self.time[self.timestep] = self.ds.current_time.in_units('s').v
            self.ll_all[self.timestep],self.ur_all[self.timestep] = np.array(ll_all),np.array(ur_all)
            min_time_0 = 1e10
            self.end = False
            self.calc_count = 0
            if self.reset is not None:
                self.restart(self.reset)
            if rank ==0:
                for halo in self.idl:
                    for time in self.idl[halo]:
                      min_time_0 = min(min_time_0,time)
            min_time_0 = comm.bcast(min_time_0,root=0)
            if os.path.exists(savestring +  '/' + 'halotree_%s_final.npy' % (fldn)) and self.reset is None:
                     min_time_0 = min(self.all_times)
            if fake:# and min_time_0 == timestep:
                tind = self.all_times.index(self.timestep)
                self.direction = max(self.all_times[tind-1]-self.all_times[tind],1)
                self.send_halotree()
                self.make_lengths()
                # self.reorder_halos()
                for halo in self.plot_list:
                    if halo in self.halotree:
                      if self.timestep in self.halotree[halo] and  self.lenhalo[halo] >= 2:
                            tree_halo = self.halotree[halo][self.timestep]
                            rvir = tree_halo['Halo_Radius']
                            if halo not in self.rvir_0:
                                self.rvir_0[halo] = 4*rvir
                self.reverse_ids()
                self.plot_halo()
            # if video and timestep - min_time_0 < self.interval/4:
            #     self.make_lengths()
            #     #self.make_arbor()
            #     #self.plot_halo()
            #min_time_0 = 6
            for timestepi in self.all_times:
                self.timestep = timestepi
                len_now = len(np.array(self.all_times)[np.array(self.all_times)>=self.timestep])
                if os.path.exists(fld_list[self.timestep]):
                  self.ll_all[self.timestep],self.ur_all[self.timestep] = np.array(ll_all),np.array(ur_all)
                  tind = self.all_times.index(self.timestep)
                  self.direction = max(self.all_times[tind-1]-self.all_times[tind],1)
                  #print(self.direction)
                  if self.timestep < min_time_0 and make_tree and not END and red_list_1[self.timestep] <29  and ((self.codetp != 'AREPO' and self.codetp != 'ART') or timestepi >5):
                   if self.calc_count ==0:
                       self.send_halotree()
                   self.make_active_list(backlook=self.direction)
                   #print(self.mass_list)
                   if len(self.mass_list) > 0:
                    self.calc_count += 1
                    if rank==0:
                      if not self.traceback and self.track_mem:
                          tracemalloc.start(50)
                          self.traceback = True
                      print('Timestep',self.timestep,'calculating....')
                    if rank==0 and self.tracktime:
                      print('Begining of iteration',time_sys.time()-self.time0)
                      self.time0 = time_sys.time()
                      self.time2 = time_sys.time()
                    self.plot = plot
                    self.ds,self.meter = open_ds(self.timestep,self.codetp,direction=self.direction)
                    self.kg_sun = self.ds.mass_unit.in_units('Msun').v/self.ds.mass_unit.in_units('kg').v
                    self.time[self.timestep] = self.ds.current_time.in_units('s').v
                    self.taken = []
                    # Opens the simulation at the given timestep
                    # _m propagation is positive, backward propagation is negative
                    tind = self.all_times.index(self.timestep)
                    self.direction = max(self.all_times[tind-1]-self.all_times[tind],1)
                    # Every few timesteps, the halofinder is rerun to see if any halo escaped the amalysis if multitree is True
                    # self.reorder_halos()
                    # if rank==0 and self.tracktime:
                    #   print('Loaded simulation',time_sys.time()-self.time0)
                    #   self.time0 = time_sys.time()
                    # if rank==0:
                    #     print(len_now%self.interval,self.interval)
                    if self.verbose and rank==0:
                        print(len_now%self.interval,0.2*self.interval,0.8*self.interval,\
                            len_now%self.interval > max(0.2*self.interval,3),len_now%self.interval < min(0.8*self.interval,self.interval-3))
                    if (len_now%self.interval == 0)  and multitree and self.timestep > 0 and not self.final_reverse:
                        self.fof[self.timestep] = True
                        # self.reorder_halos()
                        if trackbig != None:
                            track = self.halotree[self.halo_num[0]][self.timestep+self.direction]
                        self.dens[self.timestep] = Initial_Halo_Tree(codetp=self.codetp,skip_large=skip_large,ds=self.ds,lu=self.meter,\
                            from_tree=from_tree,last_timestep=self.timestep,verbose=self.verbose,refined=refined,trackbig=track,oden0=self.mean_cden)
                        ll_all,ur_all = self.dens[self.timestep].ll_all,self.dens[self.timestep].ur_all
                        self.fof_halo = {}
                        self.make_active_list(backlook=self.direction)
                        self.evolve_com(self.halo_num)
                        if rank==0 and self.tracktime:
                          print('FoF Complete',time_sys.time()-self.time0)
                          self.time0 = time_sys.time()
                    elif (self.timestep%min(7,int(self.interval/2)) == 0) and self.ds.current_redshift >6 \
                        and self.timestep >min(7,int(self.interval/2-1)) \
                         and (len_now%self.interval > max(0.2*self.interval,3) and len_now%self.interval < min(0.8*self.interval,self.interval-3)):
                        self.fof[self.timestep] = True
                        self.make_active_list(backlook=self.direction)
                        arg_mass1 = np.argsort(self.mass_list)[::-1][:20]
                        arg_mass2 = np.arange(len(self.mass_list))[self.mass_list >1e9]
                        arg_mass = np.append(arg_mass1,arg_mass2)
                        arg_mass = np.unique(arg_mass)
                        #arg_mass = arg_mass1
                        self.evolve_com(self.halo_num)
                        self.dens[self.timestep] = Initial_Halo_Tree(codetp=self.codetp,skip_large=skip_large,ds=self.ds,lu=self.meter,\
                            from_tree=from_tree,last_timestep=self.timestep,verbose=self.verbose,refined=refined,trackbig=track,oden0=self.mean_cden,\
                            icenters=self.new_coml[arg_mass],iradii=3*self.rad_list[arg_mass])
                        self.fof_halo = {}
                        if rank==0 and self.tracktime:
                          print('FoF Complete',time_sys.time()-self.time0)
                          self.time0 = time_sys.time()
                    else:
                        self.fof[self.timestep] = False
                        self.make_active_list(backlook=self.direction)
                        self.evolve_com(self.halo_num)
                    if (len_now%self.interval == 1 or self.timestep==0) and multitree:
                        ntries = [1,7,4]
                    elif len_now%self.interval == 2:
                        ntries = [1,7,4]
                    else:
                        ntries = [1,7,4]
                    self.ll_all[self.timestep],self.ur_all[self.timestep] = np.array(ll_all),np.array(ur_all)
                    # Self.force bypasses the density calculation when set to True
                    # self.force = np.full(len(self.halo_num),False)
                    if rank==0:
                        print('Begining Halo Finding for %s Halos for Timestep %s' % (len(self.halo_num),self.timestep),time_sys.time()-self.time0)
                    # Finds progenitor galxies of the current halo list in the current timestep
                    if rank==0 and self.tracktime:
                        self.time4 = time_sys.time()
                    idl,massl,nl,rad,com,vcom,rvir_l,cden_l,cdens,hullv,more_var_l,var_names_scalar,var_names_vec,more_vec = self.determine_progenitors(ntries=ntries,check=True)
                    # Checks for overlapping halos
                    if len(cdens) >0:
                            if len_now <= 2:
                                self.indices = get_non_coincident_halo_index(idl,massl,nl,rad*self.meter,self.lenhalo,self.escape(massl,rad),\
                                    vcom*self.meter,com*self.meter,rvir_l*self.meter,cden_l,margin=0.3,margin2=0.3)
                            else:
                                self.indices = get_non_coincident_halo_index(idl,massl,nl,rad*self.meter,self.lenhalo,self.escape(massl,rad),\
                                    vcom*self.meter,com*self.meter,rvir_l*self.meter,cden_l,margin=0.3,margin2=0.3)
                            #self.indices = np.arange(len(massl))
                            if rank==0 and self.tracktime:
                                print('Elimitated Overlapping Halos',time_sys.time()-self.time0)
                                self.time0 = time_sys.time()
                            if rank ==0 and self.verbose:
                                id_original = np.arange(len(massl))
                                for id_ind in id_original:
                                    if id_ind not in self.indices:
                                        prog,r = nl[id_ind]
                                        if prog ==0:
                                            print('Halo %s eliminated' % r)
                    else:
                             self.indices = np.arange(len(massl))
                    #self.taken = np.array(nl)[self.indices,1]
                    # Builds halo tree and updates attribute variables
                    #if rank==0:
                    self.build_tree(nl,massl,idl,rad,com,vcom,rvir_l,cden_l,cdens,hullv,more_var_l,var_names_scalar,var_names_vec,more_vec)
                    nl,massl,idl,rad,com,vcom,rvir_l,cden_l,cdens,hullv,more_var_l,var_names_scalar,var_names_vec,more_vec =0,0,0,0,0,0,0,0,0,0,0,0,0,0
                    #self.join_halotree()
                    #self.halotree = comm.bcast(self.halotree,root=0)
                    self.make_lengths()
                    if rank ==0 and self.tracktime:
                            print('Updated Halo Tree',time_sys.time()-self.time0)
                            self.time0 = time_sys.time()
                    # New co-progenitors are forward modeled back into the descendent halos, broken halo trees are removed
                    if (len_now%self.interval == 2) and timestepi > 10 and len_now !=2:
                        self.timestep = timestepi
                        self.plot = False
                        self.make_force = False
                        self.forward_model_co_progenitors(inters=0.2)
                        self.make_force = True
                        if rank==0 and self.tracktime:
                          print('Completed Forward Modelling',time_sys.time()-self.time0)
                          self.time0 = time_sys.time()
                        self.timestep = timestepi
                        tind = self.all_times.index(self.timestep)
                        self.direction = max(self.all_times[tind-1]-self.all_times[tind],1)
                        self.timestep = timestepi
                        #if rank ==0:
                        self.delete_halos(timestepi)
                        #self.join_halotree()
                        #self.halotree = comm.bcast(self.halotree,root=0)
                        self.send_halotree()
                        self.make_active_list(backlook=self.direction)
                        del self.ds
                    # Plots the halos for creating videos
                    if video and 1==2:
                        if len_now%(int(1*self.interval)) == 2:
                            my_storage = {}
                            tmax = min(timestep-1,timestepi+(2*self.interval*self.direction)+1)
                            trange = np.array(self.all_times)[(np.array(self.all_times)>=timestepi)*(np.array(self.all_times)<=tmax)]
                            if rank ==0:
                                np.save(savestring +  '/' + 'idl_%s.npy' % (fldn),self.idl)
                                np.save(savestring +  '/' + 'halotree_%s.npy' % (fldn),self.halotree)
                                np.save(savestring +  '/' + 'hullv_%s.npy' % (fldn),self.hullv)
                            if rank ==min(nprocs-1,1) and find_stars:
                                #if len(self.star_id['ids']) >0:
                                    np.save(savestring +  '/' + 'star_id_%s.npy' % (fldn),self.star_id)
                            self.reverse_ids()
                            split_array = np.array_split(trange,int(max(len(trange)/nprocs-1,1)))
                            for t_split_a in split_array:
                                my_storage = {}
                                for sto,timestep_p in yt.parallel_objects(t_split_a,\
                                    self.nprocs, storage = my_storage):
                                #for timestep_p in range(timestepi,min(timestep-1,timestepi+(self.interval))):
                                    self.timestep = timestep_p
                                    self.ds,self.meter = open_ds(self.timestep,self.codetp,direction=self.direction)
                                    self.plot_halo_2()
                                my_storage = {}
                            if rank==0 and self.tracktime:
                              print('Made Images',time_sys.time()-self.time0)
                              self.time0 = time_sys.time()
                        # Adds the last frames of the plots into a video
                        if len_now > 3 and len_now%(1*self.interval) == 2:
                            self.make_avi()
                            if rank==0 and self.tracktime:
                              print('Made Video',time_sys.time()-self.time0)
                              self.time0 = time_sys.time()
                    if rank==0 and self.tracktime:
                      print('Finished Iteration Time:',time_sys.time()-self.time2)
                  elif self.final_reverse:
                      self.time[self.timestep] = float(time_list_1[self.timestep])*3.1536e13
                      if ((len_now%self.interval == 0)  and multitree and self.timestep > 3)\
                        or ((self.timestep%min(7,int(self.interval/2)) == 0) and self.ds.current_redshift >6 and self.timestep >=min(2,int(self.interval/2))):
                          self.fof[self.timestep] = True
                      else:
                          self.fof[self.timestep] = False
                  elif self.timestep >= min_time_0 and make_tree:
                      #self.ds,self.meter = open_ds(self.timestep,self.codetp,direction=self.direction,skip=True)
                      # self.time[self.timestep] = self.ds.current_time.in_units('s').v
                      self.time[self.timestep] = float(time_list_1[self.timestep])*3.1536e13
                      if len_now%self.interval == 0 and multitree and refined:
                        refined_region = np.load(savestring + '/' + 'Refined/' + 'refined_region_%s.npy' % (self.timestep),allow_pickle=True).tolist()
                        ll_all,ur_all = refined_region
                      self.fof[self.timestep] = False
                      self.ll_all[self.timestep],self.ur_all[self.timestep] = np.array(ll_all),np.array(ur_all)
                      if video:
                         if len_now == 3:
                             self.make_active_list()
                             #self.plot_list = np.arange(3).astype(str)
                             for halo in self.plot_list:
                                 if halo in self.halotree:
                                   if self.timestep in self.halotree[halo] and self.lenhalo[halo] >= 3:
                                         # Halo plots all particles around the target halo and saves the size of the region
                                         # so that the boundaries are consitently large
                                         tree_halo = self.halotree[halo][self.timestep]
                                         rvir = tree_halo['Halo_Radius']
                                         if halo not in self.rvir_0:
                                             self.rvir_0[halo] = self.plot_width*rvir
                         # if timestep - min_time_0 < self.interval/4:
                         #      self.plot_halo()
                         # if self.timestep == min_time_0 and self.timestep != min(self.all_times):
                         #     my_storage = {}
                         #     for sto,timestep_p in yt.parallel_objects(np.array(self.all_times)[np.array(self.all_times) >= min_time_0],\
                         #        nprocs, storage = my_storage):
                         #         if timestep_p < timestep:
                         #             self.timestep = timestep_p
                         #             self.ds,self.meter = self.ds_list[timestep_p]['ds'],self.ds_list[timestep_p]['meter']
                         #             self.res = min(20*self.ds.index.get_smallest_dx(),2e-3)
                         #             self.plot_halo_2()
                         #     self.make_avi(erase=False)

                  if (len(self.time) == 2 or len(self.time)==3 or (len(self.time)%self.interval == 2)) and min_time_0 > timestepi:
                          # self.reorder_halos()
                          self.timestep = timestepi
                          lenhalo = {}
                          if rank==0:
                              for halo in self.halotree:
                                  lenhalo[halo] = len(self.idl[halo])
                          lenhalo = comm.bcast(lenhalo,root=0)
                          for halo in self.plot_list:
                              if halo in self.halotree:
                                if self.timestep in self.halotree[halo] and lenhalo[halo] >= 2:
                                      tree_halo = self.halotree[halo][self.timestep]
                                      rvir = tree_halo['Halo_Radius']
                                      if halo not in self.rvir_0:
                                          self.rvir_0[halo] = self.plot_width*rvir
                          if not self.final_reverse:
                            self.ds,self.meter = open_ds(self.timestep,self.codetp,direction=self.direction)
                            if rank ==0:
                                np.save(savestring +  '/' + 'idl_%s.npy' % (fldn),self.idl)
                                np.save(savestring +  '/' + 'halotree_%s.npy' % (fldn),self.halotree)
                                np.save(savestring +  '/' + 'hullv_%s.npy' % (fldn),self.hullv)
                            if rank == min(nprocs-1,1) and find_stars:
                                np.save(savestring +  '/' + 'star_id_%s.npy' % (fldn),self.star_id)
                            self.reverse_ids()
                            self.plot_halo()
                            if rank==0 and self.tracktime:
                              print('Made Images',time_sys.time()-self.time0)
                              self.time0 = time_sys.time()
                if rank==0:
                    print('Timestep',self.timestep,'complete')
                    if self.traceback:
                        self.snaps.append(tracemalloc.take_snapshot())
                        back = self.snaps[-1].statistics("traceback")
                        print("\n*** top 10 stats ***")
                        for i in range(10):
                           b = back[i]
                           print("%s b" % b.size)
                           for l in b.traceback.format():
                               print(l)
        if make_tree:
                if refined:
                    self.ll_all,self.ur_all = self.make_region_list()
                    # if rank ==0:
                    #     print(self.ll_all)
                self.plot = False
                if os.path.exists(savestring +  '/' + 'max_time_%s.txt' % (fldn)):
                    min_time_0 = np.loadtxt(savestring +  '/' + 'max_time_%s.txt' % (fldn))
                else:
                    min_time_0 = min(self.all_times)
                if min_time_0 != max(self.all_times):
                    if rank == 0:
                        if os.path.exists(savestring +  '/' + 'max_time_%s.txt' % (fldn)):
                            print('Continuing Reverse Modelling at Timestep %s' % min_time_0)
                        else:
                            #if rank ==0:
                            self.delete_halos(min_time_0)
                            self.reset_forced()
                            np.savetxt(savestring +  '/' + 'max_time_%s.txt' % (fldn),np.array([min_time_0]))
                    min_time_0 = comm.bcast(min_time_0,root=0)
                    #self.join_halotree()
                    #self.halotree = comm.bcast(self.halotree,root=0)
                    min_time_0 = np.maximum(min_time_0 - 4,0)
                    self.timestep = int(np.array(min_time_0))
                    self.end = True
                    self.ds,self.meter = open_ds(self.timestep,self.codetp)
                    self.kg_sun = self.ds.mass_unit.in_units('Msun').v/self.ds.mass_unit.in_units('kg').v
                    self.time
                    self.forward_model_co_progenitors(inters=10000)
                    if rank==0:
                        print('Reverse modelling complete',time_sys.time()-self.time0)
                        self.time0 = time_sys.time()
                min_time_0 = min(self.all_times)
                self.timestep = min_time_0
                self.direction = skip
                #self.join_halotree()
                if rank == min(nprocs-1,1):
                    self.star_id = {}
                min_time_0 = comm.bcast(min_time_0,root=0)
                if rank==0:
                    if find_stars:
                        self.star_id = np.load(savestring +  '/' + 'star_id_%s.npy' % (fldn),allow_pickle=True).tolist()
                    self.delete_halos(self.timestep,final=True)
                    if self.tracktime and rank==0:
                        print('Delete halos complete',time_sys.time()-self.time0)
                        self.time0 = time_sys.time()
                    self.prefall_mass()
                    self.clear_halos()
                    self.connect_halos()
                    if self.tracktime and rank==0:
                        print('Connect halos complete',time_sys.time()-self.time0)
                        self.time0 = time_sys.time()
                    self.reorder_halos()
                    if self.tracktime and rank==0:
                        print('Reorder halos complete',time_sys.time()-self.time0)
                        self.time0 = time_sys.time()
                        print(self.halotree.keys())
                #self.halotree = comm.bcast(self.halotree,root=0)
                if rank==0:
                    np.save(savestring +  '/' + 'halotree_%s_final.npy' % (fldn),self.halotree)
                    np.save(savestring +  '/' + 'idl_%s_final.npy' % (fldn),self.idl)
                    np.save(savestring +  '/' + 'hullv_%s_final.npy' % (fldn),self.hullv)
                    if find_stars:
                        np.save(savestring +  '/' + 'star_id_%s_final.npy' % (fldn),self.star_id)
                    if self.tracktime:
                        print('Final save complete',time_sys.time()-self.time0)
                        self.time0 = time_sys.time()
        # Completes the video and erases the images
        if video:
            if self.tracktime and rank==0:
                print('Starting Video',time_sys.time()-self.time0)
                self.time0 = time_sys.time()
                # if rank==0:
                #     partf = 'particle_plots_%s' % fldn
                #     savestg = savestring + '/'+ partf + '/'
                    # for halo in self.plot_list:
                    #     os.system('rm' + ' ' + savestg + 'Particle_Plot_%s_*_vid.png' % halo)
            my_storage = {}
            self.make_lengths()
            if rank ==0:
                for halo in self.plot_list:
                    for time in self.all_times:
                     if time in self.halotree[halo]:
                        tree_halo = self.halotree[halo][time]
                        rvir = tree_halo['Halo_Radius']
                        if halo not in self.rvir_0:
                            self.rvir_0[halo] = self.plot_width*rvir
            self.rvir_0 = comm.bcast(self.rvir_0,root=0)
            self.reverse_ids()
            split_array = np.array_split(self.all_times,int(max(len(self.all_times)/min(nprocs,15)+1,1)))
            for t_split_a in split_array:
                self.send_halotree(times=t_split_a)
                my_storage = {}
                for sto,timestep_p in yt.parallel_objects(t_split_a,min(nprocs,15), storage = my_storage):
                    if timestep_p < timestep:
                        self.timestep = timestep_p
                        self.ds,self.meter = open_ds(self.timestep,self.codetp,skip=True)
                        self.plot_halo_2()
                my_storage = {}
            if self.tracktime and rank==0:
                print('Finished Images',time_sys.time()-self.time0)
                self.time0 = time_sys.time()
            self.make_avi(erase=False)
            time_sys.sleep(20)
            if self.tracktime and rank==0:
                print('Finished Video',time_sys.time()-self.time0)

    def reverse_ids(self):
        halos = None
        if rank == 0:
            self.ids_reverse = {}
            for v in self.idl:
                for time in self.idl[v]:
                    if time not in self.ids_reverse:
                        self.ids_reverse[time] = {}
                    self.ids_reverse[time][v] = self.idl[v][time]
            np.save(savestring +  '/' + 'ridl_%s.npy' % (fldn),self.ids_reverse)
            halos = np.array(list(self.halotree.keys()))
        halos = comm.bcast(halos,root=0)

    def prune_halotree(self):
        if rank !=0:
            halos_2 = np.array(list(self.halotree.keys()))
            tmin = self.timestep+self.direction*3
            for halo in halos_2:
                if tmin in self.halotree[halo]:
                    if tmin not in self.halotree[halo]['Solved']:
                        del self.halotree[halo][tmin]
                        del self.hullv[halo][tmin]

    def add_halos(self):
            halos_2 = None
            if rank ==0:
                halos_2 = np.array(list(self.halotree.keys()))
            halos_2 = comm.bcast(halos_2,root=0)
            split_halos = np.array_split(halos_2,max(len(halos_2)/500,1))
            for halo_l in split_halos:
                halos_chunk = {}
                hull_v_chunk = {}
                if rank ==0:
                    for halo in halo_l:
                        if self.timestep in self.halotree[halo]:
                            halos_chunk[halo] = {}
                            halos_chunk[halo][self.timestep] = self.halotree[halo][self.timestep]
                            hull_v_chunk[halo] = {}
                            hull_v_chunk[halo][self.timestep] = self.hullv[halo][self.timestep]
                halos_chunk = comm.bcast(halos_chunk,root=0)
                hull_v_chunk = comm.bcast(hull_v_chunk,root=0)
                if rank != 0:
                    for halo in halos_chunk:
                        # if halo not in self.halotree:
                        #     # if rank ==2:
                        #     #     print(halo)
                        #     #     print(self.halotree.keys())
                        #     #     print(self.halotree[halo])
                        #     self.halotree[halo] = {}
                        #     self.hullv[halo] = {}
                        self.halotree[halo][self.timestep] = halos_chunk[halo][self.timestep]
                        self.hullv[halo][self.timestep] = hull_v_chunk[halo][self.timestep]


    def send_halotree(self,times=[]):
            halos_2 = np.array([])
            if rank ==0:
                halos_2 = np.array(list(self.halotree.keys()))
            new_tree = {}
            new_hullv = {}
            halos_2 = comm.bcast(halos_2,root=0)
            if len(times) ==0:
                min_time = min(self.timestep,self.timestep+self.direction*4)
                max_time = max(self.timestep,self.timestep+self.direction*4)
                time_range = np.arange(min_time,max_time+1)
            else:
                time_range = times
            #print(time_range,self.timestep,self.timestep+self.direction*3)
            split_halos = np.array_split(halos_2,max(len(halos_2)/500,1))
            for halo_l in split_halos:
                halos_chunk = {}
                hull_v_chunk = {}
                if rank ==0:
                    for halo in halo_l:
                            halos_chunk[halo] = {}
                            hull_v_chunk[halo] = {}
                            for time in time_range:
                                if time in self.halotree[halo]:
                                    halos_chunk[halo][time] = self.halotree[halo][time]
                                    hull_v_chunk[halo][time] = self.hullv[halo][time]
                            time_0 = self.halotree[halo]['Solved'][0]
                            time_1 = self.halotree[halo]['Solved'][1]
                            if time_0 not in halos_chunk[halo] and time_0 in self.halotree[halo]:
                                halos_chunk[halo][time_0] = self.halotree[halo][time_0]
                                hull_v_chunk[halo][time_0] = self.hullv[halo][time_0]
                            if time_1 not in halos_chunk[halo] and time_1 in self.halotree[halo]:
                                halos_chunk[halo][time_1] = self.halotree[halo][time_1]
                                hull_v_chunk[halo][time_1] = self.hullv[halo][time_1]
                            halos_chunk[halo]['Solved'] = self.halotree[halo]['Solved']
                            halos_chunk[halo]['Consec_Forced'] = self.halotree[halo]['Consec_Forced']
                            halos_chunk[halo]['Forced'] = self.halotree[halo]['Forced']
                            halos_chunk[halo]['Num_Progs'] = self.halotree[halo]['Num_Progs']
                halos_chunk = comm.bcast(halos_chunk,root=0)
                hull_v_chunk = comm.bcast(hull_v_chunk,root=0)
                if rank != 0:
                    for halo in halos_chunk:
                        new_tree[halo] = halos_chunk[halo]
                        new_hullv[halo] = hull_v_chunk[halo]
            if rank != 0:
                self.halotree = new_tree
                self.hullv = new_hullv
                # if rank ==2:
                #   print(self.halotree)
                new_tree,new_hull = {},{}

    def join_halotree(self,small_range=False,times=[]):
        if self.direction >0:
            check = 0
            self.prune_halotree()
            check = comm.bcast(check,root=0)
        else:
            check = 0
            self.prune_halotree()
            self.add_halos()
            check = comm.bcast(check,root=0)


    def reset_forced(self):
        if rank ==0:
            for k in self.halotree:
                self.halotree[k]['Consec_Forced'] = 0

    def open_all_ds(self,codetp):
        # The conversion factor from code_length to physical unit is not correct in AGORA's GADGET3, GADGET4, and AREPO
        self.ds_list = {}
        for timestep in range(len(fld_list)):
            self.ds_list[timestep] = {}
            if codetp == 'AREPO':
                self.ds_list[timestep]['ds'] = yt.load(fld_list[timestep], unit_base = {"length": (1.0, "Mpccm/h")})
            elif codetp == 'GADGET3' or codetp == 'GADGET4':
                self.ds_list[timestep]['ds'] = yt.load(fld_list[timestep], unit_base = {"length": (1.0, "Mpccm/h"),"UnitVelocity_in_cm_per_s":(1e3)})
            else:
                self.ds_list[timestep]['ds'] = yt.load(fld_list[timestep])
            left0 = self.ds_list[timestep]['ds'].domain_left_edge
            right0 = self.ds_list[timestep]['ds'].domain_right_edge
            width = 1 #(right0-left0)[0].v
            meter = self.ds_list[timestep]['ds'].length_unit.in_units('m').v/width
            self.ds_list[timestep]['meter'] = meter

    def delete_halos(self,timestepi,final=False):
        dellist = []
        # if not final:
        if final and refined:
            len_i = len(self.idl)
            for halo in self.idl:
                t_list = list(self.idl[halo].keys())
                for t in t_list:
                    ll,ur = self.ll_all[t],self.ur_all[t]
                    com = self.halotree[halo][t]['Halo_Center']
                    #print((np.sum(com > ll)==3)*(np.sum(com < ur)==3))
                    if not (np.sum(com > ll)==3)*(np.sum(com < ur)==3):
                        del self.halotree[halo][t]
                        del self.idl[halo][t]
                        del self.hullv[halo][t]
                        if find_stars and rank==0:
                            if halo in self.star_id['ids']:
                                if t in self.star_id['ids'][halo]:
                                    del self.star_id['ids'][halo][t]
                                    del self.star_id['energies'][halo][t]
            halo_l = list(self.idl.keys())
            for halo in halo_l:
                if len(self.idl[halo])==0:
                    del self.halotree[halo]
                    del self.idl[halo]
                    del self.hullv[halo]
                    if find_stars:
                        if halo in self.star_id['ids']:
                            del self.star_id['ids'][halo]
                            del self.star_id['energies'][halo]
            print('Deleting Outside of Region',len_i,len(self.idl))
        for halo in self.idl:
            min_halo_time = min(list(self.idl[halo].keys()))
            min_del = min(max(min(5,self.interval/2),5),len(self.all_times)-2)
            if len(self.idl[halo]) <= min_del and min_halo_time > timestepi + 2*self.direction and (min_halo_time > self.interval/2 or len(self.all_times) <10):
                dellist.append(halo)
        for halo in dellist[::-1]:
            if rank ==0:
                del self.halotree[halo]
                del self.idl[halo]
                del self.hullv[halo]
            if find_stars and (rank==min(nprocs-1,1) or final):
                if halo in self.star_id['ids']:
                    del self.star_id['ids'][halo]
                    del self.star_id['energies'][halo]
            if rank==0 and self.verbose:
                print('Halo %s removed' % halo)
        if final:
            if rank==0 and self.verbose:
                print('Final delete round')
            dellist = []
            halos = list(self.halotree.keys())
            end_time = np.array([])
            end_com = np.array([])
            end_length = {}
            end_length_2 = np.array([])
            change_list = np.array([])
            start_time,start_com = np.array([]),np.array([])
            mid_time,mid_com = np.array([]),np.array([])
            radii = {}
            volume = {}
            halo_time = {}
            com = {}
            vcom = {}
            mass = {}
            dellist2 = []
            k_mass = {}
            k_times = {}
            for k in halos:
                end_length_2 = np.append(end_length_2,len(self.idl[k].keys()))
            arg = np.argsort(end_length_2)
            for k in np.array(halos)[arg]:
                #print(self.idl[k].keys())
                k_time = np.array(list(self.idl[k].keys()))
                arg_k = np.argsort(k_time)
                k_time = k_time[arg_k]
                k_times[k] = k_time
                k_mass[k] = np.array([])
                change_list = np.append(change_list,k)
                current_time = min(k_time)
                fin_time = max(k_time)
                end_time = np.append(end_time,current_time)
                twothirds_time_i = int(min(max(2*len(k_time)/3,0),len(k_time)-1))
                #print(twothirds_time_i,k_time[twothirds_time_i])
                twothirds_time = int(k_time[twothirds_time_i])
                start_time = np.append(start_time,fin_time)
                mid_time = np.append(mid_time,twothirds_time)
                if len(end_com) ==0:
                    end_com = np.array([self.halotree[k][current_time]['Halo_Center']])
                    mid_com = np.array([self.halotree[k][twothirds_time]['Halo_Center']])
                else:
                    end_com = np.vstack((end_com,self.halotree[k][current_time]['Halo_Center']))
                    mid_com = np.vstack((mid_com,self.halotree[k][twothirds_time]['Halo_Center']))
                if len(start_com) ==0:
                    start_com = np.array([self.halotree[k][fin_time]['Halo_Center']])
                else:
                    start_com = np.vstack((start_com,self.halotree[k][fin_time]['Halo_Center']))
                end_length[k] = len(self.idl[k].keys())
                # end_length_2 = np.append(end_length_2,len(self.idl[k].keys()))
                for htime in k_time:
                    htime = int(htime)
                    if self.halotree[k][htime] ['bound_mass'] != None:
                        k_mass[k] = np.append(k_mass[k],self.halotree[k][htime]['Halo_Mass'])
                    else:
                        k_mass[k] = np.append(k_mass[k],self.halotree[k][htime]['Halo_Mass'])
                    if not htime in radii:
                        radii[htime] = np.array([self.halotree[k][htime]['Halo_Radius']])
                        halo_time[htime]  = np.array([k])
                        com[htime] = np.array([self.halotree[k][htime]['Halo_Center']])
                        vcom[htime] = np.array([self.halotree[k][htime]['Vel_Com']])
                        mass[htime] = np.array([self.halotree[k][htime]['Halo_Mass']])
                        volume[htime] = np.array([self.halotree[k][htime]['hull_volume']])
                    else:
                        radii[htime] = np.append(radii[htime],self.halotree[k][htime]['Halo_Radius'])
                        halo_time[htime] = np.append(halo_time[htime],k)
                        com[htime] = np.vstack((com[htime],self.halotree[k][htime]['Halo_Center']))
                        vcom[htime] = np.vstack((vcom[htime],self.halotree[k][htime]['Vel_Com']))
                        mass[htime] = np.append(mass[htime],self.halotree[k][htime]['Halo_Mass'])
                        volume[htime] = np.append(volume[htime],self.halotree[k][htime]['hull_volume'])
            # arg = np.argsort(end_length_2)
            # end_time = end_time[arg].astype(int)
            # end_com = end_com[arg]
            # start_time = start_time[arg].astype(int)
            # start_com = start_com[arg]
            # change_list = change_list[arg]
            start_time = start_time.astype(int)
            end_time = end_time.astype(int)
            for i,k0 in enumerate(change_list):
                dist_end = np.linalg.norm(com[end_time[i]]-end_com[i],axis=1)
                dist_start = np.linalg.norm(com[start_time[i]]-start_com[i],axis=1)
                bool_same = halo_time[end_time[i]] != k0
                bool_dist = dist_end < 0.8*radii[end_time[i]]
                bool_start = halo_time[start_time[i]][dist_start < 0.8*radii[start_time[i]]]
                smaller = False
                for k1 in halo_time[end_time[i]][bool_same*bool_dist]:
                    if not smaller and k1 != k0 and k1 not in dellist:
                        if k1 in bool_start and k_mass[k1].sum() > 2*k_mass[k0].sum():
                            smaller = True
                if smaller and abs(end_time[i]-start_time[i]) > 0 and len(self.all_times) >10:
                    dellist.append(k0)
            for i,k0 in enumerate(change_list):
                if k0 not in dellist and len(self.idl[k0].keys()):
                    #dist_end = np.linalg.norm(com[end_time[i]]-end_com[i],axis=1)
                    dist_start = np.linalg.norm(com[start_time[i]]-start_com[i],axis=1)
                    dist_mid = np.linalg.norm(com[mid_time[i]]-mid_com[i],axis=1)
                    #bool_dist = dist_end < 0.3*radii[end_time[i]]
                    bool_start_1 = halo_time[start_time[i]][dist_start < 0.4*radii[start_time[i]]]
                    bool_start_2 = halo_time[mid_time[i]][dist_mid < 0.4*radii[mid_time[i]]]
                    #print(bool_start_1,bool_start_2,k0)
                    bool_start = np.unique(np.append(bool_start_1,bool_start_2))
                    count = 0
                    for k1 in bool_start:
                        j = np.where(change_list==k1)[0][0]
                        k0_ind = np.where(halo_time[min(start_time[j],start_time[i])]==k0)[0]
                        if k1 != k0 and count ==0 and k1 not in dellist and len(self.idl[k1].keys()) >0:# and start_time[i]==start_time[j]:
                            k0_ind = np.where(halo_time[min(start_time[j],start_time[i])]==k0)[0]
                            k1_ind = np.where(halo_time[min(start_time[j],start_time[i])]==k1)[0]
                            #bool_same_rad_1 = abs(radii[start_time[i]][k0_ind]/radii[start_time[i]][k1_ind]-1) <0.3
                            mass1 = mass[min(start_time[j],start_time[i])][k0_ind]
                            mass2 = mass[min(start_time[j],start_time[i])][k1_ind]
                            mass_same = abs(mass1/mass2 - 1) < 0.2
                            #print(mass_same)
                            if mass_same.size > 0:
                                bool_smaller = (not mass_same and k_mass[k1].sum() > k_mass[k0].sum()) or \
                                     (mass_same and start_time[j] > start_time[i]) or (mass_same and start_time[j] == start_time[i] and k_mass[k1].sum() > k_mass[k0].sum())
                            else:
                                bool_smaller = False
                            #bool_smaller = mass[min(start_time[j],start_time[i])][k0_ind] < k_mass[k0].sum() and end_length[k1] > end_length[k0]#mass_same and
                            if bool_smaller:# and bool_same_rad:
                                #print(k0,k1,mass1,mass2)
                                for htime_2 in np.arange(max(end_time[i],end_time[j]),max(start_time[i],start_time[j])+1):
                                  # if [k1,htime_2] in  dellist2:
                                  #     print([k1,htime_2])
                                  # else:
                                  #     print([k1,htime_2],dellist2)
                                  if htime_2 in halo_time and [k1,htime_2] not in dellist2 and htime_2 in self.idl[k0] and htime_2 in self.idl[k1]:
                                    if k0 in halo_time[htime_2] and k1 in halo_time[htime_2]:
                                        k0_ind_2 = np.where(halo_time[htime_2]==k0)[0]
                                        k1_ind_2 = np.where(halo_time[htime_2]==k1)[0]
                                        # if htime_2 in radii:
                                          # if k0_ind_2 in radii[htime_2] and k1_ind_2 in radii[htime_2]:
                                        if count <3:
                                            bool_close_2 = np.linalg.norm(com[htime_2][k0_ind_2]-com[htime_2][k1_ind_2])/radii[htime_2][k0_ind_2] < 0.2
                                            bool_same_rad_2 = abs(radii[htime_2][k0_ind_2]/radii[htime_2][k1_ind_2]-1) <0.15
                                            if volume[htime_2][k0_ind_2] != None and volume[htime_2][k1_ind_2] != None:
                                                bool_same_volume = abs(volume[htime_2][k0_ind_2]/volume[htime_2][k1_ind_2]-1) <0.15
                                            else:
                                                bool_same_volume = False
                                            v0 = vcom[htime_2][k0_ind_2][0]
                                            v1 = vcom[htime_2][k1_ind_2][0]
                                            normv0 = np.linalg.norm(v0)
                                            normv1 = np.linalg.norm(v1)
                                            d_vcom = abs(np.dot(v1,v0)/(normv0*normv1) -1)
                                            bool_same_vcom = d_vcom < 0.01 and abs(normv0/normv1-1) < 0.05
                                            #print(d_vcom,v0,v1)
                                            if (bool_close_2 and bool_same_vcom) or (bool_close_2 and bool_same_rad_2) or (bool_close_2 and bool_same_volume): #bool_same_rad_2 and
                                                count+=1
                                  if count >=3 and htime_2 in self.idl[k0]:
                                        dellist2.append([k0,htime_2])
                                        k_mass[k0][np.where(k_times[k0]==htime_2)[0]] = 0
                                start_time[i] = k_times[k0][k_mass[k0]>0].max()
                                end_time[i] = k_times[k0][k_mass[k0]>0].min()
            for halo in dellist:
                del self.halotree[halo]
                del self.idl[halo]
                del self.hullv[halo]
                if find_stars and rank==0:
                    if halo in self.star_id['ids']:
                        del self.star_id['ids'][halo]
                        del self.star_id['energies'][halo]
                if rank==0 and self.verbose:
                    print('Halo %s removed' % halo)
            for dels in dellist2:
                del self.halotree[dels[0]][dels[1]]
                del self.idl[dels[0]][dels[1]]
                del self.hullv[dels[0]][dels[1]]
                if find_stars and rank==0:
                    if dels[0] in self.star_id['ids']:
                        if dels[1] in self.star_id['ids'][dels[0]]:
                            del self.star_id['ids'][dels[0]][dels[1]]
                            del self.star_id['energies'][dels[0]][dels[1]]
                if rank==0 and self.verbose:
                    print('Halo %s, timestep %s removed' % (dels[0],dels[1]))
            dellist3 = []
            for halo in self.idl:
              if len(list(self.idl[halo].keys())) >0:
                min_halo_time = min(list(self.idl[halo].keys()))
                if len(self.idl[halo]) <= 5:# and min_halo_time > timestepi + 2*self.direction and (min_halo_time > self.interval/2 or len(self.all_times) <10):
                    dellist3.append(halo)
              else:
                  dellist3.append(halo)
            for halo in dellist3:
                del self.halotree[halo]
                del self.idl[halo]
                del self.hullv[halo]
                if find_stars and rank==0:
                    if halo in self.star_id['ids']:
                        del self.star_id['ids'][halo]
                        del self.star_id['energies'][halo]
                if rank==0 and self.verbose:
                    print('Halo %s removed' % halo)
            if rank ==0:
                print('%s halos remain' % len(self.halotree))

    def restart(self,min_time):
        halos = list(self.halotree.keys())
        halos_new = {}
        idl_new = {}
        hullv_new = {}
        star_id_ids_new = {}
        star_id_energies_new = {}
        for halo in halos:
          if max(list(self.idl[halo].keys())) > min_time:
            halos_new[halo] = {}
            idl_new[halo] = {}
            for time in self.idl[halo]:
                if time > min_time:
                    if rank ==0 and find_stars:
                        if halo in self.star_id['ids']:
                            star_id_ids_new[halo] = {}
                            star_id_ids_energies_new[halo] = {}
                            if time in self.star_id['ids'][halo]:
                                star_id_ids_new[halo][time] = self.star_id['ids'][halo][time]
                                star_id_energies_new[halo][time] = self.star_id['energies'][halo][time]
                    idl_new[halo][time] = self.idl[halo][time]
                    hullv_new[halo][time] = self.hullv[halo][time]
                    halos_new[halo][time] = self.halotree[halo][time]
        for halo in halos_new:
            for index in self.halotree[halo]:
                if index not in self.idl[halo]:
                    if min(list(self.idl[halo].keys())) > min_time:
                        halos_new[halo][index] = self.halotree[halo][index]
                    elif index == "Consec_Forced":
                        halos_new[halo][index] = max(self.halotree[halo][index]-3,0)
                    else:
                        halos_new[halo][index] = self.halotree[halo][index]
        self.halotree = halos_new
        self.idl = idl_new
        self.hullv = hullv_new
        if rank ==0 and find_stars:
            self.star_id['ids'] = star_id_ids_new
            self.star_id['energies'] = star_id_energies_new

    def connect_halos(self,timing=10):
        halos = list(self.halotree.keys())
        halos_new = {}
        idl_new = {}
        hullv_new = {}
        star_id_ids_new = {}
        star_id_energies_new = {}
        change_list = np.array([])
        end_mass = np.array([])
        end_time = np.array([])
        end_com = np.array([])
        masses = {}
        radii = {}
        halo_time = {}
        com = {}
        numprog = {}
        for k in halos:
            if len(k.split('_')) == 1:
                change_list = np.append(change_list,k)
                current_time = max(list(self.idl[k].keys()))
                end_time = np.append(end_time,current_time)
                end_mass = np.append(end_mass,self.halotree[k][current_time]['Halo_Mass'])
                if len(end_com) ==0:
                    end_com = np.array([self.halotree[k][current_time]['Halo_Center']])
                else:
                    end_com = np.vstack((end_com,self.halotree[k][current_time]['Halo_Center']))
            for htime in list(self.idl[k].keys()):
                if not htime in masses:
                    masses[htime] = np.array([self.halotree[k][htime]['Halo_Mass']])
                    radii[htime] = np.array([self.halotree[k][htime]['Halo_Radius']])
                    halo_time[htime]  = np.array([k])
                    com[htime] = np.array([self.halotree[k][htime]['Halo_Center']])
                else:
                    masses[htime] = np.append(masses[htime],self.halotree[k][htime]['Halo_Mass'])
                    radii[htime] = np.append(radii[htime],self.halotree[k][htime]['Halo_Radius'])
                    halo_time[htime] = np.append(halo_time[htime],k)
                    com[htime] = np.vstack((com[htime],self.halotree[k][htime]['Halo_Center']))
            numprog[k] = self.halotree[k]['Num_Progs']
        if self.verbose and rank==0:
            print('Collected centers, masses, radii, and times')
        arg = np.argsort(end_mass)
        change_list = change_list[arg]
        end_mass = end_mass[arg]
        end_time = end_time[arg]
        end_com = end_com[arg]
        replacement = {}
        for i,k0 in enumerate(change_list):
            dist = np.linalg.norm(com[end_time[i]]-end_com[i],axis=1)
            bool_same = halo_time[end_time[i]] != k0
            bool_dist = dist < 0.8*radii[end_time[i]]
            bool_mass = self.halotree[k0]['Pre_Infall_Mass'] < 0.8*masses[end_time[i]]
            bool_total = bool_same*bool_dist*bool_mass
            if bool_total.sum() > 0:
                overlap = []
                for k1 in halo_time[end_time[i]][bool_total]:
                    overlap.append(np.isin(self.idl[k0][end_time[i]],self.idl[k1][end_time[i]]).sum()/len(self.idl[k0][end_time[i]]))
                overlap = np.array(overlap)
                bool_over = overlap > 0
                if bool_over.sum() > 0:
                    dist_desc = dist[bool_total][bool_over].min()
                    over_ind = np.arange(len(overlap[bool_over]))[dist[bool_total][bool_over]==dist_desc]
                    desc = halo_time[end_time[i]][bool_total][bool_over][over_ind]
                    if len(desc) != 0:
                        if desc[0] in replacement:
                            desc_k0 = replacement[desc[0]][1]
                        else:
                            desc_k0 = desc[0]
                        new_k = desc_k0+'_'+str(numprog[desc_k0])
                        if new_k.split('_')[0] != k0:
                            replacement[k0] = [new_k,desc_k0]
                            numprog[desc_k0] += 1
                            numprog[new_k] = 0
                        # if rank==0 and self.verbose:
                        #     print(k0,new_k,numprog[desc_k0],numprog[new_k])
        if self.verbose and rank==0:
            print('Calculated replacement list')
        repeat = True
        while repeat:
            lenr = 0
            rhalo_list = list(replacement.keys())
            for rhalo in replacement:
              replace = False
              for i in range(len(replacement[rhalo][1].split('_'))+1)[::-1]:
                if not replace:
                    chain = '_'.join(replacement[rhalo][1].split('_')[0:i])
                if chain in replacement and not replace:
                    stop = False
                    if replacement[chain][1] in replacement:
                        if chain == replacement[replacement[chain][1]][1]:
                            stop = True
                    if not stop:
                        new_k0 = replacement[rhalo][1].replace(chain,replacement[chain][0])
                        if new_k0 in numprog:
                            new_k = new_k0+'_'+str(numprog[new_k0])
                        else:
                            new_k = new_k0+'_'+str(0)
                            numprog[new_k0] = 0
                        numprog[new_k0] += 1
                        numprog[new_k] = 0
                        replacement[rhalo] = [new_k,new_k0]
                        lenr += 1
                        replace = True
            if lenr == 0:
                repeat = False
        if self.verbose and rank==0:
            print('Calculated replacement list for deeper levels')
        for k in halos:
            halo_split = k.split('_')
            if halo_split[0] in replacement and len(halo_split) >1:
                new_k = replacement[halo_split[0]][0]+"_"+"_".join(halo_split[1:])
                new_k0 = "_".join(new_k.split('_')[:-1])
                if new_k0 not in numprog:
                    if "_".join(halo_split[:-1]) in numprog:
                        numprog[new_k0] = numprog["_".join(halo_split[:-1])]
                    else:
                        numprog[new_k0] = 0
                replacement[k] = [new_k,1]
                numprog[new_k] = numprog[k]
                numprog[new_k0] += 1
        if self.verbose and rank==0:
            print('Created new labels for halos')
        for k in halos:
            if k in replacement:
                hullv_new[replacement[k][0]] = self.hullv[k]
                halos_new[replacement[k][0]] = self.halotree[k]
                idl_new[replacement[k][0]] = self.idl[k]
                halos_new[replacement[k][0]]['Num_Progs'] = numprog[replacement[k][0]]
                if rank ==0 and find_stars:
                    if k in self.star_id['ids']:
                        star_id_ids_new[replacement[k][0]] = self.star_id['ids'][k]
                        star_id_energies_new[replacement[k][0]] = self.star_id['energies'][k]
                if rank==0 and self.verbose:
                    print('Halo',k,'replaced by',replacement[k][0],'with',numprog[replacement[k][0]],'progentor(s)')
            else:
               halos_new[k] = self.halotree[k]
               halos_new[k]['Num_Progs'] = numprog[k]
               idl_new[k] = self.idl[k]
               hullv_new[k] = self.hullv[k]
               if rank == 0 and find_stars:
                   if k in self.star_id['ids']:
                       star_id_ids_new[k] = self.star_id['ids'][k]
                       star_id_energies_new[k] = self.star_id['energies'][k]
        self.halotree = halos_new
        self.idl = idl_new
        self.hullv = hullv_new
        if rank ==0 and find_stars:
            self.star_id['ids'] = star_id_ids_new
            self.star_id['energies'] = star_id_energies_new


    def prefall_mass(self):
        halos = np.array(list(self.halotree.keys()))
        masses = {}
        radii = {}
        halo_time = {}
        com = {}
        for k in halos:
            for htime in list(self.idl[k].keys()):
                if not htime in masses:
                    if self.halotree[k][htime]['bound_mass'] != None:
                        masses[htime] = np.array([self.halotree[k][htime]['Halo_Mass']])
                    else:
                        masses[htime] = np.array([self.halotree[k][htime]['Halo_Mass']])
                    radii[htime] = np.array([self.halotree[k][htime]['Halo_Radius']])
                    halo_time[htime]  = np.array([k])
                    com[htime] = np.array([self.halotree[k][htime]['Halo_Center']])
                else:
                    if self.halotree[k][htime]['bound_mass'] != None:
                        masses[htime] = np.append(masses[htime],self.halotree[k][htime]['Halo_Mass'])
                    else:
                        masses[htime] = np.append(masses[htime],self.halotree[k][htime]['Halo_Mass'])
                    #masses[htime] = np.append(masses[htime],self.halotree[k][htime]['Halo_Mass'])
                    radii[htime] = np.append(radii[htime],self.halotree[k][htime]['Halo_Radius'])
                    halo_time[htime] = np.append(halo_time[htime],k)
                    com[htime] = np.vstack((com[htime],self.halotree[k][htime]['Halo_Center']))
        pre_infall_mass = {}
        end_mass = {}
        for k in halos:
            found_mass = False
            ordered_time = np.array(list(self.idl[k].keys()))
            arg_time = np.argsort(ordered_time)[::-1]
            ordered_time = ordered_time[arg_time]
            for htime in ordered_time:
              if not found_mass:
                if htime == max(list(self.idl[k].keys())):
                    if self.halotree[k][htime]['bound_mass'] != None:
                        end_mass[k] = self.halotree[k][htime]['Halo_Mass']
                    else:
                        end_mass[k] = self.halotree[k][htime]['Halo_Mass']
                if not found_mass:
                    if self.halotree[k][htime]['bound_mass'] != None:
                        mass_bool = masses[htime] >= 0.9*self.halotree[k][htime]['Halo_Mass']
                    else:
                        mass_bool = masses[htime] >= 0.9*self.halotree[k][htime]['Halo_Mass']
                    if len(masses[htime]>0):
                        dist = np.linalg.norm(self.halotree[k][htime]['Halo_Center']-com[htime],axis = 1)
                    else:
                        dist = np.linalg.norm(self.halotree[k][htime]['Halo_Center']-com[htime])
                    dist_bool = dist < 1.5*radii[htime]
                    bool_same = halo_time[htime] != k
                    bool_close = bool_same*mass_bool*dist_bool
                    if bool_close.sum() ==0:
                        found_mass = True
                        if self.halotree[k][htime]['bound_mass'] != None:
                            pre_infall_mass[k] = self.halotree[k][htime]['Halo_Mass']
                        else:
                            pre_infall_mass[k] = self.halotree[k][htime]['Halo_Mass']
            if not found_mass and self.halotree[k]['Solved'][1] in self.halotree[k]:
                timestep_1 = self.halotree[k]['Solved'][1]
                if self.halotree[k][timestep_1]['bound_mass'] != None:
                    pre_infall_mass[k] = self.halotree[k][timestep_1]['Halo_Mass']
                else:
                    pre_infall_mass[k] = self.halotree[k][timestep_1]['Halo_Mass']
            elif not found_mass:
                timestep_2 = max(list(self.idl[k].keys()))
                if self.halotree[k][timestep_2]['bound_mass'] != None:
                    pre_infall_mass[k] = self.halotree[k][timestep_2]['Halo_Mass']
                else:
                    pre_infall_mass[k] = self.halotree[k][timestep_2]['Halo_Mass']
                #pre_infall_mass[k] = self.halotree[k][max(list(self.idl[k].keys()))]['Halo_Mass']
            # if rank==0 and self.verbose:
            #     if end_mass[k] != pre_infall_mass[k]:
            #         print(end_mass[k],pre_infall_mass[k],k)
            self.halotree[k]['Pre_Infall_Mass'] = pre_infall_mass[k]


    def clear_halos(self):
        halos = np.array(list(self.halotree.keys()))
        end_mass = np.array([])
        for k in halos:
            current_time = max(list(self.idl[k].keys()))
            #end_mass = np.append(end_mass,self.halotree[k][current_time]['Halo_Mass'])
            end_mass = np.append(end_mass,self.halotree[k]['Pre_Infall_Mass'])
        arg = np.argsort(end_mass)[::-1]
        halos_new = {}
        idl_new = {}
        hullv_new = {}
        star_id_ids_new = {}
        star_id_energies_new = {}
        new_list = np.arange(len(halos)).astype(str)
        for i in range(len(arg)):
            halos_new[new_list[i]] = self.halotree[halos[arg[i]]]
            idl_new[new_list[i]] = self.idl[halos[arg[i]]]
            halos_new[new_list[i]]['Num_Progs'] = 0
            hullv_new[new_list[i]] = self.hullv[halos[arg[i]]]
            if rank ==0 and find_stars:
                if halos[arg[i]] in  self.star_id['ids']:
                    star_id_ids_new[new_list[i]] = self.star_id['ids'][halos[arg[i]]]
                    star_id_energies_new[new_list[i]] = self.star_id['energies'][halos[arg[i]]]
        self.halotree = halos_new
        self.idl = idl_new
        self.hullv = hullv_new
        if rank ==0 and find_stars:
            self.star_id['ids'] = star_id_ids_new
            self.star_id['energies'] = star_id_energies_new


    def reorder_halos(self):
        halos = np.array(list(self.halotree.keys()))
        halos_new = {}
        idl_new = {}
        hullv_new = {}
        star_id_ids_new = {}
        star_id_energies_new = {}
        max_len = 0
        replacement = {}
        prog_list = {}
        expanded_list = []
        prog = {}
        end_time = np.array([])
        end_mass = np.array([])
        len_list = np.array([])
        for k in halos:
            max_len = max(max_len,len(k.split('_')))
            current_time = max(list(self.idl[k].keys()))
            end_time = np.append(end_time,current_time)
            #end_mass = np.append(end_mass,self.halotree[k][current_time]['Halo_Mass'])
            end_mass = np.append(end_mass,self.halotree[k]['Pre_Infall_Mass'])
            len_list = np.append(len_list,len(k.split('_')))
        arg1 = np.argsort(end_time)[::-1]
        end_mass = end_mass[arg1]
        end_time = end_time[arg1]
        len_list = len_list[arg1]
        halos = halos[arg1]
        main_halos = halos[len_list==1]
        arg2 = np.argsort(end_mass[len_list==1])[::-1]
        main_halos = main_halos[arg2].tolist()
        maxhalo = 0
        for k in halos:
            expanded_list.append(k.split('_'))
            prog_list[k] = 0
        for root in range(max_len):
            for e_k in expanded_list:
                if len(e_k) == root+1:
                    if root ==0:
                        halo_k0 = "_".join(e_k)
                        new_halo = str(main_halos.index(e_k[0]))
                        replacement[halo_k0] = str(new_halo)
                        prog[str(new_halo)] = 0
                        maxhalo = max(maxhalo,main_halos.index(e_k[0]))
                    else:
                        halo_k = "_".join(e_k[0:root])
                        halo_k0 = "_".join(e_k)
                        if halo_k in replacement:
                            halo_k = replacement[halo_k]
                        elif halo_k not in prog:
                            halo_k = str(maxhalo+1)
                            maxhalo += 1
                            prog[halo_k] = 0
                        new_halo = halo_k+'_'+str(prog[halo_k])
                        replacement[halo_k0] = new_halo
                        prog[halo_k] += 1
                        prog[new_halo] = 0
        replace_list = natsorted(list(replacement.keys()))
        for old_halo in replace_list:
            new_halo = replacement[old_halo]
            halos_new[new_halo] = self.halotree[old_halo]
            idl_new[new_halo] = self.idl[old_halo]
            hullv_new[new_halo] = self.hullv[old_halo]
            halos_new[new_halo]['Num_Progs'] = prog[new_halo]
            if rank == 0 and find_stars:
                if old_halo in self.star_id['ids']:
                    star_id_ids_new[new_halo] = self.star_id['ids'][old_halo]
                    star_id_energies_new[new_halo] = self.star_id['energies'][old_halo]
            if rank==0 and self.verbose:
                #print(set(list(self.halotree.keys()))-set(list(self.hullv.keys())))
                print(old_halo,'replaced by',new_halo,'with',prog[new_halo],'progenitor(s)')
        self.halotree = halos_new
        self.idl = idl_new
        self.hullv = hullv_new
        if rank ==0 and find_stars:
            self.star_id['ids'] = star_id_ids_new
            self.star_id['energies'] = star_id_energies_new

    def make_region_list(self):
        region_list = []
        all_times = np.array(self.all_times)
        for times in all_times:
            #len_now = len(all_times[all_times>=times])
            #if (len_now%self.interval ==0 or times == max(all_times)) and times >0:
            region_list.append(times)
            if not os.path.exists(savestring + '/' + 'Refined/'+ 'refined_region_%s.npy' % (times)):
                    fr = Find_Refined(code, fld_list, times, savestring + '/' + 'Refined')
        region_list = comm.bcast(region_list,root=0)
        region_list.sort()
        region_list = np.array(region_list)
        ll_reg = {}
        ur_reg = {}
        for reg_time in region_list:
            fr = np.load(savestring + '/' + 'Refined/' + 'refined_region_%s.npy' % (reg_time),allow_pickle=True).tolist()
            ll_reg[reg_time],ur_reg[reg_time] = fr
        ll_all = {}
        ur_all = {}
        self.time = {}
        for time in self.all_times:
            region_idx = region_list[region_list >=time].min()
            ll_all[time] = ll_reg[region_idx]
            ur_all[time] = ur_reg[region_idx]
            self.time[time] = time
            if time not in self.fof:
                self.fof[time] = False
        return ll_all,ur_all


    # Makes folders and directories as needed

    def make_arbor(self):
        my_storage = {}
        if rank==0:
            if not os.path.exists(savestring + '/arbor_tree.h5'):
                if os.path.exists(string + '/rockstar_halos' + '/trees/' + 'tree_0_0_0.dat'):
                    halos = ytree.load(string + '/rockstar_halos' + '/trees/' + 'tree_0_0_0.dat')
                    fn = halos.save_arbor(filename=savestring + '/arbor_tree.h5')
        # for sto, times in yt.parallel_objects(range(len(fld_list)),nprocs, storage = my_storage):
        #     if not os.path.exists(savestring + '/' + sim + '/arbor_%s.h5' % (times+offset)):
        #         if os.path.exists(string + '/' + sim + '/rockstar_halos' + '/' + 'out_%s.list' % (times+offset)):
        #            halos = ytree.load(string + '/' + sim + '/rockstar_halos' + '/' + 'out_%s.list' % (times+offset))
        #            fn = halos.save_arbor(filename=savestring + '/' + sim + '/arbor_%s.h5' % (times+offset))


    # Converts images of halos into videos in parallel with one core per halo
    def make_avi(self,erase=False):
        partf = 'particle_plots_%s' % fldn
        savestg = savestring + '/'+ partf + '/'
        my_storage = {}
        plot_list = np.array([])
        plot_list = np.append(plot_list,self.plot_list)
        for sto, halo in yt.parallel_objects(plot_list,self.nprocs, storage = my_storage):
            try:
                video_name = savestg+'Particle_video-%s.avi' % halo
                video_name_2 = savestg+'Particle_video-%s.mp4' % halo
                files = glob.glob(savestg+'Particle_Plot_%s_*_vid.png' % (halo))
                files = natsorted(files)
                frame = cv2.imread(files[0])
                height, width, layers = frame.shape
                video = cv2.VideoWriter(video_name, 0, 10, (width,height))
                for file in files:
                    video.write(cv2.imread(file))
                cv2.destroyAllWindows()
                video.release()
                if os.path.exists(video_name_2):
                    os.system('rm'+' '+video_name_2)
                convert_avi_to_mp4(video_name,savestg+'Particle_video-%s' % halo)
                if erase:
                    os.system('rm' + ' ' + savestg + 'Particle_Plot_%s_*_vid.png' % halo)
                    os.system('rm' + ' ' + video_name)
            except:
                pass
        plot_list = comm.bcast(plot_list,root=0)

    # Creates plots of targets halos and their surrounding enviornment
    def plot_halo_2(self):
        self.colors = list(mcolors.TABLEAU_COLORS)[1:]
        self.color_0 = list(mcolors.TABLEAU_COLORS)[0]
        #self.meter = self.ds.length_unit.in_units('m').v
        #ensure_dir(self.savestg)
        if os.path.exists(savestring + '/arbor_tree.h5') and self.compare:
            self.halo_0 = []
            self.halos_ar = ytree.load(savestring +  '/arbor_tree.h5')
            self.halo_tree_list = self.halos_ar['Tree_root_ID']
        # Index to insure that halos are plotted with consistent colors even as the haloList
        # changes
        index = np.arange(len(self.halotree))
        plot_list = np.array([])
        plot_list = np.append(plot_list,self.plot_list)
        for halo in plot_list:
                if halo != 'all' and len(self.plot_list) >0:
                    if halo in self.halotree:
                      if self.timestep in self.halotree[halo] and  self.lenhalo[halo] >= 3:
                            # Halo plots all particles around the target halo and saves the size of the region
                            # so that the boundaries are consitently large
                            tree_halo = self.halotree[halo][self.timestep]
                            rvir = np.array(tree_halo['Halo_Radius'])
                            center = np.array(tree_halo['Halo_Center'])
                            if halo not in self.rvir_0:
                                self.rvir_0[halo] = self.plot_width*rvir
                            pos_min = center - self.rvir_0[halo]
                            pos_max = center + self.rvir_0[halo]
                            self.plotting(np.array(pos_min),np.array(pos_max),halo,index)
                else:
                    pos_min,pos_max = self.ll_all[self.timestep],self.ur_all[self.timestep]
                    self.plotting(np.array(pos_min),np.array(pos_max),halo,index)


    # Creates plots of targets halos and their surrounding enviornment
    def plot_halo(self):
        self.colors = list(mcolors.TABLEAU_COLORS)[1:]
        self.color_0 = list(mcolors.TABLEAU_COLORS)[0]
        #self.meter = self.ds.length_unit.in_units('m').v
        #ensure_dir(self.savestg)
        if os.path.exists(savestring +  '/arbor_tree.h5') and self.compare:
            self.halo_0 = []
            self.halos_ar = ytree.load(savestring + '/arbor_tree.h5')
            self.halo_tree_list = self.halos_ar['Tree_root_ID']
        # Index to insure that halos are plotted with consistent colors even as the haloList
        # changes
        index = np.arange(len(self.halotree))
        plot_list = np.array([])
        plot_list = np.append(plot_list,self.plot_list)
        my_storage = {}
        for sto, halo in yt.parallel_objects(plot_list,min(self.nprocs,30), storage = my_storage):
                if halo != 'all' and len(self.plot_list) >0:
                    if halo in self.halotree:
                      if self.timestep in self.halotree[halo] and self.lenhalo[halo]:
                            # Halo plots all particles around the target halo and saves the size of the region
                            # so that the boundaries are consitently large
                            tree_halo = self.halotree[halo][self.timestep]
                            rvir = np.array(tree_halo['Halo_Radius'])
                            center = np.array(tree_halo['Halo_Center'])
                            if halo not in self.rvir_0:
                                self.rvir_0[halo] = self.plot_width*rvir
                            pos_min = center - self.rvir_0[halo]
                            pos_max = center + self.rvir_0[halo]
                            self.plotting(np.array(pos_min),np.array(pos_max),halo,index)
                else:
                    pos_min,pos_max = self.ll_all[self.timestep],self.ur_all[self.timestep]
                    self.plotting(np.array(pos_min),np.array(pos_max),halo,index)
        plot_list = comm.bcast(plot_list,root=0)

    def plotting(self,ll,ur,halo,index,multi=True,compare=True):
                width = (ur-ll).max()
                center = (ll+ur)/2
                non_sphere_0 = non_sphere
                if resave:
                    mass,pos,vel,ids = pickup_particles_backup(self.timestep,ll-width*.2,ur+width*.2,self.ds,stars=False)
                else:
                    reg = self.ds.region(center,ll-width*.2,ur+width*.2)
                    mass,pos,vel,ids = pickup_particles(reg,self.codetp,stars=False)
                    del reg
                del mass
                del vel
                pos = pos/self.meter
                pos_in = (np.sum(pos > ll,axis=1)==3)*(np.sum(pos < ur,axis=1)==3)
                pos,ids = pos[pos_in],ids[pos_in]
                comps = [0]
                idl = np.load(savestring + '/ridl_%s.npy' % (fldn),allow_pickle=True).tolist()[self.timestep]
                path1 = os.path.exists(savestring + '/arbor_tree.h5')
                path2 = os.path.exists(savestring +  '/arbor_%s.h5' % (self.timestep+offset))
                titles = ['This Work','Rockstar','Rockstar+CT']
                # if compare and path2 and not path1:
                #     fig,ax = plt.subplots(2,3,figsize=(27,18),dpi=100)
                #     comps = [0,1]
                #     qcom = [0,1]
                # elif compare and path2 and path1:
                #     fig,ax = plt.subplots(3,3,figsize=(27,27),dpi=100)
                #     comps = [0,1,2]
                #     qcom = [0,1,2]
                #     halos_ar = ytree.load(savestring + '/' + sim + '/arbor_tree.h5')
                #     halo_tree_list = halos_ar['Tree_root_ID']
                if compare and path1 and self.compare:
                    fig,ax = plt.subplots(2,3,figsize=(27,18),dpi=100)
                    comps = [0,1]
                    qcom = [0,2]
                else:
                    fig,ax = plt.subplots(1,2,figsize=(18,9),dpi=100)
                    ax = [ax]
                    qcom = [0]
                if len(pos) > 10000:
                    cut = int(np.ceil(len(pos)/10000))
                else:
                    cut = 1
                trans = min(10000*cut/(len(pos)),.02)
                for q in comps:
                    ax[q][0].scatter(pos[::cut,0],pos[::cut,1],color=self.color_0,alpha=trans,marker=".")
                    ax[q][1].scatter(pos[::cut,0],pos[::cut,2],color=self.color_0,alpha=trans,marker=".")
                    #ax[q][2].scatter(pos[::cut,1],pos[::cut,2],color=self.color_0,alpha=trans,marker=".")
                    ax[q][0].set_xlim(ll[0],ur[0])
                    ax[q][0].set_ylim(ll[1],ur[1])
                    ax[q][1].set_xlim(ll[0],ur[0])
                    ax[q][1].set_ylim(ll[2],ur[2])
                    #ax[q][2].set_xlim(ll[1],ur[1])
                    #ax[q][2].set_ylim(ll[2],ur[2])
                    #ax[q][0].set_title('%s : %s Myr' % (titles[qcom[q]],np.round(self.ds.current_time.in_units('Myr'),2)))
                    ax[q][0].set_xlabel('x')
                    ax[q][0].set_ylabel('y')
                    ax[q][1].set_xlabel('x')
                    ax[q][1].set_ylabel('z')
                    #ax[q][2].set_xlabel('y')
                    #ax[q][2].set_ylabel('z')
                    ax[q][0].set_aspect('equal')
                    ax[q][1].set_aspect('equal')
                    #ax[q][2].set_aspect('equal')
                    coms = np.array([])
                    halo_num = []
                    rads = []
                    halo_color = []
                    halo_name = []
                    halo_cden = []
                    hulls = {}
                    # Loads all the halos and assigns them their unique color
                    if qcom[q]==0:
                        for i,halos in enumerate(list(self.halotree.keys())):
                            if self.timestep in self.halotree[halos] and self.lenhalo[halos] >= 1:
                                if len(coms) >0:
                                    coms = np.vstack((coms,self.halotree[halos][self.timestep]['Halo_Center']))
                                else:
                                    coms = self.halotree[halos][self.timestep]['Halo_Center']
                                halo_num.append(halos)
                                rads0 = np.array([key[1:] for key in self.halotree[halos][self.timestep].keys() if key[0]=='r'])
                                rads0 = rads0[np.argsort(rads0.astype(float))]
                                rad_avg = self.halotree[halos][self.timestep]['Halo_Radius']
                                cden0 = self.halotree[halos][self.timestep]['cden']
                                # if not non_sphere or len(self.hullv[halos][self.timestep])>0:
                                #     if (rads0.astype(float)>=199).sum() >0:
                                #         rad_avg = self.halotree[halos][self.timestep]['r'+str(rads0[rads0.astype(float)>=199][0])]
                                #         cden0 = int(round(float(rads0[rads0.astype(float)>=199][0])))
                                rads.append(rad_avg)
                                halo_color.append(index[i])
                                halo_name.append(halos)
                                halo_cden.append(cden0)
                                # print(self.hullv[halos][self.timestep])
                                # print(np.isin(ids,self.hullv[halos][self.timestep]))
                                #print(pos[np.isin(ids,self.hullv[halos][self.timestep])])
                                if non_sphere and len(self.hullv[halos][self.timestep])>0:
                                    hulls[halos] = pos[np.isin(ids,self.hullv[halos][self.timestep])]
                                else:
                                    hulls[halos] = np.array([])
                        rads = np.array(rads)
                        halo_name = np.array(halo_name)
                        halo_cden = np.array(halo_cden)
                    elif qcom[q] ==1:
                        halos = ytree.load(savestring + '/arbor_%s.h5' % (self.timestep+offset))
                        coms = halos['position'].v
                        rads = (halos['virial_radius']/halos.box_size).v
                        mass = halos['mass'].v
                        bool = mass > self.minmass
                        halo_num = np.arange(bool.sum())
                        halo_color = np.arange(bool.sum()).tolist()
                        coms,rads = coms[bool],rads[bool]
                    elif qcom[q] ==2:
                         if len(self.halo_0) == 0:
                             self.halo_0 = list(self.halos_ar.select_halos("tree['tree','Snap_idx']==%s" % (self.timestep+offset)))
                         rads = np.array([])
                         mass = np.array([])
                         coms = np.array([])
                         halo_color = []
                         for halo_j in self.halo_0:
                                if len(coms) > 0:
                                    coms = np.vstack((coms,halo_j['position'].v))
                                else:
                                    coms = (halo_j['position'].v)[np.newaxis,:]
                                rads = np.append(rads,(halo_j['virial_radius']/self.halos_ar.box_size).v)
                                mass = np.append(mass,halo_j['mass'].v)
                                halo_color.append(np.where(self.halo_tree_list==halo_j['Tree_root_ID'])[0][0])
                         bool = mass > self.minmass
                         halo_num = np.arange(bool.sum())
                         halo_color = np.array(halo_color)[bool]
                         coms,rads = coms[bool],rads[bool]
                    if len(rads) >= 1:
                        if len(rads) ==1:
                            dist = np.linalg.norm(coms-center)
                        else:
                            dist = np.linalg.norm(coms-center,axis=1)
                        arg = np.argsort(dist)
                        bool_dist = (np.sum(coms[arg]+rads[arg,np.newaxis] >= ll,axis=1) == 3)*(np.sum(coms[arg]-rads[arg,np.newaxis] <= ur,axis=1) == 3)
                        halos_in = np.array(halo_num)[arg][bool_dist]
                        halo_color = np.array(halo_color)[arg][bool_dist]+1
                        #print(dist[arg],halos_in,rvir)
                        if self.verbose and ((multi == False and rank==0) or (multi == True)):
                            print(halo,halos_in)
                        # Plots all halos that fall within the region inlcuding coloring their bound paricles
                        if qcom[q]==1 or qcom[q]==2:
                            dist2 = cdist(coms,pos[::int(np.ceil(cut/5))])
                        # if rank ==0:
                        #     print(idl.keys(),halos_in)
                        for i,halo_i in enumerate(halos_in):
                          try:
                            if qcom[q]==0:
                                id_bool = np.isin(ids,idl[halo_i])
                            if qcom[q]==1 or qcom[q]==2:
                                id_bool = dist2[halo_i,:] < rads[halo_i]
                            if id_bool.sum() >= 1:#125/cut:
                                color_count = halo_color[i]
                                if qcom[q]==0:
                                    pos_halo = pos[id_bool]
                                if qcom[q] ==1 or qcom[q] ==2:
                                    pos_halo = pos[::int(np.ceil(cut/5))][id_bool]
                                center = coms[arg][bool_dist][i]
                                color = self.colors[color_count%len(self.colors)]
                                rvir = rads[arg][bool_dist][i]
                                ax[q][0].scatter(pos_halo[:,0],pos_halo[:,1],color=color,alpha=0.1,marker=".")
                                if non_sphere_0:
                                    if len(hulls[halo_i]) > 4:
                                        #print(hulls[i][0])
                                        hull = ConvexHull(hulls[halo_i][:,[0,1]])
                                        hull_vert = np.append(hull.vertices,hull.vertices[0])
                                        ax[q][0].plot(hulls[halo_i][hull_vert][:,0],hulls[halo_i][hull_vert][:,1],color=color)
                                    else:
                                        circle1 = plt.Circle((center[0],center[1]), rvir, color=color, fill=False)
                                        ax[q][0].add_patch(circle1)
                                else:
                                    circle1 = plt.Circle((center[0],center[1]), rvir, color=color, fill=False)
                                    ax[q][0].add_patch(circle1)
                                ax[q][1].scatter(pos_halo[:,0],pos_halo[:,2],color=color,alpha=0.1,marker=".")
                                if non_sphere_0:
                                    if len(hulls[halo_i]) > 4:
                                        hull = ConvexHull(hulls[halo_i][:,[0,2]])
                                        hull_vert = np.append(hull.vertices,hull.vertices[0])
                                        ax[q][1].plot(hulls[halo_i][hull_vert][:,0],hulls[halo_i][hull_vert][:,2],color=color)
                                    else:
                                        circle2 = plt.Circle((center[0],center[2]), rvir, color=color, fill=False)
                                        ax[q][1].add_patch(circle2)
                                else:
                                    circle2 = plt.Circle((center[0],center[2]), rvir, color=color, fill=False)
                                    ax[q][1].add_patch(circle2)
                                name = halo_i
                                ax[q][1].text(center[0],center[2],'%s' % (name),fontsize=9)
                          except:
                                pass
                if fake:
                            pos_list_fake = np.array(np.load(savestring +'/'+'pos_list.npy',allow_pickle=True).tolist()[self.timestep])/self.meter
                            #print(pos_list_fake)
                            bool_pos_dist = (np.sum(pos_list_fake >= ll,axis=1) == 3)*(np.sum(pos_list_fake <= ur,axis=1) == 3)
                            print(pos_list_fake[bool_pos_dist])
                            ax[q][0].scatter(pos_list_fake[bool_pos_dist][:,0],pos_list_fake[bool_pos_dist][:,1],s=40,marker='x',color='black')
                                # ax[q][2].scatter(pos_halo[:,1],pos_halo[:,2],color=color,alpha=0.1,marker=".")
                                # if non_sphere:
                                #         if len(hulls[halo_i]) > 4:
                                #             hull = ConvexHull(hulls[halo_i][:,[1,2]])
                                #             hull_vert = np.append(hull.vertices,hull.vertices[0])
                                #             ax[q][2].plot(hulls[halo_i][hull_vert][:,1],hulls[halo_i][hull_vert][:,2],color=color)
                                #         else:
                                #             circle3 = plt.Circle((center[1],center[2]), rvir, color=color, fill=False)
                                #             ax[q][2].add_patch(circle3)
                                # else:
                                #     circle3 = plt.Circle((center[1],center[2]), rvir, color=color, fill=False)
                                #     ax[q][2].add_patch(circle3)
                                # cden = round(halo_cden[arg][bool_dist][i])
                                # name = halo_name[arg][bool_dist][i]
                                # ax[q][2].text(center[1],center[2]-0.02*width,'%s Den: %s' % (name,cden),fontsize='x-small')
                if not multi:
                    if rank==0:
                        plt.savefig(self.savestg+'/'+'Particle_Plot_%s_%s_vid.png' % (halo,self.timestep))
                else:
                    plt.savefig(self.savestg+'/'+'Particle_Plot_%s_%s_vid.png' % (halo,self.timestep))
                plt.close()

    def make_lengths(self):
        self.lenhalo = {}
        self.lenhalo = comm.bcast(self.lenhalo,root=0)
        if rank ==0:
            for halo in self.halotree:
                timesteps = np.array([x for x in self.halotree[halo].keys() if isinstance(x,numbers.Integral)])
                # if self.direction>0:
                #     self.lenhalo[halo] = len(timesteps[timesteps>self.timestep])
                # else:
                #     self.lenhalo[halo] = len(timesteps[timesteps<self.timestep])
                self.lenhalo[halo] = len(timesteps)
        self.lenhalo = comm.bcast(self.lenhalo,root=0)

    # Find progentors from the new center of mass
    def determine_progenitors(self,ntries=[1,2],maxpart=1.5e50,check=False,reg=True,scale_on=False):
        idl,massl,nl,rad,com,vcom,rvir_l,cdens,cden_l,scaling_l = [],np.array([]),[],np.array([]),[],[],[],np.array([]),[],[]
        hullv = []
        more_var_l = []
        var_names_scalar = []
        var_names_vec = []
        more_vec = []
        snaps = []
        self.com_ids = {}
        len_fof = 0
        if self.fof[self.timestep] and self.direction >0:# and self.progs:
            fof_r = np.arange(len(self.dens[self.timestep].rads))
            for i in fof_r:
                if len(rad) == 0:
                     com = np.array([self.dens[self.timestep].coms[i]])
                     vcom = np.array([self.dens[self.timestep].vcom[i]])
                     rvir_l = np.array([self.dens[self.timestep].rvir_l[i]])
                     cden_l = np.array([self.dens[self.timestep].cden_l[i]])
                     more_var_l = np.array([self.dens[self.timestep].more_variables[i]])
                     nl = np.array([[-1,self.halo_num[0]]])
                else:
                    com = np.vstack((com,self.dens[self.timestep].coms[i]))
                    vcom = np.vstack((vcom,self.dens[self.timestep].vcom[i]))
                    rvir_l = np.vstack((rvir_l,self.dens[self.timestep].rvir_l[i]))
                    cden_l = np.vstack((cden_l,self.dens[self.timestep].cden_l[i]))
                    more_var_l = np.vstack((more_var_l,self.dens[self.timestep].more_variables[i]))
                    nl = np.vstack((nl,[-1,self.halo_num[0]]))
                var_names_scalar.append(self.dens[self.timestep].var_names_scalar[i])
                var_names_vec.append(self.dens[self.timestep].var_names_vec[i])
                more_vec.append(self.dens[self.timestep].more_vec[i])
                rad = np.append(rad,self.dens[self.timestep].rads[i])
                massl = np.append(massl,self.dens[self.timestep].massl[i])
                cdens = np.append(cdens,self.dens[self.timestep].cden[i])
                idl.append(self.dens[self.timestep].idl[i])
                hullv.append(self.dens[self.timestep].hullv[i])
                len_fof +=1
        self.dens[self.timestep] = None
        if len(self.rad_list) > 0:
            if len(self.mass_list) > 1000:
                bool_min_part = self.mass_list > 0 #minmass
                self.long = True
            else:
                bool_min_part = self.mass_list > 0
                self.long= False
            halo_list = np.arange(len(self.halo_num))
            self.solved_list = {}
            if rank==0 and self.tracktime:
                print(self.mass_list,np.round(self.cden).astype(int),self.halo_num)
                print('Mean Cden: %s' % self.mean_cden)
                print('Mean Forced/Length: %s, Mean Consecutive Forced: %s' % (self.forced_count.mean(),self.con_forced_count.mean()))
                if self.verbose:
                    print(self.halo_num[self.forced_count>0])
                    print(self.halo_num[self.con_forced_count>0])
                print(self.con_forced_count[self.con_forced_count>0],'Max: %s' % min(self.interval+2,len(self.all_times)/2))
                #print(len(self.rad_list),len(self.rad_list_all))
            # bool_
            # Initialization of variables describing halos in this timestep
            #m200 = np.array([])
            # Find progenitors with one core per descendant
            self.multi = False
            rad_boost = np.array([])
            for r in halo_list:
              rad_boost = np.append(rad_boost,self.get_radmax(r,3.5))
            #if rank ==0:
            #print(self.box_width,rad_boost)
            if self.direction >0 or len(self.rad_list) < 1000:
                halo_id,ll_total,ur_total,volume_list = get_all_regions_1(self.new_coml,rad_boost*self.box_width,halo_list,max_rad=self.box_width.max())
            else:
                halo_id,ll_total,ur_total,volume_list = get_all_regions_1(np.vstack((self.all_centers[:6],self.new_coml)),\
                    np.append(1.5*self.all_radii[:6],rad_boost*self.box_width),\
                    np.append(np.full(6,None),halo_list),max_rad=self.rad_list.max())
            if self.direction >0:
                minsplit= max(min(min(1000,20*nprocs),int(len(volume_list)/5)),nprocs)
            else:
                minsplit= max(min(max(min(200,nprocs*10),nprocs),int(len(volume_list)/5)),nprocs)
            # if rank ==0:
            #     print(volume_list)
            #minsplit = min(1000,self.nprocs*5)
            if len(volume_list) >0:
                split_volumes = equisum_partition(volume_list,np.maximum(int(len(volume_list)/minsplit),1))
            else:
                split_volumes = np.array([halo_list])
            #split_volumes = np.array([np.arange(len(volume_list))])
            out_round = 1
            halo_complete = []
            split_volumes = comm.bcast(split_volumes,root=0)
            for outer_ind in range(len(split_volumes)):
              if rank==0 and self.tracktime:
                  self.time_out = time_sys.time()
              out_list = split_volumes[outer_ind]
              # if rank ==0:
              #     print(volume_list[out_list],volume_list[out_list].sum())
              # if rank==0:
              #     print(split_volumes[outer_ind])
              if len(out_list) >0:
                self.halotry = {}
                h_list = np.array([])
                for i in out_list:
                    h_list = np.append(h_list,halo_id[i])
                h_list = np.unique(h_list)
                for pz in h_list:
                    if pz not in halo_complete:
                        self.halotry[pz] = [0,-1,100]
                        halo_complete.append(pz)
                if rank==0 and self.tracktime:
                    self.time9 = time_sys.time()
                self.my_storage_reg = 0
                self.my_storage_reg = get_all_regions_2(halo_id,ll_total,ur_total,out_list,\
                        self.ds,self.codetp,volume_list,nprocs=self.nprocs,verb=self.verbose,timestep=self.timestep)
                if rank==0 and self.tracktime:
                    print('Regions Created','Time:',time_sys.time()-self.time9)
                    self.time9 = time_sys.time()
                if rank==0 and self.tracktime:
                        self.time0 = time_sys.time()
                if rank==0 and self.tracktime:
                    self.time1 = time_sys.time()
                self.finished = False
                round_dh = 1
                while not self.finished:
                    current_list = np.array([])
                    for r in self.halotry:
                        if (self.halotry[r][0] < self.halotry[r][2]) and self.halotry[r][1]==-1:
                           current_list = np.append(current_list,r)
                    arg = np.argsort(self.mass_list[current_list.astype(int)])[::-1]
                    current_list = current_list[arg]
                    # if rank==0:
                    #     print(current_list)
                    if len(current_list) ==0:
                        self.finished = True
                    if not self.finished:
                        current_list = current_list.astype(int)
                        jobs = {}
                        ranks = np.arange(nprocs)
                        joblen = np.zeros(len(ranks))
                        sto = {}
                        split_jobs = {}
                        split_len = np.zeros(len(ranks))
                        for o in ranks:
                          jobs[o] = []
                          for r in current_list:
                            if r in self.my_storage_reg['rank'][o]:
                                jobs[o].append(r)
                                joblen[o] += 1
                                sto[r] = {}
                                sto[r]['rank'] = o
                        for o in jobs:
                          if joblen[o] >0 and joblen[o].max() >0:
                            split_jobs[o] = np.array_split(jobs[o],max(joblen[o].min()/5,1))
                            split_len[o] += len(split_jobs[o])
                        split_jobs = comm.bcast(split_jobs,root=0)
                        # segments = int(np.ceil(len(current_list)/(2*self.nprocs)))
                        # split_index = np.array_split(current_list,max(segments,1))
                        for split_i in range(int(split_len.max())):
                            ntries_0 = np.copy(ntries)
                            # if split_i < split_len[rank]:
                            #     print(rank,split_jobs[rank][split_i])
                            # else:
                            #     print(rank,[])
                            if split_i < split_len[rank]:
                             if self.optimization:
                                 time7 = time_sys.time()
                                 time_split = np.zeros(len(split_jobs[rank][split_i]))
                                 time_i = 0
                             for r in split_jobs[rank][split_i]:
                                if self.optimization:
                                    time8 = time_sys.time()
                                if not self.progs:
                                    halo_index = self.halo_num[r]
                                else:
                                    halo_index = r
                                #print('0',r,rank)
                            # for sto, r in yt.parallel_objects(h_index.astype(int),\
                            #     self.nprocs, storage = my_storage):
                                self.reg_info = {}
                                self.reg_info['pos'],self.reg_info['mass'],self.reg_info['vel'],self.reg_info['ids'] =\
                                 np.array([]),np.array([]),np.array([]),np.array([])
                                self.reg_info['sids'] = np.array([])
                                for y in self.my_storage_reg['halo'][r]:
                                        v = self.my_storage_reg[y]
                                        if len(self.reg_info['mass'])==0:
                                            self.reg_info['pos'] = v['pos']
                                            self.reg_info['vel'] = v['vel']
                                            if find_stars:
                                                self.reg_info['spos'] = v['spos']
                                                self.reg_info['svel'] = v['svel']
                                        else:
                                            self.reg_info['pos'] = np.vstack((self.reg_info['pos'],v['pos']))
                                            self.reg_info['vel'] = np.vstack((self.reg_info['vel'],v['vel']))
                                            if find_stars:
                                                self.reg_info['spos'] = np.vstack((self.reg_info['spos'],v['spos']))
                                                self.reg_info['svel'] = np.vstack((self.reg_info['svel'],v['svel']))
                                        self.reg_info['mass'] = np.append(self.reg_info['mass'],v['mass'])
                                        self.reg_info['ids'] = np.append(self.reg_info['ids'],v['ids'])
                                        if find_stars:
                                            self.reg_info['sids'] = np.append(self.reg_info['sids'],v['sids'])
                                        v = None
                                bool_contained = (np.sum(self.reg_info['pos']/self.meter >= self.new_coml[r]-1.25*rad_boost[r]*self.rad_list[r],axis=1) ==3)*\
                                    (np.sum(self.reg_info['pos']/self.meter <= self.new_coml[r]+1.25*rad_boost[r]*self.rad_list[r],axis=1) ==3)
                                self.reg_info['ids'] = self.reg_info['ids'][bool_contained]
                                self.reg_info['mass'] = self.reg_info['mass'][bool_contained]
                                self.reg_info['pos'] = self.reg_info['pos'][bool_contained]
                                self.reg_info['vel'] = self.reg_info['vel'][bool_contained]
                                self.reg_info['id_sample'] = self.reg_info['ids'][np.isin(self.reg_info['ids'],self.idl_all)]
                                self.reg_info['halo'] = self.halo_num[r]
                                if find_stars:
                                    if len(self.reg_info['sids'])>0:
                                        bool_contained = (np.sum(self.reg_info['spos']/self.meter >= self.new_coml[r]-rad_boost[r]*self.rad_list[r],axis=1) ==3)\
                                            *(np.sum(self.reg_info['spos']/self.meter <= self.new_coml[r]+rad_boost[r]*self.rad_list[r],axis=1) ==3)
                                        self.reg_info['sids'] = self.reg_info['sids'][bool_contained]
                                        self.reg_info['spos'] = self.reg_info['spos'][bool_contained]
                                        self.reg_info['svel'] = self.reg_info['svel'][bool_contained]
                                #print('1',r,rank)
                                # if self.tracktime:
                                #     self.time9 = time_sys.time()
                                self.rad_boost = rad_boost[r]
                                main_prog,prog_stats,halotry_r,scaling = \
                                        self.find_progenitor_halos(r,ntries=ntries_0,check=check)
                                # if time_sys.time()-self.time9>5 and self.tracktime:
                                #     print('Find_Progenitors: %s' % r,'Time:',time_sys.time()-self.time9)
                                #     self.time9 = time_sys.time()
                                #print('2',r,rank)
                                self.reg_info = {}
                                sto[r]['halotry'] = [r,halotry_r]
                                sto[r]["massl"] = np.array([])
                                sto[r]['rank'] = comm.rank
                                sto[r]['Solved'] = False
                                sto[r]['scaling'] = scaling
                                sto[r]['com_ids'] = self.com_ids[r]
                                #sto[r]['var_names_scalar'] = var_names_scalar
                                if main_prog != -1:
                                    sto[r]["idl"] = []
                                    sto[r]["hullv"] = []
                                    sto[r]["nl"] = []
                                    sto[r]["rad"] = np.array([])
                                    sto[r]['com'] = []
                                    sto[r]['vcom'] = []
                                    sto[r]['rvir_l'] = []
                                    sto[r]['cden_l'] = []
                                    sto[r]['cden'] = []
                                    sto[r]['more_variables'] = []
                                    sto[r]['var_names_scalar'] = []
                                    sto[r]['var_names_vec'] = []
                                    sto[r]['more_vec'] = []
                                    # sto[r]['m200'] = []
                                    for key in prog_stats.keys():
                                        if 'Solved' in prog_stats[key]:
                                            sto[r]['Solved'] = prog_stats[key]['Solved']
                                        sto[r]["rad"] = np.append(sto[r]["rad"],prog_stats[key]["rad"])
                                        sto[r]["idl"].append(prog_stats[key]["idl"])
                                        sto[r]["hullv"].append(prog_stats[key]["hullv"])
                                        sto[r]["massl"] = np.append(sto[r]["massl"],prog_stats[key]["mass"])
                                        # try:
                                        #     test = prog_stats[key]["m200"]
                                        # except:
                                        #     print(prog_stats[key])
                                        # sto[r]["m200"] = np.append(sto[r]["m200"],prog_stats[key]["m200"])
                                        if self.progs:
                                            sto[r]["nl"].append([key,self.halo_num[prog_stats[key]["prog"]]])
                                        else:
                                            sto[r]["nl"].append([self.halo_num[r][0],self.halo_num[r][1]])
                                        sto[r]["cden"] = np.append(sto[r]["cden"],prog_stats[key]["cden"])
                                        #print(prog_stats[key])
                                        sto[r]['var_names_scalar'].append(prog_stats[key]['var_names_scalar'])
                                        sto[r]['var_names_vec'].append(prog_stats[key]['var_names_vec'])
                                        sto[r]['more_vec'].append(prog_stats[key]['more_vec'])
                                        if len(sto[r]['com']) == 0:
                                            sto[r]['com'] = prog_stats[key]["com"]
                                            sto[r]['vcom'] = prog_stats[key]["vcom"]
                                            sto[r]['rvir_l'] = prog_stats[key]["rvir_l"]
                                            sto[r]['cden_l'] = prog_stats[key]["cden_l"]
                                            sto[r]['more_variables'] = prog_stats[key]['more_variables']
                                        else:
                                            sto[r]['com'] = np.vstack((sto[r]['com'],prog_stats[key]["com"]))
                                            sto[r]['vcom'] = np.vstack((sto[r]['vcom'],prog_stats[key]["vcom"]))
                                            sto[r]['rvir_l'] = np.vstack((sto[r]['rvir_l'],prog_stats[key]["rvir_l"]))
                                            sto[r]['cden_l'] = np.vstack((sto[r]['cden_l'],prog_stats[key]["cden_l"]))
                                            sto[r]['more_variables'] = np.vstack((sto[r]['more_variables'],prog_stats[key]['more_variables']))
                                if self.optimization:
                                    time_split[time_i] += time_sys.time() - time8
                                    time_i += 1
                            if self.optimization:
                                print(rank,time_split,sum(time_split))
                                split_len_1 = np.zeros(len(ranks))
                                time7 = time_sys.time() - time7
                                time_array = np.zeros(len(ranks))
                                for i in ranks:
                                  time_array[i] = comm.bcast(time7,root=i)
                                  if split_i < split_len[i]:
                                      split_len_1[i] += len(split_jobs[i][split_i])
                                if rank ==0:# and self.verbose:
                                    print('time',time_array)
                                    print('len',split_len_1)
                            for i in ranks:
                              if split_i < split_len[i]:
                                v = None
                                for r in split_jobs[i][split_i]:
                                    #r = h_index[t].astype(int)
                                    v = comm.bcast(sto[r],root=int(i))
                                    self.com_ids[r] = v['com_ids']
                                    halo = self.halo_num[v['halotry'][0]]
                                    vrank = v['rank']
                                    self.solved_list[halo] = comm.bcast(v['Solved'],root=vrank)
                                    self.halotree[halo]['Forced'] = comm.bcast(self.halotree[halo]['Forced'],root=vrank)
                                    self.halotree[halo]['Consec_Forced'] = comm.bcast(self.halotree[halo]['Consec_Forced'],root=vrank)
                                    self.halotry[v['halotry'][0]] = v['halotry'][1]
                                    if rank ==0 and v['Solved'] and self.direction >0 and scale_on:
                                        if len(scaling_l) ==0 and len(v['scaling']) ==5:
                                            scaling_l = v['scaling'][np.newaxis,:]
                                        elif len(scaling_l) >0 and len(v['scaling']) ==5:
                                            scaling_l = np.vstack((scaling_l,v['scaling']))
                                    # if v['Solved']:
                                    #     print(v['more_variables'])
                                    for t in range(len(v['massl'])):
                                        idl.append(v["idl"][t])
                                        hullv.append(v["hullv"][t])
                                        massl = np.append(massl,v["massl"][t])
                                        # m200 = np.append(m200,v["m200"])
                                        rad = np.append(rad,v["rad"][t])
                                        cdens = np.append(cdens,v['cden'][t])
                                        # if rank==0:
                                        #     print(v["nl"][t])
                                        #nl.append(v["nl"][t])
                                        if len(com) ==0:
                                            nl = np.array(v["nl"][t])[np.newaxis,:]
                                            com = v['com'][t]
                                            vcom = v['vcom'][t]
                                            rvir_l = v['rvir_l'][t]
                                            cden_l = v['cden_l'][t][np.newaxis,:]
                                            more_var_l = v['more_variables'][t][np.newaxis,:]
                                        else:
                                            nl = np.vstack((nl,np.array(v["nl"][t])))
                                            com = np.vstack((com,v['com'][t]))
                                            vcom = np.vstack((vcom,v['vcom'][t]))
                                            rvir_l = np.vstack((rvir_l,v['rvir_l'][t]))
                                            cden_l = np.vstack((cden_l,v['cden_l'][t]))
                                            more_var_l = np.vstack((more_var_l,v['more_variables'][t]))
                                        var_names_scalar.append(v['var_names_scalar'][0])
                                        var_names_vec.append(v['var_names_vec'][0])
                                        more_vec.append(v['more_vec'][0])
                                    v = None
                                    # if rank==0:
                                    #     print(var_names_scalar)
                            split_jobs = comm.bcast(split_jobs,root=0)
                            if rank==0 and self.tracktime:
                                #print(self.halotry)
                                if self.direction >0 and scale_on:
                                    #print(scaling_l)
                                    np.savetxt('%s/scaling_%s.txt' % (self.savestg,self.timestep),scaling_l)
                                print('Group %s of %s Done in Determing Halos of Round %s' % (split_i+1,int(split_len.max()),round_dh),'Time:',\
                                    time_sys.time()-self.time0)
                                self.time0 = time_sys.time()
                        if rank==0 and self.tracktime:
                            print('Round %s Done in Determing Halos' % (round_dh),'Time:',time_sys.time()-self.time1,'Found:',len(massl))
                            self.time1 = time_sys.time()
                            round_dh += 1
                self.my_storage_reg = 0
                sto = {}
              if rank==0 and self.tracktime:
                    print('Batch %s of %s Done in Determing Halos' % (out_round,len(split_volumes)),'Time:',time_sys.time()-self.time_out,'Found:',len(massl))
                    self.time_out = time_sys.time()
                    out_round += 1
            # if rank ==0:
            #     #print(var_names_scalar)
            #     compare(snaps)
            if len(massl)>1:
                nl = self.find_best_progenitor(idl,massl,nl,rad,com,vcom,cdens,len_fof)
            return idl,massl,nl,rad,com,vcom,rvir_l,cden_l,cdens,hullv,more_var_l,var_names_scalar,var_names_vec,more_vec
        else:
            return idl,massl,nl,rad,com,vcom,rvir_l,cden_l,cdens,hullv,more_var_l,var_names_scalar,var_names_vec,more_vec

    def find_best_progenitor(self,idl,massl,nl,rad,com,vcom,cdens,len_fof):
        halo_list = np.arange(len(self.halo_num))
        ranks = np.arange(nprocs)
        jobs,sto = job_scheduler(halo_list)
        for rank_now in ranks:
            if rank == rank_now:
                for r in jobs[rank]:
                    if r in self.com_ids:
                        if len(self.com_ids[r]) >0:
                            change = False
                            mass_ratio = massl/self.mass_list[r]#self.halotree[self.halo_num[r]][self.timestep+self.direction]['Halo_Mass']
                            mass_ratio_diff = abs(mass_ratio - 1)
                            inside_box = self.com_ids[r]
                            cden_last = self.halotree[self.halo_num[r]][self.timestep+self.direction]['cden']
                            cden_rat_last = np.maximum(cdens/cden_last,1)
                            cden_rat_last_2 = np.maximum(cden_last/cdens,1)
                            if len(com) >1:
                                rad_dist = np.linalg.norm(com-self.com_ids[r],axis=1)/self.rad_list[r]
                                vcom_diff = np.dot(vcom,self.vcom_list[r])/(np.linalg.norm(self.vcom_list[r])*np.linalg.norm(vcom,axis=1))
                            elif len(com)==1:
                              rad_dist = np.array([np.linalg.norm(com-self.com_ids[r])/self.rad_list[r]])
                              vcom_diff = np.dot(vcom,self.vcom_list[r])/(np.linalg.norm(self.vcom_list[r])*np.linalg.norm(vcom))
                            cost_t = 1e10*np.ones(len(massl))
                            ind_prog = np.arange(len(massl))[(nl[:,1]==self.halo_num[r])*(nl[:,0]=='0')]
                            for t in range(len(massl)):
                                if (nl[t][1]==self.halo_num[r] or rad_dist[t] < 1) and t >= len_fof:
                                    membership_r_1 = cden_rat_last[t]*np.isin(idl[t],self.idl_last[r]).sum()/\
                                        min(self.maxids,len(self.idl_last[r]))
                                    if membership_r_1 > 0.75 and mass_ratio_diff[t]/max(cden_rat_last[t],cden_rat_last_2[t]) <0.15:
                                        cost_t[t] = (1+10*mass_ratio_diff[t])**2 + 1/membership_r_1**2 + 100*abs(vcom_diff[t]-1) + 10*rad_dist[t]
                            if len(ind_prog)>0:
                                if cost_t.min()<cost_t[ind_prog[0]]:
                                    change = True
                                    sto[r]['last_halo'] = ind_prog[0]
                                    sto[r]['last_cost'] = cost_t[ind_prog[0]]
                            elif cost_t.min() < 1e10:
                                change = True
                                sto[r]['last_halo'] = None
                            if change:
                                    #print(self.halo_num[r],np.arange(len(massl))[cost_t==cost_t.min()][0],cost_t[cost_t<1e10],nl[:,0][cost_t<1e10],nl[:,1][cost_t<1e10],np.arange(len(massl))[cost_t<1e10],cost_t[ind_prog])
                                    sto[r]['best'] = np.arange(len(massl))[cost_t==cost_t.min()][0]
                                    sto[r]['cost'] = cost_t[cost_t==cost_t.min()]
        for rank_now in ranks:
            for r in jobs[rank_now]:
                sto[r] = comm.bcast(sto[r],root=rank_now)
        for r in halo_list:
            if 'best' in sto[r]:
                if sto[r]['best'] != sto[r]['last_halo']:
                    if nl[sto[r]['best']][0] != '0': # not taken
                        if sto[r]['last_halo'] != None:
                            nl[sto[r]['last_halo']][0] = '-1'
                        nl[sto[r]['best']][1] = self.halo_num[r]
                        nl[sto[r]['best']][0] = '0'
                        if rank ==0 and self.verbose:
                            print('not taken',self.halo_num[r])
                    elif nl[sto[r]['best']][0] == '0' and sto[r]['last_halo'] != None: #flip
                        halo_best = nl[sto[r]['best']][1]
                        halo_last = nl[sto[r]['last_halo']][1]
                        r_best = halo_list[self.halo_num==halo_best][0]
                        if 'best' in sto[r_best]:
                            if sto[r_best]['best'] == sto[r]['last_halo']:
                                nl[sto[r]['best']][1] = self.halo_num[r]
                                if rank ==0 and self.verbose:
                                    print('flip',self.halo_num[r],self.halo_num[r_best])
        return nl









    # Updates the halo tree and the variables used to find halos in the next timestep
    # Adds new branches to the halo tree with co-progentors
    def build_tree(self,nl,massl,idl,rad,com,vcom,rvir_l,cden_l,cdens,hullv,more_var_l,var_names_scalar,var_names_vec,more_vec,coprog_on=True,overwrite=False,limited=False,reformed=False,save=True,save_now=False):
            jlist = []
            halonum = []
            # if rank==0:
            #     print(nl,self.indices)
            # if rank ==0:
            #     print(cden_l)
            # Adjusts arrays so that they properly stack in every situation
            rvir_200_i = np.arange(len(oden_list))[abs(np.array(oden_list).astype(float)-200)==abs(np.array(oden_list).astype(float)-200).min()].min()
            if len(rad) > 0:
                if len(rad) ==1:
                 if len(com) ==3:
                    com = np.array([com])
                    vcom = np.array([vcom])
                elif len(com) == 1:
                    com = com[0]
                    vcom = vcom[0]
                if rvir_l.ndim ==1:
                    rvir_l = rvir_l[np.newaxis,:]
                for j in self.indices:
                    prog,r = nl[j]
                    prog,r = int(prog),str(r)
                    # If the halo is known and data from this timestep is missing
                    if self.timestep not in self.halotree[r].keys() and prog == 0 and not overwrite \
                            and r not in self.taken:
                        if rank==0 and self.verbose:
                            print('Halo %s added at timestep %s' % (r,self.timestep))
                        self.halotree[r][self.timestep] = {}
                        self.halotree[r][self.timestep]['NumParts'] = len(idl[j])
                        self.halotree[r][self.timestep]['Halo_Mass'] = massl[j]
                        self.halotree[r][self.timestep]['Halo_Radius'] = rad[j]
                        self.halotree[r][self.timestep]['cden'] = cdens[j]
                        #self.halotree[r][self.timestep]['m%s' % str(int(cden_l[j][rvir_200_i]))] = m200[j]
                        oden_count = 0
                        oden_taken = []
                        for rad_i in range(len(oden_list)):
                            if str(int(cden_l[j][rad_i])) not in oden_taken:
                                oden = str(int(round(cden_l[j][rad_i])))
                            else:
                                oden = str(int(round(cden_l[j][rad_i])))+'.'+str(oden_count)
                            self.halotree[r][self.timestep]['r'+oden] = rvir_l[j][rad_i]
                            if oden in oden_taken:
                                oden_count += 1
                            oden_taken.append(oden)
                        for f,var in enumerate(var_names_scalar[j]):
                            if var not in self.halotree[r][self.timestep]:
                                self.halotree[r][self.timestep][var] = more_var_l[j][f]
                        # if rank==0:
                        #     print(len(var_names_vec[j]),len(more_vec[j]))
                        for f,var in enumerate(var_names_vec[j]):
                            if var not in self.halotree[r][self.timestep] and var != 'star_id' and var != 'star_energy':
                                self.halotree[r][self.timestep][var] = more_vec[j][f]
                            if rank ==min(nprocs-1,1) and var == 'star_id':
                                if r not in self.star_id['ids']:
                                    self.star_id['ids'][r] = {}
                                self.star_id['ids'][r][self.timestep] = more_vec[j][f]
                            if rank ==min(nprocs-1,1) and var == 'star_energy':
                                if r not in self.star_id['energies']:
                                    self.star_id['energies'][r] = {}
                                self.star_id['energies'][r][self.timestep] = more_vec[j][f]
                        self.halotree[r][self.timestep]['Halo_Center'] = com[j]
                        self.halotree[r][self.timestep]['Vel_Com'] = vcom[j]
                        self.hullv[r][self.timestep] = hullv[j]
                        if rank==0:
                            self.idl[r][self.timestep] = idl[j]
                        if self.solved_list[r]:
                            if self.direction >0:
                                self.halotree[r]['Solved'][0] = min(self.timestep,min(self.halotree[r]['Solved']))
                            elif self.direction <0:
                                self.halotree[r]['Solved'][1] = max(self.timestep,max(self.halotree[r]['Solved']))
                        jlist.append(j)
                        halonum.append(r)
                    # If the halo is known and the data must be overwritten
                    elif self.timestep in self.halotree[r].keys() and prog ==0 and overwrite:
                        if rank==0 and self.verbose:
                            print('Halo %s replacement at timestep %s' % (r,self.timestep))
                        self.halotree[r][self.timestep] = {}
                        self.halotree[r][self.timestep]['NumParts'] = len(idl[j])
                        self.halotree[r][self.timestep]['Halo_Mass'] = massl[j]
                        self.halotree[r][self.timestep]['Halo_Radius'] =  rad[j]
                        self.halotree[r][self.timestep]['cden'] = cdens[j]
                        #self.halotree[r][self.timestep]['m%s' % str(int(cden_l[j][rvir_200_i]))] = m200[j]
                        oden_count = 0
                        oden_taken = []
                        for rad_i in range(len(oden_list)):
                            if str(int(cden_l[j][rad_i])) not in oden_taken:
                                oden = str(int(round(cden_l[j][rad_i])))
                            else:
                                oden = str(int(round(cden_l[j][rad_i])))+'.'+str(oden_count)
                            self.halotree[r][self.timestep]['r'+oden] = rvir_l[j][rad_i]
                            if oden in oden_taken:
                                oden_count += 1
                            oden_taken.append(oden)
                        for f,var in enumerate(var_names_scalar[j]):
                            if var not in self.halotree[r][self.timestep]:
                                self.halotree[r][self.timestep][var] = more_var_l[j][f]
                        for f,var in enumerate(var_names_vec[j]):
                            if var not in self.halotree[r][self.timestep] and var != 'star_id' and var != 'star_energy':
                                self.halotree[r][self.timestep][var] = more_vec[j][f]
                            if rank ==min(nprocs-1,1) and var == 'star_id':
                                if r not in self.star_id['ids']:
                                    self.star_id['ids'][r] = {}
                                self.star_id['ids'][r][self.timestep] = more_vec[j][f]
                            if rank ==min(nprocs-1,1) and var == 'star_energy':
                                if r not in self.star_id['energies']:
                                    self.star_id['energies'][r] = {}
                                self.star_id['energies'][r][self.timestep] = more_vec[j][f]
                        self.halotree[r][self.timestep]['Halo_Center'] = com[j]
                        self.halotree[r][self.timestep]['Vel_Com'] = vcom[j]
                        self.hullv[r][self.timestep] = hullv[j]
                        if rank==0:
                            self.idl[r][self.timestep] = idl[j]
                        if self.solved_list[r]:
                            if self.direction >0:
                                self.halotree[r]['Solved'][0] = min(self.timestep,min(self.halotree[r]['Solved']))
                            elif self.direction <0:
                                self.halotree[r]['Solved'][1] = max(self.timestep,max(self.halotree[r]['Solved']))
                        jlist.append(j)
                        halonum.append(r)
                for j in self.indices:
                    prog,r = nl[j]
                    prog,r = int(prog),str(r)
                    if prog <0 and coprog_on:
                        t = str(self.new_halo+1)
                        if rank==0 and self.verbose:
                            print('New halo %s added at timestep %s' % (t,self.timestep))
                        self.halotree[t] = {}
                        self.halotree[t]['Num_Progs'] = 0
                        self.halotree[t]['Forced'] = 0
                        self.halotree[t]['Consec_Forced'] = 0
                        self.halotree[t]['Solved'] = [self.timestep,self.timestep]
                        self.halotree[t][self.timestep] = {}
                        self.halotree[t][self.timestep]['NumParts'] = len(idl[j])
                        self.halotree[t][self.timestep]['Halo_Mass'] = massl[j]
                        self.halotree[t][self.timestep]['Halo_Radius'] =  rad[j]
                        self.halotree[t][self.timestep]['cden'] = cdens[j]
                        #self.halotree[t][self.timestep]['m%s' % str(int(cden_l[j][rvir_200_i]))] = m200[j]
                        oden_count = 0
                        oden_taken = []
                        for rad_i in range(len(oden_list)):
                            if str(int(cden_l[j][rad_i])) not in oden_taken:
                                oden = str(int(round(cden_l[j][rad_i])))
                            else:
                                oden = str(int(round(cden_l[j][rad_i])))+'.'+str(oden_count)
                            self.halotree[t][self.timestep]['r'+oden] = rvir_l[j][rad_i]
                            if oden in oden_taken:
                                oden_count += 1
                            oden_taken.append(oden)
                        for f,var in enumerate(var_names_scalar[j]):
                            if var not in self.halotree[t][self.timestep]:
                                self.halotree[t][self.timestep][var] = more_var_l[j][f]
                        for f,var in enumerate(var_names_vec[j]):
                            if var not in self.halotree[t][self.timestep] and var != 'star_id'  and var != 'star_energy':
                                self.halotree[t][self.timestep][var] = more_vec[j][f]
                            if rank ==min(nprocs-1,1) and var == 'star_id':
                                if t not in self.star_id['ids']:
                                    self.star_id['ids'][t] = {}
                                self.star_id['ids'][t][self.timestep] = more_vec[j][f]
                            if rank ==min(nprocs-1,1) and var == 'star_energy':
                                if t not in self.star_id['energies']:
                                    self.star_id['energies'][t] = {}
                                self.star_id['energies'][t][self.timestep] = more_vec[j][f]
                        self.halotree[t][self.timestep]['Halo_Center'] = com[j]
                        self.halotree[t][self.timestep]['Vel_Com'] = vcom[j]
                        self.hullv[t] = {}
                        self.hullv[t][self.timestep] = hullv[j]
                        if rank==0:
                            self.idl[t] = {}
                            self.idl[t][self.timestep] = idl[j]
                        jlist.append(j)
                        halonum.append(t)
                        self.new_halo += 1
                    # If the halo is a new progenitor of another halo
                    elif coprog_on and prog > 0:
                        # # Ensures that new progentors are not written twice
                        # if bool_coprog:
                            cprog = False
                            numprog = self.halotree[r]['Num_Progs']
                            if self.timestep not in self.halotree[r].keys() and self.timestep + self.direction in self.halotree[r].keys() and reformed:
                                newr = r
                                if rank==0 and self.verbose:
                                    print('Halo %s added at timestep %s as a reformed co-progenitor' % (r,self.timestep))
                                self.halotree[newr]['Forced'] += 1
                                cprog = True
                            elif self.timestep in self.halotree[r].keys():
                                newr = r + "_" + str(numprog)
                                if rank==0 and self.verbose:
                                    print('Halo co-progenitor %s added at timestep %s' % (newr,self.timestep))
                                self.halotree[r]['Num_Progs'] += 1
                                self.halotree[newr] = {}
                                self.halotree[newr]['Num_Progs'] = 0
                                self.halotree[newr]['Forced'] = 0
                                self.halotree[newr]['Consec_Forced'] = 0
                                self.halotree[newr]['Solved'] = [self.timestep,self.timestep]
                                self.hullv[newr] = {}
                                if rank==0:
                                    self.idl[newr] = {}
                                cprog = True
                            if cprog:
                                self.halotree[newr][self.timestep] = {}
                                self.halotree[newr]['Forced'] = 0
                                self.halotree[newr]['Consec_Forced'] = 0
                                self.halotree[newr]['Solved'] = [self.timestep,self.timestep]
                                self.halotree[newr][self.timestep]['NumParts'] = len(idl[j])
                                self.halotree[newr][self.timestep]['Halo_Mass'] = massl[j]
                                self.halotree[newr][self.timestep]['Halo_Radius'] =  rad[j]
                                self.halotree[newr][self.timestep]['cden'] = cdens[j]
                                #self.halotree[newr][self.timestep]['m%s' % str(int(cden_l[j][rvir_200_i]))] = m200[j]
                                oden_count = 0
                                oden_taken = []
                                for rad_i in range(len(oden_list)):
                                    if str(int(cden_l[j][rad_i])) not in oden_taken:
                                        oden = str(int(round(cden_l[j][rad_i])))
                                    else:
                                        oden = str(int(round(cden_l[j][rad_i])))+'.'+str(oden_count)
                                    self.halotree[newr][self.timestep]['r'+oden] = rvir_l[j][rad_i]
                                    if oden in oden_taken:
                                        oden_count += 1
                                    oden_taken.append(oden)
                                for f,var in enumerate(var_names_scalar[j]):
                                    if var not in self.halotree[newr][self.timestep]:
                                        self.halotree[newr][self.timestep][var] = more_var_l[j][f]
                                for f,var in enumerate(var_names_vec[j]):
                                    if var not in self.halotree[newr][self.timestep] and var != 'star_id' and var != 'star_energy':
                                        self.halotree[newr][self.timestep][var] = more_vec[j][f]
                                    if rank ==min(nprocs-1,1) and var == 'star_id':
                                        if newr not in self.star_id['ids']:
                                            self.star_id['ids'][newr] = {}
                                        self.star_id['ids'][newr][self.timestep] = more_vec[j][f]
                                    if rank ==min(nprocs-1,1) and var == 'star_energy':
                                        if newr not in self.star_id['energies']:
                                            self.star_id['energies'][newr] = {}
                                        self.star_id['energies'][newr][self.timestep] = more_vec[j][f]
                                self.halotree[newr][self.timestep]['Halo_Center'] = com[j]
                                self.halotree[newr][self.timestep]['Vel_Com'] = vcom[j]
                                self.hullv[newr][self.timestep] = hullv[j]
                                if rank==0:
                                    self.idl[newr][self.timestep] = idl[j]
                                jlist.append(j)
                                halonum.append(newr)
            if (save and (self.timestep%self.interval == 0 or self.timestep%4 ==0)) or (save_now):
                if rank ==0:
                    print('Saving...')
                    np.save(savestring +  '/' + 'halotree_%s.npy' % (fldn),self.halotree)
                if rank ==0:
                    np.save(savestring +  '/' + 'idl_%s.npy' % (fldn),self.idl)
                    np.save(savestring +  '/' + 'hullv_%s.npy' % (fldn),self.hullv)
                if rank ==min(nprocs-1,1):
                    if len(self.star_id['ids']) >0:
                        np.save(savestring +  '/' + 'star_id_%s.npy' % (fldn),self.star_id)
                if rank ==0:
                    print('Saved',len(list(self.halotree.keys())),'halos')
                if self.direction <0 and self.end:
                    np.savetxt(savestring +  '/' + 'max_time_%s.txt' % (fldn),np.array([self.timestep]))
            if save and self.timestep%8 ==0:
                if rank ==0:
                    np.save(savestring +  '/' + 'halotree_%s_backup.npy' % (fldn),self.halotree)
                if rank ==0:
                    np.save(savestring +  '/' + 'idl_%s_backup.npy' % (fldn),self.idl)
                    np.save(savestring +  '/' + 'hullv_%s_backup.npy' % (fldn),self.hullv)
                if rank ==min(nprocs-1,1):
                    if len(self.star_id['ids']) >0:
                        np.save(savestring +  '/' + 'star_id_%s_backup.npy' % (fldn),self.star_id)
                if rank ==0:
                    print('Backed up',len(list(self.halotree.keys())),'halos')


            # Updates to the progentitor-finding variables


    # Redundant function with the one in Initial_Halo_Finder but limits output to positions and ids
    def get_mass_pos_vel_ids(self,reg,center,radius,tp = 1,make_volume=True):
        dm_name_dict = {'ENZO':'DarkMatter','GEAR': 'DarkMatter',\
         'GADGET3': 'DarkMatter', 'GADGET4': 'DarkMatter', 'AREPO': 'DarkMatter',\
          'GIZMO': 'DarkMatter', 'RAMSES': 'DM',\
           'ART': 'darkmatter', 'CHANGA': 'DarkMatter'}
        mass = reg[(dm_name_dict[self.codetp],'particle_mass')].in_units('Msun').v
        pos = reg[(dm_name_dict[self.codetp],'particle_position')].in_units('m')
        if self.codetp == 'CHANGA':
            ids = reg[(dm_name_dict[self.codetp],'iord')].v.astype(int)
        else:
            ids = reg[(dm_name_dict[self.codetp],'particle_index')].v.astype(int)
        if make_volume:
            bool1 = (np.sum(pos > center - radius,axis=1)==3)*(np.sum(pos < center + radius,axis=1)==3)
            pos,ids = pos[bool1],ids[bool1]
        return pos.v,ids

    def get_radmax(self,r,radmax):
        if self.timestep+self.direction in self.time:
            dt1 = abs(self.time[self.timestep] - self.time[self.timestep+self.direction])
            move_dist = np.abs(np.linalg.norm(self.vcom_list[r]) * dt1)/self.rad_list[r]
            if move_dist > 0.1:
                rad_out = max(3*move_dist,radmax)
            else:
                rad_out = max(1.85,radmax)
            # if 3*move_dist > 1.85:
            #     print(3*move_dist,radmax,rad_out)
        else:
            rad_out = radmax
        return rad_out

    # Advances halo forward one timestep without requiring that the halo be self bound
    # but has a calculable overdensity radius
    def propagate_halo(self,DH,r,tp=1,oden=200,reg=False):
        halo = self.halo_num[r]
        var_names_scalar,var_names_vec = make_var_names()
        more_variables_i = [None]*len(var_names_scalar)
        more_variables_vec = [None]*len(var_names_vec)
        crit_den = univDen(self.ds)
        mass0,pos0,vel0,ids = reg['mass'],reg['pos'],reg['vel'],reg['ids']
        if find_stars:
            spos,svel,sids = reg['spos'],reg['svel'],reg['sids']
        bool_ids = np.isin(ids,self.idl_solved[r])
        prog_stats = {}
        prog_stats[0] = {}
        prog_stats[0]["prog"] = r
        mass_1 = 0
        if bool_ids.sum()>1:
            pos_id,mass_id,vel_id = pos0[bool_ids].v,mass0[bool_ids],vel0[bool_ids].v
            com0 = FindCOM(mass0[bool_ids],pos0[bool_ids])
            com = FindCOM(mass0[bool_ids],pos0[bool_ids])
            self.com_ids[r] = com.v/self.meter
            radmax_r = self.box_width[r]*2
            bool_in = (np.sum(pos0.v/self.meter > com0.v/self.meter - radmax_r, axis=1)==3)*\
                (np.sum(pos0.v/self.meter < com0.v/self.meter + radmax_r, axis=1)==3)
            mass0,pos0,vel0,ids = mass0[bool_in],pos0[bool_in],vel0[bool_in],ids[bool_in]
            # Checks for the particles in the descendant timestep
            Solved = self.halotree[halo]['Solved']
            Consec_Forced = self.halotree[halo]['Consec_Forced']
            con_dir = self.timestep+self.direction
            bool_ids = np.isin(ids,self.idl_solved[r])
            did_cut = False
            #bool_ids = np.isin(ids,self.idl_solved[r])
            #bool_ids_2 = np.isin(ids,self.idl_all)
            #bool_ids = bool_ids or bool_ids_2
            if self.direction>0:
                con_dir_2 = Solved[0]
            elif self.direction <0:
                con_dir_2 = Solved[1]
                if con_dir_2 not in self.halotree[halo]:
                    con_dir_2 = con_dir
                    self.halotree[halo]['Solved'][1] = con_dir
            rad_before = self.halotree[halo][con_dir_2]['Halo_Radius']
            factor = self.halotree[halo][con_dir_2]['Halo_Mass']/(self.kg_sun*mass_id.sum())
            mass_id *= factor
            if find_stars:
                #print(len(sids),len(mass0),factor)
                if len(sids)>0:
                #print(len(sids))
                    radmax_s = self.get_radmax(r,3.5)
                    bool_s = (np.sum(spos.v/self.meter > com.v/self.meter - self.rad_list[r]*radmax_s, axis=1)==3)*\
                        (np.sum(spos.v/self.meter < com.v/self.meter + self.rad_list[r]*radmax_s, axis=1)==3)
                    #print('here',bool_s.sum())
                    #print(spos.v/self.meter,com.v/self.meter - self.rad_list[r]*radmax_s,com.v/self.meter + self.rad_list[r]*radmax_s)
                    spos,svel,sids = spos[bool_s],svel[bool_s],sids[bool_s]
            if len(mass0)>50000 and bool_ids.sum()>1:
                mass0,booli = cut_particles(pos0.v,mass0,com0.v,ids,idl_i=self.idl_solved[r])
                pos0,vel0,ids = pos0[booli],vel0[booli],ids[booli]
                did_cut = True
                bool_ids = np.isin(ids,self.idl_solved[r])
        #print('here',bool_ids.sum())
        if bool_ids.sum() > 1:
            rmax = 1.3*max(self.halotree[halo][con_dir_2]['Halo_Radius'],self.box_width[r])*self.meter
            mean_v = (vel0[bool_ids]*mass0[bool_ids,np.newaxis]).sum(axis=0)/mass0[bool_ids].sum()
            rs = np.linalg.norm(com.v-pos0.v,axis=1)
            arg_r = np.argsort(rs)
            mass0,pos0,vel0,ids,rs = mass0[arg_r],pos0[arg_r],vel0[arg_r],ids[arg_r],rs[arg_r]
            bool_ids = bool_ids[arg_r]
            M = mass0[arg_r].cumsum()
            PE = self.G*M/rs
            KE = 0.5*np.linalg.norm(vel0.v-mean_v.v,axis=1)**2
            es = PE-KE
            es[es <0] = 0
            #print(PE,KE)
            bool1 = rs <= rmax
            Bound = KE < PE
            # if ids[bool1*Bound].sum() > max(0.5*bool_ids.sum(),100):
            bool_bound = bool1*Bound
            dist_bigger = np.linalg.norm(self.new_coml_all[self.mass_list_all>self.mass_list[r]]-com.v/self.meter,axis=1)
            dist_bool  = np.sum(dist_bigger < 1.5*self.rad_list_all[self.mass_list_all>self.mass_list[r]]) == 0
            dist_bool_2 = np.sum(dist_bigger < 0.25*self.rad_list_all[self.mass_list_all>self.mass_list[r]]) == 0
            # if dist_bool_2:
            #     cdenmin = 800
            # else:
            #     cdenmin = 1500
            cdenmin = max(oden_list)*1.05
            inside_ratio = np.isin(ids[bool1*Bound],ids[bool_ids]).sum()/len(ids[bool_ids])
            if Consec_Forced > 2:
                if dist_bool:
                    if inside_ratio > 0.8:
                        for i in range(4):
                          if inside_ratio > 0.8:
                            com = FindCOM(es[bool_bound],pos0[bool_bound])
                            mean_v = (vel0[bool_bound]*mass0[bool_bound,np.newaxis]).sum(axis=0)/mass0[bool_bound].sum()
                            rs = np.linalg.norm(com.v-pos0.v,axis=1)
                            arg_r = np.argsort(rs)
                            mass0,pos0,vel0,ids,rs = mass0[arg_r],pos0[arg_r],vel0[arg_r],ids[arg_r],rs[arg_r]
                            bool_ids = bool_ids[arg_r]
                            M = mass0[arg_r].cumsum()
                            PE = self.G*M/rs
                            KE = 0.5*np.linalg.norm(vel0.v-mean_v.v,axis=1)**2
                            es = PE-KE
                            es[es <0] = 0
                            bool1 = rs <= rmax
                            Bound = KE < PE
                            inside_ratio = np.isin(ids[bool1*Bound],ids[bool_ids]).sum()/len(ids[bool_ids])
                            if inside_ratio > 0.8:
                                bool_bound = bool1*Bound
                            if bool_ids.sum() < self.maxids/2 and inside_ratio > 0.8:
                                try:
                                    bool_ids_l = np.random.choice(len(ids[bool_bound]),min(len(ids[bool_bound]),int(self.maxids/2)),\
                                        p=rs[bool_bound]/rs[bool_bound].sum(),replace=False)
                                    bool_ids = np.logical_or(np.isin(ids,ids[bool_ids_l]),bool_ids)
                                except:
                                    bool_ids_l = np.random.choice(len(ids[bool_bound]),min(len(ids[bool_bound]),int(self.maxids/2)),\
                                        replace=False)
                                    bool_ids = np.logical_or(np.isin(ids,ids[bool_ids_l]),bool_ids)
            mass,pos,vel,es = mass0[bool_bound],pos0[bool_bound],vel0[bool_bound],-1*es[bool_bound]
            oden_list0 = np.append(np.array(oden_list),oden)
            #print(rs[0],rmax,rvir)
            #print(cden,rvir/self.meter,len(mass),halo,'inital',self.halotree[halo][con_dir]['r200'])

                #print(rs[0],rmax,rvir)
                #print(cden,rvir/self.meter,len(mass),halo,'final',self.halotree[halo][con_dir]['r200'])
            # Generates halo properties if the halo has sufficient density for the propagation
            # print((Bound*bool1).sum(),bool1.sum(),mass0[bool1*Bound].sum()*self.kg_sun,self.mass_list[r],\
            #      (cden.max() >  150 and len(mass)> 3 and cden.min() < 800),cden)
            # print(cden)
            # es[r>1.05*rvir.max()] = 1e20
            # if len(mass[es<0]) >100:
            #     r_per = np.percentile(r,95)
            #     es[r>r_per] = 1e20
            # if len(mass[es<0]) >100:
            #     r_per = np.percentile(r,90)
            #     es[r>1.2*r_per] = 1e20
            #es[r>1.05*rvir.max()] =1e20
            rvir,cden,mden = Find_virRad(com,pos,mass,self.ds,oden=oden_list0,radmax=rmax)
            #print(cden,len(mass[es<0]))
            bool_worked = cden.max() >  75 and len(mass[es<0])> 4 and cden.min() < cdenmin
            if bool_worked:
                #rs2 = np.linalg.norm(com.v-pos.v,axis=1)
                rad_i = np.arange(len(oden_list0))[abs(cden-oden)==abs(cden-oden).min()].max()
                rad_0 = np.arange(len(oden_list0))[abs(cden-self.mean_cden)==abs(cden-self.mean_cden).min()].max()
                rad_200 = np.arange(len(oden_list0))[abs(cden-200)==abs(cden-200).min()].max()
                rad_max = rvir[rad_i]#,rvir[rad_0])
                #print(self.halo_num[r],rmax/self.meter,self.halotree[halo][con_dir_2]['Halo_Radius'],rvir[rad_i]/self.meter,rvir[rad_0]/self.meter,rvir[rad_200]/self.meter,rvir/self.meter)
                used_hull = False
                hull_on = False
                if non_sphere and len(rs[bool_bound][rs[bool_bound] >0]) >4 and hull_on:
                    target = max(oden,min(cden))
                    mass_1,cden_1,rvir0,estep_f,esr,hull = make_hull(rs[bool_bound],pos,mass,es,target,crit_den,rmax=rvir[rad_i])
                    #print(cden_1)
                    if cden_1 >  75 and rvir0/self.meter > 0.5*min(rad_before,self.rad_list[r]) and cden_1 <700:
                        used_hull = True
                        more_variables_i,more_variables_vec = more_var_maker(pos,mass,vel,rvir0,esr,es,estep_f,hull,com,self.meter,self.kg_sun,cden_1,self.ds,radmax=rmax)
                if not used_hull or cden_1 > max(oden_list) or mass_1 < 0.75*mass[rs[bool_bound] < rad_max].sum():
                    estep_f = 0
                    rad_200_i = np.arange(len(oden_list0))[abs(cden-200)==abs(cden-200).min()].min()
                    rad_i_2 = np.arange(len(oden_list0))[abs(cden-oden)==abs(cden-oden).min()].min()
                    rvir0 = rad_max
                    mass_1 = mass[rs[bool_bound] < rad_max].sum()
                    if rad_max == rvir[rad_i]:
                        cden_1 = cden[rad_i]
                    if rad_max == rvir[rad_0]:
                        cden_1 = cden[rad_0]
                    used_hull = False
                if used_hull:
                    vel = vel[esr<estep_f]
                    mass = mass[esr<estep_f]
                    pos = pos[esr<estep_f]
                #     #print(a,b,c)
                # if not used_hull:
                #     vel = vel[rs[bool_bound]<rvir0]
                #     mass = mass[rs[bool_bound]<rvir0]
                #cden.max() >  75 and len(mass[es<0])> 4 and cden.min() < cdenmin
                bool_worked = cden.max() >  75 and len(mass)> 1 and cden.min() < cdenmin and rvir0/self.meter > 0.5*rad_before
                # if not bool_worked:
                #     print(self.halo_num[r],cden.max(),cden.min(),(rvir0/self.meter)/(rad_before),len(mass))
                #     print(self.halo_num[r],cden.max() >  75,len(mass)> 1,cden.min() < cdenmin,rvir0/self.meter > 0.5*rad_before)
            if bool_worked:#\
                    #and rvir0/self.meter <  1.5*self.halotree[halo][con_dir_2]['Halo_Radius']:
                    vcom = ((vel*mass[:,np.newaxis]).sum(axis=0)/\
                        mass.sum())
                    found_sol = True
                    #print('ere',rvir0/(rad_before*self.meter),rs[bool_ids].max()/(rad_before*self.meter))
            elif len(mass0)> 1:
                    mass0,pos0,vel0,ids = reg['mass'],reg['pos'],reg['vel'],reg['ids']
                    bool_in = (np.sum(pos0.v/self.meter > com0.v/self.meter- radmax_r, axis=1)==3)*\
                        (np.sum(pos0.v/self.meter < com0.v/self.meter+ radmax_r, axis=1)==3)
                    mass0,pos0,vel0,ids = mass0[bool_in],pos0[bool_in],vel0[bool_in],ids[bool_in]
                    if len(mass0)>50000 and bool_ids.sum()>1:
                        mass0,booli = cut_particles(pos0.v,mass0,com0.v,ids,idl_i=self.idl_solved[r])
                        pos0,vel0,ids = pos0[booli],vel0[booli],ids[booli]
                        did_cut = True
                        #bool_ids = np.isin(ids,self.idl_solved[r])
                    #com = FindCOM(mass0[bool_ids],pos0[bool_ids])
                    #rmax = np.linalg.norm(com-pos0[bool_ids],axis=1)
                    #rmax = rmax.max()
                    #rmax = 1.4*self.halotree[halo][con_dir_2]['r200']*self.meter
                    bool_ids = np.isin(ids,self.idl_solved[r])
                    rs = np.linalg.norm(com0.v-pos0.v,axis=1)
                    rmax = rs[bool_ids].max()
                    bool_bound = rs <= rmax
                    mass,pos,vel = mass0[bool_bound],pos0[bool_bound],vel0[bool_bound]
                    rvir,cden,mden = Find_virRad(com0,pos,mass,self.ds,oden=oden_list0,radmax=rmax)
                    #print(self.halo_num[r],rvir/self.meter,cden)
                    rad_i = np.arange(len(oden_list0))[abs(cden-oden)==abs(cden-oden).min()].min()
                    #rad_0 = np.arange(len(oden_list0))[abs(cden-self.mean_cden)==abs(cden-self.mean_cden).min()].min()
                    #rad_last = np.arange(len(rvir))[abs(rvir-self.rad_list[r]*self.meter)==abs(rvir-self.rad_list[r]*self.meter).min()].min()
                    # if len(np.arange(len(rvir))[cden >self.mean_cden]) >0:
                    #     rad_last = np.arange(len(rvir))[cden >self.mean_cden][abs(rvir[cden >self.mean_cden]-rad_before*self.meter)==abs(rvir[cden >self.mean_cden]-rad_before*self.meter).min()].min()
                    #     rad_i = rad_last
                    estep_f = 0
                    #rad_200_i = np.arange(len(oden_list0))[abs(cden-200)==abs(cden-200).min()].min()
                    # if cden[rad_200_i] >175:
                    #print(cden,rvir/self.meter,self.rad_list[r],rvir[rad_i]/self.meter,rad_before,cden[rad_i],self.halo_num[r])
                    #print(rvir[rad_i]/self.meter,rvir[rad_last]/self.meter,self.rad_list[r],self.halo_num[r])
                    # if rvir[rad_i] > rvir[rad_last]:
                    #     rad_i = rad_last
                    rvir0 = rad_before*self.meter#rvir[rad_i]
                    rad_max = rad_before*self.meter#max(rvir0,rad_before*self.meter)
                    cden_1 = cden[rad_i]
                    mass_1 = min(mass[rs[bool_bound] < rvir[rad_i]].sum(),self.mass_list[r]/self.kg_sun)
                    #hull = ConvexHull(pos[rs[bool_bound] < 1.1*rvir[rad_i]])
                    used_hull = False
                    found_sol = False
                    vcom = ((vel0[bool_ids][rs[bool_ids] < 1.1*rad_max]*mass0[bool_ids][rs[bool_ids] < 1.1*rad_max][:,np.newaxis]).sum(axis=0)/\
                        mass0[bool_ids][rs[bool_ids] < 1.1*rad_max].sum())
                    #print('here',rvir0/(rad_before*self.meter),rs[bool_ids].max()/(rad_before*self.meter))
                # if used_hull:
                #     mass = mass[es<estep_f]
                #     vel = vel[es<estep_f]
                #     weighted_pos = pos[es<estep_f]*mass[:,np.newaxis]/mass.sum()
                #     eigvals, eigvecs = np.linalg.eig(np.cov(weighted_pos.T))
                #     a = max(eigvals)
                #     b = min(eigvals)
                #     c = eigvals[(eigvals!=a)*(eigvals!=b)][0]
                #if rad_i > rad_0:
                # cden_rat = cden[rad_i]/self.cden[r]
                # mass_rat = mass[rs[bool_bound] < rvir[rad_i]].sum()*self.kg_sun/(self.mass_list[r])
                # #print(mass_rat,cden_rat)
                # mass_rat *= cden_rat
                # if did_cut:
                #     print(mass_1*self.kg_sun >  minmass/2 and cden_1 <max(oden_list) and cden_1 >100)
            var_names_scalar,more_variables_i = add_virial_mass(cden[:-1],np.array(mden[:-1])*self.kg_sun,more_variables_i)
            #print(mass_1*self.kg_sun,minmass/2,cden_1,found_sol)
            #print('near',len(sids), mass_1*self.kg_sun,minmass/2)
            if mass_1*self.kg_sun >  minmass/2:# and cden_1 <max(oden_list)*1.05 and cden_1 >75:# and mass_rat < 3 and 1/mass_rat < 3:
                    if find_stars and len(sids)>0:
                        #print(len(mass),len(pos))
                        #bool_stars = rs<rvir0
                        sid_i,s_energy_i = find_star_energy(pos_id,mass_id,vel_id,com.v,spos.v,svel.v,sids)
                        #print(vel_id,svel.v)
                        #print(s_energy_i)
                        if len(sid_i)>0:
                            #print(halo,len(sid_i))
                            var_names_vec.append('star_id')
                            var_names_vec.append('star_energy')
                            more_variables_vec.append(sid_i)
                            more_variables_vec.append(s_energy_i)
                    #print(self.halo_num[r],rvir0/self.meter,cden_1)
                    prog_stats[0]['rvir_l'] = rvir[np.newaxis,:-1]/self.meter
                    prog_stats[0]['com'] = com[np.newaxis,:].v/self.meter
                    prog_stats[0]['vcom'] = vcom[np.newaxis,:].v/self.meter
                    prog_stats[0]['idl'] = ids[bool_ids][rs[bool_ids] < 1.1*rad_before*self.meter]
                    prog_stats[0]['mass'] = mass_1*self.kg_sun
                    #prog_stats[0]['m200'] = mass[rs[bool_bound] < rvir[rad_200_i]].sum()*self.kg_sun
                    #print(halo,mass_rat,cden[rad_i],self.cden[r])
                    prog_stats[0]['cden'] = cden_1
                    prog_stats[0]['cden_l'] = cden[np.newaxis,:-1]
                    prog_stats[0]['more_variables'] = np.array(more_variables_i)[np.newaxis,:]
                    prog_stats[0]['var_names_scalar'] = var_names_scalar
                    prog_stats[0]['var_names_vec'] = var_names_vec
                    prog_stats[0]['more_vec'] = more_variables_vec
                    if non_sphere:
                        if used_hull:
                            prog_stats[0]['hullv'] = ids[hull.vertices]
                        elif len(pos0[bool_ids][rs[bool_ids] < 1.1*rad_before*self.meter]) >=4:
                            ids_pos = pos0[bool_ids][rs[bool_ids] < 1.1*rad_before*self.meter]
                            prog_stats[0]['hullv'] = ids[bool_ids][rs[bool_ids] < 1.1*rad_before*self.meter][ConvexHull(ids_pos).vertices]
                            #print(prog_stats[0]['hullv'])
                        else:
                            prog_stats[0]['hullv'] = []
                    else:
                        prog_stats[0]['hullv'] = []
                    # if found_sol:
                    #     prog_stats[0]['rad'] = rvir0/self.meter
                    # else:
                    prog_stats[0]['rad'] = rvir0/self.meter #min(rs[bool_ids].max(),rad_max)/self.meter
                    #print('pere',rs[bool_ids].max()/(rad_before*self.meter))
                    self.halotree[halo]['Forced'] += 1
                    if dist_bool:
                        self.halotree[halo]['Consec_Forced'] += 1
                    if (not self.multi or (self.multi and rank==0)) and self.verbose:
                        print('Halo %s Progagated, Cden: %s Target: %s Rad %s' % (self.halo_num[r],cden[rad_i],oden,prog_stats[0]['rad']))
                        print('Forced: %s' % self.halotree[halo]['Forced'])
                    return 0,prog_stats
                # else:
                #     #print(cden,self.halo_num[r],len(mass))
                #     return -1,prog_stats
            else:
            #print(cden,self.halo_num[r],len(mass))
                    if (not self.multi or (self.multi and rank==0)) and self.verbose:
                        print('Halo %s Not Progagated, Cden_max: %s' % (self.halo_num[r],cden.max()))
                    return -1,prog_stats
        else:
            return -1,prog_stats

    # Takes new halos and propgates them forward in time to catch subhallos and branches
    # missed by the orginal checks in previous timesteps. This runs everytime a new halo is
    # found.
    def forward_model_co_progenitors(self,inters=1):
        #new_halos = list(set(self.halo_num)-set(halo_num))
        time_0 = max(list(self.time.keys()))
        time_i = self.timestep
        # self.make_active_list()
        # self.evolve_com(self.halo_num)
        time_min = time_i
        time_max = min(time_i+(inters*skip*self.interval)+1,time_0)
        intimes = np.array(self.all_times)[::-1]
        intimes_2 = intimes.tolist()
        booltime = (intimes > time_min)*(intimes <=time_max)
        tind = intimes_2.index(min(intimes[booltime]))
        self.direction = min(intimes_2[tind-1]-intimes_2[tind],-1)
        self.send_halotree()
        #self.timestep = intimes[booltime][0]
        # print(intimes[booltime],time_min,time_max,self.all_times)
        for timeback in intimes[booltime]:
          self.time6 = time_sys.time()
          self.timestep = timeback
          tind = intimes_2.index(self.timestep)
          self.direction = min(intimes_2[tind-1]-intimes_2[tind],-1)
          self.make_active_list(backlook=self.direction)
          self.evolve_com(self.halo_num)
          # print(self.direction)
          new_halos = []
          all_halos = []
          if red_list_1[self.timestep] < 35:
            # for halos in self.halo_num:
                #if rank==0 and self.verbose:
                #    print(halos,self.timestep + 1 in self.halotree[halos],self.timestep in self.halotree[halos],\
                #        self.timestep + 2 not in self.halotree[halos])
                # if  intimes_2[tind-1] in self.halotree[halos] and timeback not in self.halotree[halos] and self.lenhalo[halos] > 1:
                #     new_halos.append(halos)
                # if intimes_2[tind-1] in self.halotree[halos] or timeback in self.halotree[halos]:
                #     all_halos.append(halos)
            #self.make_active_list()
            # if rank ==0:
            #     print(self.halo_num,self.new_coml)
            halo_r = self.halo_r.astype(int) #np.arange(len(self.halo_num))[np.isin(self.halo_num,new_halos)]
            if len(halo_r) > 0:
                self.rad_list_all,self.mass_list_all = np.copy(self.rad_list),np.copy(self.mass_list_all)
                self.halo_num,self.com_list,self.rad_list, = self.halo_num[halo_r],self.com_list[halo_r],self.rad_list[halo_r]
                self.npart_list,self.mass_list,self.vcom_list = self.npart_list[halo_r],self.mass_list[halo_r],self.vcom_list[halo_r]
                self.cden = self.cden[halo_r]
                self.forced_count,self.con_forced_count = self.forced_count[halo_r],self.con_forced_count[halo_r]
                self.box_width = self.box_width[halo_r]
                # Enusres that the halo-finding runs forward with time
                self.new_coml = self.new_coml[halo_r]
                self.ds,self.meter = open_ds(self.timestep,self.codetp,direction=self.direction)
                massl = np.array([])
                cdenl = np.array([])
                idl = []
                nl0 = []
                idl0,massl0,nl,rad,com,vcom,rvir_l,cden_l,cdens,hullv,more_var_l,var_names_scalar,var_names_vec,more_vec = self.determine_progenitors(ntries=[1,7,4])
                if rank==0 and self.tracktime:
                    self.time_forward = time_sys.time()
                lenmassl = len(massl0)
                self.indices = np.array([])
                for g in range(len(massl0)):
                    massl = np.append(massl,massl0[g])
                    idl.append(idl0[g])
                    nl0.append([nl[g][0],nl[g][1]])
                if 1==1:#lenmassl >0:
                    halo_l = list(self.halotree.keys())
                    these_halos = np.array([])
                    for h in halo_l:
                        if self.timestep in self.halotree[h]:
                            these_halos = np.append(these_halos,h)
                    jobs,sto_false = job_scheduler(np.arange(len(these_halos)))
                    ranks = np.arange(nprocs)
                    sto = {}
                    sto_l = ['rad','com','vcom','rvir_l','cden_l','cden','massl','nl0','hullv']
                    for ranki in ranks:
                        sto[ranki] = {}
                        for stoi in sto_l:
                            if stoi != 'hullv' and stoi != 'nl0':
                                sto[ranki][stoi] = np.array([])
                            else:
                                sto[ranki][stoi] = []
                    for rank_now in ranks:
                        if rank == rank_now:
                            for t in jobs[rank]:
                                h = these_halos[t]
                                rvir_l_i = np.zeros(len(oden_list))
                                cden_l_i = 5e3*np.ones(len(oden_list))
                                key_list = list(self.halotree[h][self.timestep].keys())
                                r_keys = [x[1:] for x in key_list if x[0] =='r']
                                for l,oden in enumerate(r_keys):
                                    rvir_l_i[l] = self.halotree[h][self.timestep]['r'+str(oden)]
                                    cden_l_i[l] = int(float(oden))
                                if len(sto[rank]['rad'])==0:
                                    sto[rank]['com'] = np.array([self.halotree[h][self.timestep]['Halo_Center']])
                                    sto[rank]['vcom'] = np.array([self.halotree[h][self.timestep]['Vel_Com']])
                                    sto[rank]['rvir_l'] = rvir_l_i[np.newaxis,:]
                                    sto[rank]['cden_l'] = cden_l_i[np.newaxis,:]
                                else:
                                    sto[rank]['com'] = np.vstack((sto[rank]['com'],np.array([self.halotree[h][self.timestep]['Halo_Center']])))
                                    sto[rank]['vcom'] = np.vstack((sto[rank]['vcom'],np.array([self.halotree[h][self.timestep]['Vel_Com']])))
                                    sto[rank]['rvir_l'] = np.vstack((sto[rank]['rvir_l'],rvir_l_i[np.newaxis,:]))
                                    sto[rank]['cden_l'] = np.vstack((sto[rank]['cden_l'],cden_l_i[np.newaxis,:]))
                                sto[rank]['rad'] = np.append(sto[rank]['rad'],self.halotree[h][self.timestep]['Halo_Radius'])
                                sto[rank]['cden'] = np.append(sto[rank]['cden'],self.halotree[h][self.timestep]['cden'])
                                sto[rank]['massl'] = np.append(sto[rank]['massl'],self.halotree[h][self.timestep]['Halo_Mass'])
                                sto[rank]['nl0'].append([0,h])
                                sto[rank]['hullv'].append(self.hullv[h][self.timestep])
                    if rank ==0:
                        for h in these_halos:
                                idl.append(self.idl[h][self.timestep])
                    idl = comm.bcast(idl,root=0)
                    for rank_now in ranks:
                            sto[rank_now] = comm.bcast(sto[rank_now], root=rank_now)
                            massl = np.append(massl,sto[rank_now]['massl'])
                            if len(rad)==0:
                                com = sto[rank_now]['com']
                                vcom = sto[rank_now]['vcom']
                                rvir_l = sto[rank_now]['rvir_l']
                                cden_l = sto[rank_now]['cden_l']
                            elif len(sto[rank_now]['com'])>0:
                                com = np.vstack((com,sto[rank_now]['com']))
                                vcom = np.vstack((vcom,sto[rank_now]['vcom']))
                                rvir_l = np.vstack((rvir_l,sto[rank_now]['rvir_l']))
                                cden_l = np.vstack((cden_l,sto[rank_now]['cden_l']))
                            for t in range(len(sto[rank_now]['rad'])):
                                hullv.append(sto[rank_now]['hullv'][t])
                                nl0.append(sto[rank_now]['nl0'][t])
                            rad = np.append(rad,sto[rank_now]['rad'])
                            cdens = np.append(cdens,sto[rank_now]['cden'])
                            sto[rank_now] = None
                            # hullv.append(self.hullv[h][self.timestep])
                            # nl0.append([0,h])
                            # key_list = list(self.halotree[h][self.timestep].keys())
                            # r_keys = [x[1:] for x in key_list if x[0] =='r']
                            # rvir_l_i = np.zeros(len(oden_list))
                            # cden_l_i = 5e3*np.ones(len(oden_list))
                            # for t,oden in enumerate(r_keys):
                            #     rvir_l_i[t] = self.halotree[h][self.timestep]['r'+str(oden)]
                            #     cden_l_i[t] = int(float(oden))
                            # # if rank ==0:
                            # #     print(rvir_l_i,rvir_l.shape,r_keys)
                            # if len(rad)==0:
                            #     com = np.array([self.halotree[h][self.timestep]['Halo_Center']])
                            #     vcom = np.array([self.halotree[h][self.timestep]['Vel_Com']])
                            #     rvir_l = rvir_l_i[np.newaxis,:]
                            #     cden_l = cden_l_i[np.newaxis,:]
                            # else:
                            #     com = np.vstack((com,self.halotree[h][self.timestep]['Halo_Center']))
                            #     vcom = np.vstack((vcom,self.halotree[h][self.timestep]['Vel_Com']))
                            #     rvir_l = np.vstack((rvir_l,rvir_l_i))
                            #     cden_l = np.vstack((cden_l,cden_l_i))
                            # rad = np.append(rad,self.halotree[h][self.timestep]['Halo_Radius'])
                            # cdens = np.append(cdens,self.halotree[h][self.timestep]['cden'])
                            # #m200 = np.append(m200,self.halotree[h][self.timestep]['m200'])
                # if 1==1:
                #     hullv = comm.bcast(hullv,root=0)
                #     idl = comm.bcast(idl,root=0)
                #     massl = comm.bcast(massl,root=0)
                #     nl0 = comm.bcast(nl0,root=0)
                #     rad = comm.bcast(rad,root=0)
                #     com = comm.bcast(com,root=0)
                #     vcom = comm.bcast(vcom,root=0)
                #     rvir_l = comm.bcast(rvir_l,root=0)
                #     #m200 = comm.bcast(m200,root=0)
                #     cden_l = comm.bcast(cden_l,root=0)
                sto = None
                if rank==0 and self.tracktime:
                  print('Gathering tree data Time:',time_sys.time()-self.time_forward)
                  self.time_forward = time_sys.time()
                nl0 = np.array(nl0)
                if len(massl) >0:
                    self.indices = get_non_coincident_halo_index(idl, massl, nl0,rad*self.meter, self.lenhalo,self.escape(massl,rad),\
                        vcom*self.meter,com*self.meter,rvir_l*self.meter,cden_l,margin=0.15,margin2=0.15)
                    #self.indices = self.indices[self.indices < lenmassl]
                else:
                    self.indices = np.arange(len(massl))
                #print('here')
                # Forward-propagated halos are added to the tree and plots are remade
                if rank==0 and self.tracktime:
                  print('Removing overlaping halos Time:',time_sys.time()-self.time_forward)
                  self.time_forward = time_sys.time()
                save_now = False
                if timeback == intimes[booltime].max():
                    save_now = True
                #if rank==0:
                self.build_tree(nl0,massl,idl,rad,com,vcom,rvir_l,cden_l,cdens,hullv,more_var_l,var_names_scalar,var_names_vec,more_vec,limited=False,coprog_on=False,save_now=save_now)
                #self.join_halotree()
                #self.halotree = comm.bcast(self.halotree,root=0)
                self.make_lengths()
                nl0,massl,idl,rad,com,vcom,rvir_l,cden_l,cdens,hullv,more_var_l,var_names_scalar,var_names_vec,more_vec =0,0,0,0,0,0,0,0,0,0,0,0,0,0
                if rank==0 and self.tracktime:
                  print('Building halotree Time:',time_sys.time()-self.time_forward)
            if rank==0 and self.tracktime:
              print('Forward Modeling done for timestep %s Time:' % self.timestep,time_sys.time()-self.time6)
              self.time6 = time_sys.time()
            if rank==0:
                print('Timestep',self.timestep,'complete')
                if self.traceback and self.track_mem:
                    self.snaps.append(tracemalloc.take_snapshot())
                    back = self.snaps[-1].statistics("traceback")
                    print("\n*** top 10 stats ***")
                    for i in range(10):
                       b = back[i]
                       print("%s b" % b.size)
                       for l in b.traceback.format():
                           print(l)
        # The atributes are reset to continue backward propagation

        self.timestep = time_i
        #self.make_active_list()
        #tind = self.all_times.index(self.timestep)
        #self.direction = self.all_times[tind-1]-self.all_times[tind]


    # Finds the escape velocity of each halo
    def escape(self,massl,rad):
        escape_v = np.array([])
        G = 6.67e-11 * u.m**3/u.s**2/u.kg
        for i in range(len(massl)):
            escape_v = np.append(escape_v,np.sqrt(2*G*massl[i]/self.kg_sun).v)
        return escape_v


    # Updates active halo-finding atributes from the halo tree at the current timestep
    def make_active_list(self,backlook=0,min_time=None):
        self.join_halotree(small_range=True)
        self.mass_list,self.rad_list,self.com_list,self.vcom_list,self.npart_list,self.halo_num=\
            np.array([]),np.array([]),[],[],np.array([]),[]
        self.cden = np.array([])
        self.forced_count = np.array([])
        self.con_forced_count = np.array([])
        self.lookback = []
        self.idl_solved = []
        self.idl_last = []
        self.idl_all = np.array([])
        self.all_masses = np.array([])
        self.all_cden = np.array([])
        self.all_centers = np.array([])
        self.all_radii = np.array([])
        self.box_width = np.array([])
        self.make_lengths()
        min_time_0 = {}
        pass_cond_0 = {}
        pass_cond_1 = {}
        self.oden_vir = oden_vir(self.ds)
        self.halo_r = np.array([])
        if rank ==0:
            for v in self.halotree.keys():
                time_list = [x for x in self.halotree[v].keys() if isinstance(x,numbers.Integral)]
                if self.direction > 0:
                    min_time = max(min(time_list),self.timestep + backlook)
                    pass_cond = (min_time <= (self.timestep + backlook)) and (min_time in self.halotree[v])
                    pass_cond1 = self.timestep not in self.halotree[v]
                else:
                    min_time = self.timestep + backlook
                    pass_cond = min_time in self.halotree[v] and self.timestep > min(self.all_times)
                    pass_cond1 = (self.timestep not in self.halotree[v]) and (self.lenhalo[v] > 0)
                    # min_time = min(max(list(self.idl[v].keys())),self.timestep + backlook)
                    # pass_cond = (min_time in self.halotree[v]) and (min_time >= self.timestep + backlook)
                min_time_0[v] = min_time
                pass_cond_0[v] = pass_cond
                pass_cond_1[v] = pass_cond1
                if self.timestep + backlook in list(self.idl[v].keys()):
                    self.all_masses = np.append(self.all_masses,self.halotree[v][self.timestep + backlook]['Halo_Mass'])
                    self.all_cden = np.append(self.all_cden,self.halotree[v][self.timestep + backlook]['cden'])
                    self.all_radii = np.append(self.all_radii,self.halotree[v][self.timestep + backlook]['Halo_Radius'])
                    if len(self.all_centers) == 0:
                        self.all_centers = self.halotree[v][self.timestep + backlook]['Halo_Center']
                    else:
                        self.all_centers = np.vstack((self.all_centers,self.halotree[v][self.timestep + backlook]['Halo_Center']))
        min_time_0 = comm.bcast(min_time_0,root=0)
        pass_cond_0 = comm.bcast(pass_cond_0,root=0)
        self.all_masses = comm.bcast(self.all_masses,root=0)
        self.all_cden = comm.bcast(self.all_cden,root=0)
        self.all_radii = comm.bcast(self.all_radii,root=0)
        self.all_centers = comm.bcast(self.all_centers,root=0)
        for v in self.halotree.keys():
            min_time = min_time_0[v]
            pass_cond = pass_cond_0[v]
            if rank ==0 and min_time in self.idl[v]:
                self.idl_all = np.append(self.idl_all,np.random.choice(self.idl[v][min_time],size=min(5,len(self.idl[v][min_time])),replace=False))
            if pass_cond:# or (self.timestep in self.idl[v]):
                # if rank==0 and self.verbose:
                #     print(v,self.timestep,len(self.com_list),self.halotree[v][self.timestep]['Halo_Center'])
                rads = np.array([key[1:] for key in self.halotree[v][min_time].keys() if key[0]=='r'])
                rads = rads[np.argsort(rads.astype(float))]
                if (rads.astype(float)<=199).sum() >0:
                    rad_avg = self.halotree[v][min_time]['r'+str(rads[rads.astype(float)<=199][0])]
                else:
                    rad_avg = 0
                if 'd_com_coe' not in self.halotree[v][min_time]:
                    print(v,min_time,'Fail',self.halotree[v][min_time])
                if self.halotree[v][min_time]['d_com_coe'] != None:
                    rad_avg += max(self.halotree[v][min_time]['d_com_coe'],self.halotree[v][min_time]['d_com_cohe'])*self.halotree[v][min_time]['Halo_Radius']
                rad_avg = max(rad_avg,self.halotree[v][min_time]['Halo_Radius'])
                if self.timestep + 2*self.direction in self.halotree[v]:
                    rad_avg = max(rad_avg,self.halotree[v][self.timestep + 2*self.direction]['Halo_Radius'])
                self.box_width = np.append(self.box_width,rad_avg)
                if len(self.mass_list) == 0:
                    self.com_list = np.array([self.halotree[v][min_time]['Halo_Center']])
                    self.vcom_list = np.array([self.halotree[v][min_time]['Vel_Com']])
                else:
                    self.com_list = np.vstack((self.com_list,self.halotree[v][min_time]['Halo_Center']))
                    self.vcom_list = np.vstack((self.vcom_list,self.halotree[v][min_time]['Vel_Com']))
                self.mass_list = np.append(self.mass_list,self.halotree[v][min_time]['Halo_Mass'])
                self.rad_list = np.append(self.rad_list,self.halotree[v][min_time]['Halo_Radius'])
                self.npart_list = np.append(self.npart_list,int(self.halotree[v][min_time]['NumParts']))
                if self.lenhalo[v] >0:
                    self.forced_count = np.append(self.forced_count,int(self.halotree[v]['Forced'])/self.lenhalo[v])
                else:
                    self.forced_count = np.append(self.forced_count,0)
                self.con_forced_count = np.append(self.con_forced_count,int(self.halotree[v]['Consec_Forced']))
                self.cden = np.append(self.cden,self.halotree[v][min_time]['cden'])
                self.halo_num.append(v)
                if rank == 0 and pass_cond_1[v]:
                        self.halo_r = np.append(self.halo_r,len(self.mass_list)-1)
                        try:
                            if self.direction >0:
                                self.idl_solved.append(self.idl[v][self.halotree[v]['Solved'][0]])
                            else:
                                self.idl_solved.append(self.idl[v][self.halotree[v]['Solved'][1]])
                        except:
                            self.idl_solved.append(self.idl[v][min_time])
                        self.idl_last.append(self.idl[v][min_time])
                if self.direction > 0:
                    self.lookback.append(min_time - self.timestep)
                else:
                    self.lookback.append(self.direction + backlook)
        self.idl_solved = comm.bcast(self.idl_solved,root=0)
        self.idl_last = comm.bcast(self.idl_last,root=0)
        self.halo_r = comm.bcast(self.halo_r,root=0)
        self.idl_all = comm.bcast(self.idl_all,root=0)
        if len(self.rad_list) > 0:
            radmin = self.rad_list[self.rad_list > 0].min()
            self.rad_list = np.maximum(self.rad_list,radmin)
        self.mean_cden = self.oden_vir
        #if len(self.all_masses) >0:
        #    self.mean_cden = (self.all_cden*self.all_masses).sum()/self.all_masses.sum()
        self.halo_num,self.lookback = np.array(self.halo_num),np.array(self.lookback)
        self.rad_list_all = np.copy(self.rad_list)
        self.mass_list_all = np.copy(self.mass_list)
        self.halo_num_all = np.copy(self.halo_num)



    # Tries up to seven configurations of detect_halos to try to find the progenitors to a tolerance
    def find_progenitor_halos(self,r,ntries=[1,2],check=False):
            trylist = ['First_Try','Second_Try','Third_Try','Fourth_Try','Fifth_Try','Sixth_Try','Seventh_Try']
            bool_larger = self.rad_list > 2*self.rad_list[r]
            #inside_halo = np.sum(np.linalg.norm(self.new_coml[r]-self.new_coml[bool_larger],axis=1)/self.rad_list[bool_larger] < 1.1)
            #print(inside_halo,self.halo_num[r])
            # if inside_halo == 0:
            bool_mass = self.mass_list_all>1.3*self.mass_list[r]
            dist_bigger = np.linalg.norm(self.new_coml_all[bool_mass]-self.new_coml[r],axis=1)
            dist_bool  = dist_bigger < 1*self.rad_list_all[bool_mass]
            self.dist_bool = dist_bigger < 1.1*self.rad_list_all[bool_mass]
            cden0 = self.cden[r]# max(self.cden[r],175)#max(min(self.cden[r],min(self.mean_cden,300)+150),175)
            cden_max = self.oden_vir+10#210#max(min(self.mean_cden,300),210)
            cden_min = self.oden_vir-10
            target = self.oden_vir
            if dist_bool.sum() != 0:
                dist_bool_1 = np.copy(dist_bool)
                dist_bool_ind = np.arange(len(dist_bool))
                #print(dist_bool[dist_bool ==True])
                if self.direction >0:
                    for i_close,halo_near in enumerate(self.halo_num_all[bool_mass][dist_bool_1]):
                        #print(halo_near,self.halo_num[r],self.lenhalo[halo_near],self.lenhalo[self.halo_num[r]])
                        if self.lenhalo[halo_near] < self.lenhalo[self.halo_num[r]]:
                            i_now = dist_bool_ind[dist_bool_1][i_close]
                            dist_bool[i_now] = False
                #print(dist_bool[dist_bool ==True])
            dist_long = dist_bool.sum() == 0
            #print(dist_long)
            if dist_long:
                cden_change = min((target-cden0)/10,-10)
                rad_now = self.box_width[r]
            else:
                cden_change = 0
                rad_now = self.rad_list[r]
            if self.halotry[r][0] ==1:
                rad_now = self.box_width[r]
                #dist_bigger_bool = dist_bigger < 1.5*self.rad_list_all[bool_mass]
                #print(dist_bigger[dist_bigger_bool]/self.rad_list_all[bool_mass][dist_bigger_bool],self.mass_list_all[bool_mass][dist_bigger_bool]/self.mass_list[r])
            #print(self.halo_num[r],np.sum(dist_bigger<1.5*self.rad_list_all[bool_mass]),cden_change,cden0)
            # else:
            #     cden0 = max(min(self.cden[r],min(self.mean_cden,300)+400),175)
            #     cden_max = max(min(self.mean_cden+100,300),210)
            # if not self.halo_num[r] in np.arange(10).astype(str):
            #     cden_max = max(min(self.mean_cden,300)+50,300)
            # else:
            if cden0 > cden_max or cden0 < cden_min:
                if np.sign(self.oden_vir-cden0) < 0:
                    cdenr = cden0 + cden_change
                else:
                    cdenr = cden0 + np.sign(target-cden0)*min(5,abs(target-cden0))
                cden_plus = cden0
                cden_minus = cden0 - np.sign(target-cden0)*5
            else:
                cden_plus = cden0 + 5
                cdenr = cden0
                cden_minus = cden0 - 5
            rmaxs = [2.8, 2.8, 2.5, 3, 3, 2.5,2.5,2.8] # size of the radius to search for halos
            nclusters = [6, 5, 1, 6, 2, 1, 10,5] # maximum number of halos to search for
            thresholds = [.3, .75, .85, .85, .501, .65, .85,.501] # how much of the progenitor must be part of the descendent
            second_thresholds = [0.4, 0.15, 0.5, 0.2, .5, .2, .35,0.35] # Limits how much mass can be lost to progentors
            odenlist = [150,cdenr,cdenr,200,cden_plus,cdenr,200,cdenr]
            dense_rad = [False,False,False,False,False,False,False,False]
            # Confirms local halos found with the halo finder are gravitationally contained for progenitor checks
            parent_r = r
            # if self.direction >0:
            nclusters_i = int(max(2*np.log10(max(self.mass_list[r],1))-6,1))
            # else:
            #     nclusters_i = min(max(int(2*np.log10(self.mass_list[r]))-6,1),5)
            if self.progs:
              if len(self.halo_num[r].split('_')) >1:
                split_halo = self.halo_num[r].split('_')
                parent = split_halo[0]
                for x in split_halo[1:-1]:
                  parent += '_'+x
                if parent_r in self.halo_num:
                    parent_r = self.halo_num.index(parent_r)
            # if self.fof[self.timestep] and self.direction >0:# and self.progs:
            #     self.add_fof(r)
                # self.direction = self.lookback[r]
            lenid = len(self.idl_last[r])
            self.cut1 = 1
            idls_ind = [None,self.idl_last[r][0::self.cut1]]
            if lenid > self.maxids and self.progs:
                rand_ids = np.random.choice(np.arange(len(self.idl_last[r])),size =self.maxids, replace=False)
                ids_r = self.idl_last[r][rand_ids]
                ids_r = np.append(ids_r,self.idl_last[r][np.isin(self.idl_last[r],self.idl_all)])
                ids_r = np.unique(ids_r)
                idls_ind = [None,self.idl_last[r][rand_ids]]
                self.cut1 = lenid/len(rand_ids)
            idls = [1,1,1,0,1,1,1,1]
            ntries_main = np.copy(ntries).tolist()
            ntries_extra = []
            if not check:
                ntries_main = np.array(list(set(ntries_main)-set(ntries_extra)))
            else:
                ntries_main = np.append(np.array(ntries_main),np.array(ntries_extra))
            ntries_main = pd.unique(ntries_main)
            main_prog = -1
            prog_stats = {}
            first_round = True
            if ntries_main[self.halotry[r][0]]  == 4:
                first_round = False
            use_hulls=True
            if len(ntries_main) >0:
                i = int(ntries_main[self.halotry[r][0]])
                second_i = second_thresholds[i]
                if self.lenhalo[self.halo_num[r]] ==2 and self.halotry[r][0]  != 0:
                    second_i += 0
                if self.progs:
                    radmax_now = self.get_radmax(r,rmaxs[i])
                    ind = self.halo_num[r]+'_'+str(i)+'_'+str(self.timestep)
                else:
                    ind = self.halo_num[r][1]+'_'+str(i)+'_'+str(self.timestep)+'_prog'
                    radmax_now = rmaxs[i]
                DH = detect_halos(self.ds,self.meter,self.new_coml[r],rad_now,ind=ind,\
                    multi_solver=self.multi,plot=self.plot,radmax=radmax_now,nclustermax=nclusters_i,codetp=self.codetp,idl_i=idls_ind[idls[i]],\
                        oden=odenlist[i],hmass=self.mass_list[r],reg=self.reg_info,first_round=first_round,use_hulls=use_hulls)
                scaling = DH.scaling
                if self.progs:
                    main_prog,prog_stats,mass_prog,frac_prog = self.get_progenitors(DH,r,\
                        threshold=thresholds[i], second_threshold=second_i)
                # else:
                #     main_prog,prog_stats = self.get_progs(DH,r)
            # Forces the propagation of the halo if self.force is true
            push = False
            if main_prog != -1:
                key = list(prog_stats.keys())[0]
                self.halotree[self.halo_num[r]]['Consec_Forced'] = 0
                dist_bigger = np.linalg.norm(self.new_coml_all[self.mass_list_all>self.mass_list[r]]-self.new_coml[r],axis=1)
                dist_bool  = np.sum(dist_bigger < 1*self.rad_list_all[self.mass_list_all>self.mass_list[r]]) == 0
                if dist_bool:
                    prog_stats[key]['Solved'] = True
            elif main_prog == -1 and self.halotry[r][0] + 1 == len(ntries_main) and len(ntries_main) >0 and self.progs\
                and (self.lenhalo[self.halo_num[r]] > 2 or self.halo_num[r]=='0') and self.halotree[self.halo_num[r]]['Consec_Forced'] <= min(self.interval+2,len(self.all_times)/2):
                  if self.make_force:
                    # if cden0 > cden_max+10 or cden0 < 199:
                    #     cdenr = cden0 + np.sign(target-cden0)*10
                    # else:
                    cdenr = cden0
                    push = True
                    main_prog,prog_stats = self.propagate_halo(DH,r,reg=self.reg_info,oden=cdenr)
                    if self.verbose and main_prog != -1 and (not self.multi or (self.multi and rank==0)):
                        print('Forced particles in halo %s to go next timestep' % self.halo_num[r])
                    elif self.verbose and main_prog == -1 and (not self.multi or (self.multi and rank==0)):
                        print('Forced propagation failed for halo %s. No virial radius' % self.halo_num[r])
                #print(self.halo_num[r],self.halotree[self.halo_num[r]]['Solved'])
            if self.verbose and not push and self.progs:
                if main_prog ==-1 and (not self.multi or (self.multi and rank==0)) and self.halotry[r][0] + 1 == len(ntries_main):
                    print('Fail',trylist[self.halotry[r][0]],mass_prog,frac_prog,self.mass_list[r],self.npart_list[r],self.halo_num[r],cden0,self.cden[r])
                elif (not self.multi or (self.multi and rank==0)) and main_prog !=-1:
                    i = int(ntries_main[self.halotry[r][0]])
                    print('Success,',trylist[self.halotry[r][0]],'The mass fraction of the main progenitor is: ',prog_stats[0]['mass'][0]/self.mass_list[r],\
                            main_prog,len(prog_stats),self.halo_num[r],prog_stats[0]['cden'],self.cden[r])
                    if len(prog_stats) > 1:
                        for i in prog_stats:
                            if i > 0:
                                print('The mass fraction of co-progenitor %s of %s is :' % (i,self.halo_num[prog_stats[i]['prog']]), prog_stats[i]['mass'][0]/self.mass_list[r])
                            if i < 0:
                                print('New halo created with mass %s' % prog_stats[i]['mass'][0])
            halotry_r = [self.halotry[r][0]+1,main_prog,len(ntries_main)]
            return main_prog,prog_stats,halotry_r,scaling

    # Projects the center of mass forward or backward in time
    def evolve_com(self,halo_num):
        new_com_l = []
        for k,halo in enumerate(halo_num):
          if self.timestep in self.halotree[halo]:
             new_com_l.append(self.halotree[halo][self.timestep]['Halo_Center'])
          else:
            com_0 = self.halotree[halo][self.timestep+self.direction]['Halo_Center']
            vcom_0 = self.halotree[halo][self.timestep+self.direction]['Vel_Com']
            dt1 = self.time[self.timestep] - self.time[self.timestep+self.direction]
            accel = 0
            dt2 = 1
            if self.lenhalo[halo] > 1 and self.timestep+2*self.direction in self.halotree[halo]:
                vcom_1 = self.halotree[halo][self.timestep+2*self.direction]['Vel_Com']
                dt2 = self.time[self.timestep + self.direction] - self.time[self.timestep + 2*self.direction]
                if dt2 != 0:
                    accel = (vcom_0 - vcom_1)/dt2
            new_com_l.append(com_0 + vcom_0*dt1 + 0.5*accel*(dt1**2))
        self.new_coml = np.array(new_com_l)
        self.new_coml_all = np.array(new_com_l)


    # Creates a halo list from the node list created in the Initial_Halo_Finder class
    def list_halos_from_nodes(self,fnodes):
        com_list,rad_list,mass_list,id_list,npart_list,vcom_list,halo_num,cdens = [],[],[],[],[],[],[],[]
        self.halotree = {}
        self.hullv = {}
        index = list(fnodes.keys())
        index = natsorted(index)
        self.idl = {}
        self.lenhalo = {}
        if rank==min(1,nprocs):
            self.star_id = {}
            self.star_id['ids'] = {}
            self.star_id['energies'] = {}
        for halo in index:
            self.hullv[str(halo)] = {}
            if rank ==0:
                self.idl[str(halo)] = {}
            self.halotree[str(halo)] = {}
            self.halotree[str(halo)][self.timestep] = {}
            for key in fnodes[halo].keys():
                if key != 'Part_Ids' and key != 'hullv' and key not in self.halotree[str(halo)] \
                    and key != 'star_id' and key != 'star_energy':
                    self.halotree[str(halo)][self.timestep][key] = fnodes[halo][key]
            self.halotree[str(halo)]['Num_Progs'] = 0
            self.halotree[str(halo)]['Forced'] = 0
            self.halotree[str(halo)]['Consec_Forced'] = 0
            self.halotree[str(halo)]['Solved'] = [self.timestep,self.timestep]
            self.lenhalo[str(halo)] = 1
            com_list.append(fnodes[halo]['Halo_Center'].tolist())
            rad_list.append(fnodes[halo]['Halo_Radius'])
            mass_list.append(fnodes[halo]['Halo_Mass'])
            npart_list.append(fnodes[halo]['NumParts'])
            vcom_list.append(fnodes[halo]['Vel_Com'])
            cdens.append(fnodes[halo]['cden'])
            halo_num.append(halo)
            self.hullv[str(halo)][self.timestep] = fnodes[halo]['hullv']
            if rank ==0:
                self.idl[str(halo)][self.timestep] = fnodes[halo]['Part_Ids']
            if rank ==min(nprocs-1,1):
                if 'star_id' in fnodes[halo]:
                    self.star_id['ids'][str(halo)] = {}
                    self.star_id['energies'][str(halo)] = {}
                    self.star_id['ids'][str(halo)][self.timestep] = fnodes[halo]['star_id']
                    self.star_id['energies'][str(halo)][self.timestep] = fnodes[halo]['star_energy']
        if rank==0:
            np.save(savestring +  '/' + 'halotree_%s.npy' % (fldn),self.halotree)
            np.save(savestring +  '/' + 'idl_%s.npy' % (fldn),self.idl)
            np.save(savestring +  '/' + 'hullv_%s.npy' % (fldn),self.hullv)
        if rank == min(nprocs-1,1):
            if len(self.star_id['ids']) > 0:
                np.save(savestring +  '/' + 'star_id_%s.npy' % (fldn),self.star_id)
        self.com_list =  np.array(com_list)
        self.vcom_list = np.array(vcom_list)
        self.npart_list = np.array(npart_list)
        self.rad_list = np.array(rad_list)
        self.mass_list = np.array(mass_list)
        self.halo_num = np.array(halo_num)
        self.cden = np.array(cdens)

    # Treats every local halo as a possible progenitor and confirms their attributes
    # def add_fof(self,r):
    #     com_list,rad_list = self.dens[self.timestep].coms,self.dens[self.timestep].rads
    #     fof_list  = np.arange(len(rad_list))
    #     if len(rad_list) >0:
    #         dist = np.linalg.norm(com_list-self.new_coml[r],axis=1)
    #         bool_dist = (dist < 4*self.rad_list[r])
    #         if self.rad_list[r] != max(self.rad_list):
    #             bool_dist = (dist < 4*self.rad_list[r])
    #         else:
    #             bool_dist = np.full(len(fof_list),True)
    #         self.fof_r = fof_list[bool_dist]
    #     else:
    #         self.fof_r = fof_list
    #     self.fof_r = np.array([])

        # # print(self.fof_r,rank,r)

    def get_progs(self,DH,r):
        prog_stats = {}
        coms,rads,massl,idl,vcom,rvir_l,cdens = DH.coms,DH.rads,DH.massl,DH.idl,DH.vcom,DH.rvir_l,DH.cdens
        #m200 = DH.m200
        max_in = 0
        main_prog = -1
        for k in range(len(rads)):
            membership_main = (np.isin(idl[k],self.idl_solved[r])).sum()/len(idl[k])
            if membership_main > max_in:
                main_prog = k
                max_in = membership_main
        # if main_prog != -1:
        #     print(self.halo_num[r],main_prog)
        if main_prog != -1:
            membership_desc = (np.isin(idl[main_prog],main_ids)).sum()/len(idl[main_prog])
            if (self.halo_num[r][0] < 0 and membership_desc == 0) or (self.halo_num[r][0]>0 and membership_desc >0):
                num_prog = 0
                prog_stats[num_prog] = {}
                prog_stats[num_prog]["prog"] = self.halo_num[r][1]
                prog_stats[num_prog]['com'],prog_stats[num_prog]['rad'],prog_stats[num_prog]['mass'] = \
                        coms[main_prog][np.newaxis,:],np.array([rads[main_prog]]),np.array([massl[main_prog]])
                prog_stats[num_prog]['idl'],prog_stats[num_prog]['vcom'] =\
                      idl[main_prog],vcom[main_prog][np.newaxis,:]
                prog_stats[num_prog]['rvir_l'] = rvir_l[main_prog][np.newaxis,:]
                prog_stats[num_prog]['cden'] = np.array([cdens[main_prog]])
                #prog_stats[num_prog]['m200'] = np.array([m200[main_prog]])
        return main_prog,prog_stats


    # Finds all co-progenitors and checks the smaller co-progenitors for consistency
    def get_progenitors(self,DH,r,threshold=.85,second_threshold=0.1):
        # Adds halo-finding results to kmeans results for progenitor searches
        coms,rads,massl,idl,vcom,rvir_l,cdens = DH.coms,DH.rads,DH.massl,DH.idl,DH.vcom,DH.rvir_l,DH.cdens
        hullv = DH.hull_l
        com_ids = DH.com_ids
        self.com_ids[r] = DH.com_ids
        #m200 = DH.m200
        more_var_l = DH.more_variables
        var_names_scalar = DH.var_names_scalar
        more_var_vec = DH.more_vec
        var_names_vec = DH.var_names_vec
        cden_l = DH.cden_l
        lenidl = len(rads)
        # count = 0
        # if self.fof[self.timestep] and self.direction >0:
        #     for i in self.fof_r:
        #             if len(rads) == 0:
        #                  coms = np.array([self.dens[self.timestep].coms[i]])
        #                  vcom = np.array([self.dens[self.timestep].vcom[i]])
        #                  rvir_l = np.array([self.dens[self.timestep].rvir_l[i]])
        #                  cden_l = np.array([self.dens[self.timestep].cden_l[i]])
        #                  more_var_l = np.array([self.dens[self.timestep].more_variables[i]])
        #             else:
        #                 coms = np.vstack((coms,self.dens[self.timestep].coms[i]))
        #                 vcom = np.vstack((vcom,self.dens[self.timestep].vcom[i]))
        #                 rvir_l = np.vstack((rvir_l,self.dens[self.timestep].rvir_l[i]))
        #                 cden_l = np.vstack((cden_l,self.dens[self.timestep].cden_l[i]))
        #                 more_var_l = np.vstack((more_var_l,self.dens[self.timestep].more_variables[i]))
        #             var_names_scalar.append(self.dens[self.timestep].var_names_scalar[i])
        #             var_names_vec.append(self.dens[self.timestep].var_names_vec[i])
        #             more_var_vec.append(self.dens[self.timestep].more_vec[i])
        #             rads = np.append(rads,self.dens[self.timestep].rads[i])
        #             massl = np.append(massl,self.dens[self.timestep].massl[i])
        #             cdens = np.append(cdens,self.dens[self.timestep].cden[i])
        #             #m200 = np.append(m200,self.dens[self.timestep].m200[i])
        #             idl[count+lenidl] = self.dens[self.timestep].idl[i]
        #             hullv[count+lenidl] = self.dens[self.timestep].hullv[i]
        #             count += 1
        Solved = self.halotree[self.halo_num[r]]['Solved']
        if self.direction >0:
            max_time = Solved[0]
        if self.direction <0:
            max_time = Solved[1]
        if max_time not in self.halotree[self.halo_num[r]]:
            max_time = self.timestep + self.direction
        if self.direction>0:
            con_dir_2 = Solved[0]
        elif self.direction <0:
            con_dir_2 = Solved[1]
        mass_ratio_2 = massl/self.halotree[self.halo_num[r]][max_time]['Halo_Mass']
        mass_ratio = massl/self.halotree[self.halo_num[r]][con_dir_2]['Halo_Mass']#massl/self.mass_list[r]#self.halotree[self.halo_num[r]][self.timestep+self.direction]['Halo_Mass']
        mass_ratio_diff = abs(mass_ratio - 1)
        mass_ratio_last = massl/self.mass_list[r]
        mass_ratio_diff_last = abs(mass_ratio_last - 1)
        if len(coms) >1:
            rad_dist = np.linalg.norm(coms-com_ids,axis=1)/self.rad_list[r]
            vcom_diff = np.dot(vcom,self.vcom_list[r])/(np.linalg.norm(self.vcom_list[r])*np.linalg.norm(vcom,axis=1))
        elif len(coms)==1:
          rad_dist = np.array([np.linalg.norm(coms-com_ids)/self.rad_list[r]])
          vcom_diff = np.dot(vcom,self.vcom_list[r])/(np.linalg.norm(self.vcom_list[r])*np.linalg.norm(vcom))
        # if len(coms) >= 1:
        #    print(self.halo_num[r],vcom_diff)
        arg = np.argsort(mass_ratio_diff)
        index = np.arange(len(rads))[arg]
        dist_1 = np.linalg.norm(self.new_coml-com_ids,axis=1)/self.rad_list[r]
        #print(self.halo_num[r],np.linalg.norm(self.com_list[r]-com_ids)/self.box_width[r], np.linalg.norm(self.new_coml[r]-com_ids)/self.box_width[r],np.log10(self.mass_list[r]))
        dist_b = dist_1 <4
        dist_c = dist_1 <2
        prog_stats = {}
        main_prog = -1
        progs = []
        non_progs = []
        check_l = ['0','1','2']# np.arange(100).astype(str)#['0','1','2']
        membership_max = 0
        recovered = False
        mass_prog = []
        frac_prog = []
        real_prog = []
        # print(self.halo_num[r],index,lenidl,len(rads))
        #membership_max = 0
        membership_max_1 = 0
        # if len(mass_ratio_diff) > 0:
        #     diff_min = min(mass_ratio_diff)
        #     rad_min = min(rad_dist)
        # Begins searching for progenitors with halos that have masses closest to the descendant
        main_list = []
        rad_lim = 1e10
        diff_min = 1e10
        cost_function_all = 1e10
        cden_last = self.halotree[self.halo_num[r]][self.timestep+self.direction]['cden']
        cden_max = self.halotree[self.halo_num[r]][max_time]['cden']
        cden_rat_last = np.maximum(cdens/cden_last,1)
        cden_rat_last_2 = np.maximum(cden_last/cdens,1)
        cden_rat_max = np.maximum(cdens/cden_max,1)
        mass_ratio_l = np.maximum(1/mass_ratio,1)
        membership_all = np.zeros(len(index))
        for i,k in enumerate(index):
            factor = 1
            overlap = False
          # if k < lenidl:
            # membership_r_0 = mass_ratio_l[k]*cden_rat_max[k]*np.isin(idl[k],self.idl[self.halo_num[r]][max_time]).sum()/min(self.maxids/(self.cut1*self.interval),\
            #     len(self.idl[self.halo_num[r]][max_time]))
            # if self.direction > 0:
            membership_r_1 = cden_rat_last[k]*np.isin(idl[k],self.idl_last[r]).sum()/\
                min(self.maxids,len(self.idl_last[r])) #mass_ratio_l[k]*
            #membership_r_1 /= mass_ratio_l[k]*cden_rat_last[k]
            # else:
            #     membership_r_1 = 0
            # for z in np.arange(len(self.halo_num))[dist_b]:
            #     if z != r and membership_r_1 >0 and self.lenhalo[self.halo_num[z]]>self.lenhalo[self.halo_num[r]]:
            #         cden_last_z = self.halotree[self.halo_num[z]][self.timestep+self.direction]['cden']
            #         cden_rat_last_z = np.maximum(cdens[k]/cden_last_z,1)
            #         mass_ratio_z = 1 #np.maximum(self.mass_list[z]/massl[k],1)
            #         membership_other = mass_ratio_z*cden_rat_last_z*np.isin(idl[k],self.idl_last[z]).sum()\
            #             /min(len(self.idl_last[z]),self.maxids)
            #         if membership_other > membership_r_1:
            #             overlap = True
            #             factor = 1.5
            membership_all[k] = membership_r_1
            #membership_r = membership_r_1 #max(membership_r_0,membership_r_1)
            #membership_max_1 = max(membership_r,membership_max_1)
            #print(membership_r,r,i,len(index),len(DH.coms))
            # if self.halo_num[r]=='0':
            #     print(main_prog,membership_r,mass_ratio_diff[k],np.linalg.norm(coms[k]-self.new_coml[r])/self.rad_list[r] <0.75)
            if membership_r_1 > threshold:# or (k >= lenidl and membership_r_1>0 and self.direction>0)):# and membership_r_1 >= membership_all.max():
                cost_function_k = (1+10*mass_ratio_diff[k])**2 + 1/membership_all[k]**2 + 100*abs(vcom_diff[k]-1) + 10*rad_dist[k]
                # if self.halo_num[r] in check_l:
                #     print(self.halo_num[r],cost_function_k,k,mass_ratio_diff[k],membership_all[k],abs(vcom_diff[k]-1),rad_dist[k],mass_ratio_diff[k]/max(cden_rat_last[k],cden_rat_last_2[k]))
                #     print(self.halo_num[r],cost_function_k,k,mass_ratio_diff[k],membership_all[k],10*abs(vcom_diff[k]-1),10*rad_dist[k])
            # if membership_r > threshold  and mass_ratio_diff[k] < second_threshold\
            #     and mass_ratio_diff[k] <= diff_min*1.2 and membership_r > membership_max\
            #     and rad_dist[k] < min(second_threshold,0.5) and rad_dist[k] < 1.2*rad_lim\
            #     and mass_ratio_diff[k] < 3 and vcom_diff[k] > 0.8:
                if con_dir_2 != self.timestep+self.direction and self.dist_bool.sum()==0:
                    second_k = second_threshold+0.5
                    if self.halotree[self.halo_num[r]]['Consec_Forced'] >0:
                        recovered = True
                elif con_dir_2 != self.timestep+self.direction:
                    second_k = second_threshold+.1
                    #print(self.halo_num[z])
                else:
                    second_k = second_threshold
                #if cost_function_k <= cost_function_all and (mass_ratio_diff[k]/max(cden_rat_last[k],cden_rat_last_2[k]) < second_k/factor or mass_ratio_diff_last[k]/max(cden_rat_last[k],cden_rat_last_2[k]) < second_k/factor): #/max(cden_rat_last[k],cden_rat_last_2[k])
                if cost_function_k <= cost_function_all and (mass_ratio_diff[k] < second_k/factor or mass_ratio_diff_last[k] < second_k/factor):
                #and rad_dist[k] <= rad_lim*1.2 and
            # if  main_prog == -1 and k < lenidl and membership_r > threshold  and mass_ratio_diff[k] < second_threshold:
                # if membership_r > membership_max and mass_ratio_diff[k] <= diff_min*1.2:
                    main_prog = k
                    # print(r,membership_r,membership_max,mass_ratio_diff[k])
                    #main_list.append(k)
                    #membership_max = max(membership_r,membership_max)
                    cost_function_all = min(cost_function_all,cost_function_k)
                    rad_lim = min(rad_dist[k],rad_lim)
                    diff_min = min(mass_ratio_diff[k],diff_min)
        if ((self.halo_num[r] in check_l) and (len(mass_ratio_diff)>0)):# and self.reset is not None:
             #print(main_prog,membership_max,rad_lim,diff_min,vcom_diff[main_prog],membership_max_1)
             print(self.halo_num[r],main_prog,mass_ratio_diff[main_prog],abs(vcom_diff[main_prog]-1),rad_dist[main_prog],membership_all[main_prog],len(index),overlap)
                    #print(self.halo_num[r],k,main_prog,mass_ratio_diff[k],abs(vcom_diff[k]-1),rad_dist[k],membership_all[k],len(index),overlap)
             #print((1+10*mass_ratio_diff[main_prog])**2,1/membership_all[main_prog]**2,100*abs(vcom_diff[main_prog]-1),10*rad_dist[main_prog])
        #print(membership_all)
        # if main_prog == -1 and len(coms)>0:
        #     print(self.halo_num[r],membership_all[index],mass_ratio_diff[index])
        if main_prog != -1:
          dist_p = np.linalg.norm(coms-coms[main_prog],axis=1)/rads[main_prog]
          for i,k in enumerate(index):
            this_prog = r
            membership_max = 0
            if k != main_prog and self.direction > 0:# and k not in main_list:
                #membership_r_0 = cden_rat_max[k]*np.isin(idl[k],self.idl[self.halo_num[r]][max_time]).sum()/min(self.maxids/(self.cut1*self.interval),\
                #    len(self.idl[self.halo_num[r]][self.timestep+self.direction]))
                membership_r_1 = cden_rat_last[k]*np.isin(idl[k],self.idl_last[r]).sum()/\
                    min(self.maxids,len(self.idl_last[r]))
                membership_r =  membership_r_1 #max(membership_r_0,membership_r_1)
                #print(membership_r,r,i,len(index),len(DH.coms))
                for z in np.arange(len(self.halo_num))[dist_b]:
                    if z != r:
                        membership_other = np.isin(idl[k],self.idl_last[z]).sum()\
                            /min(len(self.idl_last[z]),self.maxids)
                        if membership_other > membership_max and self.mass_list[z] > 1.5*massl[k] :
                            membership_max = max(membership_max,membership_other)
                            this_prog = z
                # Checks if the halo is most associated with this descendant rather than another

                        #print(self.halo_num[r],k,mass_ratio_diff,membership_r,massl[k],self.halotree[self.halo_num[r]][self.timestep+self.direction]['Halo_Mass'])
                # if r == self.r_min:
                #     print(membership_r,membership_max,k,lenidl)
                if max(membership_r,membership_max) <= 0.01 and k < lenidl and self.direction > 0:
                        non_progs.append(k)
                        mass_prog.append(massl[k])
                        frac_prog.append(membership_r)
                        real_prog.append(this_prog)
                elif max(membership_r,membership_max) <= 0.01 and k >= lenidl and self.direction > 0:# and r == self.r_min:
                        non_progs.append(k)
                        mass_prog.append(massl[k])
                        frac_prog.append(membership_r)
                        real_prog.append(this_prog)
                elif massl[k]/self.mass_list[this_prog] < 0.3 and max(membership_r,membership_max) > 0.01\
                    and self.direction > 0 and rad_dist[k] < 2.5:# and membership_r:# > 0.8*membership_max:
                        progs.append(k)
                        real_prog.append(this_prog)
                elif k >= lenidl and self.direction > 0:# and r == self.r_min:
                        non_progs.append(k)
                        mass_prog.append(massl[k])
                        frac_prog.append(membership_r)
                        real_prog.append(this_prog)
                # elif massl[k]/self.mass_list[this_prog] < 1 - second_threshold and membership_max > 0.1 and this_prog != r\
                #     and self.direction > 0 and this_prog == z:
                #         progs.append(k)
                #         real_prog.append(this_prog)
                        # print(r,this_prog,massl[k]/self.mass_list[this_prog],membership_r)
                else:
                    real_prog.append(r)
            else:
                real_prog.append(r)
        # if main_prog != -1:
        #      if abs(cdens[main_prog] - self.cden[r]) >49:
        #          main_prog = -1
        # Checks if the mass of the main progenitor is off what is expected, but passes
        # if there are candidate progenitors.
        if main_prog != -1:
            progs,non_progs,real_prog = np.array(progs),np.array(non_progs),np.array(real_prog)
            num_prog = 0
            prog_stats[num_prog] = {}
            prog_stats[num_prog]["prog"] = r
            prog_stats[num_prog]['com'],prog_stats[num_prog]['rad'],prog_stats[num_prog]['mass'] = \
                    coms[main_prog][np.newaxis,:],np.array([rads[main_prog]]),np.array([massl[main_prog]])
            prog_stats[num_prog]['idl'],prog_stats[num_prog]['vcom'] =\
                  idl[main_prog],vcom[main_prog][np.newaxis,:]
            prog_stats[num_prog]['rvir_l'] = rvir_l[main_prog][np.newaxis,:]
            prog_stats[num_prog]['cden_l'] = cden_l[main_prog][np.newaxis,:]
            prog_stats[num_prog]['cden'] = np.array([cdens[main_prog]])
            prog_stats[num_prog]['hullv'] = hullv[main_prog]
            prog_stats[num_prog]['more_variables'] = more_var_l[main_prog][np.newaxis,:]
            prog_stats[num_prog]['var_names_scalar'] = var_names_scalar[main_prog]
            prog_stats[num_prog]['var_names_vec'] = var_names_vec[main_prog]
            prog_stats[num_prog]['more_vec'] = more_var_vec[main_prog]
            #prog_stats[num_prog]['m200'] = np.array([m200[main_prog]])
            if (len(progs) > 0 or len(non_progs) > 0) and self.direction >0:
                # Checks all progenitors to see if they are findable and consistent with all checksself.`
                # Also attempts to find the entire progenitor by searching around its center of mass.`
             if len(progs) > 0:
                arg_prog_0 = progs
                arg_prog_1 = np.full(len(massl),False)
                # arg_prog_0 = progs[progs < lenidl]
                # arg_prog_1 = progs[progs >= lenidl]
                arg_prog_m_0 = np.argsort(massl[arg_prog_0])
                arg_prog_m_1 = np.argsort(massl[arg_prog_1])
                for i in arg_prog_0[arg_prog_m_0]:
                  #print(1,cdens[i],self.rad_boost,(np.linalg.norm(coms[i]-self.new_coml[r]) + 1.75*rads[i])/self.rad_list[r])
                  if cdens[i] > 75 and real_prog[i] ==r and (np.linalg.norm(coms[i]-self.new_coml[r]) + 1.75*rads[i])/self.rad_list[r] < self.rad_boost:
                     #print('yes')
                     DH_2 = detect_halos(self.ds,self.meter,coms[i],rads[i],ind=str(r)+'_'+str(i),\
                         multi_solver=self.multi,plot=self.plot,radmax=1.75,nclustermax=1,codetp=self.codetp,idl_i=idl[i],oden=cdens[i],reg=self.reg_info,hmass=massl[i])
                     found_co = -1
                     for g in range(len(DH_2.rads)):
                         if found_co == -1:
                             membership_co = (np.isin(DH_2.idl[g],idl[i])).sum()/len(DH_2.idl[g])
                             membership_desc = (np.isin(DH_2.idl[g],self.idl_solved[r])).sum()/len(DH_2.idl[g])
                             dist_p = (np.linalg.norm(self.new_coml-DH_2.coms[g],axis=1)/self.rad_list).min() >0.25
                             if np.linalg.norm(DH_2.coms[g] - coms[i])/rads[i] <.25 and \
                                     membership_desc > 0 and DH_2.massl[g]/self.mass_list[r] <1 and membership_co > 0.5\
                                     and DH_2.cdens[g] > 75 and dist_p:
                                num_prog += 1
                                prog_stats[num_prog] = {}
                                prog_stats[num_prog]["prog"] = real_prog[i]
                                prog_stats[num_prog]['com'],prog_stats[num_prog]['rad'],prog_stats[num_prog]['mass'] =\
                                    DH_2.coms[g][np.newaxis,:],np.array([DH_2.rads[g]]),np.array([DH_2.massl[g]])
                                prog_stats[num_prog]['idl'],prog_stats[num_prog]['vcom'] =\
                                      DH_2.idl[g],DH_2.vcom[g][np.newaxis,:]
                                prog_stats[num_prog]['rvir_l'] = DH_2.rvir_l[g][np.newaxis,:]
                                prog_stats[num_prog]['cden_l'] = DH_2.cden_l[g][np.newaxis,:]
                                prog_stats[num_prog]['cden'] = np.array([DH_2.cdens[g]])
                                prog_stats[num_prog]['hullv'] = DH_2.hull_l[g]
                                prog_stats[num_prog]['more_variables'] = DH_2.more_variables[g][np.newaxis,:]
                                prog_stats[num_prog]['var_names_scalar'] = DH_2.var_names_scalar[g]
                                prog_stats[num_prog]['var_names_vec'] = DH_2.var_names_vec[g]
                                prog_stats[num_prog]['more_vec'] = DH_2.more_vec[g]
                                #prog_stats[num_prog]['m200'] = np.array([DH_2.m200[g]])
                                found_co = g
                     #print(found_co,'prog',self.halo_num[r],len(DH_2.rads))
                for i in arg_prog_1[arg_prog_m_1]:
                    dist_p = (np.linalg.norm(self.new_coml-coms[i],axis=1)/self.rad_list).min() >0.25
                    if dist_p:
                        num_prog += 1
                        prog_stats[num_prog] = {}
                        prog_stats[num_prog]["prog"] = real_prog[i]
                        prog_stats[num_prog]['com'],prog_stats[num_prog]['rad'],prog_stats[num_prog]['mass'] =\
                            coms[i][np.newaxis,:],np.array([rads[i]]),np.array([massl[i]])
                        prog_stats[num_prog]['idl'],prog_stats[num_prog]['vcom'] =\
                              idl[i],vcom[i][np.newaxis,:]
                        prog_stats[num_prog]['rvir_l'] = rvir_l[i][np.newaxis,:]
                        prog_stats[num_prog]['cden_l'] = cden_l[i][np.newaxis,:]
                        prog_stats[num_prog]['cden'] = np.array([cdens[i]])
                        prog_stats[num_prog]['hullv'] = hullv[i]
                        prog_stats[num_prog]['more_variables'] = more_var_l[i][np.newaxis,:]
                        prog_stats[num_prog]['var_names_scalar'] = var_names_scalar[i]
                        prog_stats[num_prog]['var_names_vec'] = var_names_vec[i]
                        prog_stats[num_prog]['more_vec'] = more_var_vec[i]
                    #prog_stats[num_prog]['m200'] = np.array([m200[i]])
             if len(non_progs) > 0:
                # arg_prog_0 = non_progs
                # arg_prog_1 = np.full(len(massl),False)
                arg_prog_0 = non_progs[non_progs < lenidl]
                arg_prog_1 = non_progs[non_progs >= lenidl]
                arg_prog_m_0 = np.argsort(massl[arg_prog_0])
                arg_prog_m_1 = np.argsort(massl[arg_prog_1])
                num_prog = -1
                for i in arg_prog_0[arg_prog_m_0]:
                  #print(2,self.rad_boost,(np.linalg.norm(coms[i]-self.new_coml[r]) + 1.75*rads[i])/self.rad_list[r])
                  if cdens[i] > 75 and (np.linalg.norm(coms[i]-self.new_coml[r]) + 1.75*rads[i])/self.rad_list[r] < self.rad_boost and real_prog[i] ==r:
                    DH_2 = detect_halos(self.ds,self.meter,coms[i],rads[i],ind=str(r)+'_'+str(i),\
                        multi_solver=self.multi,plot=self.plot,radmax=1.75,nclustermax=1,codetp=self.codetp,idl_i=idl[i],oden=cdens[i],reg=self.reg_info,hmass=massl[i])
                    found_co = -1
                    for g in range(len(DH_2.rads)):
                        if found_co == -1:
                            membership_co = (np.isin(DH_2.idl[g],idl[i])).sum()/len(DH_2.idl[g])
                            membership_desc = (np.isin(DH_2.idl[g],self.idl_solved[r])).sum()/len(DH_2.idl[g])
                            dist_p = (np.linalg.norm(self.new_coml-DH_2.coms[g],axis=1)/self.rad_list).min() >1
                            if np.linalg.norm(DH_2.coms[g] - coms[i])/rads[i] < .25 and \
                                    membership_desc == 0 and DH_2.cdens[g] > 75 and membership_co > 0.5 and dist_p:
                               num_prog -= 1
                               prog_stats[num_prog] = {}
                               prog_stats[num_prog]["prog"] = real_prog[i]
                               prog_stats[num_prog]['com'],prog_stats[num_prog]['rad'],prog_stats[num_prog]['mass'] =\
                                   DH_2.coms[g][np.newaxis,:],np.array([DH_2.rads[g]]),np.array([DH_2.massl[g]])
                               prog_stats[num_prog]['idl'],prog_stats[num_prog]['vcom'] =\
                                     DH_2.idl[g],DH_2.vcom[g][np.newaxis,:]
                               prog_stats[num_prog]['rvir_l'] = DH_2.rvir_l[g][np.newaxis,:]
                               prog_stats[num_prog]['cden_l'] = DH_2.cden_l[g][np.newaxis,:]
                               prog_stats[num_prog]['cden'] = np.array([DH_2.cdens[g]])
                               prog_stats[num_prog]['hullv'] = DH_2.hull_l[g]
                               prog_stats[num_prog]['more_variables'] = DH_2.more_variables[g][np.newaxis,:]
                               prog_stats[num_prog]['var_names_scalar'] = DH_2.var_names_scalar[g]
                               prog_stats[num_prog]['var_names_vec'] = DH_2.var_names_vec[g]
                               prog_stats[num_prog]['more_vec'] = DH_2.more_vec[g]
                               #prog_stats[num_prog]['m200'] = np.array([DH_2.m200[g]])
                               found_co = g
                    #print(found_co,'non_prog',len(DH_2.rads))
                for i in arg_prog_1[arg_prog_m_1]:
                  dist_p = (np.linalg.norm(self.new_coml-coms[i],axis=1)/self.rad_list).min() >1
                  if cdens[i] > 150 and dist_p:
                    num_prog -= 1
                    prog_stats[num_prog] = {}
                    prog_stats[num_prog]["prog"] = real_prog[i]
                    prog_stats[num_prog]['com'],prog_stats[num_prog]['rad'],prog_stats[num_prog]['mass'] =\
                        coms[i][np.newaxis,:],np.array([rads[i]]),np.array([massl[i]])
                    prog_stats[num_prog]['idl'],prog_stats[num_prog]['vcom'] =\
                          idl[i],vcom[i][np.newaxis,:]
                    prog_stats[num_prog]['rvir_l'] = rvir_l[i][np.newaxis,:]
                    prog_stats[num_prog]['cden_l'] = cden_l[i][np.newaxis,:]
                    prog_stats[num_prog]['cden'] = np.array([cdens[i]])
                    prog_stats[num_prog]['hullv'] = hullv[i]
                    prog_stats[num_prog]['more_variables'] = more_var_l[i][np.newaxis,:]
                    prog_stats[num_prog]['var_names_scalar'] = var_names_scalar[i]
                    prog_stats[num_prog]['var_names_vec'] = var_names_vec[i]
                    prog_stats[num_prog]['more_vec'] = more_var_vec[i]
                    #prog_stats[num_prog]['m200'] = np.array([m200[i]])
        return main_prog,prog_stats,mass_prog,frac_prog


class detect_halos():
    """
    This class detects halos in and around a given center and radius. It can occasionally become
    difficult to detect several halos at once or small halos in the presence of larger halos so
    the finding algorithm runs again if no halos are found near the expected locationself.

    The plotting option allows for visual inspection of the halos.

    Parameters
    ----------
    ds : yt simulation object
    center : array
        The center of the investigation region

    radius : float
        The radius of the investigation region (note that the initial investigation is 2.8*radius)

    res : float
        The minimum resolution for yt regions

    ind : int
        This labels plots by region and is unneccessary if not plotting

    multi_solver : bool
        This sets whether energy calculation are performed in parallel. This requires that multiple
        processors are running this class.

    plot : bool
        Turns on or off plotting of investigation regions for visual inspection.

    codetp : str
        The code used to creathe simulation

    idl_i : list of ints
        The unique ids of a halo. Used to focus the halo search

    Returns
    -------
    None.

    """
    def __init__(self,ds,lu,center,radius = 3.05176e-05,ind=10,multi_solver=False,plot=False,\
                radmax=2,nclustermax=3,codetp='ENZO',idl_i=None,oden=200,undense=False,hmass =0,\
                timing=0,reg=False,first_round=True,scaling_count=False,use_hulls=True):
        # Initialization of shared attributes
        self.final_nodes = {}
        self.ds = ds
        self.lu = lu
        self.use_hulls = use_hulls
        self.crit_den = univDen(self.ds)
        self.kg_sun = self.ds.mass_unit.in_units('Msun').v/self.ds.mass_unit.in_units('kg').v
        self.center,self.radius = center,radius
        self.multi = multi_solver
        self.codetp = codetp
        self.radmax = radmax
        self.idl_i = idl_i
        self.reg = reg
        self.hmass = hmass
        self.first_round = first_round
        # Set current mean density attribute
        self.uden = univDen(self.ds)
        # Create yt region
        if scaling_count:
            len_scaling = len(self.reg['mass'])
        time3 = time_sys.time()
        self.scaling = np.array([])
        # self.numsegs = 0
        # if hmass > 5e9:
        #     self.numsegs = 4
        # if hmass > 5e11:
        #     self.numsegs = 6
        # if len(self.reg['mass']) ==0:
        #     print('fail',self.reg)
        #     print(self.center,self.radius,radmax)
        #     reg_i = get_region(self.ds,self.center,self.radius*radmax)
        #     self.reg['mass'],self.reg['pos'],self.reg['vel'],self.reg['ids'] = pickup_particles(reg_i,self.codetp)
        #reg = get_region(self.ds,self.center,self.radius*radmax,multi=self.numsegs)
        # if time_sys.time()-time3 >10 and timing:
        #         print('First Region Pull',time_sys.time()-time3)
        #         time3 = time_sys.time()
        #reg = self.ds.sphere(self.center,max(self.radius*radmax,res))
        # Set mass, position, velocity, and particle id list attributes
        cut_size = 50
        self.get_mass_pos_vel_ids(self.center*self.lu,self.radius*radmax*self.lu,undense=undense,dense=True,cut_size=cut_size)
        # del reg
        if time_sys.time()-time3 >timing and timing:
                print('First Particle Pull',time_sys.time()-time3)
        if scaling_count:
                len_summary = len(self.mass)
                self.scaling = np.array([self.lenpart,len_summary,len_scaling/(-1*self.tscale),len_scaling/(time_sys.time()-time3)])
                #print('First Particle Pull',len_scaling,len_scaling/(time_sys.time()-time3),rank)
        time3 = time_sys.time()
        # Initialization of the plot of all the particles (from three orthogonal views) and a plot
        # of kinetic energy versus potential energy
        if plot and (not self.multi or (self.multi and rank==0)):
            colors = list(mcolors.TABLEAU_COLORS)
            self.color_count = 0
            fig,self.ax = plt.subplots(4,1,figsize=(9,27),dpi=200)
            #self.ax[1].scatter(self.pos[:,0]/self.lu,self.pos[:,1]/self.lu,color=colors[0],alpha=0.05,marker=".")
            self.ax[1].set_xlabel('x')
            self.ax[1].set_ylabel('y')
            self.ax[1].set_aspect('equal')
            #self.ax[2].scatter(self.pos[:,0]/self.lu,self.pos[:,2]/self.lu,color=colors[0],alpha=0.05,marker=".")
            self.ax[2].set_xlabel('x')
            self.ax[2].set_ylabel('z')
            self.ax[2].set_aspect('equal')
            #self.ax[3].scatter(self.pos[:,1]/self.lu,self.pos[:,2]/self.lu,color=colors[0],alpha=0.05,marker=".")
            self.ax[3].set_xlabel('y')
            self.ax[3].set_ylabel('z')
            self.ax[3].set_aspect('equal')
            self.mink,self.minp,self.maxk,self.maxp = 1e99,1e99,1e-99,1e-99
        # Run the halo finding code to return a list and set of halo charertistic attributes
        self.cluster_type = 'kmean'
        self.var_names_scalar,self.var_names_vec0 = make_var_names()
        self.coms,self.rads,self.massl,self.tmassl,partl,self.idl,self.rvir_l,\
                self.cdens,self.cden_l,self.hull_l,\
                self.more_variables,self.var_names_scalar,self.more_vec,self.var_names_vec = self.find_halo_cores(ncluster=min(6,nclustermax),\
                plot=plot,ind=ind,idl_i=self.idl_i,oden=oden)
        mincom = 0
        moredense = False
        cut_2 = int(cut_size*1.5)
        if time_sys.time()-time3 >timing and timing:
                print('First Check',time_sys.time()-time3)
        if scaling_count:
                self.scaling = np.append(self.scaling,len_scaling/(time_sys.time()-time3))
                #print('First Check',len_scaling,len_scaling/(time_sys.time()-time3),rank)
        time3 = time_sys.time()
        if len(self.coms)>0:
            mincom = ((abs(self.coms-self.com_ids)/self.radius).min() <0.5).sum()
        if plot and (not self.multi or (self.multi and rank==0)):
            partf = 'particle_plots_%s' % fldn
            self.savestg = savestring  + '/' + partf
            self.ax[0].set_xscale('log')
            self.ax[0].set_yscale('log')
            self.ax[0].set_xlabel('PE [J]')
            self.ax[0].set_ylabel('KE [J]')
            self.ax[0].set_xlim(self.minp,self.maxp)
            self.ax[0].set_ylim(self.mink,self.maxk)
            self.ax[0].plot(np.linspace(min(self.mink,self.minp),max(self.maxk,self.maxp),1000),\
                np.linspace(min(self.mink,self.minp),max(self.maxk,self.maxp),1000),'-',color='black')
            plt.savefig(self.savestg+'/'+'EnergyComparison_%s.png' % ind)
            plt.close()
        # Find the velcoity of the center of mass of the halos for evolution
        self.find_vcom()
        # Delete spurious attributes
        del self.pos,self.mass,self.vel,self.ids,self.center,self.radius,self.ds,self.lenpart,self.lu
        if plot and (not self.multi or (self.multi and rank==0)):
            del self.ax,self.mink,self.minp,self.maxk,self.maxp,self.color_count
        del self.multi

    # Finds the center of mass velocity of particles with a given set of ids from the id list of the regions
    # Sets the result as an attribute
    def find_vcom(self):
        vcom = []
        for i in range(len(self.idl)):
            index = np.isin(self.ids,self.idl[i])
            vel = self.vel[index]
            with np.errstate(divide='ignore'):
                vcom.append(((vel*self.mass[index,np.newaxis]).sum(axis=0)/self.mass[index].sum())/self.lu)
        self.vcom = np.array(vcom)

    # Gathers the positions, masses, velocities, and particle ids for all particles in a region of a given type
    def get_mass_pos_vel_ids(self,center,radius,tp = 1,cut_size=700,dense=False,undense=False,multi=False,overlap=False,timing=0):
        if timing:
           time4 = time_sys.time()
        mass,pos,vel,ids = self.reg['mass'],self.reg['pos'],self.reg['vel'],self.reg['ids']
        if find_stars:
            spos,svel,sids = self.reg['spos'],self.reg['svel'],self.reg['sids']
        if overlap:
            bool1 = (np.sum(pos > center - radius,axis=1)==3)*(np.sum(pos < center + radius,axis=1)==3)
            pos,mass,vel,ids = pos[bool1],mass[bool1],vel[bool1],ids[bool1]
            if timing and time_sys.time()-time4 >timing:
                    print('Get Local Partciles',time_sys.time()-time4)
                    time4 = time_sys.time()
            ind = np.array(pd.DataFrame(ids).reset_index().groupby([0])['index'].min().to_list()).astype(int)
            mass,pos,vel,ids = mass[ind],pos[ind],vel[ind],ids[ind]
            # ids,ind = np.unique(ids,return_index=True)
            # mass,pos,vel = mass[ind],pos[ind],vel[ind]
            if timing and time_sys.time()-time4 >timing:
                    print('Get Unique',time_sys.time()-time4)
                    time4 = time_sys.time()
        #print(mass.shape,pos.shape,vel.shape,ids.shape)
        # time4 = time_sys.time()
        bool = np.full(len(pos),True)
        self.lenpart = bool.sum()
        if self.idl_i is not None:
            bool_ids = np.isin(ids,self.idl_i)
            if bool_ids.sum() > 1:
                self.com_ids = FindCOM(mass[bool_ids], pos[bool_ids])
                #print(self.com_ids,np.linalg.norm(pos[bool_ids]-self.com_ids,v,axis=1).max())
                max_id_radius = np.linalg.norm(pos[bool_ids].v-self.com_ids.v,axis=1).max()
                self.com_ids = self.com_ids.v/self.lu
                radius = 1.75*max(radius,max_id_radius)/self.radmax
            else:
                self.com_ids = center/self.lu
        else:
            self.com_ids = center/self.lu
        #print(center,self.com_ids*self.lu)
        self.minmass_2 = 0
        mass1 = mass.sum()
        self.is_cut = False
        #dist_pos = np.linalg.norm(pos.v - self.com_ids*self.lu,axis=1)
        # print(radius,dist_pos)
        #bool1 = dist_pos  <= radius
        bool1 = np.array([False])
        if len(mass)>0:
            bool1 = (np.sum(pos > self.com_ids*self.lu - radius,axis=1)==3)*(np.sum(pos < self.com_ids*self.lu + radius,axis=1)==3)
            pos,mass,vel,ids = pos[bool1],mass[bool1],vel[bool1],ids[bool1]
        if find_stars:
            if len(sids) >0:
                bools = (np.sum(spos > self.com_ids*self.lu - radius,axis=1)==3)*(np.sum(spos < self.com_ids*self.lu + radius,axis=1)==3)
                spos,svel,sids = spos[bools],svel[bools],sids[bools]
            if len(sids) > 0:
                self.spos,self.svel,self.sids = spos.v,svel.v,sids
            else:
                self.spos,self.svel,self.sids = np.array([]),np.array([]),[]
        self.lenpart = bool1.sum()
        if self.idl_i is not None and self.reg['id_sample'] is not None:
            self.idl_i = np.append(self.idl_i,ids[np.isin(ids,self.reg['id_sample'])])
            self.idl_i = np.unique(self.idl_i)
        #print(self.com_ids,center/self.lu,radius)
        if timing and time_sys.time()-time4 >timing:
                print('Get New Center',time_sys.time()-time4)
                time4 = time_sys.time()
        if undense:
            density2 = 1
        else:
            density2 = 1
        self.tscale = time_sys.time()
        if self.lenpart >0:
            if mass.min() > 100:
                self.minmass_2 = 0#mass[mass >100].min()
        if self.lenpart > 10000:
            if dense == True:
                density = 2
            else:
                density = 1
            cutmin = int(density2*max(density*250*np.log10(len(mass))/6,density*100))
            lenmass = len(mass)
            segments = max(int(lenmass/5e6),1)
            array_index = np.arange(lenmass)
            np.random.shuffle(array_index)
            split_index = np.array_split(array_index,segments)
            ids_new = np.array([])
            for index_i in split_index:
                massi,booli = cut_particles(pos.v[index_i],mass[index_i],self.com_ids*self.lu,ids[index_i],idl_i=self.idl_i,\
                    cut_size=min(max(cutmin/segments,1),max(cut_size/segments,1)),dense=dense,segments=segments)
                if len(ids_new)==0:
                    pos_new = pos[index_i][booli]
                    vel_new = vel[index_i][booli]
                    ids_new = ids[index_i][booli]
                    mass_new = massi
                else:
                    pos_new = np.vstack((pos_new,pos[index_i][booli]))
                    vel_new = np.vstack((vel_new,vel[index_i][booli]))
                    ids_new = np.append(ids_new,ids[index_i][booli])
                    mass_new = np.append(mass_new,massi)
            pos,vel,ids,mass = pos_new,vel_new,ids_new,mass_new
            if timing and time_sys.time()-time4 >timing:
                    print('Cut Particles',time_sys.time()-time4)
                    time4 = time_sys.time()
            self.is_cut = True
            # if self.reg['halo'] != None:
            #     print(len(self.idl_i),len(mass),lenmass,self.reg['halo'])
        self.tscale -= time_sys.time()
        if len(mass) >0:
            self.pos,self.mass,self.vel,self.ids = pos.v,mass,vel.v,ids
        else:
            self.pos,self.mass,self.vel,self.ids = pos,mass,vel,ids


    # Solves for the virial radius of a group of particles given the radius from the
    # of mass of each particle and the particle masses as well as the desired density
    # at the virial radius in units of multiples of the critical density.
    # Returns viral radius and the critical density at the outer radius (in cases where it
    # is higher than the minimum for the given inputs at the edge)

    # Solves for the center of mass for a given set of positions and corresponding masses


    # Finds and returns halos within a given investigation regions
    # Parameters
    # ncluster: number of kmeans clusters
    # oden: The density of the halo as a function of the mean density of the universe
    def find_halo_cores(self,ncluster=2,oden=150,plot=False,ind=10,idl_i=None,eps=.5,sparse=1,timing=0):
        coms,rads,massl,cdens,tmassl,partl,idl,rvir_l,cden_l,mass_l = [],[],[],[],[],[],{},[],[],[]
        more_variables = []
        var_names_scalar = []
        var_names_vec = []
        more_vec = []
        hull_l = {}
        if len(self.mass)>0:
            if timing:
                time3 = time_sys.time()
            if len(self.mass) >ncluster+5:

                # Sovlves the kinetic and potential energy based on given particle propertiesis
                # This is the energy of the entire region based on its center of mass so it may
                # not show bound or unbound elements properly.


                colors = list(mcolors.TABLEAU_COLORS)
                #colors = ['black','red','blue','green','orange','yellow','pink']
                # This normalizes positions and log (energy) for a kmeans clustering calculation
                # Kmeans will minimize the Euclidian distance which will group particles by their
                # location as well as their energy, which typically will find candidate self-bound regions
                reduce_size = 1000
                reduce_size_2 = 1000
                combined = np.array([])
                if ncluster > 1:
                    if len(self.mass) > reduce_size*ncluster*sparse:
                        #mass_2,bool_cut = cut_particles(self.pos,self.mass,self.center,[],cut_size=300)
                        #rand_int =np.arange(len(self.pos))[bool_cut]
                        rand_int = np.random.choice(np.arange(len(self.mass)), size=reduce_size*ncluster*sparse,replace=False)
                        # if idl_i is not None:
                        #     ids_int = np.arange(len(self.pos))[np.isin(self.ids,idl_i)]
                        #     ids_int = np.append(rand_int,ids_int)
                        #     rand_int = np.arange(len(self.pos))[np.unique(ids_int)]
                        # arg_mass = np.argsort(self.mass)
                        # add_mass = np.arange(len(self.mass))[self.mass>self.mass[arg_mass[-1*min(200,len(self.mass))]]]
                        # rand_int = pd.unique(np.append(rand_int,add_mass))
                        Eint = np.full(len(self.mass),-1)
                        Eint[rand_int] = 0
                        mass_2 = self.mass[rand_int]*len(self.mass)/(reduce_size*ncluster*sparse)
                    else:
                        Eint = (np.zeros(len(self.mass))).astype(int)
                        rand_int = np.arange(len(Eint))
                        mass_2 = self.mass
                    com = FindCOM(self.mass[rand_int],self.pos[rand_int])
                    rel_pos = self.pos[rand_int]-com
                    Energies = Find_KE_PE(mass_2,self.pos[rand_int],self.vel[rand_int],multi=self.multi)
                    if (timing) and time_sys.time()-time3 >timing:
                        print('Found Energy Initial',time_sys.time()-time3,'Sparse:',sparse,'Length:',len(mass_2))
                    time3 = time_sys.time()
                    KE,PE = Energies.KE,Energies.PE
                    abs_rel_pos = np.linalg.norm(rel_pos,axis=1).mean()
                    combined = np.vstack(((rel_pos/abs_rel_pos).T,np.log10(PE/PE.mean())-1,np.log10(KE/PE.mean())-1)).T
                    combined = np.minimum(combined,1e20)
                    combined = np.maximum(combined,-1e20)
                    if idl_i is not None and len(idl_i) > 1:
                        bool = np.isin(self.ids[rand_int],idl_i)
                        if len(combined[np.logical_not(bool)]) > 100:
                            #combined = combined[np.logical_not(bool)]
                            if self.cluster_type == 'kmean':
                                clustering = KMeans(n_clusters=ncluster, random_state=0, n_init='auto',init='k-means++').fit(combined)
                            elif self.cluster_type == 'DBSCAN':
                                clustering = DBSCAN(eps =eps, min_samples=5).fit(combined)
                            #Eint[rand_int][np.logical_not(bool)] = clustering.labels_+1
                            Eint[rand_int] = clustering.labels_ + 1
                        else:
                            if self.cluster_type == 'kmean':
                                clustering = KMeans(n_clusters=ncluster, random_state=0, n_init='auto',init='k-means++').fit(combined)
                            elif self.cluster_type == 'DBSCAN':
                                clustering = DBSCAN(eps = eps, min_samples=5).fit(combined)
                            Eint[rand_int] = clustering.labels_
                    else:
                        if self.cluster_type == 'kmean':
                            clustering = KMeans(n_clusters=ncluster, random_state=0, n_init='auto',init='k-means++').fit(combined)
                        elif self.cluster_type == 'DBSCAN':
                            clustering = DBSCAN(eps = eps, min_samples=5).fit(combined)
                        Eint[rand_int] = clustering.labels_
                    if (timing)  and time_sys.time()-time3 >5:
                        print('Found Energy Cluster',time_sys.time()-time3,"Size :",combined.shape)
                    time3 = time_sys.time()
                # Enables searching by just the bound particles for a new halo is the broad search fails
                elif idl_i is not None:
                    Eint = (-2*np.ones(len(self.pos))).astype(int)
                    bool = np.isin(self.ids,idl_i)
                    Eint[bool] = 0
                    if (timing)  and time_sys.time()-time3 >timing:
                        print('Found Eint_bool Cluster',time_sys.time()-time3,"Size :",combined.shape)
                    time3 = time_sys.time()
                # Enables looking at an area as a single cluster
                else:
                    Eint = np.ones(len(self.pos)).astype(int)
                untaged = np.full(len(self.pos),True)
                unt_ind = np.arange(len(self.pos))
                #len_cluster = np.array([len(Eint[Eint==x]) for x in pd.unique(Eint)])
                mass_cluster = np.array([self.mass[Eint==x].sum() for x in pd.unique(Eint)])
                #arg = np.argsort(len_cluster)[::-1][:30]
                arg = np.argsort(mass_cluster)[::-1][:30]
                eint_list = pd.unique(Eint)[arg]
                eint_list = np.append(eint_list[eint_list>-2],-2)
                #oden_list = [100,150,200,250,300,500,700]
                # print(len(np.unique(Eint)[arg]))
                # Each kmeans grouping is looped through to find candidate halos
                for i in eint_list: #list(range(Eint.max()+1)):
                    if len(self.mass[(Eint==i)*untaged]) >40 and i != -1:
                        # First particles are selected that are within a bounding region that includes
                        # all the particles in the cluster that are not in another confirmed halo.
                        if i >-2:
                            com = FindCOM(self.mass[(Eint==i)*untaged],self.pos[(Eint==i)*untaged])
                        else:
                            com = self.com_ids*self.lu
                        rad = np.linalg.norm(self.pos[untaged]-com,axis=1)
                        if len(rad[(Eint[untaged]==i)]) >0:
                            radmax = min(rad[(Eint[untaged]==i)].max(),1.2*self.radius*self.lu)
                            halo_bool = (rad <= radmax*1.5)
                            if halo_bool.sum() >0:
                                # This selects whether clusters use a multithreaded calculation based on
                                # the number of particles in the cluster.
                                if len(self.mass[untaged][halo_bool]) > 10000:
                                    multi2 = True * self.multi
                                else:
                                    multi2 = False
                                # This solves the potential and kinetic energy of the cluster and finds the bound
                                # members of the cluster
                                full_len = len(self.pos[untaged][halo_bool])
                                if full_len > reduce_size_2*sparse:
                                    rand_int = np.random.choice(np.arange(full_len),p=self.mass[untaged][halo_bool]/self.mass[untaged][halo_bool].sum(),size=reduce_size_2*sparse,replace=False)
                                    mass_2 = self.mass[untaged][halo_bool][rand_int]*full_len/(reduce_size_2*sparse)
                                else:
                                    rand_int =  np.arange(full_len)
                                    mass_2 = self.mass[untaged][halo_bool][rand_int]
                                Energies = Find_KE_PE(mass_2,self.pos[untaged][halo_bool][rand_int],self.vel[untaged][halo_bool][rand_int],multi=multi2)
                                if (timing) and time_sys.time()-time3 >timing:
                                    print('Found Energies',i,time_sys.time()-time3)
                                time3 = time_sys.time()
                                KE,PE = Energies.KE,Energies.PE
                                es = KE-PE
                                if mass_2[es <0].sum()*self.kg_sun > minmass/2:
                                    # This recalculates the center of mass of the halo as the center of the bound partciles
                                    # Then the virial radius is determined from that center. The virial mass includes all bound
                                    # and unbound particles within the radius since these all contribute to collapse.
                                    #print(len(self.pos[untaged][halo_bool][rand_int][es <0]),len(mass_2[es <0]))
                                    # if len(mass_2[es <0]) >500:
                                    #     densities = stats.gaussian_kde(self.pos[untaged][halo_bool][rand_int][es <0].T,weights=self.mass[untaged][halo_bool][rand_int][es <0])
                                    #     argden = np.argsort(densities)
                                    #     com1 = self.pos[untaged][halo_bool][rand_int][es <0][argden[-4:]].mean(axis=0)
                                    #else:
                                    #com1 = self.FindCOM(mass_2[es <0],self.pos[untaged][halo_bool][rand_int][es <0])
                                    com1 = FindCOM(-1*es[es <0],self.pos[untaged][halo_bool][rand_int][es <0])
                                    #print(com,com1)
                                    radmax3 = np.linalg.norm(self.pos[untaged][halo_bool][rand_int][es <0]-com1,axis=1).max()
                                    rad = np.linalg.norm(self.pos[untaged]-com1,axis=1)
                                    radmax_all,cden,mden = Find_virRad(com1,self.pos,self.mass,self.ds,oden=[self.crit_den],radmax=radmax)
                                    if (timing)  and time_sys.time()-time3 >timing:
                                        print('Confirmed Energies',i,time_sys.time()-time3,'Sparse:',sparse)
                                        #print('Radii:',radmax_all,'Cden:',cden)
                                    time3 = time_sys.time()
                                    if cden.max() > 50:
                                        radmax2 = radmax_all.max()
                                        if radmax2 < 0.05*radmax:
                                            radmax = max(radmax2,radmax)
                                        else:
                                            radmax = radmax2
                                        #radmax = max(radmax,self.radius*self.lu)
                                        radmax = min(radmax,radmax3)
                                        halo_bool = (rad <= radmax*1.05)
                                        Energies = Find_KE_PE(self.mass[untaged][halo_bool],self.pos[untaged][halo_bool],self.vel[untaged][halo_bool],multi=multi2)
                                        KE,PE = Energies.KE,Energies.PE
                                        es = KE-PE
                                        com2 = FindCOM(-1*es[es <0],self.pos[untaged][halo_bool][es <0])
                                        if (timing)  and time_sys.time()-time3 >timing:
                                            print('Confirmed Energies 2',i,time_sys.time()-time3,'len:',len(self.mass[untaged][halo_bool]))
                                            #print('Radii:',radmax_all,'Cden:',cden)
                                        time3 = time_sys.time()
                                        #com2 = self.FindCOM(self.mass[untaged][halo_bool][es <0],self.pos[untaged][halo_bool][es <0])
                                        check_halo = np.linalg.norm(com2/self.lu-self.com_ids)/self.radius <1.5 and \
                                            self.mass[unt_ind[untaged][halo_bool][es <0]].sum()*self.kg_sun > minmass/2
                                        # if not check_halo:
                                        #      print('check_one (2,%s)' % (minmass/2),np.linalg.norm(com2/self.lu-self.com_ids)/self.radius,self.mass[unt_ind[untaged][halo_bool][es <0]].sum()*self.kg_sun)
                                        if check_halo:
                                            #rad2 = np.linalg.norm(self.pos[untaged][halo_bool][es <0]-com2,axis=1)
                                            # r = np.linalg.norm(com2-self.pos[untaged][halo_bool][es <0],axis=1)
                                            # rad_all = np.linalg.norm(com2-self.pos,axis=1)
                                            oden3 = oden
                                            # if cden_0.min() >250:
                                            #     rvir_i = np.arange(len(oden_list_0))[abs(cden_0-oden3)==abs(cden_0-oden3).min()].min()
                                            # else:
                                            oden_list_0 = np.copy(np.array(oden_list))
                                            oden_list_0 = np.append(oden_list_0,oden3)
                                            rvir_0,cden_0,mden_0 = Find_virRad(com2,self.pos[untaged][halo_bool][es <0],self.mass[untaged][halo_bool][es <0],self.ds,oden=oden_list_0,radmax=radmax)
                                            #if oden >= 349:
                                            #    oden3 = 349
                                            rvir_i = np.arange(len(oden_list_0))[abs(cden_0-oden3)==abs(cden_0-oden3).min()].min()
                                            rvir = rvir_0[rvir_i].min()
                                            rvir_200_i = np.arange(len(oden_list))[abs(np.array(oden_list).astype(float)-200)==abs(np.array(oden_list).astype(float)-200).min()].min()
                                            r200 = rvir_0[rvir_200_i].min()
                                            cden_f = cden_0[rvir_i].min()
                                            r = np.linalg.norm(com2-self.pos[untaged][halo_bool][es <0],axis=1)
                                            if (timing)  and time_sys.time()-time3 >timing:
                                                print('Final Energies',i,time_sys.time()-time3,'Length:',len(self.mass[untaged][halo_bool]))
                                                #print('Radii:',rvir,'Cden:',cden)
                                            time3 = time_sys.time()
                                            #check_halo_2 = cden_f > 75 and cden_f <max(oden_list)*1.05
                                            check_halo_2 = cden_f > 75 and cden_f <max(oden_list)*1.05 and len(self.pos[untaged][halo_bool][es <0][r<rvir]) >0
                                            # if not check_halo_2:
                                            #       print('check_two %s' % (max(oden_list)*1.05),len(self.pos[untaged][halo_bool][es <0][r<rvir_0.max()]),cden_f)
                                            #     print('check_two (175-%s,2)' % (max(oden_list)*1.05), oden,cden_f,sum(es<0),len(self.pos[untaged][halo_bool][es <0][r<rvir]))
                                            #     print('check_two_cont',rad.max()/(self.radius*self.lu),radmax/(self.radius*self.lu),rvir/(self.radius*self.lu))
                                            if check_halo_2:
                                                # Plotting the halos and their particles
                                                # Saving the partciles and halo information based on the bound particles within the virial radius
                                                used_hull = False
                                                check_halo_3 = False
                                                more_variables_i = [None]*len(self.var_names_scalar)
                                                more_variables_vec = [None]*len(self.var_names_vec0)
                                                if non_sphere and self.use_hulls:
                                                    r2 = np.linalg.norm(com2-self.pos[untaged][halo_bool],axis=1)
                                                    # if abs(oden-200)>15:
                                                    #    target = max(oden + np.sign(200-oden)*15,min(cden))
                                                    # else:
                                                    target = max(oden,min(cden_0))
                                                    mass_1,cden_1,rvir_1,estep_f,esr,hull = make_hull(r2,self.pos[untaged][halo_bool],self.mass[untaged][halo_bool],es,target,self.crit_den,rmax=rvir)
                                                    # centroid = np.mean(self.pos[untaged][halo_bool][es<estep_f][hull.vertices],axis=0)
                                                    # d_cen = np.linalg.norm(centroid-com2)/rvir_0.max()
                                                    #print(centroid,com2,d_cen)
                                                    check_halo_3 = cden_1 > 75 and cden_1 <max(oden_list)*1.05 and mass_1 > minmass/2 and (not non_sphere or len(unt_ind[untaged][halo_bool][es<estep_f]) >4)
                                                    if check_halo_3:
                                                        used_hull = True
                                                    # else:
                                                    #     target = max(oden,min(cden))
                                                    #     mass_1,cden_1,rvir_1,estep_f,esr,hull = make_hull(r2,self.pos[untaged][halo_bool],self.mass[untaged][halo_bool],es,target,self.crit_den,rmax=1.05*rvir_0.max())
                                                    #     check_halo_3 = cden_1 > 175 and cden_1 <max(oden_list)*1.05 and mass_1 > minmass/2 and (not non_sphere or len(unt_ind[untaged][halo_bool][es<estep_f]) >4)
                                                    #     if check_halo_3:
                                                    #         used_hull = True
                                                if not check_halo_3 or not used_hull or mass_1 < 0.5*self.mass[unt_ind[untaged][halo_bool][es <0][r<rvir]].sum():
                                                    massl.append(self.mass[unt_ind[untaged][halo_bool][es <0][r<rvir]].sum())
                                                    idl[len(coms)] = self.ids[unt_ind[untaged][halo_bool][es <0][r<rvir]]
                                                    partl.append(len(self.mass[unt_ind[untaged][halo_bool][es <0][r<rvir]]))
                                                    hull_l[len(coms)] = []
                                                elif used_hull:
                                                    cden_f = cden_1
                                                    idl[len(coms)] = self.ids[unt_ind[untaged][halo_bool][esr<estep_f]]
                                                    hull_l[len(coms)] = self.ids[unt_ind[untaged][halo_bool][esr<estep_f][hull.vertices]]
                                                    partl.append(len(self.mass[unt_ind[untaged][halo_bool][esr<estep_f]]))
                                                    massl.append(mass_1)
                                                    rvir = rvir_1
                                                    more_variables_i,more_variables_vec = more_var_maker(self.pos[untaged][halo_bool],self.mass[untaged][halo_bool],\
                                                        self.vel[untaged][halo_bool],rvir,esr,es,estep_f,hull,com2,self.lu,self.kg_sun,cden_1,self.ds,radmax=radmax)
                                                    # print(a/b,a/c,dist_com_1,dist_com_2,fill_sphere,fill_ellipsoid)
                                                #m200.append(self.mass[unt_ind[untaged][halo_bool][es <0][r<r200]].sum())

                                                rvir2,cden2,mden2 = Find_virRad(com2,self.pos,self.mass,self.ds,oden=[700],radmax=radmax)
                                                if (timing)  and time_sys.time()-time3 >timing:
                                                    print('Final Shape',i,time_sys.time()-time3)
                                                    #print('Radii:',rvir,'Cden:',cden)
                                                time3 = time_sys.time()
                                                rvir_i = rvir_0.tolist()[:-1]
                                                cden_i = cden_0.tolist()[:-1]
                                                var_names_scalar_i,more_variables_i = add_virial_mass(cden_i,np.array(mden_0[:-1])*self.kg_sun,more_variables_i)
                                                rvir_l.append(rvir_i)
                                                cden_l.append(cden_i)
                                                var_name_vec_i = np.copy(self.var_names_vec0).tolist()
                                                more_variables.append(more_variables_i)
                                                mass_l_i = []
                                                if find_stars and len(self.sids)>0:
                                                    sid_i,s_energy_i = find_star_energy(self.pos[untaged][halo_bool][es<0],self.mass[untaged][halo_bool][es<0],self.vel[untaged][halo_bool][es<0],com2,self.spos,self.svel,self.sids)
                                                    if len(sid_i)>0:
                                                        var_name_vec_i.append('star_id')
                                                        var_name_vec_i.append('star_energy')
                                                        more_variables_vec.append(sid_i)
                                                        more_variables_vec.append(s_energy_i)
                                                        #print(var_name_vec_i)
                                                for rad_i in rvir_i:
                                                    mass_l_i.append(self.mass[unt_ind[untaged][halo_bool][es <0][r<rad_i]].sum())
                                                if plot and (not self.multi or (self.multi and rank==0)):
                                                    pos_halo = self.pos[untaged][halo_bool][es <0][r<rvir]
                                                    self.color_count += 1
                                                    self.ax[0].scatter(PE[es <0][r<rvir],KE[es <0][r<rvir],color=colors[self.color_count],alpha=0.15,marker=".")
                                                    self.ax[1].scatter(pos_halo[:,0]/self.lu,pos_halo[:,1]/self.lu,color=colors[self.color_count],alpha=0.15,marker=".")
                                                    circle1 = plt.Circle((com2[0]/self.lu,com2[1]/self.lu), rvir/self.lu, color=colors[self.color_count], fill=False)
                                                    self.ax[1].add_patch(circle1)
                                                    self.ax[2].scatter(pos_halo[:,0]/self.lu,pos_halo[:,2]/self.lu,color=colors[self.color_count],alpha=0.15,marker=".")
                                                    circle2 = plt.Circle((com2[0]/self.lu,com2[2]/self.lu), rvir/self.lu, color=colors[self.color_count], fill=False)
                                                    self.ax[2].add_patch(circle2)
                                                    self.ax[3].scatter(pos_halo[:,1]/self.lu,pos_halo[:,2]/self.lu,color=colors[self.color_count],alpha=0.15,marker=".")
                                                    circle3 = plt.Circle((com2[1]/self.lu,com2[2]/self.lu), rvir/self.lu, color=colors[self.color_count], fill=False)
                                                    self.ax[3].add_patch(circle3)
                                                    self.mink,self.minp = min(KE.min(),self.mink), min(PE.min(),self.minp)
                                                    self.maxk,self.maxp = max(KE.max(),self.maxk),max(PE.max(),self.maxp)
                                                # if cden2 > oden_list[-2]:
                                                #     untaged[unt_ind[untaged][halo_bool][es <0][r<rvir2]] = False
                                                # if self.mass[unt_ind[untaged][halo_bool][es <0][r<rvir]].sum() > 3*self.hmass and not self.first_round and cden2 >650:
                                                #     untaged[unt_ind[untaged][halo_bool][es <0][r<rvir2]] = False
                                                coms.append(com2)
                                                rads.append(rvir)
                                                more_vec.append(more_variables_vec)
                                                # cden_0,ind_u =np.unique(cden_0,return_index=True)
                                                # rvir_0 = rvir_0[ind_u]
                                                #print(more_variables_i)
                                                cdens.append(cden_f)
                                                mass_l.append(mass_l_i)
                                                var_names_scalar.append(var_names_scalar_i)
                                                var_names_vec.append(var_name_vec_i)
                                                r = np.linalg.norm(com2-self.pos,axis=1)
                                                # Saves the total mass within the virial radius including non-bound partciles
                                                tmassl.append(self.mass[r <rvir].sum())
                                                if (timing)  and time_sys.time()-time3 >timing:
                                                    print('Complete',time_sys.time()-time3)
                                                time3 = time_sys.time()
        if len(rads) > 0:
            return np.array(coms)/self.lu,np.array(rads)/self.lu,np.array(massl)*self.kg_sun,np.array(tmassl)*self.kg_sun,np.array(partl),idl,np.array(rvir_l)/self.lu,np.array(cdens),np.array(cden_l),hull_l,np.array(more_variables),var_names_scalar,more_vec,var_names_vec
        else:
            return np.array(coms),np.array(rads),np.array(massl),np.array(tmassl),np.array(partl),idl,np.array(rvir_l),np.array(cdens),np.array(cden_l),hull_l,np.array(more_variables),var_names_scalar,more_vec,var_names_vec

class Initial_Halo_Tree():
    """
    This class detects takes a timestep and searches for all halos that are associated with simulation
    volumes that have halos that were found with other halo-finding codes.

    Parameters
    ----------
    timestep: int
        The timestep of the simulation where the tree starts. Should be the latest timestep
        with halo-finding data.

    plot : bool
        Turns on or off plotting of investigation regions for visual inspection.

    codetp : str
        The code used to create simulation

    skip_large : bool
        Only runs on a subset of small halos for easy testing and debugging

    from_tree : bool
        Boolean for whether to take halos from a preexisting list

    coms : list or array (3 x n)
        Manual list of centers of mass of halos. Default: None

    rads : list or array (1d)
        Manual list of halo radii. Default: None

    last_timestep : int
        Last snapshot of the simulation. Used as a starting point.

    verbose : bool
        Boolean for whether to output debugging text

    Returns
    -------
    self.final_nodes: A dictionary of halo parameters organized by halo number with:

    'Halo_Center' : np.array
        The center of the halo

    'Halo_Radius' : float
        The virial radius of the halo (default is r_100)

    'Halo Mass' : float
        The mass of the particles bound to the halo

    'Part_Ids' : list of ints
        The particle ids of the particles bound to the halo

    'NumParts' : int
        The number of particles bound to the halos

    'Vel_Com' : np.array
        The velocity of the halo

    self.current_time: The current age of the universe in seconds for time evolution calculations

    """
    def __init__(self,plot=False,codetp='ENZO',skip_large=False,from_tree=True,ds=False,\
            last_timestep=None,verbose=False,refined=True,trackbig=None,undense=True,lu=False,oden0=False,\
            icenters=None,iradii=None):
        timestep = last_timestep
        self.codetp= codetp
        self.ll,self.ur = [],[]
        self.ds,self.lu = ds,lu
        if not oden2:
            self.oden_vir = oden_vir(self.ds)
        else:
            self.oden_vir = oden2
        if from_tree:
            halotree = np.load(string +  "/halotree.npy", allow_pickle = True, encoding = "latin1").tolist()
            com_list,rad_list = self.list_halos_from_tree(halotree,last_timestep)
        else:
            dens = Over_Density_Finder(timestep,self.ds,self.lu,codetp=self.codetp,refined=refined,verbose=verbose,trackbig=trackbig,\
                icenters=icenters,iradii=iradii)
            self.ll_all,self.ur_all = dens.ll_all,dens.ur_all
            com_list,rad_list,mass_list = dens.centers,dens.radii,dens.masses
        self.final_nodes = {}
        self.kg_sun = self.ds.mass_unit.in_units('Msun').v/self.ds.mass_unit.in_units('kg').v
        self.current_time = self.ds.current_time.in_units('s').v
        # Gathers centers, radii, and masses from a halo tree file
        # Determins the number of particles in each halo to determine which halos need
        # parallel energy solving and which can be handles by a single core.
        #len_part = get_particle_number(self.ds,com_list,rad_list,self.codetp)
        f_com_list,f_rad_list,f_mass_list,f_ids,f_vcom,f_rvir_list,f_cden,f_cden_list,f_var_names_scalar = [],[],[],[],[],[],[],[],[]
        f_hull = []
        f_more_l = []
        f_more_vec = []
        f_var_names_vec = []
        if len(rad_list) >0:
            bool_mini = np.full(len(rad_list),True)#len_part > 50
            # Halos are handled individually by a core
            if skip_large:
                halo_group = np.arange(len(com_list))[bool_mini][0:40]
            else:
                halo_group = np.arange(len(com_list))[bool_mini]
            if rank==0:
                print('Beginning Halo Confirmation')
            halo_id,ll_total,ur_total,volume_list = get_all_regions_1(com_list,3.5*rad_list,halo_group,max_rad=rad_list.max())
            #volumes = rad_boost**3
            minsplit = max(min(min(1000,nprocs*30),int(len(volume_list)/5)),nprocs)
            # if rank ==0:
            #     print(volume_list)
            if len(volume_list) >0:
                 split_volumes = equisum_partition(volume_list,np.maximum(int(len(volume_list)/minsplit),1))
            else:
                 split_volumes = np.arange(len(volume_list))
            #split_volumes = [0]
            halo_complete = np.array([])
            for g in range(len(split_volumes)):
              # if len(out_list)>0:
                #now_list = halo_group[out_list]
                h_list = split_volumes[g].astype(int)#p.arange(len(halo_id)).astype(int)
                now_list = np.array([])
                for i in h_list:
                    for halo_i in halo_id[i]:
                        if halo_i not in halo_complete:
                            now_list = np.append(now_list,halo_i)
                now_list = np.unique(now_list).astype(int)
                halo_complete = np.append(halo_complete,now_list)
                halo_complete = np.unique(halo_complete).astype(int)
                my_storage_reg = 0
                my_storage_reg = get_all_regions_2(halo_id,ll_total,ur_total,h_list,\
                        self.ds,self.codetp,volume_list,nprocs=nprocs,timestep=timestep)
                #print(my_storage_reg)
                jobs = {}
                ranks = np.arange(nprocs)
                joblen = np.zeros(len(ranks))
                sto = {}
                for o in ranks:
                  jobs[o] = []
                  for i in now_list:
                    if i in my_storage_reg['rank'][o]:
                        jobs[o].append(i)
                        joblen[o] += 1
                        sto[i] = {}
                        sto[i]['rank'] = o
                split_jobs = {}
                split_len = np.zeros(len(ranks))
                for o in jobs:
                  if joblen[o] >0 and joblen[o].max() >0:
                    split_jobs[o] = np.array_split(jobs[o],max(joblen[o].max()/5,1))
                    split_len[o] += len(split_jobs[o])
                # segments = int(np.ceil(len(current_list)/(2*self.nprocs)))
                # split_index = np.array_split(current_list,max(segments,1))
                for split_i in range(int(split_len.max())):
                  # if split_i < split_len[rank]:
                  #       print(rank,split_jobs[rank][split_i])
                  # else:
                  #       print(rank,[])
                  if split_i < split_len[rank]:
                    for i in split_jobs[rank][split_i]:
                          # print('0',i,rank)
                        # for sto, r in yt.parallel_objects(h_index.astype(int),\
                        #     self.nprocs, storage = my_storage):
                          self.reg_info = {}
                          self.reg_info['pos'],self.reg_info['mass'],self.reg_info['vel'],self.reg_info['ids'] =\
                             np.array([]),np.array([]),np.array([]),np.array([])
                          self.reg_info['sids'] = np.array([])
                          for y in my_storage_reg['halo'][i]:
                                    v = my_storage_reg[y]
                                    #print(v)
                                    if len(self.reg_info['mass'])==0:
                                        self.reg_info['pos'] = v['pos']
                                        self.reg_info['vel'] = v['vel']
                                        if find_stars:
                                            self.reg_info['spos'] = v['spos']
                                            self.reg_info['svel'] = v['svel']

                                    else:
                                        self.reg_info['pos'] = np.vstack((self.reg_info['pos'],v['pos']))
                                        self.reg_info['vel'] = np.vstack((self.reg_info['vel'],v['vel']))
                                        if find_stars:
                                            self.reg_info['spos'] = np.vstack((self.reg_info['spos'],v['spos']))
                                            self.reg_info['svel'] = np.vstack((self.reg_info['svel'],v['svel']))
                                    self.reg_info['mass'] = np.append(self.reg_info['mass'],v['mass'])
                                    self.reg_info['ids'] = np.append(self.reg_info['ids'],v['ids'])
                                    if find_stars:
                                        self.reg_info['sids'] = np.append(self.reg_info['sids'],v['sids'])
                                    v = None
                          # print(i,self.reg_info['mass'].sum())
                          self.reg_info['id_sample'] = None
                          self.reg_info['halo'] = None
                          # if i%50 ==0:
                          #      print(i+1,'out of',len(com_list))
                          # if dens.masses[i] > 1e12:
                          #     nclusters = 30
                          # elif dens.masses[i] > 1e11:
                          #     nclusters = 20
                          # elif dens.masses[i] > 1e10:
                          #     nclusters = 15
                          # elif dens.masses[i] > 1e9:
                          #     nclusters = 10
                          # else:
                          #     nclusters = 6
                          nclusters = max(int(2*np.log10(mass_list[i]))-6,1)
                          oden0 = self.oden_vir
                          oden0_1 = self.oden_vir
                          # if not oden0:
                          #     oden0 = oden1
                          #     oden0_1 = oden2
                          # else:
                          #     oden0 = min(oden0,300)
                          #     oden0_1 = oden0+(oden2-oden1)
                          plot_i = plot
                          # if mass_list[i] >max(mass_list)/10:
                          #   plot_i = True
                          DH = detect_halos(self.ds,self.lu,com_list[i],rad_list[i],nclustermax=nclusters,ind=str(i)+'_'+str(timestep),multi_solver=False,plot=plot_i,codetp=self.codetp,\
                            oden=oden0,hmass=mass_list[i],reg=self.reg_info,first_round=False)
                          sto[i]['skip'] = True
                          var_names_scalar = []
                          var_names_vec = []
                          more_vec_i = []
                          # if rank==0:
                          #      print(DH.rvir_l)
                          g1list =[]
                          ll_halo = com_list[i]-3.2*rad_list[i]
                          ur_halo = com_list[i]+3.2*rad_list[i]
                          arg_mass = np.argsort(DH.massl)[::-1]
                          if len(DH.coms) > 0:
                              for g in arg_mass:
                                if len(g1list)==0:
                                  if len(DH.rads) >1:
                                      in_bound_ll = DH.coms[g]-DH.rads[g]
                                      in_bound_ur = DH.coms[g]+DH.rads[g]
                                  else:
                                      in_bound_ll = DH.coms-DH.rads
                                      in_bound_ur = DH.coms+DH.rads
                                  in_bounds = (np.sum(in_bound_ll > ll_halo).sum()==3)*(np.sum(in_bound_ur < ur_halo).sum()==3)
                                  if in_bounds:
                                      com_i,rads_i,massl_i,rvir_l_i =  DH.coms[g][np.newaxis,:],\
                                            np.array([DH.rads[g]]),np.array([DH.massl[g]]),DH.rvir_l[g][np.newaxis,:]
                                      idl_i,vcom_i =[DH.idl[g]],DH.vcom[g][np.newaxis,:]
                                      more_i = DH.more_variables[g][np.newaxis,:]
                                      hull_i = [DH.hull_l[g]]
                                      cden_i = np.array([DH.cdens[g]])
                                      var_names_scalar.append(DH.var_names_scalar[g])
                                      var_names_vec.append(DH.var_names_vec[g])
                                      more_vec_i.append(DH.more_vec[g])
                                      #m200_i = np.array([DH.m200[g]])
                                      cden_l_i = DH.cden_l[g][np.newaxis,:]
                                      g1list.append(g)
                                      sto[i]['skip'] = False
                          if not sto[i]['skip']:
                              for t in range(len(DH.rads)):
                                  if len(DH.rads) >1:
                                      in_bound_ll = DH.coms[t]-DH.rads[t]
                                      in_bound_ur = DH.coms[t]+DH.rads[t]
                                  else:
                                      in_bound_ll = DH.coms-DH.rads
                                      in_bound_ur = DH.coms+DH.rads
                                  in_bounds = (np.sum(in_bound_ll > ll_halo).sum()==3)*(np.sum(in_bound_ur < ur_halo).sum()==3)
                                  if len(DH.rads) >1:
                                      ll_halo = DH.coms[t]-1.25*DH.rads[t]
                                      ur_halo = DH.coms[t]+1.25*DH.rads[t]
                                  else:
                                      ll_halo = DH.coms-1.25*DH.rads
                                      ur_halo = DH.coms+1.25*DH.rads
                                  if t not in g1list and DH.massl[t] > DH.massl.max()/5 and in_bounds:
                                      # To make sure the new halos are properly calculated, the detection algorithm is run a second time
                                      #but centered on the halos from the first run.
                                      DH_2 = detect_halos(self.ds,self.lu,DH.coms[t],DH.rads[t],ind=str(i)+'_'+str(t)+'_'+str(timestep),\
                                          multi_solver=False,plot=plot_i,radmax=1.25,nclustermax=1,codetp=self.codetp,idl_i=DH.idl[t],\
                                            oden=DH.cdens[t],hmass=DH.massl[t],reg=self.reg_info,first_round=False)
                                      for g in range(len(DH_2.rads)):
                                        if abs(DH_2.massl[g]/DH.massl[t] - 1) <0.25:
                                          if np.linalg.norm(DH_2.coms[g] - DH.coms[t])/DH.rads[t] <0.25 and DH_2.massl[g]< DH.massl.max():
                                              if len(DH_2.rads) >1:
                                                  in_bound_ll = DH_2.coms[g]-DH_2.rads[g]
                                                  in_bound_ur = DH_2.coms[g]+DH_2.rads[g]
                                              else:
                                                  in_bound_ll = DH_2.coms-DH_2.rads
                                                  in_bound_ur = DH_2.coms+DH_2.rads
                                              in_bounds = (np.sum(in_bound_ll > ll_halo).sum()==3)*(np.sum(in_bound_ur < ur_halo).sum()==3)
                                              if in_bounds:
                                                  if len(g1list) ==0:
                                                      com_i,rads_i,massl_i,rvir_l_i =  DH.coms[g][np.newaxis,:],\
                                                            np.array([DH.rads[g]]),np.array([DH.massl[g]]),DH.rvir_l[g][np.newaxis,:]
                                                      idl_i,vcom_i =[DH.idl[g]],DH.vcom[g][np.newaxis,:]
                                                      hull_i = [DH.hull_l[g]]
                                                      cden_i = np.array([DH.cdens[g]])
                                                      more_i = DH.more_variables[g][np.newaxis,:]
                                                      #m200_i = np.array([DH.m200[g]])
                                                      cden_l_i = DH.cden_l[g][np.newaxis,:]
                                                  com_i,rads_i,massl_i,vcom_i,rvir_l_i = np.vstack((com_i,DH_2.coms[g])),np.append(rads_i,DH_2.rads[g])\
                                                        ,np.append(massl_i,DH_2.massl[g]),np.vstack((vcom_i,DH_2.vcom[g])),np.vstack((rvir_l_i,DH_2.rvir_l[g]))
                                                  idl_i.append(DH_2.idl[g])
                                                  hull_i.append(DH_2.hull_l[g])
                                                  cden_i = np.append(cden_i,DH_2.cdens[g])
                                                  more_i = np.vstack((more_i,DH_2.more_variables[g]))
                                                  var_names_scalar.append(DH_2.var_names_scalar[g])
                                                  var_names_vec.append(DH_2.var_names_vec[g])
                                                  more_vec_i.append(DH_2.more_vec[g])
                                                  #m200_i = np.append(m200_i,DH_2.m200[g])
                                                  cden_l_i = np.vstack((cden_l_i,DH_2.cden_l[g]))
                                                  g1list.append(t)
                              for t in range(len(DH.rads)):
                                  if len(DH.rads) >1:
                                      in_bound_ll = DH.coms[t]-DH.rads[t]
                                      in_bound_ur = DH.coms[t]+DH.rads[t]
                                  else:
                                      in_bound_ll = DH.coms-DH.rads
                                      in_bound_ur = DH.coms+DH.rads
                                  in_bounds = (np.sum(in_bound_ll > ll_halo).sum()==3)*(np.sum(in_bound_ur < ur_halo).sum()==3)
                                  if len(DH.rads) >1:
                                      ll_halo = DH.coms[t]-1.25*DH.rads[t]
                                      ur_halo = DH.coms[t]+1.25*DH.rads[t]
                                  else:
                                      ll_halo = DH.coms-1.25*DH.rads
                                      ur_halo = DH.coms+1.25*DH.rads
                                  if t not in g1list and in_bounds:
                                      # To make sure the new halos are properly calculated, the detection algorithm is run a second time
                                      #but centered on the halos from the first run.
                                      DH_2 = detect_halos(self.ds,self.lu,DH.coms[t],DH.rads[t],ind=str(i)+'_'+str(t)+'_'+'0'+'_'+str(timestep),\
                                          multi_solver=False,plot=plot_i,radmax=1.75,nclustermax=max(int(nclusters/2),1),\
                                            codetp=self.codetp,idl_i=DH.idl[t],oden=oden0_1,hmass=DH.massl[t],reg=self.reg_info,first_round=False)
                                      for g in range(len(DH_2.rads)):
                                        #if abs(DH_2.massl[g]/DH.massl[t] - 1) <0.5:
                                          if np.linalg.norm(DH_2.coms[g] - DH.coms[t])/DH.rads[t] <0.5 and DH_2.massl[g]< DH.massl.max():
                                              if len(DH_2.rads) >1:
                                                  in_bound_ll = DH_2.coms[g]-DH_2.rads[g]
                                                  in_bound_ur = DH_2.coms[g]+DH_2.rads[g]
                                              else:
                                                  in_bound_ll = DH_2.coms-DH_2.rads
                                                  in_bound_ur = DH_2.coms+DH_2.rads
                                              in_bounds = (np.sum(in_bound_ll > ll_halo).sum()==3)*(np.sum(in_bound_ur < ur_halo).sum()==3)
                                              if in_bounds:
                                                  if len(g1list) ==0:
                                                      com_i,rads_i,massl_i,rvir_l_i =  DH.coms[g][np.newaxis,:],\
                                                            np.array([DH.rads[g]]),np.array([DH.massl[g]]),DH.rvir_l[g][np.newaxis,:]
                                                      idl_i,vcom_i =[DH.idl[g]],DH.vcom[g][np.newaxis,:]
                                                      hull_i = [DH.hull_l[g]]
                                                      cden_i = np.array([DH.cdens[g]])
                                                      #m200_i = np.array([DH.m200[g]])
                                                      cden_l_i = DH.cden_l[g][np.newaxis,:]
                                                      more_i = DH.more_variables[g][np.newaxis,:]
                                                  com_i,rads_i,massl_i,vcom_i,rvir_l_i = np.vstack((com_i,DH_2.coms[g])),np.append(rads_i,DH_2.rads[g])\
                                                        ,np.append(massl_i,DH_2.massl[g]),np.vstack((vcom_i,DH_2.vcom[g])),np.vstack((rvir_l_i,DH_2.rvir_l[g]))
                                                  idl_i.append(DH_2.idl[g])
                                                  hull_i.append(DH_2.hull_l[g])
                                                  cden_i = np.append(cden_i,DH_2.cdens[g])
                                                  more_i = np.vstack((more_i,DH_2.more_variables[g]))
                                                  var_names_scalar.append(DH_2.var_names_scalar[g])
                                                  var_names_vec.append(DH_2.var_names_vec[g])
                                                  more_vec_i.append(DH_2.more_vec[g])
                                                  #m200_i = np.append(m200_i,DH_2.m200[g])
                                                  cden_l_i = np.vstack((cden_l_i,DH_2.cden_l[g]))
                                                  g1list.append(t)
                          if not sto[i]['skip']:
                              sto[i]['com'],sto[i]['rad'],sto[i]['mass'],sto[i]['ids'],sto[i]['vcom'],sto[i]['rvir_l']= \
                                            com_i,rads_i,massl_i,idl_i,vcom_i,rvir_l_i
                              sto[i]['cden'] = cden_i
                              #sto[i]['m200'] = m200_i
                              sto[i]['cden_l'] = cden_l_i
                              sto[i]['hull_l'] = hull_i
                              sto[i]['more_variables'] = more_i
                              sto[i]['var_names_scalar'] = var_names_scalar
                              sto[i]['more_vec'] = more_vec_i
                              sto[i]['var_names_vec'] = var_names_vec
                  # if rank ==0:
                  #      print('Completed batch %s of %s' % (g+1,len(split_volumes)))
                  for i in ranks:
                      # if rank ==0:
                      #     print(i)
                      if split_i < split_len[i]:
                        v = None
                        for r in split_jobs[i][split_i]:
                         v = comm.bcast(sto[r],root=int(i))
                         if not v['skip']:
                             for j in range(len(v['mass'])):
                                f_vcom.append(v['vcom'][j])
                                f_com_list.append(v['com'][j])
                                f_rad_list.append(v['rad'][j])
                                f_mass_list.append(v['mass'][j])
                                f_ids.append(v['ids'][j])
                                f_rvir_list.append(v['rvir_l'][j])
                                f_cden_list.append(v['cden_l'][j])
                                f_cden.append(v['cden'][j])
                                f_hull.append(v['hull_l'][j])
                                f_more_l.append(v['more_variables'][j])
                                f_var_names_scalar.append(v['var_names_scalar'][j])
                                f_var_names_vec.append(v['var_names_vec'][j])
                                f_more_vec.append(v['more_vec'][j])
                                #f_m200.append(v['m200'][j])
        sto = {}
        # if rank==0:
        #      print(f_cden)
        # if rank==0:
        #     print(var_names_scalar)
        arg = np.argsort(np.array(f_mass_list))[::-1]
        # Checks for duplicate halos by comparing id lists
        nl0 = []
        lengths = {}
        for f in range(len(f_mass_list)):
            nl0.append([0,str(f)])
            lengths[str(f)] = 1
        nl0 = np.array(nl0)
        # if rank==0:
        #     print(np.array(f_rvir_list))
        indices = get_non_coincident_halo_index(f_ids,np.array(f_mass_list),\
                nl0,np.array(f_rad_list)*self.lu,lengths,np.full(len(f_mass_list),1e60),np.array(f_vcom)*self.lu,\
                    np.array(f_com_list)*self.lu,np.array(f_rvir_list)*self.lu,np.array(f_cden_list),margin=0.2,margin2=0.2)
        # Creates the output attributes
        self.idl = []
        self.hullv = []
        #radii,cden = find_radii(f_com_list,f_rad_list,indices,self.codetp,self.ds)
        for j in indices:
                index = str(len(self.final_nodes))
                self.final_nodes[index] = {}
                self.final_nodes[index]['Halo_Center'] = f_com_list[j]
                oden_count = 0
                oden_list = []
                for rad_i,oden in enumerate(f_cden_list[j]):
                    if oden not in oden_list:
                        oden_0 = str(int(round(oden)))
                    else:
                        oden_0 = str(int(round(oden)))+'.'+str(oden_count)
                        oden_count +=1
                    oden_list.append(oden_0)
                    self.final_nodes[index]['r'+oden_0] = f_rvir_list[j][rad_i]
                self.final_nodes[index]['Halo_Radius'] = f_rad_list[j]
                self.final_nodes[index]['Halo_Mass'] = f_mass_list[j]
                self.final_nodes[index]['Part_Ids'] = f_ids[j]
                self.final_nodes[index]['hullv'] = f_hull[j]
                self.final_nodes[index]['NumParts'] = len(f_ids[j])
                self.final_nodes[index]['Vel_Com'] = f_vcom[j]
                self.final_nodes[index]['cden'] = f_cden[j]
                #print(f_var_names_vec[j],f_more_vec[j])
                for f,var in enumerate(f_var_names_scalar[j]):
                    self.final_nodes[index][var] = f_more_l[j][f]
                # if rank==0:
                #      print(f_var_names_vec[j],f_more_vec[j])
                for f,var in enumerate(f_var_names_vec[j]):
                    # if rank ==0:
                    #     print(f,var)
                    #     print(f_more_vec[j][f])
                    self.final_nodes[index][var] = f_more_vec[j][f]
                #self.final_nodes[index]['cden_l'] = f_cden_list[j]
                #rvir_200_i = np.arange(len(oden_list))[abs(np.array(oden_list).astype(float)-200)==abs(np.array(oden_list).astype(float)-200).min()].min()
                #self.final_nodes[index]['m%s' % str(int(f_cden_list[j][rvir_200_i]))] = f_m200[j]
                self.idl.append(f_ids[j])
                self.hullv.append(f_hull[j])
                #self.final_nodes[index]['hullv'] = f_hull[j]
        if len(indices) > 0:
            self.vcom = np.array(f_vcom)[indices]
            self.massl = np.array(f_mass_list)[indices]
            self.coms = np.array(f_com_list)[indices]
            self.rads = np.array(f_rad_list)[indices]
            self.rvir_l = np.array(f_rvir_list)[indices]
            self.cden = np.array(f_cden)[indices]
            self.more_variables = np.array(f_more_l)[indices]
            #self.m200 = np.array(f_m200)[indices]
            self.cden_l = np.array(f_cden_list)[indices]
            self.var_names_scalar = []
            self.var_names_vec = []
            self.more_vec = []
            for x in indices:
                self.var_names_scalar.append(f_var_names_scalar[x])
            for x in indices:
                self.more_vec.append(f_more_vec[x])
                self.var_names_vec.append(f_var_names_vec[x])
        else:
            self.vcom = np.array([])
            self.massl = np.array([])
            self.coms = np.array([])
            self.rads = np.array([])
            self.rvir_l = np.array([])
            self.cden = np.array([])
            #self.m200 = np.array([])
            self.cden_l = np.array([])
            self.more_variables = np.array([])
            self.var_names_scalar = []
            self.var_names_vec = []
            self.more_vec = []

        del self.ds,self.lu,self.kg_sun


class Over_Density_Finder():
    """ Friends of friends halo-finder using yt density masks for quick halo-finding in parallel.
        Halos found are not well defined and need the detect_halos class to find their precise
        attributes.

    Parameters
    ----------

    timestep : int
        Timestep for halo-finding

    codetp : str
        The code used to create simulation

    refined : bool
        Boolean for whether to use a refined region or the entire simulation volume

    verbose : bool
        Boolean for whether to output debugging text

    Returns
    -------
    self.center : 3 x n array
        Halo centers

    self.radii : 1 x n array
        Halo radii

    self.masses : 1 x n array
        Halo particle masses

    """
    def __init__(self,timestep,ds,lu,codetp='ENZO',refined=False,verbose=False,trackbig=None,icenters=None,iradii=None):
        self.verbose = verbose
        self.codetp= codetp
        self.ds,self.lu = ds,lu
        self.pc = self.ds.length_unit.in_units('pc').v
        self.uden = univDen(self.ds)
        self.minmass = 0
        self.timestep = timestep
        self.refined = refined
        self.icenters,self.iradii = icenters,iradii
        self.radii,self.centers ,self.masses= np.array([]),np.array([]),np.array([])
        # Subdivides the region into 125 cells for parallelization
        self.kg_sun = self.ds.mass_unit.in_units('Msun').v/self.ds.mass_unit.in_units('kg').v
        self.denl =25
        self.deep = False
        self.last = False
        if not oden1:
            self.oden_vir = oden_vir(ds)
        else:
            self.oden_vir = oden1
        # if self.ds.current_redshift > 6:
        #     self.denl *= 7/(1+self.ds.current_redshift)
        # self.denl = max(self.denl,30)
        center1 = np.array([])
        if (np.array(self.icenters) != None).sum()>0:
            self.denl = self.oden_vir
            self.start = False
            numlist = [6,10,12]
            if not refined:
                ll_all,ur_all = self.ds.domain_left_edge.in_units('m').v/self.lu,self.ds.domain_right_edge.in_units('m').v/self.lu
            else:
                region_idx = self.region_number(timestep)
                fr = np.load(savestring + '/' + 'Refined/' + 'refined_region_%s.npy' % (region_idx),allow_pickle=True).tolist()
                ll_all,ur_all = fr
            for i in range(len(numlist)):
                numsegs = numlist[i]
                xx,yy,zz = np.meshgrid(np.linspace(ll_all[0],ur_all[0],numsegs),\
                  np.linspace(ll_all[1],ur_all[1],numsegs),np.linspace(ll_all[2],ur_all[2],numsegs))
                ll = np.concatenate((xx[:-1,:-1,:-1,np.newaxis],yy[:-1,:-1,:-1,np.newaxis],zz[:-1,:-1,:-1,np.newaxis]),axis=3)
                ur =  np.concatenate((xx[1:,1:,1:,np.newaxis],yy[1:,1:,1:,np.newaxis],zz[1:,1:,1:,np.newaxis]),axis=3)
                ll = np.reshape(ll,(ll.shape[0]**3,3))
                ur = np.reshape(ur,(ur.shape[0]**3,3))
                j = np.sum(self.icenters+self.iradii[:,np.newaxis] < ll[:,np.newaxis],axis=2)==0
                k = np.sum(self.icenters-self.iradii[:,np.newaxis] > ur[:,np.newaxis],axis=2)==0
                bool_bound = np.sum(j*k,axis=1)>0
                if rank==0:
                  print('Searching in',round(100*bool_bound.sum()/len(ll),2),'percent of regions')
                ll,ur = ll[bool_bound],ur[bool_bound]
                self.fof(ll[::-1],ur[::-1])
        elif not refined:
            numlist = [4,4,7,8,10,12,14,10,6,3,2]
            #self.denl =30
            self.start = True
            for i in range(len(numlist)):
                # if rank ==0:
                #     print(i,'not refined')
                numsegs = numlist[i]
                ll_all,ur_all = self.ds.domain_left_edge.in_units('m').v/self.lu,self.ds.domain_right_edge.in_units('m').v/self.lu
                xx,yy,zz = np.meshgrid(np.linspace(ll_all[0],ur_all[0],numsegs),\
                  np.linspace(ll_all[1],ur_all[1],numsegs),np.linspace(ll_all[2],ur_all[2],numsegs))
                ll = np.concatenate((xx[:-1,:-1,:-1,np.newaxis],yy[:-1,:-1,:-1,np.newaxis],zz[:-1,:-1,:-1,np.newaxis]),axis=3)
                ur = np.concatenate((xx[1:,1:,1:,np.newaxis],yy[1:,1:,1:,np.newaxis],zz[1:,1:,1:,np.newaxis]),axis=3)
                ll = np.reshape(ll,(ll.shape[0]**3,3))
                ur = np.reshape(ur,(ur.shape[0]**3,3))
                if i >2:
                    self.denl = 60
                if i >3:
                    self.denl = self.oden_vir#300 #np.minimum(self.denl*6,300)
                if (i==4 and len(self.centers>5)) or (i >3 and len(center1) ==0):
                    center1 = np.copy(self.centers)
                    radii1 = np.copy(self.radii)
                if i>=2 and len(self.centers>5) and len(center1)>0:
                    if i <7:
                        j = np.sum(center1+radii1[:,np.newaxis] < ll[:,np.newaxis],axis=2)==0
                        k = np.sum(center1-radii1[:,np.newaxis] > ur[:,np.newaxis],axis=2)==0
                        bool_bound = np.sum(j*k,axis=1)>0
                    else:
                        bool_bound = np.full(len(ll),True)
                    if rank==0:
                      print('Searching in',round(100*bool_bound.sum()/len(ll),2),'percent of regions')
                    ll,ur = ll[bool_bound],ur[bool_bound]
                    if self.denl < self.oden_vir: #300
                        self.radii,self.centers ,self.masses= np.array([]),np.array([]),np.array([])
                self.fof(ll[::-1],ur[::-1])
                self.start = False
        else:
            self.start = True
            fr = np.load(savestring + '/' + 'Refined/' + 'refined_region_%s.npy' % (timestep),allow_pickle=True).tolist()
            ll_all,ur_all = fr
            if rank==0 and self.verbose:
                print('Refined region found :',ll_all,ur_all)
            if trackbig is not None:
                numlist = [2,3,6]
            else:
                numlist = [3,2,6,3,6,10,14,6,3,2]
            if trackbig == 'First' or trackbig == None:
                for i in range(len(numlist)):
                    if i == 1:
                        self.deep = True
                    else:
                        self.deep = False
                    if i == 2:
                        self.denl = 300
                        self.deep = True
                    # if i == 6:
                    #     self.deep = True
                    numsegs = numlist[i]
                    xx,yy,zz = np.meshgrid(np.linspace(ll_all[0],ur_all[0],numsegs),\
                      np.linspace(ll_all[1],ur_all[1],numsegs),np.linspace(ll_all[2],ur_all[2],numsegs))
                    ll = np.concatenate((xx[:-1,:-1,:-1,np.newaxis],yy[:-1,:-1,:-1,np.newaxis],zz[:-1,:-1,:-1,np.newaxis]),axis=3)
                    ur =  np.concatenate((xx[1:,1:,1:,np.newaxis],yy[1:,1:,1:,np.newaxis],zz[1:,1:,1:,np.newaxis]),axis=3)
                    ll = np.reshape(ll,(ll.shape[0]**3,3))
                    ur = np.reshape(ur,(ur.shape[0]**3,3))
                    # if i >2:
                    #     self.denl = 60
                    if (i==3 and len(self.centers>5)) or (i >3 and len(center1) ==0):
                        center1 = np.copy(self.centers)
                        radii1 = np.copy(self.radii)
                    if i>1 and len(self.centers>5) and len(center1)>0:
                        if i < 7:# self.denl < self.oden_vir:
                            if i>2:
                                dx = (ur[0][0]-ll[0][0]).max()
                            else:
                                dx = 1e99
                            j = np.sum(center1 < ll[:,np.newaxis],axis=2)==0
                            k = np.sum(center1 > ur[:,np.newaxis],axis=2)==0
                            dr = radii1 < dx/4
                            # if rank==0:
                            #     print(dr.sum())
                            bool_bound = np.sum(j*k*dr,axis=1)>0
                            #bool_bound = np.sum(j*k,axis=1)>0
                        else:
                            bool_bound = np.full(len(ll),True)
                        if rank==0:
                          print('Searching in',round(100*bool_bound.sum()/len(ll),2),'percent of regions')
                        ll,ur = ll[bool_bound],ur[bool_bound]
                        if self.denl < self.oden_vir: #300
                            self.radii,self.centers ,self.masses= np.array([]),np.array([]),np.array([])
                    if i >2:
                        self.denl = self.oden_vir
                    #300 #np.minimum(self.denl*6,300)
                    # Runs the parallel freinds of friends finder
                    # if rank ==0:
                    #     print(i,'Refined',len(self.centers))
                    self.fof(ll[::-1],ur[::-1])
                    self.start = False
                    if i==2:
                        self.denl = 25
            if trackbig is not None:
                self.denl = min(self.denl*8,self.oden_vir)
                self.start = True
                numlist = [2,4,8,13,3,2,8,10]
                if trackbig == 'First':
                    ind = self.masses == self.masses.max()
                    radmax = 4*self.radii[ind]
                    cenmax = self.centers[ind][0]
                    ll_0,ur_0 = cenmax -radmax, cenmax+radmax
                    ur_all,ll_all = np.minimum(ur_0,ur_all),np.maximum(ll_0,ll_all)
                    #self.start = True
                    bool_max = (np.sum(self.centers >= ll_all,axis=1)==3)*(np.sum(self.centers <= ur_all,axis=1)==3)
                    self.centers,self.masses,self.radii = self.centers[bool_max],self.masses[bool_max],self.radii[bool_max]
                    if rank==0 and self.verbose:
                        #print(len(radii),centers/self.lu,radii/self.lu,masses)
                        print('Reduced to %s overdense volume(s) found with maximum mass %s Msun' % (len(self.radii),round(self.masses.max())))
                        print('New regions is %s %s' % (ll_all,ur_all))
                else:
                    cenmax = trackbig['Halo_Center']
                    radmax = 4*trackbig['Halo_Radius']
                    ll_all,ur_all = cenmax -radmax, cenmax+radmax
                for i in range(len(numlist)):
                    numsegs = numlist[i]
                    xx,yy,zz = np.meshgrid(np.linspace(ll_all[0],ur_all[0],numsegs),\
                      np.linspace(ll_all[1],ur_all[1],numsegs),np.linspace(ll_all[2],ur_all[2],numsegs))
                    ll = np.concatenate((xx[:-1,:-1,:-1,np.newaxis],yy[:-1,:-1,:-1,np.newaxis],zz[:-1,:-1,:-1,np.newaxis]),axis=3)
                    ur =  np.concatenate((xx[1:,1:,1:,np.newaxis],yy[1:,1:,1:,np.newaxis],zz[1:,1:,1:,np.newaxis]),axis=3)
                    ll = np.reshape(ll,(ll.shape[0]**3,3))
                    ur = np.reshape(ur,(ur.shape[0]**3,3))
                    if (i==2 and len(self.centers>5)) or (i >2 and len(center1) ==0):
                        center1 = np.copy(self.centers)
                        radii1 = np.copy(self.radii)
                    if i>1 and len(self.centers>5) and len(center1)>0:
                        j = np.sum(center1+radii1[:,np.newaxis] < ll[:,np.newaxis],axis=2)==0
                        k = np.sum(center1-radii1[:,np.newaxis] > ur[:,np.newaxis],axis=2)==0
                        bool_bound = np.sum(j*k,axis=1)>0
                        if rank==0:
                          print('Searching in',round(100*bool_bound.sum()/len(ll),2),'percent of regions')
                        ll,ur = ll[bool_bound],ur[bool_bound]
                        if self.denl < self.oden_vir:#300:
                            self.radii,self.centers ,self.masses= np.array([]),np.array([]),np.array([])
                    if i >2:
                        self.denl = self.oden_vir#300 #np.minimum(self.denl*6,300)
                    # Runs the parallel friends of friends finder
                    self.fof(ll[::-1],ur[::-1])
                    self.start = False
        self.ll_all,self.ur_all = ll_all,ur_all
        del self.uden,self.ds,self.codetp,self.lu,self.start

    def region_number(self,timestep):
        lenregion = len(savestring + '/' + 'Refined/' + 'refined_region_')
        regions = glob.glob(savestring + '/' + 'Refined/' + 'refined_region_*.npy')
        region_list = []
        for region in regions:
            region_list.append(int(region[lenregion:-4]))
        region_list.sort()
        region_list = np.array(region_list)
        region_idx = region_list[region_list >=timestep].min()
        return region_idx

    def fof(self,ll,ur,tp=1):
        start_s = 0
        my_storage = {}
        jobs,sto = job_scheduler(np.arange(len(ll)))
        ranks = np.arange(nprocs)
        for rank_now in ranks:
            if rank == rank_now:
                for r in jobs[rank]:
                    sto[r]['pass'] = False
                    # Opens simulation regions with a buffer of the maximum of 2 kp or 110% of the cell width
                    if self.start:
                        dwidth = 1.5/self.ds.length_unit.in_units('Mpccm').v
                    else:
                        dwidth = max(2/self.ds.length_unit.in_units('kpccm').v,(ur[r]-ll[r]).max()*.1)
                    #reg = self.ds.region((ll[r]+ur[r])/2,ll[r]-dwidth,ur[r]+dwidth)
                    pos_min = ll[r]-dwidth
                    pos_max = ur[r]+dwidth
                    interval = int(np.ceil(((ur[r]-ll[r]).max()+2*dwidth)*self.pc/400))
                    center = []
                    rads = np.array([])
                    massl = np.array([])
                    mass = np.array([])
                    self.minmass = 0
                    #minmass_rat = max(300/self.denl,2)
                    if self.start:
                        interval = min(30,interval)*1j
                    elif self.deep:
                        interval = min(120,interval)*1j
                    else:
                        interval = min(55,interval)*1j
                    if resave:
                        massi,pos = pickup_particles_backup(self.timestep,pos_min,pos_max,self.ds,vel_ids=False,stars=False)
                    else:
                        interval2 = 1
                        reg = self.ds.r[pos_min[0]:pos_max[0]:interval2, pos_min[1]:pos_max[1]:interval2, pos_min[2]:pos_max[2]:interval2]
                        massi,pos = pickup_particles(reg,self.codetp,vel_ids=False,stars=False)
                        #print(pos.shape)
                    if len(massi) >0:
                        sto[r]['numpart'] = len(massi)
                        xx,yy,zz = self.lu*np.mgrid[pos_min[0]:pos_max[0]:interval, pos_min[1]:pos_max[1]:interval, pos_min[2]:pos_max[2]:interval]
                        dx = np.repeat(xx[1]-xx[0],len(xx))
                        x = xx.flatten()
                        y = yy.flatten()
                        z = zz.flatten()
                        interval = int(np.ceil(((ur[r]-ll[r]).max()+2*dwidth)*self.pc/400))
                        if self.start:
                            interval = min(30,interval)
                        elif self.deep:
                            interval = min(120,interval)
                        else:
                            interval = min(55,interval)
                        xrange = np.linspace(pos_min[0],pos_max[0],interval+1)*self.lu
                        yrange = np.linspace(pos_min[1],pos_max[1],interval+1)*self.lu
                        zrange = np.linspace(pos_min[2],pos_max[2],interval+1)*self.lu
                        #print(xrange.shape,yrange.shape,zrange.shape,pos.shape,massi.shape)
                        mass = stats.binned_statistic_dd(pos, massi, bins = [xrange,yrange,zrange], statistic='sum').statistic
                        massi,pos = None,None
                        mass = mass.flatten()
                        #print(mass.max())
                        #print(gmass[gmass>5.34280319e+34])
                        density = mass/dx**3
                    # else:
                    #     reg = self.ds.r[pos_min[0]:pos_max[0]:interval, pos_min[1]:pos_max[1]:interval, pos_min[2]:pos_max[2]:interval]
                    #     #self.get_min_mass(reg)
                    #     x = reg[('index','x')].in_units('m').v.flatten()
                    #     y = reg[('index','y')].in_units('m').v.flatten()
                    #     z = reg[('index','z')].in_units('m').v.flatten()
                    #     dx = reg[('index','dx')].in_units('m').v.flatten()
                    #     mass = reg[('deposit', 'nbody_mass')].in_units('Msun')
                    #     if mass.sum() >0:
                    #         mass = mass.flatten().astype(float).v
                    if len(mass) >0:
                     grid_pos = np.vstack([x,y,z]).T
                     # if resave:
                     oden = density/self.uden
                     mass = mass*self.kg_sun
                         #print(oden.max())
                     # else:
                     #     oden = ((reg["deposit", "nbody_density"].in_units('kg/m**3').v)).flatten()/self.uden
                     if (oden >self.denl).sum() > 0:
                        bool_range = (oden>=self.denl)
                        bool_big_enough = bool_range*(mass >=self.minmass)
                    # Limits analysis to volumes with desities higher than denl
                        oden_big_r,grid_pos_big_r,dx,mass = oden[bool_big_enough],grid_pos[bool_big_enough],dx[bool_big_enough],mass[bool_big_enough]
                        bool_untaken = np.full(len(oden_big_r),True)
                        halo_assignment = np.zeros(len(oden_big_r))
                        inds = np.arange(len(oden_big_r))
                        count = 1
                    # Creates a list of connected high-density volumes
                        numloops = int(np.maximum(len(grid_pos_big_r)**2/max(50000000,len(grid_pos_big_r)),1))
                        partlen = int(len(grid_pos_big_r)/numloops)
                        #print(numloops,partlen,len(grid_pos_big_r)*partlen,r,rank)
                        for j in range(numloops):
                            ind_j  = inds[j*partlen:min((j+1)*partlen,len(oden_big_r))]
                            dist = cdist(grid_pos_big_r[ind_j],grid_pos_big_r)
                            # Volumes are connected if they are closer than a factor of dx
                            bool_close = dist < 2*np.sqrt(3) * dx[np.newaxis,:]
                            for i,t in enumerate(ind_j):
                                # If this volume is close to a discovered dense group, it is added to that group along with
                                # all the unassigned close dense volumes
                                if bool_close[i][np.logical_not(bool_untaken)].sum() > 0 and halo_assignment[i] == 0:
                                    minhalo = halo_assignment[bool_close[i]*np.logical_not(bool_untaken)].min()
                                    halo_assignment[t] = minhalo
                                    halo_assignment[bool_untaken*bool_close[i]] = minhalo
                                # If there are no close groups, a new group is created
                                elif bool_close[i][bool_untaken].sum() > 0:
                                    halo_assignment[bool_untaken*bool_close[i]] = count
                                    #ids[count] = id_grid[i][bool_untaken*bool_close]
                                    count += 1
                                bool_untaken[halo_assignment >0] = False
                    # Finds the center and radius of each group
                        for j in pd.unique(halo_assignment):
                            # Only records groups with masses above a minimum mass
                            if len(grid_pos_big_r[halo_assignment==j]) > 0 and mass[halo_assignment==j].sum() > self.minmass:
                                l_halo = grid_pos_big_r[halo_assignment==j].min(axis=0)-1.01*dx[halo_assignment==j].max()
                                r_halo = grid_pos_big_r[halo_assignment==j].max(axis=0)+1.01*dx[halo_assignment==j].max()
                                cen = (r_halo+l_halo)/2
                                bool_in_box = sum(cen > ll[r]*self.lu) ==3 and sum(ur[r]*self.lu > cen) ==3
                                bool_contained = sum(grid_pos.max(axis=0) > r_halo) ==3 and sum(grid_pos.min(axis=0) < l_halo) ==3
                                # bool_quart = np.linalg.norm(r_halo-l_halo)/2 < (grid_pos.max(axis=0) - sum(grid_pos.min(axis=0))).max()
                                if bool_in_box and bool_contained:
                                    if len(center) == 0:
                                        center = ((r_halo+l_halo)/2)[np.newaxis,:]
                                    else:
                                        center = np.vstack((center,(r_halo+l_halo)/2))
                                    rads = np.append(rads,np.linalg.norm(r_halo-l_halo)/2)
                                    massl = np.append(massl,mass[halo_assignment==j].sum())
                    # Restricts returns to non-overlapping halos to avoid edge cases.
                    # detect_halos is responsible for finding subhalos
                    if len(rads) > 0:
                        arg = np.argsort(massl)[::-1]
                        massl,center,rads = massl[arg],center[arg],rads[arg]
                        bool_nooverlap = np.full(len(massl),True)
                        dist = cdist(center,center)
                        for i in range(len(massl)):
                            bool_rad = dist[i,:i] < 1*rads[:i]
                            # bool_rad_2 = rads[:i] <
                            # if not self.start:
                            #     bool_mass = abs(massl[i]/massl[:i]-1) < 0.5
                            #     bool_rad = bool_rad*bool_mass
                            if (bool_rad).sum() > 0:
                                bool_nooverlap[i] = False
                        massl,center,rads = massl[bool_nooverlap],center[bool_nooverlap],rads[bool_nooverlap]
                        sto[r]['radii'] = rads
                        sto[r]['centers'] = center
                        sto[r]['mass'] = massl
                        sto[r]['pass'] = True
                    else:
                        sto[r]['pass'] = False
        masses = []
        self.numpart = 0
        for rank_now in ranks:
            for t in jobs[rank_now]:
                    sto[t] = comm.bcast(sto[t], root=rank_now)
                    if start_s == 0 and sto[t]['pass']:
                        centers = sto[t]['centers']
                        radii = sto[t]['radii']
                        masses = sto[t]['mass']
                        start_s = 1
                    elif start_s !=0 and sto[t]['pass']:
                        centers = np.vstack((centers,sto[t]['centers']))
                        radii = np.append(radii,sto[t]['radii'])
                        masses = np.append(masses,sto[t]['mass'])
                    self.numpart = max(self.numpart,sto[t]['numpart'])
        # Aranges halos by mass and sets them as attributes
        sto = {}
        gc.collect()
        if len(masses) > 0:
            arg = np.argsort(masses)[::-1]
            centers,radii,masses = centers[arg],radii[arg],masses[arg]
            if len(self.radii) ==0:
                self.centers,self.radii,self.masses = centers/self.lu,radii/self.lu,masses
            else:
                self.centers = np.vstack((self.centers,centers/self.lu))
                self.radii = np.append(self.radii,radii/self.lu)
                self.masses = np.append(self.masses,masses)
            bool = np.full(len(self.radii),True)
            bool_overlap = None
            N2 = len(self.centers)**2
            array_splits = np.array_split(np.arange(len(self.centers)),max(N2/4e6,1))
            for z,ind_array in enumerate(array_splits):
                jobs,sto = job_scheduler(ind_array)
                ranks = np.arange(nprocs)
                #for sto, i in yt.parallel_objects(range(len(self.centers)), nprocs, storage = my_storage):
                #for i in range(len(self.centers)):
                for rank_now in ranks:
                    if rank == rank_now:
                        for i in jobs[rank]:
                            dist = np.linalg.norm(self.centers[i]-self.centers,axis=1)
                            booli = np.full(len(self.centers),False)
                            for j in range(len(self.centers)):
                                if i != j:
                                    if dist[j]/self.radii[i] < .33 or dist[j]/self.radii[j] < .33 \
                                        and (abs(self.masses[j]/self.masses[i]-1) < 0.25 or abs(self.radii[j]/self.radii[i]-1) < 0.25):
                                        booli[j] = True
                            sto[i]['booli'] = booli
                            sto[i]['i'] = np.where(ind_array==i)[0].item()
                jobs = comm.bcast(jobs,root=0)
                for rank_now in ranks:
                    for i in jobs[rank_now]:
                            sto[i] = comm.bcast(sto[i], root=rank_now)
                            if rank != 0 and rank == rank_now:
                                sto[i] = {}
                if rank ==0:
                    bool_overlap = np.full((len(ind_array),len(self.centers)),False)
                    for rank_now in ranks:
                        for i in jobs[rank_now]:
                        #membership[v['k'],v['new_ind']] = v['membership']
                            bool_overlap[sto[i]['i']] = sto[i]['booli']
                    sto = {}
                    for h in range(len(ind_array)):
                      i = ind_array[h]
                      if bool[i]:
                        for j in np.arange(len(self.centers))[bool_overlap[h]]:
                            if bool[i] and bool[j]:
                                    if self.masses[i] > self.masses[j]:
                                        bool[j] = False
                                    elif self.masses[j] > self.masses[i]:
                                        bool[i] = False
            bool  = comm.bcast(bool,root=0)
            self.centers = self.centers[bool]
            self.radii = self.radii[bool]
            self.masses = self.masses[bool]
            arg_mass =np.argsort(self.masses)[::-1]
            self.centers,self.radii,self.masses = self.centers[arg_mass],self.radii[arg_mass],self.masses[arg_mass]
        if rank==0:# and self.verbose:
                if len(self.radii) >0:
                #print(len(radii),centers/self.lu,radii/self.lu,masses)
                    print('%s overdense volume(s) found with %s overdensity and maximum mass %s Msun' % (len(self.radii),self.denl,round(self.masses.max())))
                else:
                    print('%s overdense volume(s) found' % (len(self.radii)))


    # Returns the number of particles in a region
    def get_min_mass(self,reg):
        # dm_name_dict = {'ENZO':'DarkMatter','GEAR': 'DarkMatter',\
        #  'GADGET3': 'DarkMatter','GADGET4': 'DarkMatter', 'AREPO': 'DarkMatter',\
        #   'GIZMO': 'DarkMatter', 'RAMSES': 'DM',\
        #    'ART': 'darkmatter', 'CHANGA': 'DarkMatter'}
        # mass = reg[(dm_name_dict[self.codetp],'particle_mass')].in_units('Msun').v
        # if len(mass) > 0:
        #     self.minmass = 0 #mass[mass >100].min()
        # else:
        self.minmass = 0


class Find_KE_PE():

        """
        This class detects takes in particle states (mass, position, and velocity) and returns the kinetic
        and potential energy of each particle. There are duplicate functions for using multiple cores to
        calculate the energy.

        Parameters
        ----------
        mass : np.array()
            The mass of the particles in kg

        pos : np.array() (shape : n x 3)
            An array of the positions of the particles in meters

        vel : np.array() (shape : n x 3)
            An array of the velocities of the particles in meters/second

        Returns
        -------
        self.KE : np.array()
            An array of the kinetic energies of each particle.

        self.KE : np.array()
            An array of the potential energies of each particle.

        """

        # Gathers the energies of the particles
        def __init__(self,mass,pos,vel,multi=False):
            self.G = 6.6743e-11
            self.multi = multi
            self.KE = self.Find_KE(mass,vel)
            self.PE = self.Find_Bound(pos,mass)

        # Selects between parallelized and non paralized versions of the energy solver
        def Find_Bound(self,pos,mass,timer = False):
            if timer and rank==0:
                st = time_sys.time()
            # if self.multi:
            #     PE = self.potential_energy_multi(pos,mass, Numlength = Numlength)
            # else:
            PE = self.potential_energy_single(pos,mass, Numlength = Numlength*2)
            if timer and rank==0:
                ft = time_sys.time()
                print("The energy calculator took {:.4f} seconds".format(ft - st))
            return PE

        # Solves for the kinectic energy relative to the center of mass velocity
        def Find_KE(self,mass,vel):
            vcom = (vel*mass[:,np.newaxis]).sum(axis=0)/mass.sum()
            return 0.5*np.linalg.norm(vel-vcom,axis=1)**2

        def find_approx_orbital_energy_square(dmpos,dmmass,dmvel,starpos):
            G = 6.6743e-11
            N = len(dmpos)
            if len(dmpos) >0:
                diam = dmpos[:,0].max()- dmpos[:,0].min()
                diam = diam/N
            else:
                diam = 0
            numloops_dm = int(np.ceil(len(dmpos)/Numlength))
            numloops_star = int(np.ceil(len(starpos)/Numlength))
            pot = np.zeros(len(starpos))
            #print(N,Numlength,numloops)
            inds_dm = np.arange(len(dmpos))
            inds_star = np.arange(len(starpos))
            for i in range(numloops_star):
                ind_star  = inds_star[i*Numlength:min((i+1)*Numlength,len(starpos))]
                #print('ind dm shape', ind_dm.shape)
                for j in range(numloops_dm):
                    ind_dm  = inds_dm[j*Numlength:min((j+1)*Numlength,len(dmpos))]
                    #print('ind star shape', ind_star.shape)
                    r = cdist(starpos[ind_star],dmpos[ind_dm])
                    #print(r)
                    bool = r ==0
                    with np.errstate(divide='ignore'):
                        r = 1/np.maximum(r,diam)
                    mkg = dmmass[ind_dm]
                    r[bool+~np.isfinite(r)] = 0
                    s = np.sum((mkg*r), axis = 1)
                    if j == 0:
                        U = s
                    else:
                        U += s
                pot[ind_star] = U
            PE = -G*pot
            velcom = np.average(dmvel, axis=0, weights=dmmass)
            return PE,velcom

        def find_star_orbital_energy_square(dmpos,dmmass,dmvel,starpos, starvel, starid):
            G = 6.6743e-11
            N = len(dmpos)
            if len(dmpos) >0:
                diam = dmpos[:,0].max()- dmpos[:,0].min()
                diam = diam/N
            else:
                diam = 0
            numloops_dm = int(np.ceil(len(dmpos)/Numlength))
            numloops_star = int(np.ceil(len(starpos)/Numlength))
            pot = np.zeros(len(starpos))
            #print(N,Numlength,numloops)
            inds_dm = np.arange(len(dmpos))
            inds_star = np.arange(len(starpos))
            for i in range(numloops_star):
                ind_star  = inds_star[i*Numlength:min((i+1)*Numlength,len(starpos))]
                #print('ind dm shape', ind_dm.shape)
                for j in range(numloops_dm):
                    ind_dm  = inds_dm[j*Numlength:min((j+1)*Numlength,len(dmpos))]
                    #print('ind star shape', ind_star.shape)
                    r = cdist(starpos[ind_star],dmpos[ind_dm])
                    #print(r)
                    bool = r ==0
                    with np.errstate(divide='ignore'):
                        r = 1/np.maximum(r,diam)
                    mkg = dmmass[ind_dm]
                    r[bool+~np.isfinite(r)] = 0
                    s = np.sum((mkg*r), axis = 1)
                    #print(s)
                    if j == 0:
                        U = s
                    else:
                        U += s
                    #print(U.shape)
                # if i == 0:
                #     pot = U
                # else:
                #     pot = np.append(pot,U)
                pot[ind_star] = U
            #
            PE = -G*pot
            velcom = np.average(dmvel, axis=0, weights=dmmass)
            KE = 0.5*np.linalg.norm(starvel - velcom, axis=1)**2
            E = KE + PE
            #print(len(starid[E<0])/len(starid))
            #print(PE)
            #print(KE)
            return starid[E<0], E[E<0]

        def find_star_orbital_energy_rect(dmpos, dmmass, dmvel, starpos, starvel, starid, allow_mem = 0.5):
            """
            This function calculates the orbital energy of a list of stars with respect to all the bound dark matter particles that define a halo.
            ---
            Input:
                dmpos, dmmass, dmvel: position, mass and velocity of all bound dark matter particles
                starpos, starvel, starid: position, velocity and ID of all stars to be calculated
            ---
            Output:
                ID and orbital energy of all bound stars (E<0)
            """
            G = 6.6743e-11
            chunk_size = int((allow_mem*1e9/8)/len(dmpos)) #this means that the cdist output will take as much 'allow_mem' GB of memory (default is 0.5 GB)
            if chunk_size >= len(starpos):
                disAinv = 1/cdist(starpos, dmpos, 'euclidean')
                disAinv[~np.isfinite(disAinv)] = 0
                PE = np.sum(-G*dmmass*disAinv, axis=1)
            else:
                PE = np.zeros(len(starpos))
                for j in range(0, len(starpos), chunk_size):
                    pos_chunk = starpos[j : j + chunk_size, :]
                    disAinv = 1/cdist(pos_chunk, dmpos, 'euclidean')
                    disAinv[~np.isfinite(disAinv)] = 0
                    PE[j : j + chunk_size] = np.sum(-G*dmmass*disAinv, axis=1)
            velcom = np.average(dmvel, axis=0, weights=dmmass)
            KE = 0.5*np.linalg.norm(starvel - velcom, axis=1)**2
            E = KE + PE
            #print(PE)
            #print(KE)
            return starid[E<0], E[E<0]


        # Solves for potential energy by discretizing the calculation into smaller chucks
        # to save on memory
        def potential_energy_single(self,pos,mass,Numlength = 2000):
            N = len(pos)
            if len(pos) >0:
                diam = pos[:,0].max()-pos[:,0].min()
                diam = diam/N
            else:
                diam = 0
            numloops = int(np.ceil(N/Numlength))
            pot = np.zeros(N)
            #print(N,Numlength,numloops)
            inds = np.arange(len(pos))
            for i in range(numloops):
                ind_i  = inds[i*Numlength:min((i+1)*Numlength,len(pos))]
                for j in range(numloops):
                    ind_j  = inds[j*Numlength:min((j+1)*Numlength,len(pos))]
                    r = cdist(pos[ind_i],pos[ind_j])
                    bool = r ==0
                    with np.errstate(divide='ignore'):
                        r = 1/np.maximum(r,diam)
                    mkg = mass[ind_j]
                    r[bool+~np.isfinite(r)] = 0
                    s = np.sum((mkg*r), axis = 1)
                    if j == 0:
                        U = s
                    else:
                        U += s
                # if i == 0:
                #     pot = U
                # else:
                #     pot = np.append(pot,U)
                pot[ind_i] = U
            return self.G*pot


        # Solves for potential energy by discretizing the calculation into smaller chucks
        # to save memory and parallelized
        # def potential_energy_multi(self,pos,mass,Numlength = 20000):
        #     if len(pos) >0:
        #         diam = pos[:,0].max()-pos[:,0].min()
        #     else:
        #         diam = 0
        #     N = len(pos)
        #     diam = diam/N
        #     numloops = int(np.ceil(N/Numlength))
        #     pot = np.zeros(len(pos))
        #     my_storage = {}
        #     inds = np.arange(len(pos))
        #     for sto, i in yt.parallel_objects(range(numloops), nprocs, storage = my_storage):
        #         ind_i  = inds[i*Numlength:min((i+1)*Numlength,len(pos))]
        #         for j in range(numloops):
        #             ind_j  = inds[j*Numlength:min((j+1)*Numlength,len(pos))]
        #             r = cdist(pos[ind_i],pos[ind_j])
        #             bool = r ==0
        #             with np.errstate(divide='ignore'):
        #                 r = 1/np.maximum(r,diam)
        #             r[bool+~np.isfinite(r)] = 0
        #             mkg = mass[ind_j]
        #             r[r == np.inf] = 0
        #             s = np.sum((mkg*r), axis = 1)
        #             if j == 0:
        #                 U = s
        #             else:
        #                 U += s
        #         sto.result = {}
        #         sto.result['i'] = i
        #         sto.result['U'] = U
        #     for c, v in sorted(my_storage.items()):
        #         i = v['i']
        #         ind_i  = inds[i*Numlength:min((i+1)*Numlength,len(pos))]
        #         pot[ind_i] =  v['U']
        #
        #     return self.G*pot


def open_ds(timestep,codetp,direction=1,skip=False):
    # The conversion factor from code_length to physical unit is not correct in AGORA's GADGET3 and AREPO
    if not organize_files or skip:
        time_sys.sleep(0.02*rank)
        if codetp == 'AREPO':
            ds = yt.load(fld_list[timestep], unit_base = {"length": (1.0, "Mpccm/h")})
        elif codetp == 'GADGET3' or codetp == 'GADGET4':
            ds = yt.load(fld_list[timestep], unit_base = {"length": (1.0, "Mpccm/h"),"UnitMass_in_g": 1.989e43})
        else:
            ds = yt.load(fld_list[timestep])
        ds = make_filter(ds, codetp)
        left0 = ds.domain_left_edge
        right0 = ds.domain_right_edge
        width = 1 #(right0-left0)[0].v
        meter = ds.length_unit.in_units('mcm').v/(width*(1+ds.current_redshift))
        return ds,meter
    else:
        tmp_folder = savestring+'/tmp_files/'
        ensure_dir(tmp_folder)
        my_storage = {}
        for sto, i in yt.parallel_objects(range(10*nprocs), nprocs, storage = my_storage):
            if timestep-direction*i >= 0 and i <3:
                if timestep-direction*i in range(len(fld_list)):
                    sim_files = fld_list[timestep-direction*i].split('/')
                    current_sim_folder = '/'.join(sim_files[:-1])
                    new_sim_folder = tmp_folder+sim_files[-2]
                    #print('make',new_sim_folder)
                    if not os.path.exists(new_sim_folder):
                        os.system('cp -r %s %s' % (current_sim_folder,new_sim_folder))
                        print('cp -r %s %s' % (current_sim_folder,new_sim_folder))
            if i >= 1 and timestep+direction*i >= 0:
                if timestep+direction*i in range(len(fld_list)):
                    sim_files = fld_list[timestep+direction*i].split('/')
                    current_sim_folder = '/'.join(sim_files[:-1])
                    new_sim_folder = tmp_folder+sim_files[-2]
                    #print('erase',new_sim_folder)
                    if os.path.exists(new_sim_folder):
                        os.system('rm -r %s' % (new_sim_folder))
                        print('rm -r %s' % (new_sim_folder))
        sim_files = fld_list[timestep].split('/')
        current_sim_folder = tmp_folder+'/'.join(sim_files[-2:])
        time_now = time_sys.time()
        loaded = False
        while (time_sys.time() - time_now) < 60 and not loaded:
            if os.path.exists(current_sim_folder):
                if codetp == 'AREPO':
                    ds = yt.load(current_sim_folder, unit_base = {"length": (1.0, "Mpccm/h")})
                elif codetp == 'GADGET3' or codetp == 'GADGET4':
                    ds = yt.load(current_sim_folder, unit_base = {"length": (1.0, "Mpccm/h"),"UnitMass_in_g": 1.989e43})
                else:
                    ds = yt.load(current_sim_folder)
                ds = make_filter(ds, codetp)
                left0 = ds.domain_left_edge
                right0 = ds.domain_right_edge
                width = 1 #(right0-left0)[0].v
                meter = ds.length_unit.in_units('mcm').v/(width*(1+ds.current_redshift))
                loaded = True
                return ds,meter
            else:
                time_sys.sleep(5)



# Sorts the halos by smallest to largest and checks if a smaller halo is more
# thank 80% conincident with masses within 25%. The halo with the shortest history is removed.
# This ensures that duplicate halos are removed, including partially determined
# halos at the edge of the investigation region.
def get_non_coincident_halo_index(f_ids,f_mass_list,nl,rad3,lengths,escape,vcom,com,rad_list,cden_list,margin=0.15,margin2=0.15,oden=500,oden2=200):
    #nl = np.array(nl)
    # if rank==0:
    #     print(oden,next_oden,cden_list.shape,rad_i.shape,cden_list[IND_0,rad_i].shape,rad_list.shape)
    #     print(rad_i)
    #     print(rad_i_2)
    #     print(cden_list[IND_0,rad_i])
    #     print(cden_list[IND_0,rad_i_2])
    # if rank ==0:
    #     print(nl)
    # if rank ==0 :
    #     print(rad3)
    #     print(rad_list)
    #     print(com)
    if len(f_mass_list) >1:
        sum_test_pass = np.array([])
        next_oden = oden2#oden_list[min(np.searchsorted(np.array(oden_list),oden),len(oden_list)-1) +1]
        rad_array = np.zeros((len(f_mass_list),len(oden_list)))
        if rad_list.ndim ==1:
            rad_list = rad_list[np.newaxis,:]
        IND_0 = np.arange(len(cden_list))
        bool_big = np.abs(cden_list - np.array(oden_list)) >25
        rad_list[bool_big] = 0
        for oi,oden_i in enumerate(oden_list):
            #rad_i = np.array([min(np.searchsorted(u+1,oden_i),len(u)-1) for u in cden_list])
            #print(rad_i.shape,rad_i,rad_array.shape,rad_list.shape)
            #print(rad_list[IND_0,rad_i])
            #
            rad_array[:,oi] = rad_list[IND_0,oi]
        #rad_i_2 = np.array([min(np.searchsorted(u+1,next_oden),len(u)-1) for u in cden_list])
        if vcom.ndim >1:
            vcom_norm = np.linalg.norm(vcom,axis=1)
        else:
            vcom_norm = np.linalg.norm(vcom)
        #rad1 = rad_list[IND_0,rad_i]
        #rad2 = rad_list[IND_0,rad_i_2]
        rad3 = rad3[IND_0]
        keys = nl[:,0].astype(int)
        halo = nl[:,1].astype(str)
        weights = np.zeros(len(rad3))
        for ih,h in enumerate(halo):
            weights[ih] = max(lengths[h],1)*f_mass_list[ih]
        #halo = halo[arg]
        #keys = keys[arg]
        bool = np.full(len(f_mass_list),True)
        index0 = np.arange(len(f_mass_list))
        arg0 = np.argsort(f_mass_list[keys < 0])
        index = index0[keys < 0][arg0]
        margin0 = np.full(len(index),margin2)
        arg0 = np.argsort(f_mass_list[keys > 0])
        index = np.append(index,index0[keys > 0][arg0])
        margin0 = np.append(margin0,np.full(len(index0[keys > 0]),margin2))
        arg0 = np.argsort(weights[keys == 0])
        index = np.append(index,index0[keys == 0][arg0])
        margin0 = np.append(margin0,np.full(len(index0[keys == 0]),margin))
        if len(index) >1:
            N2 = len(com)**2
            array_splits = np.array_split(np.arange(len(index)),max(N2/4e5,1))
            for z,ind_array in enumerate(array_splits):
                jobs,sto_false = job_scheduler(ind_array)
                ranks = np.arange(nprocs)
                sto = {}
                for rank_now in ranks:
                    sto[rank_now] = {}
                    if rank == rank_now:
                        for i in jobs[rank]:
                            sto[rank_now][i] = {}
                            k = index[i]
                            new_ind = index[i+1:]
                            membership1 = np.array([np.isin(f_ids[k],f_ids[j]).sum()/len(f_ids[k]) for j in new_ind])
                            sto[rank_now][i]['array_ind'] = np.where(ind_array==i)[0].item()
                            sto[rank_now][i]['new_ind'] = new_ind
                            #sto.result['membership'] = membership1
                            bool_test = np.full(len(com),False)
                            if len(com>1):
                                dist_all = np.linalg.norm(com[k]-com,axis=1)
                                vcomn = np.dot(vcom,vcom[k])/(vcom_norm[k]*vcom_norm)
                            else:
                                dist_all = np.array([0])
                                vcomn = np.array([1])
                            vcomn_diff = abs(vcomn - 1)
                            for t,j in enumerate(new_ind):
                                if k != j and membership1[t]>0:
                                    dist = dist_all[j]
                                    rad_bool = (rad_array[k,:] >0 )*(rad_array[j,:] >0)
                                    mass_bool = abs(f_mass_list[k]/f_mass_list[j] - 1)
                                    if rad_bool.sum() >0:
                                        bool_rad = np.abs(rad_array[k,rad_bool]/rad_array[j,rad_bool]-1).min() < margin0[k] or abs(rad3[k]/rad3[j]-1) < margin0[k]
                                    else:
                                        bool_rad = abs(rad3[k]/rad3[j]-1) < margin0[k]
                                    bool_test[j] = (dist == 0) or ((dist/rad3[k] < margin0[k] or dist/rad3[j] < margin0[k]) and \
                                            (bool_rad or mass_bool) and (vcomn_diff[j] < 0.01) and abs(vcom_norm[k]/vcom_norm[j] - 1) < 0.05)
                                    # print(dist/rad3[k],dist/rad3[j],abs(rad3[k]/rad3[j]-1),np.abs(rad_array[k,rad_bool]/rad_array[j,rad_bool]-1),bool_rad,mass_bool,bool_test[j])
                                    # if (dist/rad_array[k,0] < margin0[k] or dist/rad_array[j,0] < margin0[k]):# and not bool_test[j]:
                                    #     print(dist/rad_array[k,0],dist/rad_array[j,0],abs(f_mass_list[k]/f_mass_list[j] - 1),\
                                    #         np.abs(rad_array[k,:]/rad_array[j,1]-1),abs(rad3[k]/rad3[j]-1) < margin0[k],abs(vcom_norm[k]/vcom_norm[j] - 1),bool_test[j])
                            sto[rank_now][i]['bool_test'] = bool_test
                            bool_test = 0
                jobs = comm.bcast(jobs,root=0)
                for rank_now in ranks:
                        sto[rank_now] = comm.bcast(sto[rank_now], root=rank_now)
                        if rank != 0 and rank == rank_now:
                            sto[rank_now] = {}
                if rank ==0:
                    test_pass = np.full((len(ind_array),len(com)),False)
                    for rank_now in ranks:
                        for t in jobs[rank_now]:
                            test_pass[sto[rank_now][t]['array_ind']] = sto[rank_now][t]['bool_test']
                    #sum_test_pass = np.append(sum_test_pass,test_pass.sum(axis=1))
                    #print(test_pass.sum(axis=0),test_pass.sum(axis=1))
                    #print(sum_test_pass.mean())
                    #print(len(sum_test_pass),np.percentile(sum_test_pass,25),np.percentile(sum_test_pass,50),np.percentile(sum_test_pass,75))
                    for h,i in enumerate(ind_array):
                            k = index[i]
                            new_ind = index[i+1:][test_pass[h][index[i+1:]]*bool[index[i+1:]]]
                            for j in new_ind:
                                if bool[np.where(index==k)] and bool[np.where(index==j)]:
                                      if keys[j] != 0:
                                          lenj = 0
                                      else:
                                          lenj = lengths[halo[j]]
                                      if keys[k] != 0:
                                          lenk = 0
                                      else:
                                          lenk = lengths[halo[k]]
                                      # if bool[np.where(index==k)] and bool[np.where(index==j)]:
                                      if lenj >=  lenk -2 or lenk==0:
                                          bool[np.where(index==k)] = False
                                          #print(k,j,halo[k],halo[j],lenk,lenj,bool[np.where(index==k)])
        test_pass = 0
        bool = comm.bcast(bool,root=0)
        if rank==0:
            print(len(index[bool]),'of',len(index),'halos preserved')
        if not fake:
            index = index[bool]
        return index[::-1]
    else:
        return np.arange(len(f_mass_list))

def univDen(ds):
    # Hubble constant
    H0 = ds.hubble_constant * 100 * u.km/u.s/u.Mpc
    H = H0**2 * (ds.omega_matter*(1 + ds.current_redshift)**3 + ds.omega_lambda)  # Technically H^2
    G = 6.67e-11 * u.m**3/u.s**2/u.kg
    # Density of the universe
    den = (3*H/(8*np.pi*G)).in_units("kg/m**3") / u.kg * u.m**3
    return den.v

def oden_vir(ds):
    omega_o = ds.omega_matter*(1 + ds.current_redshift)**3
    Ez = ds.omega_matter*(1 + ds.current_redshift)**3 + ds.omega_lambda
    omega_z = omega_o/Ez
    x = omega_z - 1
    oden = 18*np.pi**2 + 82*x - 39*x**2
    return oden

def find_star_energy(pos,mass,vel,com,spos,svel,sids):
        if len(sids)>5000:
            time_check = time_sys.time()
            region_ind,pos_annuli = cut_star_particles(spos,com)
            filled_reg = np.unique(region_ind).astype(int)
            #print(filled_reg)
            PE_star,velcom = Find_KE_PE.find_approx_orbital_energy_square(pos,mass,vel,pos_annuli)
            KE_all_stars = 0.5*np.linalg.norm(svel - velcom, axis=1)**2
            PE_all_stars = np.zeros(len(sids))
            for PEi in range(len(PE_star)):
                PE_all_stars[region_ind==filled_reg[PEi]] = PE_star[PEi]
            #region_ind = None
            #print(KE_all_stars,PE_all_stars)
            Star_Energy = KE_all_stars+PE_all_stars
            star_rat = KE_all_stars/np.abs(PE_all_stars)
            bool_resim = (star_rat <1.5)*(star_rat>0.75)
            not_bool_resim = np.logical_not(bool_resim)
            sid_i,s_energy_i = sids[not_bool_resim][Star_Energy[not_bool_resim] <0],Star_Energy[not_bool_resim][Star_Energy[not_bool_resim]<0]
            Star_Energy,not_bool_resim = None, None
            sid_i_1,s_energy_i_1 = Find_KE_PE.find_star_orbital_energy_square(pos,mass,vel,spos[bool_resim],svel[bool_resim],sids[bool_resim])
            sid_i = np.append(sid_i,sid_i_1)
            s_energy_i = np.append(s_energy_i,s_energy_i_1)
            KE_all_stars,PE_all_stars,bool_resim= None,None,None
            #sid_i,s_energy_i = self.sids[Star_Energy <0],Star_Energy[Star_Energy<0]
            # time_check_0 = time_sys.time()-time_check
            # time_check = time_sys.time()
            # #print(len(sid_i)/len(self.sids),len(PE_star),len(region_ind),len(self.sids))
            # sid_i_2,s_energy_i_2 = Find_KE_PE.find_star_orbital_energy_square(self.pos[untaged][halo_bool][es<0],self.mass[untaged][halo_bool][es<0],self.vel[untaged][halo_bool][es<0],self.spos,self.svel,self.sids)
            # time_check_1 = time_sys.time()-time_check
            # if len(sid_i_2)>0:
            #     print(1-len(sid_i)/len(sid_i_2),len(sid_i)-len(sid_i_2),len(sid_i_2),(time_check_1-time_check_0)/time_check_0)
        else:
            sid_i,s_energy_i = Find_KE_PE.find_star_orbital_energy_square(pos,mass,vel,spos,svel,sids)
        return sid_i,s_energy_i

def cut_star_particles(pos,center):
    vec = {}
    vec[0] = vecs_calc(2)
    vec[1] = vecs_calc(3)
    vec[2] = vecs_calc(3)
    dist_pos = np.linalg.norm(pos-center,axis=1)
    inner = np.array([100,50,12])
    annuli = np.linspace(0,dist_pos.max()/3,inner[0])
    annuli2 = np.linspace(dist_pos.max()/3,2*dist_pos.max()/3,inner[1])[1:]
    annuli3 = np.linspace(2*dist_pos.max()/3,dist_pos.max(),inner[2])[1:]
    annuli = np.append(annuli,annuli2)
    annuli = np.append(annuli,annuli3)
    index = np.arange(len(pos))
    region_ind = np.zeros(len(pos))
    pos_annuli = np.array([])
    count = 0
    for i in range(len(annuli)-1):
        bool_in_0 = (dist_pos <= annuli[i+1])*(dist_pos > annuli[i])
        current_group = np.arange(len(inner))[(i >= np.cumsum(inner)-inner)*(i < np.cumsum(inner))][0]
        pos_norm = dist_pos[bool_in_0][:,np.newaxis]
        pos_in = pos[bool_in_0]
        vec_ang = np.dot((pos_in-center),vec[current_group].T)
        if len(np.unique(vec_ang)) != len(vec_ang):
            vec_ang += vec_ang.min()*1e-5*np.random.random(len(vec_ang[0]))[np.newaxis,:]
        pos_group = np.where(vec_ang == vec_ang.max(axis=1)[:,np.newaxis])[1]
        for t in range(len(vec[current_group])):
            current_index = index[bool_in_0][pos_group==t]
            if len(current_index) >0:
                region_ind[current_index] = count
                if len(pos_annuli) ==0:
                    pos_annuli = center+vec[current_group][t]*(annuli[i+1]+annuli[i])/2
                else:
                    pos_annuli = np.vstack((pos_annuli,center+vec[current_group][t]*(annuli[i+1]+annuli[i])/2))
                count += 1
    return region_ind,pos_annuli


def cut_particles(pos,mass,center,ids,idl_i=None,cut_size=700,dense=False,segments=1,timing=0):
        if timing:
            time5 = time_sys.time()
            time6 = time_sys.time()
        bool = np.full(len(pos),True)
        vec = {}
        if not dense:
            vec[0] = vecs_calc(1)
            vec[1] = vecs_calc(1)
            vec[2] = vecs_calc(1)
        else:
            vec[0] = vecs_calc(1)
            vec[1] = vecs_calc(2)
            vec[2] = vecs_calc(2)
        dist_pos = np.linalg.norm(pos-center,axis=1)
        inner = np.array([40,20,12])
        annuli = np.linspace(0,dist_pos.max()/3,inner[0])
        annuli2 = np.linspace(dist_pos.max()/3,2*dist_pos.max()/3,inner[1])[1:]
        annuli3 = np.linspace(2*dist_pos.max()/3,dist_pos.max(),inner[2])[1:]
        annuli = np.append(annuli,annuli2)
        annuli = np.append(annuli,annuli3)
        index = np.arange(len(pos))
        for i in range(len(annuli)-1):
            bool_in_0 = (dist_pos <= annuli[i+1])*(dist_pos > annuli[i])
            cutlength = cut_size*annuli[i+1]/annuli[-1]
            current_group = np.arange(len(inner))[(i >= np.cumsum(inner)-inner)*(i < np.cumsum(inner))][0]
            pos_norm = dist_pos[bool_in_0][:,np.newaxis]
            pos_in = pos[bool_in_0]
            vec_ang = np.dot((pos_in-center),vec[current_group].T)
            if len(np.unique(vec_ang)) != len(vec_ang):
                vec_ang += vec_ang.min()*1e-5*np.random.random(len(vec_ang[0]))[np.newaxis,:]
            pos_group = np.where(vec_ang == vec_ang.max(axis=1)[:,np.newaxis])[1]
            for t in range(len(vec[current_group])):
                current_index = index[bool_in_0][pos_group==t]
                if len(current_index) > max(cutlength,max(10/segments,1)):
                    mass_tot = mass[current_index].sum()
                    cut = int(np.ceil(len(current_index)/cutlength))
                    bool[current_index] = False
                    rand_ind = np.random.choice(current_index,size=len(current_index),replace=False)
                    bool[rand_ind[0::cut]] = True
                    mass_in = mass[rand_ind[0::cut]].sum()
                    mass[rand_ind[0::cut]] *= mass_tot/mass_in
            if timing and time_sys.time()-time5 >timing:
                print('Make Annuli',time_sys.time()-time5)
                time5 = time_sys.time()
        del pos
        mass[index[np.logical_not(bool)]] *= 1e-10
        if idl_i is not None:
                bool = np.logical_or(np.isin(ids,idl_i),bool)
        if timing and time_sys.time()-time6 >timing:
            print('Cut particles interior',time_sys.time()-time6)
        return mass[bool],bool

def make_hull(r,pos,mass,es,oden,crit_den,rmax=1e99,user=False):
    # if user:
    #     cut_e = r.max()+1
    #     esr = r
    # else:
    cut_e = 0#r.max()+1   #es[r>rmax] = 1e99
    esr = es*(r/rmax)**-2 #r
    esr[es >=0] = cut_e
    count = 0
    if len(mass[esr<cut_e]) >4:
        hull = ConvexHull(pos[esr <cut_e])
        hull2 = hull
        cden_1 = (mass[esr <cut_e].sum()/hull.volume)/crit_den
        mass_1 = mass[esr <cut_e].sum()
        cden_2 = np.copy(cden_1)
        mass_2 = np.copy(mass_1)
        inter = 0
        rlast = r[esr<cut_e].max()
        rlast_2 = np.copy(rlast)
        estep_f = cut_e
        # if cden_1 > oden:
        #     oden *=1.1
        #es*(r/r.max())**-2
        if cden_1 < oden:
            arg_es = np.argsort(esr)
            es_sort = esr[arg_es]
            es_sort = es_sort[es_sort<cut_e]
            es_last = len(es_sort) -1
            step = int(len(es_sort)/6)
            count = 1
            if len(es_sort) >4:
                while abs(cden_1 -oden) > 5  and inter <50:
                    step_i = min(max(int(es_last+np.sign(cden_1-oden)*step),0),len(es_sort) -1)
                    es_last = step_i
                    #print(es_sort,step_i)
                    estep = es_sort[step_i]
                    inter += 1
                    if len(mass[esr <estep]) > 4:
                        hull = ConvexHull(pos[esr <estep])
                        cden_0 = (mass[esr <estep].sum()/hull.volume)/crit_den
                        dv = (4/3 * np.pi)*((r[esr<estep].max()/r[es <=0].max())**3-(rlast/r[es <=0].max())**3)
                        #dm = (mass[esr <estep].sum()-mass_1)/mass_2
                        dcdends = -1*(cden_0-cden_1)/(np.sign(cden_1-oden)*step)
                        if dv >= 0 and np.sign(oden-cden_1) >0 and r[esr<estep].max() > np.percentile(r[es <=0],99):
                            oden *= 1.05
                            step = max(int(len(es_sort)/(5*1.1**inter)),step)
                            #step = min(int(len(es_sort)*0.01*min(abs(cden_1-oden)/2,30)),step)
                        elif abs(dcdends) >0:
                            step = min(int(len(es_sort)*0.01*min(abs(cden_1-oden)/2,30)),step)
                        if np.sign(cden_0-oden) != np.sign(cden_1-oden):
                            step /= (1.75+0.1*count)
                            count += 1
                        else:
                            step /= 1.2
                        # if rank==0:
                        #      print(estep,inter,step_i,step,np.sign(oden-cden_1),cden_1,cden_0,dv,r[esr<estep].max()/np.percentile(r[es <=0],90),oden)
                        cden_1 = cden_0
                        mass_1 = mass[esr <estep].sum()
                        rlast = r[esr<estep].max()
                        estep_f = estep
        rvir = rlast
        cden_f = cden_1
        # if cden_f >500:
        #     cden_f = cden_2
        #     rvir = rlast_2
        #     mass_1 = mass_2
        #     estep = cut_e
        #     hull = hull2
        # if cden_2 <oden:
        #     print(cden_1,cden_1/cden_2,mass_1/mass_2,cden_2,oden,count)
        #print(oden,cden_2,cden_1,mass_2,mass_1,estep_f,inter,len(pos),len(pos[esr <estep_f]))
        return mass_1,cden_f,rvir,estep_f,esr,hull
    else:
        return 0,0,0,0,esr,0

def Find_virRad(com,pos,mass,ds,oden=[200],radmax=1e50):
    r = np.linalg.norm(com-pos,axis=1)
    r,mass = r[(r < radmax)*(r>0)],mass[(r < radmax)*(r>0)]
    arg_r = np.argsort(r)
    mass_cum = mass[arg_r].cumsum()
    V = 4/3 * np.pi * (r[arg_r])**3
    radius,cden,mden = np.zeros(len(oden)),np.zeros(len(oden)),np.zeros(len(oden))
    check = np.zeros(len(oden))
    masst = mass[arg_r].sum()
    if V.sum() > 0:
        den = mass_cum/V
        uden = univDen(ds)
        fden = np.array(den/uden)
        bool_den = (fden>min(oden_list))*(fden<max(oden_list))
        radius[0] = r.max()
        r = r[arg_r][bool_den]
        fden = fden[bool_den]
        for t,od in enumerate(oden):
            close = np.abs(fden-oden[t]) < 20
            if close.sum() >0:
                closest = np.abs(fden-oden[t]).min() == np.abs(fden-oden[t])
                radius[t] = r[closest].max()
                cden[t] = fden[r==radius[t]].min()
                mden[t] = mass_cum[bool_den][r==radius[t]].max()
                check[t] = 1
            elif len(fden[(fden >= oden[t])]) > 2 and len(fden) >0:
                radius[t] = r[(fden >= oden[t])*(fden < max(oden_list))].max()
                cden[t] = fden[r==radius[t]].min()
                mden[t] = mass_cum[bool_den][r==radius[t]].max()
                check[t] = 3
            elif len(fden[(fden <= oden[t])]) > 2 and len(fden) >0:
                radius[t] = r[(fden <= oden[t])*(fden > min(oden_list))].min()
                cden[t] = fden[r==radius[t]].min()
                mden[t] = mass_cum[bool_den][r==radius[t]].max()
                check[t] = 2
            else:
                radius[t] = radius[t-1]
                if len(fden[r==radius[t]])> 0:
                    cden[t] = fden[r==radius[t]].min()
                    mden[t] = mass_cum[bool_den][r==radius[t]].max()
                else:
                    cden[t] = 0
                    mden[t] = 0
                check[t] = 4
        #print(radius,cden,check)
        if len(oden) >1:
            return radius, cden, mden
        else:
            return radius[0],cden[0],mden[0]
    else:
        return radius, cden,mden

# Creates a list of particle counts
def get_particle_number(ds,com_list,rad_list,codetp):
        len_part = []
        my_storage = {}
        rad_list = np.array(rad_list)
        minrad = 0
        if len(rad_list) >0:
            minrad = rad_list[rad_list >0].min()
        for sto, k in yt.parallel_objects(range(len(rad_list)), nprocs, storage = my_storage):
            center, radius = com_list[k],rad_list[k]
            res = 20*ds.index.get_smallest_dx()
            reg = get_region(ds,center,max(radius,minrad)*2.8)
            sto.result = {}
            sto.result['lenp'] = get_part_length(reg,codetp)
            del reg
        for c, v in sorted(my_storage.items()):
            len_part.append(v['lenp'])
        len_part = np.array(len_part)
        return len_part


# Returns the number of particles in a region
def get_part_length(reg,codetp,tp=1):
    dm_name_dict = {'ENZO':'DarkMatter','GEAR': 'DarkMatter',\
     'GADGET3': 'DarkMatter','GADGET4': 'DarkMatter', 'AREPO': 'DarkMatter',\
      'GIZMO': 'DarkMatter', 'RAMSES': 'DM',\
       'ART': 'darkmatter', 'CHANGA': 'DarkMatter'}
    mass = reg[(dm_name_dict[codetp],'particle_mass')].in_units('kg').v
    return len(mass)


def make_filter_star(ds_now, codetp, manual=None):
    """
    This function loads the simulation snapshot and create a star particle field (if necessary). It also changes the fieldtype name to 'star' regardless of the original name in each simulation code.

    If the user has a different fieldtype name for star particles in their simulation (which are not in the default list of this function), the user can provide their simulation's fieldtype name through the 'manual' variable.
    """
    star_name_default_dict = {'GADGET3':'PartType4','GADGET4':'PartType4','GEAR':'PartType1','AREPO':'PartType4','GIZMO':'PartType4','RAMSES':'star','ART':'stars','CHANGA':'Stars'}
    #Checking manual compatibility
    if codetp == 'ENZO' and manual != None and all(isinstance(manual_i, int) for manual_i in manual) == False:
        raise ValueError("for ENZO code, manual must have integer values")
    elif codetp in ['GADGET3','GADGET4','GEAR','AREPO','GIZMO','RAMSES','ART','CHANGA'] and manual != None and all(isinstance(manual_i, str) for manual_i in manual) == False:
        raise ValueError("for GADGET3,'GADGET4', GEAR, AREPO, GIZMO, RAMSES, ART and CHANGA codes, manual must have string values")
    #
    if codetp == 'ENZO':
        def star_init(pfilter, data):
            if manual == None:
                filter_star0 = np.logical_or.reduce((data["all", "particle_type"] == 2, data["all", "particle_type"] == 5, data["all", "particle_type"] == 7))
            elif manual != None:
                if len(manual) == 1:
                    filter_star0 = data["all", "particle_type"] == manual[0]
                elif len(manual) > 1:
                    filter_star0 = np.logical_or.reduce([data["all", "particle_type"] == manual_i for manual_i in manual])
            filter_star = np.logical_and(filter_star0,data['all', 'particle_mass'].to('Msun') > 1)
            return filter_star
        add_particle_filter("star",function=star_init,filtered_type='all',requires=["particle_type","particle_mass"])
        ds_now.add_particle_filter("star")
    elif codetp == 'ART':
        if manual == None:
            def star_init(pfilter, data):
                return (data[(pfilter.filtered_type, 'particle_mass')] > 0)
            add_particle_filter('star', function=star_init, filtered_type=star_name_default_dict[codetp], requires=['particle_mass'])
            ds_now.add_particle_filter('star')
        else:
            filter_star = ParticleUnion("star",[manual])
            ds_now.add_particle_union(filter_star)
    else:
        if manual == None:
            filter_star = ParticleUnion("star",[star_name_default_dict[codetp]])
        else:
            filter_star = ParticleUnion("star",[manual])
        ds_now.add_particle_union(filter_star)
    return ds_now


def pickup_particles(reg,codetp,stars=True,vel_ids=True):
    # 'nbody'
    if codetp == 'ENZO':
        mass_all = reg[('all','particle_mass')].in_units('Msun')
        pos_all = reg[('all','particle_position')].in_units('m')
        if vel_ids:
            vel_all = reg[('all','particle_velocity')].in_units('m/s')
            ids_all = reg[('all','particle_index')].v.astype(int)
        type_all = reg[('all','particle_type')].v.astype(int)
    if find_dm:
        dm_name_dict = {'GEAR': ["PartType5","PartType2"],\
                'GADGET3': ["PartType5","PartType1"],'GADGET4': ["PartType5","PartType1"], 'AREPO': ["PartType2","PartType1"],\
                'GIZMO': ["PartType2","PartType1"], 'RAMSES': 'DM',\
                'ART': 'darkmatter', 'CHANGA': 'DarkMatter'}
        if codetp == 'ENZO':
            dm_bool = np.logical_and(np.logical_or(type_all == 1, type_all==4), mass_all > 0.1)
            mass = mass_all[dm_bool]
            pos = pos_all[dm_bool]
            if vel_ids:
                vel = vel_all[dm_bool]
                ids = ids_all[dm_bool]
        elif codetp == 'GEAR' or codetp == 'GADGET3' or codetp == 'AREPO' or codetp == 'GIZMO' or codetp == 'GADGET4':
            for type_i in range(len(dm_name_dict[codetp])):
                if type_i == 0:
                    mass = reg[(dm_name_dict[codetp][type_i],'particle_mass')].in_units('Msun')
                    pos = reg[(dm_name_dict[codetp][type_i],'particle_position')].in_units('m')
                    if vel_ids:
                        vel = reg[(dm_name_dict[codetp][type_i],'particle_velocity')].in_units('m/s')
                        ids = reg[(dm_name_dict[codetp][type_i],'particle_index')].v.astype(int)
                else:
                    mass = np.append(mass, reg[(dm_name_dict[codetp][type_i],'particle_mass')].in_units('Msun'))
                    pos = np.append(pos, reg[(dm_name_dict[codetp][type_i],'particle_position')].in_units('m'), axis=0)
                    if vel_ids:
                        vel = np.append(vel, reg[(dm_name_dict[codetp][type_i],'particle_velocity')].in_units('m/s'), axis=0)
                        ids = np.append(ids, reg[(dm_name_dict[codetp][type_i],'particle_index')].v.astype(int))
        elif codetp == 'RAMSES' or codetp == 'ART':
            mass = reg[(dm_name_dict[codetp],'particle_mass')].in_units('Msun')
            pos = reg[(dm_name_dict[codetp],'particle_position')].in_units('m')
            if vel_ids:
                vel = reg[(dm_name_dict[codetp],'particle_velocity')].in_units('m/s')
                ids = reg[(dm_name_dict[codetp],'particle_index')].v.astype(int)
        elif codetp == 'CHANGA':
            mass = reg[(dm_name_dict[codetp],'particle_mass')].in_units('Msun')
            pos = reg[(dm_name_dict[codetp],'particle_position')].in_units('m')
            if vel_ids:
                vel = reg[(dm_name_dict[codetp],'particle_velocity')].in_units('m/s')
                ids = np.arange(len(mass)).astype(int)
    #Adding stars
    if find_stars and stars:
        star_name_dict = {'GADGET3':'PartType4','GADGET4':'PartType4','GEAR':'PartType1','AREPO':'PartType4','GIZMO':'PartType4','RAMSES':'star','ART':'stars','CHANGA':'Stars'}
        if codetp == 'ENZO':
            star_bool = np.logical_and(np.logical_or.reduce((type_all == 2, type_all == 5, type_all == 7)), mass_all > 1)
            spos = pos_all[star_bool]
            if vel_ids:
                svel = vel_all[star_bool]
                sids = ids_all[star_bool]
        else:
            try:
                spos = reg[(star_name_dict[codetp],'particle_position')].in_units('m')
                if vel_ids:
                    if codetp == 'CHANGA':
                        sids = np.arange(len(spos)) + 15368024 #only applicable when loading the whole box, issue with AGORA CHANGA data
                    else:
                        sids = reg[(star_name_dict[codetp],'particle_index')].astype(int)
                    svel = reg[(star_name_dict[codetp],'particle_velocity')].in_units('m/s')
            except:
                spos,svel,sids = np.reshape(np.array([]), (0,3)), np.reshape(np.array([]), (0,3)), np.array([])
    if vel_ids:
        if find_dm and (find_stars and stars):
            return mass.in_units('kg').v,pos,vel,ids,spos,svel,sids
        elif (find_stars and stars) and not find_dm:
            return spos,svel,sids
        elif find_dm and not (find_stars and stars):
            return mass.in_units('kg').v,pos,vel,ids
    else:
        if find_dm and (find_stars and stars):
            return mass.in_units('kg').v,pos,spos
        elif (find_stars and stars) and not find_dm:
            return spos
        elif find_dm and not (find_stars and stars):
            return mass.in_units('kg').v,pos

def pickup_min_dm(ll,ur,codetp,ds,floormass=100):
    if resave:
        mass,pos = pickup_particles_backup(last_timestep,ll,ur,ds,vel_ids=False,stars=False)
        pos = None
        lenmass = len(mass)
        if len(mass) >0:
            mass = mass[mass >floormass].min()
        else:
            mass = 1e99
    else:
        reg = ds.box(ll ,ur)
        if codetp == 'ENZO':
            mass_all = reg[('all','particle_mass')].in_units('Msun')
            type_all = reg[('all','particle_type')].v.astype(int)
        if find_dm:
            dm_name_dict = {'GEAR': ["PartType5","PartType2"],\
                    'GADGET3': ["PartType5","PartType1"],'GADGET4': ["PartType5","PartType1"], 'AREPO': ["PartType2","PartType1"],\
                    'GIZMO': ["PartType2","PartType1"], 'RAMSES': 'DM',\
                    'ART': 'darkmatter', 'CHANGA': 'DarkMatter'}
            if codetp == 'ENZO':
                dm_bool = np.logical_and(np.logical_or(type_all == 1, type_all==4), mass_all > 1)
                mass = mass_all[dm_bool]
            elif codetp == 'GEAR' or codetp == 'GADGET3' or codetp == 'AREPO' or codetp == 'GIZMO' or codetp == 'GADGET4':
                for type_i in range(len(dm_name_dict[codetp])):
                    if type_i == 0:
                        mass = reg[(dm_name_dict[codetp][type_i],'particle_mass')].in_units('Msun')
                    else:
                        mass = np.append(mass, reg[(dm_name_dict[codetp][type_i],'particle_mass')].in_units('Msun'))*mass.units
            elif codetp == 'RAMSES' or codetp == 'ART':
                mass = reg[(dm_name_dict[codetp],'particle_mass')].in_units('Msun')
            elif codetp == 'CHANGA':
                mass = reg[(dm_name_dict[codetp],'particle_mass')].in_units('Msun')
        lenmass = len(mass)
        if len(mass) >0:
            mass = mass[mass >floormass].min()
        else:
            mass = 1e99
    return mass,lenmass

def build_hierarchy(centers,radii,hindex,interior,ll_0,ur_0,halo_max=20):
    level = 0
    tagged = np.zeros(1)
    bool_not_non = hindex != None
    new_ind = np.arange(len(hindex))[bool_not_non]
    bool_in = (np.sum(centers[new_ind] - radii[new_ind,np.newaxis] \
        > ll_0,axis=1) ==3)*(np.sum(centers[new_ind] + radii[new_ind,np.newaxis] < ur_0,axis=1) ==3)
    in_interior = np.logical_not(np.isin(hindex[new_ind],interior))
    bool_in = bool_in*in_interior
    bool_in_sum = bool_in.sum()
    if bool_in_sum > halo_max:
        tagged[0] = 1
    numhalos = np.array([bool_in_sum])
    ll_list = ll_0[np.newaxis,:]
    ur_list = ur_0[np.newaxis,:]
    while tagged.sum() >0 and level <3:
        for i in range(len(ll_list)):
            ll_all = ll_list[i]
            ur_all = ur_list[i]
            # if rank ==0:
            #     print(ll_list[i],len(tagged),len(ll_list),tagged[i],numhalos[i],level)
            if tagged[i]:
                if level == 1:
                    numsegs = 2
                else:
                    numsegs = 1
                xx,yy,zz = np.meshgrid(np.linspace(ll_all[0],ur_all[0],numsegs+2),\
                np.linspace(ll_all[1],ur_all[1],numsegs+2),np.linspace(ll_all[2],ur_all[2],numsegs+2))
                ll = np.concatenate((xx[:-1,:-1,:-1,np.newaxis],yy[:-1,:-1,:-1,np.newaxis],zz[:-1,:-1,:-1,np.newaxis]),axis=3)
                ur =  np.concatenate((xx[1:,1:,1:,np.newaxis],yy[1:,1:,1:,np.newaxis],zz[1:,1:,1:,np.newaxis]),axis=3)
                diam_i = (ur_all[0]-ll_all[0])/(numsegs+1)
                ll = np.reshape(ll,(ll.shape[0]**3,3))-0.03*diam_i
                ur = np.reshape(ur,(ur.shape[0]**3,3))+0.03*diam_i
                for g in range(len(ll)):
                    bool_in = (np.sum(centers[new_ind] - radii[new_ind,np.newaxis] \
                        > ll[g],axis=1) ==3)*(np.sum(centers[new_ind] + radii[new_ind,np.newaxis] < ur[g],axis=1) ==3)
                    in_interior = np.logical_not(np.isin(hindex[new_ind],interior))
                    bool_in = bool_in*in_interior
                    if bool_in.sum() >0:
                        if bool_in.sum() >halo_max:
                            tagged = np.append(tagged,1)
                        else:
                            tagged = np.append(tagged,0)
                        numhalos = np.append(bool_in.sum(),numhalos)
                        ll_list = np.vstack((ll[g],ll_list))
                        ur_list = np.vstack((ur[g],ur_list))
            tagged[i] = 0
        level += 1
    V_all_0 = ur_list - ll_list
    radii_list = V_all_0[:,0]
    V_all =  V_all_0[:,0]*V_all_0[:,1]*V_all_0[:,2]
    Varg = ~np.isnan(radii_list)
    return ll_list[Varg],ur_list[Varg],radii_list[Varg]

def get_all_regions_1(centers,radii,hindex,overlap=True,make_numseg=False,max_rad=1):
    if len(radii) > 20:
        radii = np.maximum(np.maximum(radii,np.percentile(radii,20)),radii[radii>0].min())
    elif len(radii) > 1:
        radii = np.maximum(radii,radii[radii>0].min())
    region_number = nprocs
    total_volume = np.sum(radii**3)
    segement_volume = total_volume/region_number
    numsegs_all = np.maximum((((radii**3)/segement_volume)**(1/3)).astype(int),1)
    ll_total = np.array([])
    ur_total = np.array([])
    interior = np.array([])
    halo_id = []
    bool_not_non = hindex != None
    new_ind = np.arange(len(hindex))[bool_not_non]
    for i in range(len(radii)):
        if hindex[i]== None:
            halo_max = min(nprocs,10)
        else:
            halo_max = min(nprocs,10)
        if not make_numseg:
            numsegs = 0
        else:
            numsegs = numsegs_all[i]
        ll_0 = centers[i] - radii[i]
        ur_0 = centers[i] + radii[i]
        ll_0 -= 0.03*radii[i]
        ur_0 += 0.03*radii[i]
        h_ind = hindex[i]
        if not overlap or h_ind not in interior:
          ll_list,ur_list,radii_list = build_hierarchy(centers,radii,hindex,interior,ll_0,ur_0,halo_max=halo_max)
          for t in range(len(radii_list)):
            ll_all,ur_all = ll_list[t],ur_list[t]
            if overlap:
                bool_in = (np.sum(centers[new_ind] - radii[new_ind,np.newaxis]\
                    > ll_all,axis=1) ==3)*(np.sum(centers[new_ind] + radii[new_ind,np.newaxis] < ur_all,axis=1) ==3)
                index_in = hindex[new_ind][bool_in][-halo_max:].astype(int)
            if hindex[i] not in interior and (t == len(radii_list)-1 or len(radii_list)==1) and hindex[i] != None:
                 id_now = [hindex[i]]
            else:
                id_now = []
            index_in = np.setdiff1d(index_in,interior)
            if overlap and len(index_in) >0:
                id_now.extend(hindex[new_ind][index_in].tolist())
            if (len(id_now) >0 and hindex[i] != None and (t == len(radii_list)-1 or len(radii_list)==1)) or len(id_now) >2:
                id_now = np.unique(np.array(id_now)).tolist()
                if len(ll_total)==0:
                    ll_total = ll_all
                    ur_total = ur_all
                else:
                    ll_total = np.vstack((ll_total,ll_all))
                    ur_total = np.vstack((ur_total,ur_all))
                halo_id.append(id_now)
                interior = np.append(interior,id_now)
                interior = np.unique(interior)
                if rank==0 and hindex[i]==None:
                    print(i,id_now,t)
    volume_0 = (ur_total-ll_total)
    if volume_0.ndim >1:
        volume_list = volume_0[:,0]*volume_0[:,1]*volume_0[:,2]
    else:
        volume_list = np.array([volume_0[0]*volume_0[1]*volume_0[2]])
        ll_total = ll_total[np.newaxis,:]
        ur_total = ur_total[np.newaxis,:]
    volume_list = volume_list/volume_list[volume_list >0].min()
    volume_list =  volume_list**(1.5)
    (halo_id,ll_total,ur_total,volume_list) = comm.bcast((halo_id,ll_total,ur_total,volume_list),root=0)
    return halo_id,ll_total,ur_total,volume_list


def get_all_regions_3(halo_id,ll_total,ur_total,out_list,nprocs=10):
    volumes = ur_total-ll_total
    volumes = volumes[:,0]*volumes[:,1]*volumes[:,2]
    if len(volumes) >10:
        mean_volume = volumes[np.argsort(volumes)][-10]/20.1
    else:
        mean_volume = volumes.max()/27.1
    ind_split = np.arange(len(volumes))[volumes > 8*mean_volume]
    halo_id_f = []
    out_list_f = np.array([])
    ll_total_f = np.array([])
    ur_total_f = np.array([])
    out_list_f = np.array([])
    for ind_now in out_list:
        if ind_now in ind_split:
            numsegs = int((volumes[ind_now]/mean_volume)**(1/3))
            xx,yy,zz = np.meshgrid(np.linspace(ll_total[ind_now][0],ur_total[ind_now][0],numsegs+1),\
                np.linspace(ll_total[ind_now][1],ur_total[ind_now][1],numsegs+1),\
                np.linspace(ll_total[ind_now][2],ur_total[ind_now][2],numsegs+1))
            ll = np.concatenate((xx[:-1,:-1,:-1,np.newaxis],yy[:-1,:-1,:-1,np.newaxis],zz[:-1,:-1,:-1,np.newaxis]),axis=3)
            ur =  np.concatenate((xx[1:,1:,1:,np.newaxis],yy[1:,1:,1:,np.newaxis],zz[1:,1:,1:,np.newaxis]),axis=3)
            ll = np.reshape(ll,(ll.shape[0]**3,3))
            ur = np.reshape(ur,(ur.shape[0]**3,3))
            for t in range(len(ll)):
                if len(ll_total_f) ==0:
                    ll_total_f = ll[t,np.newaxis]
                    ur_total_f = ur[t,np.newaxis]
                else:
                    ll_total_f = np.vstack((ll_total_f,ll[t]))
                    ur_total_f = np.vstack((ur_total_f,ur[t]))
                halo_id_f.append(halo_id[ind_now])
        else:
            if len(ll_total_f) ==0:
                ll_total_f = ll_total[ind_now,np.newaxis]
                ur_total_f = ur_total[ind_now,np.newaxis]
            else:
                ll_total_f = np.vstack((ll_total_f,ll_total[ind_now]))
                ur_total_f = np.vstack((ur_total_f,ur_total[ind_now]))
            halo_id_f.append(halo_id[ind_now])
    out_list_f = np.arange(len(ll_total_f))
    return halo_id_f,ll_total_f,ur_total_f,out_list_f


def get_all_regions_2(halo_id,ll_total,ur_total,out_list,ds,codetp,volumes,nprocs=nprocs,verb=False,timestep=None):
    ranks = np.arange(nprocs)
    numhalos = 0
    for halos in halo_id:
        numhalos += len(halos)
    len_jobs = np.zeros(len(ranks))
    len_jobs_2 = np.zeros(len(ranks))
    len_regions = np.zeros(len(ranks))
    jobs = {}
    ranki = 0
    for i in ranks:
        jobs[i] = []
    if rank==0:
        for i in range(len(out_list)):
          c = out_list[i]
          V = (ur_total[c][0]-ll_total[c][0])**2
          new_rank = min(np.where(len_jobs==min(len_jobs))[0])
          jobs[new_rank].append(i)
          len_jobs[new_rank] += len(halo_id[c])
          len_jobs_2[new_rank] += len(halo_id[c])
          len_regions[new_rank] += 1
          if np.isnan(len(halo_id[c])) and rank==0:
              print(V,ur_total[c][0],ll_total[c][0],len(halo_id[c]))
              break
        for o in jobs:
           job_o = jobs[o]
           np.random.shuffle(job_o)
           jobs[o] = job_o
    jobs = comm.bcast(jobs,root=0)
    len_jobs = comm.bcast(len_jobs,root=0)
    if rank==0 and verb:
            print(len_jobs,len_jobs.sum())
            print(len_jobs_2)
            print(len_regions)
    stou = {}
    stou['rank'] = {}
    stou['halo'] = {}
    for i in ranks:
        stou['rank'][i] = np.array([])
        for c in jobs[i]:
            t = out_list[c]
            stou[t] = {}
            stou[t]['rank'] = i
            stou[t]['halo'] = halo_id[t]
            for halo_i in halo_id[t]:
                 if halo_i in stou['halo']:
                     stou['halo'][halo_i].append(t)
                 else:
                    stou['halo'][halo_i] = [t]
            stou['rank'][i] = np.append(stou['rank'][i],np.array(halo_id[t]))
    rank_list = np.array_split(ranks,max(len(ranks)/500,1))
    interval = 1
    for rank_now in rank_list:
        if rank in rank_now:
            for t in jobs[rank]:
                 i = out_list[t]
                 volume_i = volumes[i]
                 ll_vsi,ur_vsi = ll_total[i],ur_total[i]
                 width = ur_vsi-ll_vsi
                 if resave:
                     if find_stars:
                         mass0,pos0,vel0,ids0,spos0,svel0,sids0 = pickup_particles_backup(timestep,ll_vsi,ur_vsi,ds)
                     else:
                         mass0,pos0,vel0,ids0 = pickup_particles_backup(timestep,ll_vsi,ur_vsi,ds)
                 else:
                     left,right = ll_vsi,ur_vsi
                     reg = ds.r[left[0]:right[0]:interval, left[1]:right[1]:interval, left[2]:right[2]:interval]
                     if find_stars:
                         mass0,pos0,vel0,ids0,spos0,svel0,sids0 = pickup_particles(reg,codetp)
                     else:
                         mass0,pos0,vel0,ids0 = pickup_particles(reg,codetp)

                 # else:
                 #     len_vsplit = 1#max(int((volume_i**(1/3))/8),1)
                 #     if len_vsplit ==1:
                 #         left,right = ll_vsi,ur_vsi
                 #         reg = ds.r[left[0]:right[0]:interval, left[1]:right[1]:interval, left[2]:right[2]:interval]
                 #         mass0,pos0,vel0,ids0 = pickup_particles(reg,codetp)
                 #         reg = 0
                 #     else:
                 #         ll,ur = subdivide_region(ll_vsi,ur_vsi,buffer=0.01,numsegs=len_vsplit)
                 #         for vsi in range(len(ll)):
                 #            left,right = ll[vsi],ur[vsi]
                 #            #print(left,right)
                 #            center = (ll[vsi]+ur[vsi])/2
                 #            if vsi ==0:
                 #                reg = ds.r[left[0]:right[0]:interval, left[1]:right[1]:interval, left[2]:right[2]:interval]
                 #                mass0,pos0,vel0,ids0 = pickup_particles(reg,codetp)
                 #            else:
                 #                reg = ds.r[left[0]:right[0]:interval, left[1]:right[1]:interval, left[2]:right[2]:interval]
                 #                mass,pos,vel,ids = pickup_particles(reg,codetp)
                 #                bool_unique = np.logical_not(np.isin(ids,ids0))
                 #                mass0 = np.append(mass0,mass[bool_unique])
                 #                mass = 0
                 #                ids0 = np.append(ids0,ids[bool_unique])
                 #                ids = 0
                 #                vel0 = np.vstack((vel0,vel[bool_unique])).v*ytunit.m/ytunit.s
                 #                vel = 0
                 #                pos0 = np.vstack((pos0,pos[bool_unique])).v*ytunit.m
                 #                pos = 0
                 #            reg = 0
                     # if testgear:
                 # if vsi >0:
                 #     print(volume_i,len_vsplit,len(ll),len_vsplit**3)
                 #     test_bool = test_particles(ll_vsi,ur_vsi,pos0,i,ds)
                 #     if not test_bool:
                 #         print('Particles not in all quads, %s, partlen: %s' % (i,len(pos)))
                 #         # mass,pos,vel,ids = pickup_particles_backup(timestep,left,right,ds)
                 #         # test_bool = test_particles(left,right,pos,i,ds)
                 #         # if test_bool:
                 #         #     print('Missing particles resolved after loading from file: %s' % i)
                 #         # else:
                 #         #     print('Particles not in all quads after loading from file: %s' % i)
                 stou[i]['mass'] = mass0
                 stou[i]['pos'] = pos0
                 stou[i]['vel'] = vel0
                 stou[i]['ids'] = ids0
                 if find_stars:
                     stou[i]['spos'] = spos0
                     stou[i]['svel'] = svel0
                     stou[i]['sids'] = sids0
                     spos0,sids0,svel0 = [],[],[]
                 mass0,pos0,vel0,ids0 = [],[],[],[]
    jobs = comm.bcast(jobs,root=0)
    return stou

def subdivide_region(ll_total,ur_total,buffer=False,numsegs=1):
    xx,yy,zz = np.meshgrid(np.linspace(ll_total[0],ur_total[0],numsegs+1),\
        np.linspace(ll_total[1],ur_total[1],numsegs+1),\
        np.linspace(ll_total[2],ur_total[2],numsegs+1))
    ll = np.concatenate((xx[:-1,:-1,:-1,np.newaxis],yy[:-1,:-1,:-1,np.newaxis],zz[:-1,:-1,:-1,np.newaxis]),axis=3)
    ur =  np.concatenate((xx[1:,1:,1:,np.newaxis],yy[1:,1:,1:,np.newaxis],zz[1:,1:,1:,np.newaxis]),axis=3)
    ll = np.reshape(ll,(ll.shape[0]**3,3))
    ur = np.reshape(ur,(ur.shape[0]**3,3))
    if buffer and numsegs>1:
        width = (ur[0]-ll[0])*buffer
        ll -= width
        ur += width
        #print(numsegs)
    return ll,ur

def display_top(snapshot, key_type='lineno', limit=10):
    top_stats = snapshot.statistics(key_type)
    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        print("#%s: %s:%s: %.1f KiB"
              % (index, frame.filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))

def compare(snaps):
     for snapshot in snaps:
         back = snapshot.statistics("traceback")
         print("\n*** top 10 stats ***")
         for i in range(10):
            b = back[i]
            print("%s b" % b.size)
            for l in b.traceback.format():
                print(l)

def test_particles(left,right,pos,halo,ds):
    pos = pos/ds.length_unit.in_units('m')
    x,y,z = np.linspace(left[0],right[0],3),\
       np.linspace(left[1],right[1],3),np.linspace(left[2],right[2],3)
    xx,yy,zz = np.meshgrid(x,y,z)
    ll_m = np.concatenate((xx[:-1,:-1,:-1,np.newaxis],yy[:-1,:-1,:-1,np.newaxis],zz[:-1,:-1,:-1,np.newaxis]),axis=3)
    ur_m =  np.concatenate((xx[1:,1:,1:,np.newaxis],yy[1:,1:,1:,np.newaxis],zz[1:,1:,1:,np.newaxis]),axis=3)
    ll_m = np.reshape(ll_m,(ll_m.shape[0]**3,3))
    ur_m = np.reshape(ur_m,(ur_m.shape[0]**3,3))
    bool_in = True
    bool_in_l = np.array([])
    for i in range(len(ll_m)):
        bool_in_i = ((np.sum(pos >= ll_m[i],axis=1)==3)*(np.sum(pos <= ur_m[i],axis=1)==3)).sum()
        bool_in_l = np.append(bool_in_l,bool_in_i)
        bool_in *= bool_in_i >0
        # print(bool_in_l,bool_in_i)
        # if bool_in_i.sum() ==0:
        #     reg2 = ds.box(ll_m[i],ur_m[i])
        #     mass,pos,vel,ids = pickup_particles(reg2,codetp)
        #     print(len(mass))
    if not bool_in:
        print(halo,bool_in_l)
    return bool_in

def get_region(ds,center,radius,multi=False):
    if not multi:
        left = center - radius
        right = center + radius
        reg = ds.r[left[0]:right[0],left[1]:right[1],left[2]:right[2]]
    else:
        reg = {}
        numsegs = 3
        ll_all = center - radius
        ur_all = center + radius
        xx,yy,zz = np.meshgrid(np.linspace(ll_all[0],ur_all[0],numsegs),\
        np.linspace(ll_all[1],ur_all[1],numsegs),np.linspace(ll_all[2],ur_all[2],numsegs))
        ll = np.concatenate((xx[:-1,:-1,:-1,np.newaxis],yy[:-1,:-1,:-1,np.newaxis],zz[:-1,:-1,:-1,np.newaxis]),axis=3)
        ur =  np.concatenate((xx[1:,1:,1:,np.newaxis],yy[1:,1:,1:,np.newaxis],zz[1:,1:,1:,np.newaxis]),axis=3)
        ll = np.reshape(ll,(ll.shape[0]**3,3))
        ur = np.reshape(ur,(ur.shape[0]**3,3))
        for i in range(len(ll)):
            reg[i] = ds.r[ll[i,0]:ur[i,0],ll[i,1]:ur[i,1],ll[i,2]:ur[i,2]]
    return reg

def halo_ids(tree_enzo, ids_enzo):
    query = f"(tree['tree', 'mass'].to('Msun') > 1e7) & (tree['tree', 'redshift'] > {redshift}-0.001) & (tree['tree', 'redshift'] < {redshift}+0.001)"
    halos_enzo = list(tree_enzo.select_halos(query))
    tree_ids_enzo = np.array([halo['halo_id'] for halo in halos_enzo])
    idx_enzo = {}
    bad = []
    for idx in ids_enzo:
        try:
            idx_enzo[idx] = np.where(idx == tree_ids_enzo)[0][0]
        except:
            bad.append(idx)
    return halos_enzo, idx_enzo, bad

def get_com_rad(savestring,sim):
    if not os.path.exists(savestring +  "/coms.npy"):
        import pandas as pd
        import ytree
        tree_enzo = ytree.load(savestring  + "/tree_0_0_0.dat")
        ids_enzo = np.array(pd.read_csv(savestring +  "/ids_total.csv")["ENZO"])[1:].astype(int)
        halo,id,bad = halo_ids(tree_enzo, ids_enzo)
        coms = []
        rads = []
        for h in id:
            coms.append(halo[id[h]]["position"].v)
            rads.append(halo[id[h]]["Rvir"].to("unitary").v)
        if rank==0:
            np.save(savestring + '/'  + "coms.npy", np.array(coms).astype(np.float64))
            np.save(savestring + '/'  + "rads.npy", np.array(rads).astype(np.float64))
    else:
        coms = np.load(savestring + '/'  + "coms.npy", allow_pickle = True)
        rads = np.load(savestring + '/'  + "rads.npy", allow_pickle = True)
    return np.array(coms).astype(np.float64), np.array(rads).astype(np.float64)

def make_filter(ds_now, codetp):
    """
    This function loads the simulation snapshot and create a Dark Matter particle field (if necessary)
    """
    if codetp == 'ENZO':
        def darkmatter_init(pfilter, data):
            filter_darkmatter0 = np.logical_or(data["all", "particle_type"] == 1, data["all", "particle_type"] == 4)
            filter_darkmatter = np.logical_and(filter_darkmatter0,data['all', 'particle_mass'].to('Msun') > 1)
            return filter_darkmatter
        add_particle_filter("DarkMatter",function=darkmatter_init,filtered_type='all',requires=["particle_type","particle_mass"])
        ds_now.add_particle_filter("DarkMatter")
    #combine less-refined particles and refined-particles into one field for GEAR, GIZMO, AREPO, GADGET3 and GADGET4
    if codetp == 'GEAR':
        dm = ParticleUnion("DarkMatter",["PartType5","PartType2"])
        ds_now.add_particle_union(dm)
    if codetp == 'GADGET3' or codetp == 'GADGET4':
        dm = ParticleUnion("DarkMatter",["PartType5","PartType1"])
        ds_now.add_particle_union(dm)
    if codetp == 'AREPO' or codetp == 'GIZMO':
        dm = ParticleUnion("DarkMatter",["PartType2","PartType1"])
        ds_now.add_particle_union(dm)
    # if find_stars:
    #     ds_now = make_filter_star(ds_now, codetp)

    return ds_now

def ensure_dir(f):
   if rank==0:
    if not os.path.exists(f):
        os.makedirs(f)

def convert_avi_to_mp4(avi_file_path, output_name):
    os.popen("ffmpeg -y -i {input} -ac 2 -b:v 2000k -c:a aac -c:v libx264 -b:a 160k -vprofile high -bf 0 -strict experimental -f mp4 {output}.mp4".format(input = avi_file_path, output = output_name))
    return True
#
# def convert_again(num):
#     import os
#     avi_file_path = 'Particle_video-%s.avi' % num
#     output_name = 'Particle_video-%s' % num
#     os.popen("ffmpeg -y -i {input} -ac 2 -b:v 2000k -c:a aac -c:v libx264 -b:a 160k -vprofile high -bf 0 -strict experimental -f mp4 {output}.mp4".format(input = avi_file_path, output = output_name))
#
#
# convert_again(0)
# convert_again(1)
# convert_again(2)

def make_refined():
    interval,timelist = inteval_timelist(skip,fld_list)
    time0 = 0
    for times in timelist:
        len_now = len(timelist[timelist>=times])
        if (len_now%interval ==0 or times == last_timestep) and times >0:
            refined_region = None
            if not os.path.exists(savestring + '/' + 'Refined/' + 'refined_region_%s.npy' % (times)):
                    fr = Find_Refined(code, fld_list, times, savestring + '/' + 'Refined')
                    refined_region = fr.refined_region
                    # if rank==0:
                    #     #print(refined_region)
                    #     np.save(savestring + '/' + 'Refined/' + 'refined_region_%s.npy' % (times),refined_region)
                    refined_region = comm.bcast(refined_region,root=0)
            else:
                refined_region = None
                if rank ==0:
                    refined_region = np.load(savestring + '/' + 'Refined/' + 'refined_region_%s.npy' % (times),allow_pickle=True).tolist()
                refined_region = comm.bcast(refined_region,root=0)
            ll_all,ur_all = refined_region
            if rank==0:
                print('Refined region found :',ll_all,ur_all,'Timestep: ',times)

def vecs_calc(nside):
    pix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside,np.arange(pix))
    vecs = hp.ang2vec(np.array(theta),np.array(phi))
    return vecs

def equisum_partition(arr,p):
    ac = arr.cumsum()
    partsum = ac[-1]//p
    cumpartsums = np.array(range(1,p))*partsum
    inds = np.searchsorted(ac,cumpartsums)
    parts = np.split(np.arange(len(arr)),inds)
    return parts

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

def find_minmass(codetp,refined=False,numsegs_init=64):
    interval,timelist = inteval_timelist(skip,fld_list)
    mintimestep = len(timelist)
    for times in timelist:
        len_now = len(timelist[timelist>=times])
        if (len_now%interval ==0 or times == last_timestep) and times >0:
            mintimestep = min(times,mintimestep)
    ds,meter = open_ds(mintimestep,codetp)
    dd = ds.domain_dimensions[0]
    if dd <2:
        numsegs = numsegs_init
    else:
        numsegs = dd
    if not refined:
            left = ds.domain_left_edge.in_units('m').v/meter
            right = ds.domain_right_edge.in_units('m').v/meter
            center = (left+right)/2
            dx = right[0]-left[0]
            ll_all,ur_all = center-dx/2,center+dx/2
    else:
            fr = np.load(savestring + '/' + 'Refined/' + 'refined_region_%s.npy' % (mintimestep),allow_pickle=True).tolist()
            ll_all,ur_all = fr
    xx,yy,zz = np.meshgrid(np.linspace(ll_all[0],ur_all[0],numsegs),\
                np.linspace(ll_all[1],ur_all[1],numsegs),np.linspace(ll_all[2],ur_all[2],numsegs))
    ll = np.concatenate((xx[:-1,:-1,:-1,np.newaxis],yy[:-1,:-1,:-1,np.newaxis],zz[:-1,:-1,:-1,np.newaxis]),axis=3) #ll is lowerleft
    ur = np.concatenate((xx[1:,1:,1:,np.newaxis],yy[1:,1:,1:,np.newaxis],zz[1:,1:,1:,np.newaxis]),axis=3) #ur is upperright
    ll = np.reshape(ll,(ll.shape[0]*ll.shape[1]*ll.shape[2],3))
    ur = np.reshape(ur,(ur.shape[0]*ur.shape[1]*ur.shape[2],3))
    out_list = 0
    if rank ==0:
        out_list = np.random.choice(len(ll),size=min(len(ll),numsegs*50),replace=False)
    out_list = comm.bcast(out_list,root=0)
    jobs,sto = job_scheduler(out_list)
    ranks = np.arange(nprocs)
    buffer = 0
    for rank_now in ranks:
        if rank == rank_now:
            for t in jobs[rank]:
                mass,lenmass = pickup_min_dm(ll[t] - buffer,ur[t] + buffer,codetp,ds)
                sto[t] = [mass,lenmass]
    ds,meter = 0,0
    jobs = comm.bcast(jobs,root=0)
    unique_mass = np.array([])
    min_mass_num = 0
    for rank_now in ranks:
        for t in jobs[rank_now]:
                sto[t] = comm.bcast(sto[t], root=rank_now)
                if sto[t][1] > min_mass_num:
                    min_mass_num = sto[t][1]
                if sto[t][0] <1e98:
                    unique_mass = np.unique(np.append(unique_mass,np.round(sto[t][0])))
    sto = {}
    if len(unique_mass) > 1 or refined:
        if rank==0 and refined == True:
            print('This is a Zoom-in With the Following Masses:',unique_mass)
        elif rank ==0 and refined == False:
            refined = True
            print('This is a Zoom-in, finding region')
    elif rank==0:
        print('This is a not a Zoom-in Simulation')
    refined = comm.bcast(refined,root=0)
    if dd >1:
        min_mass_num = 0
    return 1*unique_mass.min(),refined,min_mass_num

def minmass_calc(code):
    num_mass = 10
    count_mass = 1
    refined = False
    while not refined and num_mass >8 and count_mass <5:
        minmass,refined,num_mass = find_minmass(code,numsegs_init=2**(5+count_mass))
        count_mass +=1
    if refined:
        make_refined()
        minmass,blank,num_mass = find_minmass(code,refined=refined,numsegs_init=2**(5))
    if rank ==0:
        print('Calculated Floor Mass:',minmass)
    return minmass,refined

def inteval_timelist(skip,fld_list):
    timelist = np.arange(len(fld_list))[::-1][::skip]
    interval = int(len(timelist)/10)
    if interval <=7:
        interval = len(timelist)-1
    return interval,timelist

def resave_particles(ranklim=3):
    save_part = savestring+'/particle_save'
    if os.path.exists(save_part+'/part_dict.npy'):
        part_dict = np.load(save_part+'/part_dict.npy',allow_pickle= True).tolist()
    else:
        part_dict = {}
    if len(part_dict) ==0:
        interval,timelist = inteval_timelist(skip,fld_list)
        ranks = np.arange(min(nprocs,ranklim))
        jobs,sto = job_scheduler(timelist,ranklim=ranklim)
        ensure_dir(save_part)
        if refined:
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
            buffer = (np.array(ur_all)-np.array(ll_all))*0.05
            ll_all,ur_all = np.array(ll_all)-buffer,np.array(ur_all)+buffer
        else:
            ds_for_resave = open_ds(0, code)
            ll_all,ur_all = np.array(ds_for_resave.domain_left_edge),np.array(ds_for_resave.domain_right_edge)
            del ds_for_resave
        jobs = comm.bcast(jobs,root=0)
        for rank_now in ranks:
            if rank == rank_now:
                for t in jobs[rank]:
                    if t not in part_dict:
                        numsegs = max(int(2+nprocs**(1/3)),3)
                        xx,yy,zz = np.meshgrid(np.linspace(ll_all[0],ur_all[0],numsegs),\
                                    np.linspace(ll_all[1],ur_all[1],numsegs),np.linspace(ll_all[2],ur_all[2],numsegs))
                        ll = np.concatenate((xx[:-1,:-1,:-1,np.newaxis],yy[:-1,:-1,:-1,np.newaxis],zz[:-1,:-1,:-1,np.newaxis]),axis=3) #ll is lowerleft
                        ur = np.concatenate((xx[1:,1:,1:,np.newaxis],yy[1:,1:,1:,np.newaxis],zz[1:,1:,1:,np.newaxis]),axis=3) #ur is upperright
                        ll = np.reshape(ll,(ll.shape[0]*ll.shape[1]*ll.shape[2],3))
                        ur = np.reshape(ur,(ur.shape[0]*ur.shape[1]*ur.shape[2],3))
                        ds,meter = open_ds(t,code)
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
                        if find_stars == True:
                            spos, svel, sids = 0,0,0
                        reg = 0
                        ds = 0
        part_dict = comm.bcast(part_dict,root=0)
        for rank_now in ranks:
            for t in jobs[rank_now]:
                    sto[t] = comm.bcast(sto[t], root=rank_now)
        jobs = comm.bcast(jobs,root=0)
        if rank==0 and len(part_dict)==0:
            np.save(save_part+'/part_dict.npy',sto)


def pickup_particles_backup(timestep,left,right,ds,stars=True,vel_ids=True):
    if os.path.exists(savestring+'/particle_save'):
        save_part = savestring+'/particle_save'
    elif os.path.exists(string+'/particle_save'):
        save_part = string+'/particle_save'
    meter = ds.length_unit.in_units('m')
    part_dict = np.load(save_part+'/part_dict.npy',allow_pickle= True).tolist()
    ind_array = np.arange(len(part_dict[timestep]))
    ll = part_dict[timestep]['ll']
    ur = part_dict[timestep]['ur']
    ind_array = np.arange(len(ll))
    bool_overlap = (np.sum( ur <= left,axis=1)==0)*(np.sum(ll >= right,axis=1)==0)
    ind_array = ind_array[bool_overlap]
    mass,pos,vel,ids = np.array([]),np.array([]),np.array([]),np.array([])
    if find_stars:
        spos,svel,sids = np.array([[]]),np.array([[]]),np.array([])
    for i in ind_array:
        part = np.load(save_part+'/part_%s_%s.npy' % (timestep,i),allow_pickle= True).tolist()
        if len(mass) ==0:
            mass = part['mass']
            pos = part['pos']
            if vel_ids:
                vel = part['vel']
                ids = part['ids']
        else:
            mass = np.append(mass,part['mass'])
            pos = np.vstack((pos,part['pos']))
            if vel_ids:
                vel = np.vstack((vel,part['vel']))
                ids = np.append(ids,part['ids'])
        if find_stars:
            if len(sids)==0:
                spos = part['spos']
                if vel_ids:
                    svel = part['svel']
                    sids = part['sids']
            else:
                spos = np.vstack((spos,part['spos']))
                if vel_ids:
                    svel = np.vstack((svel,part['svel']))
                    sids = np.append(sids,part['sids'])
    if len(mass)>0 and len(pos)>0:
        bool_in = (np.sum(pos >= left*meter,axis=1) ==3)*(np.sum(pos < right*meter,axis=1) ==3)
        if vel_ids:
            mass,pos,vel,ids = mass[bool_in],pos[bool_in],vel[bool_in],ids[bool_in]
        else:
            mass,pos = mass[bool_in],pos[bool_in]
        if find_stars and len(sids)>0:
            bool_in = (np.sum(spos >= left*meter,axis=1) ==3)*(np.sum(spos < right*meter,axis=1) ==3)
            spos,svel,sids = spos[bool_in],svel[bool_in],sids[bool_in]
    # elif len(mass)>0:
    #     print(ll,ur,left,right,bool_overlap)
    # else:
    #     print(ll,ur,left,right,bool_overlap,pos,mass)
    #print(mass)
    if vel_ids:
        if find_stars and stars:
            return mass,pos,vel,ids,spos,svel,sids
        else:
            return mass,pos,vel,ids
    else:
        if find_stars and stars:
            return mass,pos,spos
        else:
            return mass,pos

def FindCOM(mass, pos):
    if len(mass) >1:
        com = (mass*pos.T).sum(axis=1)/mass.sum()
    else:
        com = pos
    return com

def add_virial_mass(cdenl,mden,more_variables_i):
    var_names_scalar,var_names_vec = make_var_names()
    for i,cden in enumerate(cdenl):
        var_names_scalar.append('m%s' % int(np.round(cden)))
        more_variables_i.append(mden[i])
    return var_names_scalar,more_variables_i

def make_var_names():
    var_names_scalar = ['d_com_coh','d_com_coe','d_com_cohe','com_r','coe_r',\
        'coh_r','sphericity','ellip_mass_frac','hull_volume','ellipse_volume',\
        'Total_Particles','spin','bound_mass','cden_rad']
    var_names_vec = ['a','b','c','j','hull_energy_center','hull_mass_center']
    return var_names_scalar,var_names_vec

def more_var_maker(pos,mass,vel,rvir,esr,es,estep_f,hull,com2,self_lu,self_kg_sun,cden_1,ds,radmax=1e50):
    rvir_cden,cden_r,mden_r = Find_virRad(com2,pos[es<0],mass[es<0],ds,oden=[cden_1],radmax=radmax)
    #print(rvir_cden/self_lu)
    center_of_energy = FindCOM(-1*es[esr<estep_f],pos[esr<estep_f])
    center_of_mass = FindCOM(mass[esr<estep_f],pos[esr<estep_f])
    center_of_hull = pos[hull.vertices].mean(axis=0)
    meanv = np.sum(vel[esr<estep_f]*mass[esr<estep_f][:,np.newaxis],axis=0)/mass[esr<estep_f].sum()
    #j = np.sum(np.cross(mass[esr<estep_f][:,np.newaxis]*(pos[esr<estep_f]-com2),vel[esr<estep_f]-meanv),axis=0)/mass[esr<estep_f].sum()
    #print(center_of_hull)
    J = np.sum(np.cross(mass[esr<estep_f][:,np.newaxis]*(pos[esr<estep_f]-com2),vel[esr<estep_f]-meanv),axis=0)
    j = J/mass[esr<estep_f].sum()
    spin = (np.linalg.norm(J)*abs((mass[esr<estep_f]*es[esr<estep_f]).sum())**(1/2))/(6.6743e-11*mass[esr<estep_f].sum()**(5/2))
    #print(spin)
    weighted_pos = pos[esr<estep_f]
    fweights = 2*mass[esr<estep_f]*self_kg_sun/minmass
    total_particles = int(len(weighted_pos)*fweights.mean()/2) + 1
    fweights = (np.maximum(fweights,1)).astype(int)
    SA = hull.area
    sphericity = ((np.pi**(1/3))*(6*hull.volume)**(2/3))/SA
    #print(fweights)
    half_width = np.linalg.norm(pos[hull.vertices]-center_of_hull,axis=1).max()
    com_r = np.linalg.norm(pos[hull.vertices]-center_of_mass,axis=1).max()
    coe_r = np.linalg.norm(pos[hull.vertices]-center_of_energy,axis=1).max()
    #half_width = cdist(pos[hull.vertices],pos[hull.vertices]).max()/2
    weighted_pos = np.repeat((weighted_pos-center_of_energy),fweights.T,axis=0)
    if len(weighted_pos) > 70000:
        rand_int = np.random.choice(np.arange(len(weighted_pos)),size=70000,replace=False)
        weighted_pos = weighted_pos[rand_int]
    eigvals, eigvecs = np.linalg.eig(np.cov(weighted_pos.T/self_lu))
    eigvals *= (len(weighted_pos))
    a = max(eigvals)
    b = min(eigvals)
    c = eigvals[(eigvals!=a)*(eigvals!=b)][0]
    a,b,c = np.sqrt(a),np.sqrt(b),np.sqrt(c)
    dist_com_1 = np.linalg.norm(center_of_mass-center_of_energy)/rvir
    dist_com_2 = np.linalg.norm(center_of_mass-com2)/rvir
    dist_com_3 = np.linalg.norm(pos[hull.vertices].mean(axis=0)-center_of_mass)/rvir
    volume_hull = hull.volume/(self_lu**3)
    volume_half = (4/3)*np.pi*(half_width/self_lu)**3
    volume_ellpise = (4/3)*np.pi*(b/a)*(c/a)*(half_width/self_lu)**3
    #fill_sphere = volume_hull/volume_half
    fill_ellipsoid = volume_hull/volume_ellpise
    pos = pos[esr<estep_f]
    outside_pos = ((pos[:,0]-center_of_mass[0])/(half_width))**2 + ((pos[:,1]-center_of_mass[1])/((b/a)*half_width))**2 + ((pos[:,2]-center_of_mass[2])/((c/a)*half_width))**2
    #print(outside_pos)
    outside = outside_pos>1
    mass_frac = mass[esr<estep_f][outside].sum()/mass[esr<estep_f].sum()
    axis_factor = (half_width/self_lu)
    a_vec = eigvecs[eigvals == max(eigvals)][0]/np.linalg.norm(eigvecs[eigvals == max(eigvals)][0])
    a_vec *= axis_factor
    b_vec = eigvecs[eigvals == min(eigvals)][0]/np.linalg.norm(eigvecs[eigvals == min(eigvals)][0])
    b_vec *= axis_factor*b/a
    c_vec = eigvecs[(eigvals!=a)*(eigvals!=b)][0]/np.linalg.norm(eigvecs[(eigvals!=a)*(eigvals!=b)][0])
    c_vec *= axis_factor*c/a
    bound_mass = mass[esr<0].sum()*self_kg_sun
    more_variables_i = [dist_com_3,dist_com_1,dist_com_2,com_r/self_lu,coe_r/self_lu,half_width/self_lu,sphericity,1-mass_frac,\
            volume_hull,volume_ellpise,total_particles,spin,bound_mass,rvir_cden/self_lu]
    more_variables_vec = [a_vec,b_vec,c_vec,j,center_of_energy/self_lu,center_of_mass/self_lu]
    #print(more_variables_i)
    return more_variables_i,more_variables_vec

if __name__ == "__main__":


    ### Constants
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
        find_stars = False
        resave = False # recommended to turn on for particle-based codes (GEAR, GIZMO, CHANGA, GAGDET3) due to potential loading issue with yt
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
        if resave:
            resave_particles()
        Evolve_Tree(plot=False,codetp=code,skip_large=False,verbose=False,\
            from_tree=False,last_timestep=last_timestep,multitree=True,refined=refined,video=False,trackbig=False,tracktime=True)
        if organize_files and rank==0:
            tmp_folder = savestring+'/tmp_files/'
            os.system('rm -r %s' % tmp_folder)
