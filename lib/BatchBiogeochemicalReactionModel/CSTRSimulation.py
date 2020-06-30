import logging
import os
import uuid
import re
import time

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from BatchBiogeochemicalReactionModel.ode_constraints import constrain

from installed_clients.DataFileUtilClient import DataFileUtil
from installed_clients.KBaseReportClient import KBaseReport

class CSTRSimulation(object):
    """
    Build the CSTR simulation for the biogeochemical reaction models
    """
    def __init__(self, config):
        super(CSTRSimulation, self).__init__()
        self.callback_url = os.environ['SDK_CALLBACK_URL']
        self.shared_folder = config['scratch']
        logging.basicConfig(format='%(created)s %(levelname)s: %(message)s',
                            level=logging.INFO)

    def init(self, stoich_mat, num_samples=100, 
                 model_type="cybernetic", random_seed=41):
        '''
        init batch simulator
        
        Parameters
        ----------
            stoich_mat : DataFrame
                Stoichiometry of metabolic reactions. 
                It must include 'donor', 'acceptor', 'biom', and 'C' columns.
            num_samples : int
                Number of random samples. 
                If None, consider all reactions without sampling.
            model_type : str
                Either 'cybernetic' or 'kinetic'
            random_seed : int
                Random seed used to initialize the pseudo-random number
                generator.
        '''
        self.stoich_mat = stoich_mat
        self.num_reactions = stoich_mat.shape[0]
        self.num_samples = num_samples
        self.model_type = model_type
        
        assert model_type in ['cybernetic', 'kinetic'],\
            "'model_type' should be either 'cybernetic' or 'kinetic'."
        
        if num_samples is None:
            self.sample_idx = [i for i in range(self.num_reactions)]
        else:
            rand = np.random.RandomState(random_seed)
            self.sample_idx = rand.randint(self.num_reactions, 
                                           size=num_samples)
    
    def __repr__(self):
        return \
            "BatchSimulation(num_reactions={}, num_samples={}, model_type={})"\
            .format(
                self.num_reactions, self.num_samples, self.model_type
            )
    
    def run_cstr(self, params):
        uuid_string = str(uuid.uuid4())
        objects_created = []
        output_files = []

        html_folder = os.path.join(self.shared_folder, 'html')
        os.mkdir(html_folder)

        num_samples = int(params['num_samples'])
        max_growth = float(params['max_growth'])
        harvest_vol = float(params['harvest_vol'])

        kd = float(params['cell_death'])
        feed_doc = float(params['feed_doc'])
        feed_o2 = float(params['feed_o2'])
        dilution_rate = float(params['dilution_rate'])
        cc = float(params['carrying_capacity'])
        init_doc = float(params['initial_doc'])
        init_o2 = float(params['initial_o2'])
        init_biom = float(params['initial_biom'])

        end_time = float(params['end_time'])
        timestep = int(params['timestep'])
        random_seed = int(params['random_seed'])

        model_type = int(params['model_type'])

        model_type = "cybernetic" if model_type==1 else "kinetic"

        #######################################################################
        #  check out reactions from fba model
        #######################################################################
        print ("Input parameter", params['fba_model'])
        dfu = DataFileUtil(self.callback_url)
        fba_model = dfu.get_objects({'object_refs': [params['fba_model']]})['data'][0]
        print("FBA Model Name:", fba_model['data']['name'])
        # print(fba_model['data']['modelcompounds'][0:5])
        # print(fba_model['data']['modelreactions'][0:2])

        # collect the compound info
        cpdid2formula = dict()
        for compound in fba_model['data']['modelcompounds']:
            cpdid2formula[compound['id']] = compound['formula']

        # collect donor, acceptor, biom from reactions
        """
            donor : "~/modelcompounds/id/xcpd2_c0"
            acceptor : "~/modelcompounds/id/acceptor_c0"
            biom : "~/modelcompounds/id/biom_c0"
        """
        stoich_by_reactions = []
        for reaction in fba_model['data']['modelreactions']:
            stoich = dict()
            stoich["id"] = reaction["id"]
            for reagent in reaction["modelReactionReagents"]:
                cpdid = reagent['modelcompound_ref'].split('/id/')[1]
                if "acceptor" in cpdid:
                    stoich["acceptor"] = reagent['coefficient']
                elif "biom" in cpdid:
                    stoich["biom"] = reagent['coefficient']
                elif "xcpd" in cpdid:
                    stoich["donor"] = reagent['coefficient']
                    formula = cpdid2formula[cpdid]
                    stoich["formula"] = formula
                    num_carbon = re.search('C(\d*)', formula)
                    if num_carbon:
                        n_element = num_carbon.group(1)
                        if n_element=='': stoich["C"] = 1
                        else: stoich["C"] = int(n_element)
            stoich_by_reactions.append(stoich)

        #######################################################################
        # debug
        #######################################################################
        # for i,k in enumerate(cpdid2formula):
        #     print(k, cpdid2formula[k])
        #     if i == 5: break
        # print(stoich_by_reactions[0:2])
        # with open(self.shared_folder+'/stoich_by_reactions.pkl', 'wb') as f:
        #     pickle.dump(stoich_by_reactions, f, pickle.HIGHEST_PROTOCOL)

        #######################################################################
        #  batch simulation
        #######################################################################
        df = pd.DataFrame(stoich_by_reactions)
        self.init(df, num_samples=num_samples, model_type=model_type,
                  random_seed=random_seed)
        
        (ocfig, ro2fig, occsv, ro2csv) = self.run(umax=max_growth,
            vh=harvest_vol, kd=kd, feed_doc=feed_doc, feed_o2=feed_o2,
            dilution_rate=dilution_rate, cc=cc,
            init_doc=init_doc, init_o2=init_o2, init_biom=init_biom,
            end_time=end_time, timestep=timestep, fout=html_folder + "/" + fba_model['data']['name'])
    
        output_files.append({'path': ocfig, 'name': os.path.basename(ocfig),
            'label': 'Concentration profile', 'description': 'Concentration profile'})
        output_files.append({'path': ro2fig, 'name': os.path.basename(ro2fig),
            'label': 'r O2 profile', 'description': 'r O2 profile'})
        output_files.append({'path': occsv, 'name': os.path.basename(occsv),
            'label': 'Concentration profile (csv)', 'description': 'Concentration profile (csv)'})
        output_files.append({'path': ro2csv, 'name': os.path.basename(ro2csv),
            'label': 'r O2 profile (csv)', 'description': 'r O2 profile (csv)'})
        
        #######################################################################
        #  html
        #######################################################################
        html_str = '\
        <html>\
          <head><title>CSTR Simulation Report</title>\
          </head>\
          <body><br><br>{}<br>\
          <div style="display: inline;">\
          <img src="{}" alt="OC profile" style="width: 40%;">\
          </div><div style="display: inline;">\
          <img src="{}" alt="r O2 profile" style="width: 40%;"></div>\
          </body>\
          </script>\
        </html>'
        html_str = html_str.format("",
                                   os.path.basename(ocfig),
                                   os.path.basename(ro2fig))

        with open(os.path.join(html_folder, "index.html"), 'w') as index_file:
            index_file.write(html_str)

        #######################################################################
        #  create report
        #######################################################################
        report = KBaseReport(self.callback_url)
        html_dir = {
            'path': html_folder,
            'name': 'index.html',
            'description': 'CSTR simulation Report'
        }
        report_info = report.create_extended_report({
            'objects_created': [],
            'file_links': output_files,
            'html_links': [html_dir],
            'direct_html_link_index': 0,
            'report_object_name': 'cstr_sim_report',
            'workspace_name': params['workspace_name']
        })
        
        output = {
            'report_name': report_info['name'],
            'report_ref': report_info['ref'],
        }
        return output

    def get_params_by_sampling(self, umax=0.3, vh=10,
                               kd=0.01, feed_doc=0.05, feed_o2=0.25,
                               dilution_rate=0.1,
                               cc=np.inf,
                               init_doc=0.1,
                               init_o2=0.25,
                               init_biom=0.1,
                               indices=None):
        '''
        get parameters from the stoichiometry by random sampling
        
        Parameters
        ----------
            indices : list
                Sample index. If it's given, use this sample set without 
                sampling
            
        Returns
        -------
            param : dict
                Parameters extracted from the selected rows
            select_idx : np.array
                Selected row indices
        '''
        param = dict()
        
        select_idx = self.sample_idx
        if indices is not None:
            select_idx = indices
        
        tdf = self.stoich_mat.iloc[select_idx] \
            [['formula', 'donor', 'acceptor', 'biom']].copy()

        print(tdf)
        param['formula'] = tdf.formula.values
        param['yCs'] = tdf.donor.values
        param['yO2'] = tdf.acceptor.values
        param['yBiom'] = tdf.biom.values
        
        param['nc'] = self.stoich_mat.C.iloc[select_idx].tolist()
        param['noc'] = tdf.shape[0]
        
        param['umax'] = umax
        param['vh'] = vh
        param['kd'] = kd
        param['cc'] = cc
        param['D'] = dilution_rate

        # % substrate concentrations in the feed (required only when D is nonzero)
        param['yin'] = np.zeros(param['noc']+2)
        param['yin'][0:param['noc']] = feed_doc  # OC
        param['yin'][param['noc']] = feed_o2  # O2
        param['yin'][param['noc']+1] = 0  # Biomass

        # initial concentration
        param['y0'] = np.zeros(param['noc']+2)
        param['y0'][0:param['noc']] = feed_doc  # OC
        param['y0'][param['noc']] = feed_o2  # O2
        param['y0'][param['noc']+1] = init_biom  # Biomass
        
        return param, select_idx

    @constrain([0, None])
    def f(self, t, y, param):
        """
        Define the derivative function

        Parameters
        ----------
            t, y : numeric
                Time and y
            param : dict
                Parameters
            
        Returns
        -------
            dydt : np.array
                Derivatives
        
        """
        # print('y.shape:', y.shape)
        ny = len(y)
        dydt = np.zeros(ny)
        dydt0 = np.zeros(ny)

        yCs = param['yCs']
        yO2 = param['yO2']
        yBiom = param['yBiom']
        noc = param['noc']

        umax = param['umax']
        vh = param['vh']
        kd = param['kd']

        D = param['D']
        cc = param['cc']
        yin = param['yin']

        oc = y[0:noc]
        o2 = y[noc]
        biom = y[-1]

        rkin = np.zeros(noc)
        for ioc in range(noc):
            # rkin[ioc] = umax*np.exp(yCs[ioc]/vh/oc[ioc])*biom
            rkin[ioc] = umax*np.exp(yCs[ioc]/vh/oc[ioc])*np.exp(yO2[ioc]/vh/o2)*biom*(1-biom/cc)

        if param['modeltype'] == 'kinetic':
            r=rkin
        elif param['modeltype'] == 'cybernetic':
            # calculate the cybernetic variables
            roi = rkin
            roi = np.maximum(roi, np.zeros(noc))
            pu = np.maximum(roi, np.zeros(noc))
            sumpu = np.sum(pu)
            if sumpu>0:
                u = pu/sumpu
            else:
                u = np.zeros(noc)
            r = u*rkin

        Y = np.diag(yCs)
        Yaug = np.vstack([Y, yO2, yBiom])
        # dydt = np.dot(Yaug, r)
        # dydt[-1] = dydt[-1] - kd*biom
        dydt0 = np.dot(Yaug, r)

        dydt = dydt0+D*(yin-y)
        dydt[-1]=dydt0[-1]-kd*biom
        return dydt

    def get_ro2(self, Y, param, odefun):
        """
        get rO2 values
        
        Parameters
        ----------
            Y : list
                y values derived from ode function
            param : dict
                Parameters
            
        Returns
        -------
            dydt : np.array
                Derivatives
        
        """
        n = Y.shape[0]
        noc = param['noc']
        D = param['D']
        kd = param['kd']
        yin = param['yin']

        rO2 = np.zeros(n)
        rBiom = np.zeros(n)
        for i in range(n):
            y = Y[i,:]
            dydt = odefun([], y, param)
            
            dydt0=dydt-D*(yin-y)
            dydt0[-1]=dydt[-1]+kd*y[-1]
            
            # rO2[i] = dydt[noc]/y[-1] # specific rate
            rO2[i] = dydt0[noc]/y[-1]  # specific rate
            rBiom[i] = dydt0[-1]/y[-1]  # specific rate
        return rO2, rBiom
    
    def run(self, umax=0.3, vh=10, kd=0.01, feed_doc=0.05, feed_o2=0.25,
            dilution_rate=0.1,
            cc=np.inf, init_doc=0.1, init_o2=0.25, init_biom=0.1,
            end_time=10, timestep=50, interval=None,
            indices=None, fout=None):
        """
        Run a batch simulation
        
        Parameters
        ----------
            end_time : int or float
                End time for a simulation
            timestep : int
                Number of time steps
            interval : int or float
                Time interval for a simulation
                This must be larger than `end_time`
            indices : list
                Sample index. If it's given, use this sample set without sampling
            fout : str
                Output file path to save the figures
        Returns
        -------
            _ : list
                Output file paths
        """
        # Define time spans, initial values, and constants
        assert (end_time is not None) | (interval is not None), \
            "Either 'end_time' or 'interval' must be set up."
                
        if interval:
            assert end_time > interval, \
                "'end_time' must be larger than 'interval'."
            tspan = np.array([i for i in range(int(end_time / interval)+1)])*\
                interval            
        else:
            tspan = np.linspace(0, end_time, timestep)
        
        # get params
        param, _ = self.get_params_by_sampling(umax=umax, vh=vh,
                                               kd=kd, feed_doc=feed_doc,
                                               feed_o2=feed_o2,
                                               dilution_rate=dilution_rate,
                                               cc=cc, init_doc=init_doc,
                                               init_o2=init_o2,
                                               init_biom=init_biom,
                                               indices=indices)
        param['modeltype'] = self.model_type
        
        stime = time.time()
        # Solve differential equation
        sol = solve_ivp(lambda t, y: self.f(t, y, param), 
                        [tspan[0], tspan[-1]], param['y0'],
                        method='LSODA',  # RK45 , BDF, LSODA: work
                        t_eval=tspan)
        
        print("Time:", (time.time()-stime), "secs", sol.y.shape)
        # ro2 = np.abs(self.get_ro2(sol.y.T, param, odefun=self.f))
        ro2, rBiom = self.get_ro2(sol.y.T, param, odefun=self.f)
        ro2 = np.abs(ro2)

        if fout:
            ############# dynamic ##############
            fig = plt.figure(figsize=(10,10))

            _dict = {"t[h]":sol.t}
            noc = param['noc']

            ax = fig.add_subplot(3, 1, 1)
            for i in range(noc):
                _dict[param['formula'][i]] = sol.y[i,:]
                ax.plot(sol.t, sol.y[i,:])
            # ax.set_xlabel(r'$t$ [h]', fontsize=15)
            ax.set_ylabel(r'$DOC$ [mol/$m^3$]', fontsize=15)
            
            ax = fig.add_subplot(3, 1, 2)
            _dict["DO"] = sol.y[noc,:]
            ax.plot(sol.t, sol.y[noc,:])
            # ax.set_xlabel(r'$t$ [h]', fontsize=15)
            ax.set_ylabel(r'$DO$ [mol/$m^3$]', fontsize=15)

            ax = fig.add_subplot(3, 1, 3)
            _dict["Biomass"] = sol.y[noc+1,:]
            ax.plot(sol.t, sol.y[noc+1,:])
            ax.set_xlabel(r'$t$ [h]', fontsize=15)
            ax.set_ylabel(r'$Biomass$ [mol/$m^3$]', fontsize=15)

            plt.tight_layout()
            plt.savefig(fout+'_concentrations.png')
            pd.DataFrame(_dict).to_csv(fout+'_concentrations.csv', index=False)
            ############# dynamic ##############

            ############# rates ##############
            fig = plt.figure(figsize=(10,10))

            ax = fig.add_subplot(2, 1, 1)
            ax.plot(sol.t, ro2)
            ax.set_xlabel(r'$t$ [h]', fontsize=15)
            ax.set_ylabel(r'$|r_{O_2}|$ [mol/biom-mol/h]', fontsize=15)

            ax = fig.add_subplot(2, 1, 2)
            ax.plot(sol.t, rBiom)
            ax.set_xlabel(r'$t$ [h]', fontsize=15)
            ax.set_ylabel(r'$r_{Biom}$ [1/h]', fontsize=15)
            
            plt.tight_layout()
            plt.savefig(fout+'_rO2_biom.png')
            pd.DataFrame({"t[h]":sol.t, "rO2":ro2, "rBiom":rBiom}).to_csv(fout+'_rO2_biom.csv', index=False)
            ############# rates ##############

            return (fout+'_concentrations.png', fout+'_rO2_biom.png',
                    fout+'_concentrations.csv', fout+'_rO2_biom.csv')
        return ()

    def state_plotter(self, times, states, fig_num):
        num_states = np.shape(states)[0]
        num_cols = int(np.ceil(np.sqrt(num_states)))
        num_rows = int(np.ceil(num_states / num_cols))
        plt.figure(fig_num)
        plt.clf()
        fig, ax = plt.subplots(num_rows, num_cols, num=fig_num, clear=True,
                             squeeze=False)
        for n in range(num_states):
            row = n // num_cols
            col = n % num_cols
            ax[row][col].plot(times, states[n], 'k.:')
            ax[row][col].set(xlabel='Time',
                             ylabel='$y_{:0.0f}(t)$'.format(n),
                             title='$y_{:0.0f}(t)$ vs. Time'.format(n))
            
        for n in range(num_states, num_rows * num_cols):
            fig.delaxes(ax[n // num_cols][n % num_cols])

        fig.tight_layout()