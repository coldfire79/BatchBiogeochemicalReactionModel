import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

class BatchSimulation(object):
    """
    Build the batch (0-d) simulation for the biogeochemical reaction models
    """
    def __init__(self, stoich_mat, num_samples=100, 
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
        super(BatchSimulation, self).__init__()
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
        
    def get_params_by_sampling(self, umax=0.3, vh=10,
                               kd=0.01, kLa=350, o2star=0.25,
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
        param['kLa'] = kLa
        param['o2star'] = o2star
        param['cc'] = cc

        # initial concentration
        param['y0'] = np.zeros(param['noc']+2)
        param['y0'][0:param['noc']] = init_doc  # OC
        param['y0'][param['noc']] = init_o2  # O2
        param['y0'][param['noc']+1] = init_biom  # Biomass
        
        return param, select_idx

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

        kLa = param['kLa']
        o2star = param['o2star']
        cc = param['cc']

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
        dydt0 = dydt = np.dot(Yaug, r)
        dydt[ny-2]=dydt0[ny-2]+kLa*(o2star-o2)
        dydt[ny-1]=dydt0[ny-1]-kd*biom
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
        kLa = param['kLa']
        o2star = param['o2star']
        kd = param['kd']

        rO2 = np.zeros(n)
        rBiom = np.zeros(n)
        for i in range(n):
            y = Y[i,:]
            dydt = odefun([], y, param)
            
            dydt0 = dydt
            o2 = y[noc]
            biom = y[-1]
            dydt0[-2]=dydt[-2]-kLa*(o2star-o2)
            dydt0[-1]=dydt[-1]+kd*biom
            
            # rO2[i] = dydt[noc]/y[-1] # specific rate
            rO2[i] = dydt0[noc]/y[-1]  # specific rate
            rBiom[i] = dydt0[-1]/y[-1]  # specific rate
        return rO2, rBiom
    
    def run(self, umax=0.3, vh=10, kd=0.01, kLa=350, o2star=0.25,
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
                                               kd=kd, kLa=kLa, o2star=o2star,
                                               cc=cc, init_doc=init_doc,
                                               init_o2=init_o2,
                                               init_biom=init_biom,
                                               indices=indices)
        param['modeltype'] = self.model_type
        
        # Solve differential equation
        sol = solve_ivp(lambda t, y: self.f(t, y, param), 
                        [tspan[0], tspan[-1]], param['y0'], t_eval=tspan)
        
        print(sol.y.shape)
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