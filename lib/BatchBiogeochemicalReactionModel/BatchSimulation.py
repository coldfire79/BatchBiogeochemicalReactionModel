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
        
    def get_params_by_sampling(self, umax=1, vh=0.2, indices=None):
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
        param['kd'] = 0

        # initial concentration
        param['y0'] = np.zeros(param['noc']+2)
        param['y0'][0:param['noc']] = 1  # OC
        param['y0'][param['noc']] = 0  # O2
        param['y0'][param['noc']+1] = 0.01  # Biomass
        
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
        dydt = np.zeros(len(y))

        yCs = param['yCs']
        yO2 = param['yO2']
        yBiom = param['yBiom']
        noc = param['noc']

        umax = param['umax']
        vh = param['vh']
        kd = param['kd']

        oc = y[0:noc]
        biom = y[-1]

        rkin = np.zeros(noc)
        for ioc in range(noc):
            rkin[ioc] = umax*np.exp(yCs[ioc]/vh/oc[ioc])*biom

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
        dydt = np.dot(Yaug, r)
        dydt[-1] = dydt[-1] - kd*biom
        
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

        rO2 = np.zeros(n)
        for i in range(n):
            y = Y[i,:]
            dydt = odefun([], y, param);
            rO2[i] = dydt[noc]/y[-1] # specific rate
        return rO2
    
    def run(self, umax=1, vh=0.2, end_time=10, timestep=50, interval=None,
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
        param, _ = self.get_params_by_sampling(umax=umax, vh=vh, indices=indices)
        param['modeltype'] = self.model_type
        
        # Solve differential equation
        sol = solve_ivp(lambda t, y: self.f(t, y, param), 
                        [tspan[0], tspan[-1]], param['y0'], t_eval=tspan)
        
        print(sol.y.shape)
        ro2 = np.abs(self.get_ro2(sol.y.T, param, odefun=self.f))

        if fout:
            plt.figure()
            _dict = {"t[d]":sol.t}
            for i in range(param['noc']):
                _dict[param['formula'][i]] = sol.y[i,:]
                plt.plot(sol.t, sol.y[i,:])
            plt.xlabel(r'$t$ [d]', fontsize=15)
            plt.ylabel(r'$OC$ [mM]', fontsize=15)
            plt.savefig(fout+'_OC.png')
            pd.DataFrame(_dict).to_csv(fout+'_OC.csv', index=False)

            plt.figure()
            plt.plot(sol.t, ro2)
            plt.xlabel(r'$t$ [d]', fontsize=15)
            plt.ylabel(r'$|r_{O_2}|$ [mmol/C-mol-biom/d]', fontsize=15)
            plt.savefig(fout+'_rO2.png')
            pd.DataFrame({"t[d]":sol.t, "rO2":ro2}).to_csv(fout+'_rO2.csv', index=False)

            return (fout+'_OC.png', fout+'_rO2.png', fout+'_OC.csv', fout+'_rO2.csv')
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