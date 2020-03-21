# -*- coding: utf-8 -*-
#BEGIN_HEADER
import logging
import os
import uuid
import pickle
import re
import pandas as pd

from installed_clients.DataFileUtilClient import DataFileUtil
from installed_clients.KBaseReportClient import KBaseReport
from BatchBiogeochemicalReactionModel.BatchSimulation import BatchSimulation
#END_HEADER


class BatchBiogeochemicalReactionModel:
    '''
    Module Name:
    BatchBiogeochemicalReactionModel

    Module Description:
    A KBase module: BatchBiogeochemicalReactionModel
    '''

    ######## WARNING FOR GEVENT USERS ####### noqa
    # Since asynchronous IO can lead to methods - even the same method -
    # interrupting each other, you must be *very* careful when using global
    # state. A method could easily clobber the state set by another while
    # the latter method is running.
    ######################################### noqa
    VERSION = "0.0.1"
    GIT_URL = ""
    GIT_COMMIT_HASH = ""

    #BEGIN_CLASS_HEADER
    #END_CLASS_HEADER

    # config contains contents of config file in a hash or None if it couldn't
    # be found
    def __init__(self, config):
        #BEGIN_CONSTRUCTOR
        self.callback_url = os.environ['SDK_CALLBACK_URL']
        self.shared_folder = config['scratch']
        logging.basicConfig(format='%(created)s %(levelname)s: %(message)s',
                            level=logging.INFO)
        #END_CONSTRUCTOR
        pass


    def run_BatchBiogeochemicalReactionModel(self, ctx, params):
        """
        This example function accepts any number of parameters and returns results in a KBaseReport
        :param params: instance of mapping from String to unspecified object
        :returns: instance of type "ReportResults" -> structure: parameter
           "report_name" of String, parameter "report_ref" of String
        """
        # ctx is the context object
        # return variables are: output
        #BEGIN run_BatchBiogeochemicalReactionModel

        uuid_string = str(uuid.uuid4())
        objects_created = []
        output_files = []

        html_folder = os.path.join(self.shared_folder, 'html')
        os.mkdir(html_folder)

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
        batchsim = BatchSimulation(df, 
                                   num_samples=int(params['num_samples']),
                                   model_type=params['model_type'])
        print("Start", batchsim)
        (ocfig, ro2fig) = batchsim.run(end_time=float(params['end_time']),
                                       timestep=int(params['timestep']),
                                       fout=html_folder + "/" + fba_model['data']['name'])
    
        output_files.append({'path': ocfig, 'name': os.path.basename(ocfig),
            'label': 'OC profile', 'description': 'OC profile'})
        output_files.append({'path': ro2fig, 'name': os.path.basename(ro2fig),
            'label': 'r O2 profile', 'description': 'r O2 profile'})
        
        #######################################################################
        #  html
        #######################################################################
        html_str = '\
        <html>\
          <head><title>Batch Simulation Report</title>\
          </head>\
          <body><br><br>{}<br>\
          <div style="display: inline;">\
          <img src="{}" alt="OC profile" style="width: 40%;">\
          </div><div style="display: inline;">\
          <img src="{}" alt="r O2 profile" style="width: 40%;"></div>\
          </body>\
          </script>\
        </html>'
        html_str = html_str.format(batchsim,
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
            'description': 'Batch simulation Report'
        }
        report_info = report.create_extended_report({
            'objects_created': [],
            'file_links': output_files,
            'html_links': [html_dir],
            'direct_html_link_index': 0,
            'report_object_name': 'batch_sim_report',
            'workspace_name': params['workspace_name']
        })
        
        output = {
            'report_name': report_info['name'],
            'report_ref': report_info['ref'],
        }
        #END run_BatchBiogeochemicalReactionModel

        # At some point might do deeper type checking...
        if not isinstance(output, dict):
            raise ValueError('Method run_BatchBiogeochemicalReactionModel return value ' +
                             'output is not type dict as required.')
        # return the results
        return [output]
    def status(self, ctx):
        #BEGIN_STATUS
        returnVal = {'state': "OK",
                     'message': "",
                     'version': self.VERSION,
                     'git_url': self.GIT_URL,
                     'git_commit_hash': self.GIT_COMMIT_HASH}
        #END_STATUS
        return [returnVal]
