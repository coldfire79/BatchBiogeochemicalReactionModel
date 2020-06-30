# -*- coding: utf-8 -*-
import os
import time
import unittest
from configparser import ConfigParser

from BatchBiogeochemicalReactionModel.BatchBiogeochemicalReactionModelImpl import BatchBiogeochemicalReactionModel
from BatchBiogeochemicalReactionModel.BatchBiogeochemicalReactionModelServer import MethodContext
from BatchBiogeochemicalReactionModel.authclient import KBaseAuth as _KBaseAuth

from installed_clients.WorkspaceClient import Workspace


class BatchBiogeochemicalReactionModelTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        token = os.environ.get('KB_AUTH_TOKEN', None)
        config_file = os.environ.get('KB_DEPLOYMENT_CONFIG', None)
        cls.cfg = {}
        config = ConfigParser()
        config.read(config_file)
        for nameval in config.items('BatchBiogeochemicalReactionModel'):
            cls.cfg[nameval[0]] = nameval[1]
        # Getting username from Auth profile for token
        authServiceUrl = cls.cfg['auth-service-url']
        auth_client = _KBaseAuth(authServiceUrl)
        user_id = auth_client.get_user(token)
        # WARNING: don't call any logging methods on the context object,
        # it'll result in a NoneType error
        cls.ctx = MethodContext(None)
        cls.ctx.update({'token': token,
                        'user_id': user_id,
                        'provenance': [
                            {'service': 'BatchBiogeochemicalReactionModel',
                             'method': 'please_never_use_it_in_production',
                             'method_params': []
                             }],
                        'authenticated': 1})
        cls.wsURL = cls.cfg['workspace-url']
        cls.wsClient = Workspace(cls.wsURL)
        cls.serviceImpl = BatchBiogeochemicalReactionModel(cls.cfg)
        cls.scratch = cls.cfg['scratch']
        cls.callback_url = os.environ['SDK_CALLBACK_URL']
        suffix = int(time.time() * 1000)
        cls.wsName = "test_BatchBiogeochemicalReactionModel_" + str(suffix)
        ret = cls.wsClient.create_workspace({'workspace': cls.wsName})  # noqa

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, 'wsName'):
            cls.wsClient.delete_workspace({'workspace': cls.wsName})
            print('Test workspace was deleted')

    # NOTE: According to Python unittest naming rules test method names should start from 'test'. # noqa
    def test_your_method(self):
        # Prepare test objects in workspace if needed using
        # self.getWsClient().save_objects({'workspace': self.getWsName(),
        #                                  'objects': []})
        #
        # Run your method by
        # ret = self.getImpl().your_method(self.getContext(), parameters...)
        #
        # Check returned data with
        # self.assertEqual(ret[...], ...) or other unittest methods
        params={
            'workspace_name': self.wsName,
            "fba_model": "39537/18/1",
            "num_samples": "10",
            "model_type": "1",
            "max_growth": "0.3",
            "harvest_vol": "10",
            "cell_death": "0.01",
            "initial_doc": "0.1",
            "initial_o2": "0.25",
            "initial_biom": "0.1",
            "carrying_capacity": "inf",
            "o2_saturation": "0.25",
            "kLa": "350",
            "end_time": "60",
            "timestep": "100",
            "random_seed": "0"
        }
        ret = self.serviceImpl.run_BatchBiogeochemicalReactionModel(self.ctx, params)
