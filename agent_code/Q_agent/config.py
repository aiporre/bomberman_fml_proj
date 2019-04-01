explorer = {
    'train': False,
    'dqn': True,
    'base_algorithm':{
        'greedy_prob' : 0.9,
        'lr': 0.0001,
        'discount': 0.9
    },
    'base_agent':{
        'td': False,
        'td_steps': 5
    },
    'dql': {
        'max_memory': 10000,
        'batch_size': 16,
        'epochs': 1,
        'update_step_rate': 1000,
        'model_file': 'models/final_explorer.tar',
        'track' : None
    },
    'game_abstraction': {
        'patch_size': 5
    },
    'network': {
        'new_network': True,
        'state_width': 11,
        'state_height': 11,
        'lr': 0.0001,
        'num_out_channels': 6,
        'num_in_channels' : 1,
        'method': 'cnn'
    },
    'db': None # 'storage/experience_simple_agent_as_explorer_density_75.json'
}
coin = {
    'train': False,
    'dqn': True,
    'base_algorithm':{
        'greedy_prob' : 1.0,
        'lr': 0.0001,
        'discount': 0.95,
    },
    'base_agent':{
        'td': False,
        'td_steps': 5
    },
    'dql': {
        'max_memory': 10000,
        'batch_size': 16,  # {'external': 12, 'own': 20},
        'epochs': 1,
        'update_step_rate': 100,
        'model_file': 'models/final_coin_searcher.tar',
        'track': None # 'positive_reward'
    },
    'game_abstraction': {
        'patch_size': 5
    },
    'network': {
        'new_network': False,
        'state_width': 11,
        'state_height': 11,
        'lr': 0.001,
        'num_out_channels': 4,
        'num_in_channels': 2,
        'method': 'cnn'
    },
    'db': None #'storage/experience_coin.json'
}
runner = {
    'train': False,
    'dqn': True,
    'base_algorithm':{
        'greedy_prob' : 0.9,
        'lr': 0.0001,
        'discount': 0.9
    },
    'base_agent':{
        'td': False,
        'td_steps': 5
    },
    'dql': {
        'max_memory': 10000,
        'epochs': 1,
        'update_step_rate': 500,
        'model_file': 'models/final_runner.tar',
        'batch_size': 16, #{'external':10,'own':10},
        'track' : None
    },
    'game_abstraction': {
        'patch_size': 5
    },
    'network': {
        'new_network': True,
        'state_width': 11,
        'state_height': 11,
        'lr': 0.0001,
        'num_out_channels': 5,
        'num_in_channels': 1,
        'method': 'cnn'
    },
    'db': None  # 'experience_runner.json'
}
master = {
    'train': False,
    'dqn': True,
    'base_algorithm':{
        'greedy_prob' : 0.99,
        'lr': 0.001,
        'discount': 0.99
    },
    'base_agent':{
        'td': False,
        'td_steps': 5
    },
    'dql': {
        'max_memory': 100000,
        'batch_size': 16,
        'epochs': 1,
        'update_step_rate': 1000,
        'model_file': 'models/final_master.tar',
        'track' : None
    },
    'game_abstraction': {
        'patch_size': 5
    },
    'network': {
        'new_network': True,
        'state_width': 17,
        'state_height': 17,
        'lr': 0.001,
        'num_out_channels': 4,
        'num_in_channels': 2,
        'method': 'cnn'
    },
    'model_filename': 'models/master_1'
}

simple = {
    'train': False,
    'dqn': False,
    'base_algorithm':{
        'greedy_prob' : 1.0,
        'lr': 0.001,
        'discount': 0.6
    },
    'base_agent':{
        'td': False,
        'td_steps': 5
    },
    'dql': {
        'max_memory': 100,
        'batch_size': {'external':10,'own':10},
        'epochs': 100,
        'update_step_rate': 3,
        'model_file': 'models/dqn_model_explorer-simple-agent.tar',
        'track' : 'all'
    },
    'game_abstraction': {
        'patch_size': 5
    },
    'network': {
        'new_network': True,
        'state_width': 11,
        'state_height': 11,
        'lr': 0.000001,
        'num_out_channels': 5,
        'method': 'dueling_cnn'
    },
    'db': None  # 'storage/experience_simple_agent_as_explorer.json'
}


greedy_annealing = {
    'on' : False,
    'min_prob': 0.001
}

scheduler = {
    'agent_files': {
        'EXPLORE': 'models/dqn-tunning-explorer_1',
        'SEARCH_COIN': 'models/dqn-tunning-coin_searcher_1',
        'RUN_AWAY': 'models/dqn-tunning-runner_nb1'
    }
}

stats_directory = 'stats'
stats_prefix = 'dqn-exp-run--tunning-'