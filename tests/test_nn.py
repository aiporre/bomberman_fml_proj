from agent_code.Q_agent.CNN import ModelCoin
import torch
import agent_code.Q_agent.config as c


next_to_coin = '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,' \
               '-7,-7,-7,-7,-7,-7,-7,-7,-7,0,0,-7,11,12,-15,14,15,14,13,12,0,0,-7,11,-7,13,-7,14,-7,13,-7,0,0,' \
               '-7,11,12,13,13,13,13,13,12,0,0,-7,11,-7,12,-7,12,-7,12,-7,0,0,-7,11,11,11,11,11,11,11,11,0,0,-7,' \
               '10,-7,10,-7,10,-7,10,-7;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,' \
               '0,0,0,0,0,0,0,0,0,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,-1,0,0,-15,0,0,0,0,0,0,0,-1,0,-1,0,-1,0,-1,0' \
               ',-1,0,0,-1,0,0,0,0,0,0,0,0,0,0,-1,0,-1,0,-1,0,-1,0,-1,0,0,-1,0,0,0,0,0,0,0,0,0,0,-1,0,-1,0,-1,0,-1,0,-1'
def test_cnn():
    transform = Compose([MultiToTensor(c.coin['network']['state_height'],
                                       c.coin['network']['state_width'],
                                       num_in_channels=c.coin['network']['num_in_channels'])])
    
    rewards = torch.tensor([-2.1000,  5.0000, -3.5000, -1.8000, -1.6000, -1.2000, -1.9000,  2.4000,
        -1.7000, -1.3000, -2.0000, -1.5000, -1.4000])

    states = [
        '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,14,14,13,12,-15,10,9,8,7,6,-7,15,-7,13,-7,11,-7,9,-7,7,-7,-7,14,14,13,12,11,10,9,8,7,6,-7,13,-7,13,-7,11,-7,9,-7,7,-7,-7,12,12,12,12,11,10,9,8,7,6,-7,11,-7,11,-7,11,-7,9,-7,7,-7;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,-15,0,0,0,0,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,-1,0,0,0,0,0,0,0,0,0,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,-1,0,0,0,0,0,0,0,0,0,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1',
        '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,0,-7,11,12,13,-15,15,14,13,12,11,0,-7,11,-7,13,-7,14,-7,13,-7,11,0,-7,11,12,13,13,13,13,13,12,11,0,-7,11,-7,12,-7,12,-7,12,-7,11,0,-7,11,11,11,11,11,11,11,11,11,0,-7,10,-7,10,-7,10,-7,10,-7,10;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,-1,0,0,0,-15,0,0,0,0,0,0,-1,0,-1,0,-1,0,-1,0,-1,0,0,-1,0,0,0,0,0,0,0,0,0,0,-1,0,-1,0,-1,0,-1,0,-1,0,0,-1,0,0,0,0,0,0,0,0,0,0,-1,0,-1,0,-1,0,-1,0,-1,0',
        '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-7,-7,-7,-7,-7,-7,-7,0,0,0,0,6,5,4,3,2,-15,-7,0,0,0,0,-7,5,-7,3,-7,1,-7,0,0,0,0,6,5,4,3,2,1,-7,0,0,0,0,-7,5,-7,3,-7,1,-7,0,0,0,0,6,5,4,3,2,1,-7,0,0,0,0,-7,5,-7,3,-7,1,-7,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0,0,0,0,0,-15,-1,0,0,0,0,-1,0,-1,0,-1,0,-1,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,-1,0,-1,0,-1,0,-1,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,-1,0,-1,0,-1,0,-1,0,0,0,0',
        '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,13,12,11,10,9,-15,7,6,5,4,3,13,-7,11,-7,9,-7,7,-7,5,-7,3,13,12,11,10,9,8,7,6,5,4,3,13,-7,11,-7,9,-7,7,-7,5,-7,3,12,12,11,10,9,8,7,6,5,4,3,11,-7,11,-7,9,-7,7,-7,5,-7,3;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0,-15,0,0,0,0,0,0,-1,0,-1,0,-1,0,-1,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,-1,0,-1,0,-1,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,-1,0,-1,0,-1,0,-1,0',
        '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,11,10,9,8,7,-15,5,4,3,2,1,11,-7,9,-7,7,-7,5,-7,3,-7,1,11,10,9,8,7,6,5,4,3,2,1,11,-7,9,-7,7,-7,5,-7,3,-7,1,11,10,9,8,7,6,5,4,3,2,1,11,-7,9,-7,7,-7,5,-7,3,-7,1;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0,-15,0,0,0,0,0,0,-1,0,-1,0,-1,0,-1,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,-1,0,-1,0,-1,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,-1,0,-1,0,-1,0,-1,0',
        '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-7,-7,-7,-7,-7,-7,-7,-7,0,0,0,7,6,5,4,3,-15,1,-7,0,0,0,7,-7,5,-7,3,-7,1,-7,0,0,0,7,6,5,4,3,2,1,-7,0,0,0,7,-7,5,-7,3,-7,1,-7,0,0,0,7,6,5,4,3,2,1,-7,0,0,0,7,-7,5,-7,3,-7,1,-7,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0,0,0,0,-15,0,-1,0,0,0,0,-1,0,-1,0,-1,0,-1,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,-1,0,-1,0,-1,0,-1,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,-1,0,-1,0,-1,0,-1,0,0,0',
        '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,14,13,12,11,10,-15,8,7,6,5,4,-7,13,-7,11,-7,9,-7,7,-7,5,-7,14,13,12,11,10,9,8,7,6,5,4,-7,13,-7,11,-7,9,-7,7,-7,5,-7,12,12,12,11,10,9,8,7,6,5,4,-7,11,-7,11,-7,9,-7,7,-7,5,-7;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0,-15,0,0,0,0,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,0,0,0,0,0,0,0,0,0,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,0,0,0,0,0,0,0,0,0,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1',
        '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-7,-7,-7,-7,-7,-7,-7,-7,-7,0,0,-7,11,12,-15,14,15,14,13,12,0,0,-7,11,-7,13,-7,14,-7,13,-7,0,0,-7,11,12,13,13,13,13,13,12,0,0,-7,11,-7,12,-7,12,-7,12,-7,0,0,-7,11,11,11,11,11,11,11,11,0,0,-7,10,-7,10,-7,10,-7,10,-7;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,-1,0,0,-15,0,0,0,0,0,0,0,-1,0,-1,0,-1,0,-1,0,-1,0,0,-1,0,0,0,0,0,0,0,0,0,0,-1,0,-1,0,-1,0,-1,0,-1,0,0,-1,0,0,0,0,0,0,0,0,0,0,-1,0,-1,0,-1,0,-1,0,-1',
        '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,12,11,10,9,8,-15,6,5,4,3,2,-7,11,-7,9,-7,7,-7,5,-7,3,-7,12,11,10,9,8,7,6,5,4,3,2,-7,11,-7,9,-7,7,-7,5,-7,3,-7,12,11,10,9,8,7,6,5,4,3,2,-7,11,-7,9,-7,7,-7,5,-7,3,-7;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0,-15,0,0,0,0,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,0,0,0,0,0,0,0,0,0,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,0,0,0,0,0,0,0,0,0,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1',
        '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-7,-7,-7,-7,-7,-7,-7,-7,-7,0,0,8,7,6,5,4,-15,2,1,-7,0,0,-7,7,-7,5,-7,3,-7,1,-7,0,0,8,7,6,5,4,3,2,1,-7,0,0,-7,7,-7,5,-7,3,-7,1,-7,0,0,8,7,6,5,4,3,2,1,-7,0,0,-7,7,-7,5,-7,3,-7,1,-7,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0,0,0,-15,0,0,-1,0,0,-1,0,-1,0,-1,0,-1,0,-1,0,0,0,0,0,0,0,0,0,0,-1,0,0,-1,0,-1,0,-1,0,-1,0,-1,0,0,0,0,0,0,0,0,0,0,-1,0,0,-1,0,-1,0,-1,0,-1,0,-1,0,0',
        '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,14,14,13,12,11,-15,9,8,7,6,5,15,-7,13,-7,11,-7,9,-7,7,-7,5,14,14,13,12,11,10,9,8,7,6,5,13,-7,13,-7,11,-7,9,-7,7,-7,5,12,12,12,12,11,10,9,8,7,6,5,11,-7,11,-7,11,-7,9,-7,7,-7,5;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0,-15,0,0,0,0,0,0,-1,0,-1,0,-1,0,-1,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,-1,0,-1,0,-1,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,-1,0,-1,0,-1,0,-1,0',
        '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,10,9,8,7,6,-15,4,3,2,1,-7,-7,9,-7,7,-7,5,-7,3,-7,1,-7,10,9,8,7,6,5,4,3,2,1,-7,-7,9,-7,7,-7,5,-7,3,-7,1,-7,10,9,8,7,6,5,4,3,2,1,-7,-7,9,-7,7,-7,5,-7,3,-7,1,-7;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0,-15,0,0,0,0,-1,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,0,0,0,0,0,0,0,0,0,-1,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,0,0,0,0,0,0,0,0,0,-1,-1,0,-1,0,-1,0,-1,0,-1,0,-1',
        '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,0,9,8,7,6,5,-15,3,2,1,-7,0,9,-7,7,-7,5,-7,3,-7,1,-7,0,9,8,7,6,5,4,3,2,1,-7,0,9,-7,7,-7,5,-7,3,-7,1,-7,0,9,8,7,6,5,4,3,2,1,-7,0,9,-7,7,-7,5,-7,3,-7,1,-7,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0,0,-15,0,0,0,-1,0,0,-1,0,-1,0,-1,0,-1,0,-1,0,0,0,0,0,0,0,0,0,0,-1,0,0,-1,0,-1,0,-1,0,-1,0,-1,0,0,0,0,0,0,0,0,0,0,-1,0,0,-1,0,-1,0,-1,0,-1,0,-1,0']
    actions = torch.tensor([[1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
        [1]])
    next_states_no_terminal =  ['0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,14,14,13,12,11,-15,9,8,7,6,5,15,-7,13,-7,11,-7,9,-7,7,-7,5,14,14,13,12,11,10,9,8,7,6,5,13,-7,13,-7,11,-7,9,-7,7,-7,5,12,12,12,12,11,10,9,8,7,6,5,11,-7,11,-7,11,-7,9,-7,7,-7,5;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0,-15,0,0,0,0,0,0,-1,0,-1,0,-1,0,-1,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,-1,0,-1,0,-1,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,-1,0,-1,0,-1,0,-1,0', '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-7,-7,-7,-7,-7,-7,-7,0,0,0,0,6,5,4,3,2,-15,-7,0,0,0,0,-7,5,-7,3,-7,1,-7,0,0,0,0,6,5,4,3,2,1,-7,0,0,0,0,-7,5,-7,3,-7,1,-7,0,0,0,0,6,5,4,3,2,1,-7,0,0,0,0,-7,5,-7,3,-7,1,-7,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0,0,0,0,0,-15,-1,0,0,0,0,-1,0,-1,0,-1,0,-1,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,-1,0,-1,0,-1,0,-1,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,-1,0,-1,0,-1,0,-1,0,0,0,0', '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,12,11,10,9,8,-15,6,5,4,3,2,-7,11,-7,9,-7,7,-7,5,-7,3,-7,12,11,10,9,8,7,6,5,4,3,2,-7,11,-7,9,-7,7,-7,5,-7,3,-7,12,11,10,9,8,7,6,5,4,3,2,-7,11,-7,9,-7,7,-7,5,-7,3,-7;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0,-15,0,0,0,0,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,0,0,0,0,0,0,0,0,0,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,0,0,0,0,0,0,0,0,0,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1', '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,10,9,8,7,6,-15,4,3,2,1,-7,-7,9,-7,7,-7,5,-7,3,-7,1,-7,10,9,8,7,6,5,4,3,2,1,-7,-7,9,-7,7,-7,5,-7,3,-7,1,-7,10,9,8,7,6,5,4,3,2,1,-7,-7,9,-7,7,-7,5,-7,3,-7,1,-7;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0,-15,0,0,0,0,-1,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,0,0,0,0,0,0,0,0,0,-1,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,0,0,0,0,0,0,0,0,0,-1,-1,0,-1,0,-1,0,-1,0,-1,0,-1', '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-7,-7,-7,-7,-7,-7,-7,0,0,0,0,6,5,4,3,2,-15,-7,0,0,0,0,-7,5,-7,3,-7,1,-7,0,0,0,0,6,5,4,3,2,1,-7,0,0,0,0,-7,5,-7,3,-7,1,-7,0,0,0,0,6,5,4,3,2,1,-7,0,0,0,0,-7,5,-7,3,-7,1,-7,0,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0,0,0,0,0,-15,-1,0,0,0,0,-1,0,-1,0,-1,0,-1,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,-1,0,-1,0,-1,0,-1,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,-1,0,-1,0,-1,0,-1,0,0,0,0', '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,13,12,11,10,9,-15,7,6,5,4,3,13,-7,11,-7,9,-7,7,-7,5,-7,3,13,12,11,10,9,8,7,6,5,4,3,13,-7,11,-7,9,-7,7,-7,5,-7,3,12,12,11,10,9,8,7,6,5,4,3,11,-7,11,-7,9,-7,7,-7,5,-7,3;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0,-15,0,0,0,0,0,0,-1,0,-1,0,-1,0,-1,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,-1,0,-1,0,-1,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,-1,0,-1,0,-1,0,-1,0', '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,0,-7,11,12,13,-15,15,14,13,12,11,0,-7,11,-7,13,-7,14,-7,13,-7,11,0,-7,11,12,13,13,13,13,13,12,11,0,-7,11,-7,12,-7,12,-7,12,-7,11,0,-7,11,11,11,11,11,11,11,11,11,0,-7,10,-7,10,-7,10,-7,10,-7,10;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,-1,0,0,0,-15,0,0,0,0,0,0,-1,0,-1,0,-1,0,-1,0,-1,0,0,-1,0,0,0,0,0,0,0,0,0,0,-1,0,-1,0,-1,0,-1,0,-1,0,0,-1,0,0,0,0,0,0,0,0,0,0,-1,0,-1,0,-1,0,-1,0,-1,0', '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,11,10,9,8,7,-15,5,4,3,2,1,11,-7,9,-7,7,-7,5,-7,3,-7,1,11,10,9,8,7,6,5,4,3,2,1,11,-7,9,-7,7,-7,5,-7,3,-7,1,11,10,9,8,7,6,5,4,3,2,1,11,-7,9,-7,7,-7,5,-7,3,-7,1;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0,-15,0,0,0,0,0,0,-1,0,-1,0,-1,0,-1,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,-1,0,-1,0,-1,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,-1,0,-1,0,-1,0,-1,0', '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-7,-7,-7,-7,-7,-7,-7,-7,0,0,0,7,6,5,4,3,-15,1,-7,0,0,0,7,-7,5,-7,3,-7,1,-7,0,0,0,7,6,5,4,3,2,1,-7,0,0,0,7,-7,5,-7,3,-7,1,-7,0,0,0,7,6,5,4,3,2,1,-7,0,0,0,7,-7,5,-7,3,-7,1,-7,0,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0,0,0,0,-15,0,-1,0,0,0,0,-1,0,-1,0,-1,0,-1,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,-1,0,-1,0,-1,0,-1,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,-1,0,-1,0,-1,0,-1,0,0,0', '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,14,13,12,11,10,-15,8,7,6,5,4,-7,13,-7,11,-7,9,-7,7,-7,5,-7,14,13,12,11,10,9,8,7,6,5,4,-7,13,-7,11,-7,9,-7,7,-7,5,-7,12,12,12,11,10,9,8,7,6,5,4,-7,11,-7,11,-7,9,-7,7,-7,5,-7;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0,-15,0,0,0,0,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,0,0,0,0,0,0,0,0,0,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,0,0,0,0,0,0,0,0,0,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1', '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-7,-7,-7,-7,-7,-7,-7,-7,-7,-7,0,9,8,7,6,5,-15,3,2,1,-7,0,9,-7,7,-7,5,-7,3,-7,1,-7,0,9,8,7,6,5,4,3,2,1,-7,0,9,-7,7,-7,5,-7,3,-7,1,-7,0,9,8,7,6,5,4,3,2,1,-7,0,9,-7,7,-7,5,-7,3,-7,1,-7,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0,0,-15,0,0,0,-1,0,0,-1,0,-1,0,-1,0,-1,0,-1,0,0,0,0,0,0,0,0,0,0,-1,0,0,-1,0,-1,0,-1,0,-1,0,-1,0,0,0,0,0,0,0,0,0,0,-1,0,0,-1,0,-1,0,-1,0,-1,0,-1,0', '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-7,-7,-7,-7,-7,-7,-7,-7,-7,0,0,8,7,6,5,4,-15,2,1,-7,0,0,-7,7,-7,5,-7,3,-7,1,-7,0,0,8,7,6,5,4,3,2,1,-7,0,0,-7,7,-7,5,-7,3,-7,1,-7,0,0,8,7,6,5,4,3,2,1,-7,0,0,-7,7,-7,5,-7,3,-7,1,-7,0,0;0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0,0,0,-15,0,0,-1,0,0,-1,0,-1,0,-1,0,-1,0,-1,0,0,0,0,0,0,0,0,0,0,-1,0,0,-1,0,-1,0,-1,0,-1,0,-1,0,0,0,0,0,0,0,0,0,0,-1,0,0,-1,0,-1,0,-1,0,-1,0,-1,0,0']
    non_terminal_mask = torch.tensor([1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.uint8)
    coin_model = ModelCoin()
    q_max_values = torch.tensor([0.5264, 0.0000, 0.2727, 0.3478, 0.3038, 0.2727, 0.4543, 0.4463, 0.3793,
        0.2789, 0.4037, 0.3151, 0.2835])
    target_values = torch.tensor([-1.5788,  5.0000, -3.2300, -1.4556, -1.2992, -0.9300, -1.4502,  2.8418,
        -1.3245, -1.0239, -1.6003, -1.1880, -1.1193])

    states = transform(states)
    print('fitting values: ')
    states = states
    loss_val = 0
    for e in range(100):
        optimizer.zero_grad()
        predicted_values = network(states).gather(1, actions)
        loss = criterion(predicted_values, target_values.unsqueeze(1))
        loss.backward()
        optimizer.step()
        loss_val = loss.detach().cpu().numpy()
        # print('LOSS = ', loss_val, 'EPOCH = ', e)
    print('====> last loss:', loss_val)



    # data = ["".join(['{},'.format(i) for i in range(11*11)])[:-1]]
    # data2 = next_to_coin
    #
    # target = torch.tensor([[5.6]])
    # actions = torch.tensor([[[1]]],dtype=torch.long)
    # coin_model.fit(data,actions=actions, target_values=target)
    # prediction = coin_model.predict(data)
    # print('prediciton', prediction)


def test_suit_case():
    # test_coin_zone_gen()
    # test_episode_agent()
    # test_render_one_time()
    a =10
    test_cnn()
    # test_nogui_agent()


if __name__ == '__main__':
    test_suit_case()