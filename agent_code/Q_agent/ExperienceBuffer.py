from tinydb import TinyDB , Query
from tinydb.storages import MemoryStorage
import numpy as np
import json

class DataCollector(object):
    def __init__(self, file_db):
        self.db = TinyDB(storage=MemoryStorage)
        try:
            with open(file_db) as f:
                d = json.load(f)
                for e in d['_default']:
                    item = d['_default'][e]
                    self.db.insert(item)
        except Exception as e:
            print('ERROR loading the database: ', str(e))

        self.current_episode = len(self.db)-1
        self.step = -1
        self.User = Query()
        if len(self.db) == 0:
            self.create_episode()

    def create_episode(self):
        '''
        Creates a new episode
        :return:
        '''
        self.current_episode += 1
        self.db.insert({'episode': self.current_episode, 'steps': [], 'items': []})
        self.step = -1

    def track_best_state(self, item):
        done = item[-1]
        if item[2]>0 or done:
            self.track_state(item)

    def track_state(self, item):
        '''
        Process state and store in the current episode
        This must be the first
        :param state:
        :return:
        '''

        episode = self.db.get(self.User.episode == self.current_episode)
        self.step += 1
        episode['steps'].append(self.step)
        episode['items'].append(item)
        self.db.update({'episode': self.current_episode,
                        'steps': episode['steps'],
                        'items': episode['items']},
                       self.User.episode == self.current_episode)

        done = item[-1]
        if done:
            self.create_episode()

    def get_last_episode(self):
        return self.db.get(self.User.episode == self.current_episode)

    def get_mini_batch(self, batch_size):
        indices = np.random.choice(len(self.db), min(len(self.db), batch_size), replace=False).tolist()
        query = self.db.search(self.User.episode.test(lambda s: s in indices))
        items = list(map(lambda x: x['items'], query))
        return [item for sublist in items for item in sublist]

    def print_all(self):
        print(self.db.all())
