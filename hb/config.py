class Config:
    def __init__(self, camp_id):
        self.day = {
            '1458': [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            '2259': [19, 20, 21, 22, 23, 24, 25],
            '2261': [24, 25, 26, 27, 28],
            '2821': [21, 22, 23, 24, 25],
            '2997': [23, 24, 25, 26, 27],
            '3358': [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            '3386': [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            '3427': [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            '3476': [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        }
        self.data = '../data/'
        self.result = 'result/'
        self.dataset = 'ipinyou/'
        self.campaign_id = camp_id + '/'
        self.train_type = 'normal'
        self.budget_prop_list = [2, 4, 8, 16]

        if self.train_type == 'reverse':
            self.train_data = 'test.bid.all.hb'
            self.test_data = 'train.bid.all.hb'
        else:
            self.train_data = 'train.bid.all.hb'
            self.test_data = 'test.bid.all.hb'

        self.train_data_path = self.data + self.dataset + self.campaign_id + self.train_data + '.csv'
        self.test_data_path = self.data + self.dataset + self.campaign_id + self.test_data + '.csv'

        self.result_path = self.result + self.dataset + self.campaign_id + self.train_type + '/'
        self.train_log_path = self.result_path + 'train/'
        self.test_log_path = self.result_path + 'test/'
