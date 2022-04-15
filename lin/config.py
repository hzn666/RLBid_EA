import argparse


def init_parser():
    day = {
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

    parser = argparse.ArgumentParser()
    parser.add_argument('--day', default=day)

    parser.add_argument('--data', default='../data/')
    parser.add_argument('--dataset', default='ipinyou/')
    parser.add_argument('--result', default='result/')
    parser.add_argument('--campaign_id', default='1458', help='1458')
    parser.add_argument('--train_type', default='normal', help='normal or reverse')
    parser.add_argument('--budget_prop_list', default=[2, 4, 8, 16])

    return parser.parse_args()
