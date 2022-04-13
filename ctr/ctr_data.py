import csv
import operator
import argparse
import random
import numpy as np
import os


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def to_time_frac(hour, min, time_frac_dict):
    for key in time_frac_dict[hour].keys():
        if key[0] <= min <= key[1]:
            return str(time_frac_dict[hour][key])


def to_libsvm_encode(datapath, sample_type, time_frac_dict):
    print('转换为libsvm编码')
    oses = ["windows", "ios", "mac", "android", "linux"]
    browsers = ["chrome", "sogou", "maxthon", "safari", "firefox", "theworld", "opera", "ie"]

    f1s = ["weekday", "hour", "IP", "region", "city", "adexchange", "domain", "slotid", "slotwidth", "slotheight",
           "slotvisibility", "slotformat", "creative", "advertiser"]

    f1sp = ["useragent", "slotprice"]

    f2s = ["weekday,region"]

    def feat_trans(name, content):
        # 特征转换
        content = content.lower()
        # 操作系统和浏览器
        if name == "useragent":
            operation = "other"
            for o in oses:
                if o in content:
                    operation = o
                    break
            browser = "other"
            for b in browsers:
                if b in content:
                    browser = b
                    break
            return operation + "_" + browser
        # 地板价
        if name == "slotprice":
            price = int(content)
            if price > 100:
                return "101+"
            elif price > 50:
                return "51-100"
            elif price > 10:
                return "11-50"
            elif price > 0:
                return "1-10"
            else:
                return "0"

    def getTags(content):
        if content == '\n' or len(content) == 0:
            return ["null"]
        return content.strip().split(',')[:5]

    # initialize
    namecol = {}  # 名字列字典
    featindex = {}  # 特征索引字典
    maxindex = 0  # 最大索引

    fi = open(datapath + 'train.bid.all.csv', 'r')

    first = True

    featindex['truncate'] = maxindex
    maxindex += 1

    for line in fi:
        s = line.split(',')
        # 初始化名字列字典以及特征编码
        if first:
            first = False
            for i in range(0, len(s)):
                namecol[s[i].strip()] = i
                if i > 0:
                    featindex[str(i) + ':other'] = maxindex
                    maxindex += 1
            continue
        # 第一组特征
        for f in f1s:
            col = namecol[f]
            content = s[col]
            feat = str(col) + ':' + content
            if feat not in featindex:
                featindex[feat] = maxindex
                maxindex += 1
        # 第二组特征（需要转换）
        for f in f1sp:
            col = namecol[f]
            content = feat_trans(f, s[col])
            feat = str(col) + ':' + content
            if feat not in featindex:
                featindex[feat] = maxindex
                maxindex += 1

        # usertag标签 trick
        col = namecol["usertag"]
        tags = getTags(s[col])
        # for tag in tags:
        feat = str(col) + ':' + ''.join(tags)
        if feat not in featindex:
            featindex[feat] = maxindex
            maxindex += 1

    print('feature size: ' + str(maxindex))
    # 特征索引排序
    featvalue = sorted(featindex.items(), key=operator.itemgetter(1))

    # 写特征索引表
    fo = open(datapath + 'feat.bid.all.txt', 'w')
    fo.write(str(maxindex) + '\n')
    for fv in featvalue:
        fo.write(fv[0] + '\t' + str(fv[1]) + '\n')
    fo.close()

    # indexing train
    print('indexing ' + datapath + 'train.bid.all.csv')
    fi = open(datapath + 'train.bid.all.csv', 'r')
    fo = open(datapath + 'train.bid.all.txt', 'w')

    first = True
    for line in fi:
        if first:
            first = False
            continue
        s = line.split(',')
        time_frac = s[4][8: 12]
        # click + winning price + hour + time_frac
        fo.write(s[0] + ',' + s[23] + ',' + s[2] + ',' + to_time_frac(int(time_frac[0:2]), int(time_frac[2:4]),
                                                                      time_frac_dict) + ',' + str(s[4]))
        index = featindex['truncate']
        fo.write(',' + str(index))
        for f in f1s:  # every direct first order feature
            col = namecol[f]
            content = s[col]
            feat = str(col) + ':' + content
            if feat not in featindex:
                feat = str(col) + ':other'
            index = featindex[feat]
            fo.write(',' + str(index))

        for f in f1sp:
            col = namecol[f]
            content = feat_trans(f, s[col])
            feat = str(col) + ':' + content
            if feat not in featindex:
                feat = str(col) + ':other'
            index = featindex[feat]
            fo.write(',' + str(index))

        # usertag标签 trick
        col = namecol["usertag"]
        tags = getTags(s[col])
        # for tag in tags:
        feat = str(col) + ':' + ''.join(tags)
        if feat not in featindex:
            feat = str(col) + ':other'
        index = featindex[feat]
        fo.write(',' + str(index))
        fo.write('\n')
    fo.close()

    # indexing test
    print('indexing ' + datapath + 'test.bid.' + sample_type + '.csv')
    fi = open(datapath + 'test.bid.' + sample_type + '.csv', 'r')
    fo = open(datapath + 'test.bid.' + sample_type + '.txt', 'w')

    first = True
    for line in fi:
        if first:
            first = False
            continue
        s = line.split(',')
        time_frac = s[4][8: 12]
        fo.write(s[0] + ',' + s[23] + ',' + s[2] + ',' + to_time_frac(int(time_frac[0:2]), int(time_frac[2:4]),
                                                                      time_frac_dict) + ',' + str(s[4]))
        index = featindex['truncate']
        fo.write(',' + str(index))
        for f in f1s:  # every direct first order feature
            col = namecol[f]
            if col >= len(s):
                print('col: ' + str(col))
                print(line)
            content = s[col]
            feat = str(col) + ':' + content
            if feat not in featindex:
                feat = str(col) + ':other'
            index = featindex[feat]
            fo.write(',' + str(index))
        for f in f1sp:
            col = namecol[f]
            content = feat_trans(f, s[col])
            feat = str(col) + ':' + content
            if feat not in featindex:
                feat = str(col) + ':other'
            index = featindex[feat]
            fo.write(',' + str(index))
        col = namecol["usertag"]
        tags = getTags(s[col])
        # for tag in tags:
        feat = str(col) + ':' + ''.join(tags)
        if feat not in featindex:
            feat = str(col) + ':other'
        index = featindex[feat]
        fo.write(',' + str(index))
        fo.write('\n')
    fo.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../data/')
    parser.add_argument('--dataset_name', default='ipinyou/', help='ipinyou')
    parser.add_argument('--campaign_id', default='1458/', help='1458, 3427')
    parser.add_argument('--is_to_csv', default=True)

    setup_seed(1)

    args = parser.parse_args()
    print("数据集：" + args.campaign_id)
    data_path = args.data_path + args.dataset_name + args.campaign_id

    # 时间分段为96个段 每15分钟一个
    time_frac_dict = {}  # 时间分段字典
    count = 0  # 计数器
    for i in range(24):
        hour_frac_dict = {}  # 小时分段字典
        for item in [(0, 15), (15, 30), (30, 45), (45, 60)]:
            hour_frac_dict.setdefault(item, count)  # 设定默认值
            count += 1
        time_frac_dict.setdefault(i, hour_frac_dict)  # 设定默认值

    # 转换为csv
    if args.is_to_csv:
        print('转换为csv')
        file_name = 'train.log.txt'
        with open(data_path + 'train.bid.all.csv', 'w', newline='') as csv_file:  # newline防止每两行就空一行
            spam_writer = csv.writer(csv_file, dialect='excel')  # 读要转换的txt文件，文件每行各词间以@@@字符分隔
            with open(data_path + file_name, 'r') as filein:
                for i, line in enumerate(filein):
                    line_list = line.strip('\n').split('\t')
                    spam_writer.writerow(line_list)
        print('train-data读写完毕')

        file_name = 'test.log.txt'
        with open(data_path + 'test.bid.all.csv', 'w', newline='') as csv_file:  # newline防止每两行就空一行
            spam_writer = csv.writer(csv_file, dialect='excel')  # 读要转换的txt文件，文件每行各词间以@@@字符分隔
            with open(data_path + file_name, 'r') as filein:
                for i, line in enumerate(filein):
                    line_list = line.strip('\n').split('\t')
                    spam_writer.writerow(line_list)
        print('test-data读写完毕')

    to_libsvm_encode(data_path, 'all', time_frac_dict)
    os.remove(os.path.join(data_path, 'train.bid.all.csv'))
    os.remove(os.path.join(data_path, 'test.bid.all.csv'))
