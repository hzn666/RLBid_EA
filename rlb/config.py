import json

dataPath = "../data/"
projectPath = "../rlb/"
ipinyouPath = dataPath + "ipinyou/"
ipinyou_camps = ["1458", "3358", "3427", "3476"]
# ipinyou_camps = ["1458", "2259", "2261", "2821", "2997", "3358", "3386", "3427", "3476"]

ipinyou_max_market_price = 300

info_keys = ["imp_test", "cost_test", "clk_test", "imp_train", "cost_train", "clk_train", "field", "dim",
             "price_counter_train"]


# info_keys:imp_test   cost_test   clk_test    clk_train   imp_train   field   cost_train  dim  price_counter_train
def get_camp_info(camp):
    return json.load(open(ipinyouPath + camp + "/info.json", "r"))
