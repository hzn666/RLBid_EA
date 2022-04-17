# RLB

## 参数配置

* 修改 `config.py` 第6行指定数据集
```
ipinyou_camps = ["1458", "3358", "3427", "3476"]
```
* 修改 `rlb_main.py` 第11行指定预算
```
c0 = 1 / 2
```
## 运行

运行 `rlb_main.py`
```
*/RLBid_EA/rlb $ python rlb_main.py
```
```
setting                                                	  objective	  auction	 impression	    click	     cost	  win-rate	      CPM	     eCPC	 win-pctr
ipinyou, camp=1458, algo=rlb, N=1000, c0=0.5           	        415	   614638	     398927	      415	 20995703	    64.90%	    52.63	    50.59	 419.8293
```