# RLB

## Parameters config

* Alter line 6 in `config.py` to specify the dataset
```
ipinyou_camps = ["1458", "3358", "3427", "3476"]
```
* Alter line 11 in `rlb_main.py` to specify the budget condition
```
c0 = 1 / 2
```
## Run

Run `rlb_main.py`
```
*/RLBid_EA/rlb $ python rlb_main.py
```
```
setting                                                	  objective	  auction	 impression	    click	     cost	  win-rate	      CPM	     eCPC	 win-pctr
ipinyou, camp=1458, algo=rlb, N=1000, c0=0.5           	        415	   614638	     398927	      415	 20995703	    64.90%	    52.63	    50.59	 419.8293
```