# Categorial Variation Database (version 2.1)
Categorial Variation Database (version 2.1) © 2003 Copyright University of Maryland. All Rights Reserved.
Licensed under the Open Software License version 1.1

## CATVAR 2.1 Release

The Release2.1 directory contains the following files:

* Item **README**			This file!
* Item **LICENSE**           Text of Open Software License version 1.1
* Item **catvar21.signed**         Main CATVAR with credit signatures
* Item **catvar2.prep.signed**     Prep-Verb CATVAR with credit signatures
* Item **CVsearch.pl**             Main access function

## Citation

If you use this resource, please cite it as ([PDF](http://aclweb.org/anthology/N03-1013)):

```
Habash, Nizar and Bonnie Dorr. A Categorial Variation Database for English. 
In Proceedings of the North American Association for Computational Linguistics (NAACL'03), 
Edmonton, Canada, 2003, pp. 17--23.
```


```
@inproceedings{Habash:2003a,
	Address = {Edmonton, Canada},
	Author = {Nizar Habash and Bonnie Dorr},
	Booktitle = {{Proceedings of the North American Association for Computational Linguistics (NAACL'03)}},
	Pages = {17--23},
	Title = {{A Categorial Variation Database for English}},
	Year = {2003}}
```

## Main CATVAR

### Sources

Source	    |	    Counts   |	Signature(Bin) | Preparer and Source
 -------------- | ------------ | ----------- | ----------- 
 wn16.words (all)    |	**90,809 words** | 1		     | Nizar Habash, UMD, from WordNet 1.6
 *wn16.adj*    |	*19,525 words* |  1 |   Nizar Habash, UMD, from WordNet 1.6
 *wn16.adv*    |	 *3,689 words* | 1 |  Nizar Habash, UMD, from WordNet 1.6
 *wn16.noun*  |	*47,300 words* | 1 |  Nizar Habash, UMD, from WordNet 1.6
 *wn16.pn*    |	*12,590 words* | 1 |  Nizar Habash, UMD, from WordNet 1.6
 *wn16.verb*  |	 *7,705 words* | 1  |  Nizar Habash, UMD, from WordNet 1.6
 brown.words 	  | **40,334** words | 10		   | Nizar Habash, UMD, from Brown Corpus
 englex.words   | **21,351** words | 100		   | Nizar Habash, UMD, from ENGLEX
 gremio.pairs   |  **5,148** pairs | 1000		 | Greg Martin, UMD, from NOMLEX 
 lcsverbs.words |	 **4,301** words | 10000		 | Nizar Habash, UMD, from LCS Verb Lexicon
 rgreen.pairs   |  **3,019** pairs | 100000	 |	Rebecca Green, UMD, from LDOCE
 englex.pairs   |  **2,387** pairs | 1000000	 |	Nizar Habash, UMD, from ENGLEX
 habash.list    |      **6** pairs | 10000000 |	Nizar Habash, UMD,(extensions)
 
 **Total data points 	167,355**

### Cluster Profile

**Cluster Count = 51972**

**Word Count = 82676 (82675 unique)**

Cluster Size | Distribution 
-------------| ------------
1    |   38,604
2    |   6,584
3    |   2,795
4    |   1,629
5    |   907
6    |   518
7    |   335
8    |   187
9    |   152
10   |   82
11   |   61
12   |   36
13   |   25
14   |   16
15   |   15
16   |   14
17   |   4
18   |   3
19   |   1
20   |   1
22   |   1
23   |   1
24   |   1


Part of Speech | Distribution
--------------- | -----------
AJ    |  20,136 
AV    |  3,784 
N     |  49,578 
V     |  9,178 

## Prep-Verb Main CATVAR

### Sources 

Source	    |	    Counts   |	Signature(Bin) | Preparer and Source
 -------------- | ------------ | ----------- | ----------- 
LCS Verb/Prep Lexicon | 	 635 pairs/242 clusters | 10010000 |	Nizar Habash, UMD

The signature used here is a combination of LCS Lexicon and Habash
since a large amount of cleanup was required after automatic
extraction.

### Profile 

**Cluster Count = 242**
**Word Count = 877 (261 unique)**

Cluster Size | Distribution 
-------------| ------------
2	| 15
3	| 114
4	| 94
5	| 2
7	| 17

Part of Speech | Distribution
--------------- | -----------
P	| 635 
V	| 242 


## File Format

The files are in this format:
```
<file> ::= <cluster>\n
<cluster> ::= <word>{#<word>}*
<word> ::= <text>_<pos>%<signature>
```
```
example: abduct_V%61#abductor_N%33#abducted_AJ%1#abduction_N%11#abducting_AJ%1
```

## Catvar Search

CVsearch.pl is the main access function to CATVAR2.1.

```
CVsearch.pl { <word>{_<pos>}? <length>?(+||-)? || <signature> } 

<word> can be a regular expression.  Allowed regular expressions include ^ (start), _ (end) . (any) * (zero+).
<length>+ means clusters of size <length> or more
<length>- means clusters of size <length> or less
```
Passing <signature> only gives detailed information about the sources
associated with the signature.  The output of a regular search will
include the signature code in addition to a two-letter code of the
source.  The porter stem is also included in the short list of sources
per word.

```
Example:
> CVsearch.pl "avoid"
-------------------------------------
CATVAR File: catvar21.signed ... 

avoid	V	<31>	(WN BC ED NX LL avoid)
avoidance	N	<11>	(WN BC NX avoid)
avoidable	AJ	<1>	(WN avoid)
-------------------------------------
unavoidable	AJ	<1>	(WN unavoid)
unavoidably	AV	<1>	(WN unavoid)
-------------------------------------
Subtotal = 2 clusters found
-------------------------------------
CATVAR File: catvar2.prep.signed ... 

away from	P	<144>	(LL UH away from)
avoid	V	<144>	(LL UH avoid)
-------------------------------------
Subtotal = 1 clusters found
Total = 3 clusters found

> CVsearch.pl 31
1 WordNet 1.6 (WN)
2 Brown Corpus (BC)
4 Englex Dictionary (ED)
8 UMD-NOMLEX pair (NX)
16 LCS Lexicon (LL)
```


------------------------------------------------------------

Categorial Variation Database (version 2.1) © 2003 Copyright University of Maryland. All Rights Reserved.
Licensed under the Open Software License version 1.1
 
