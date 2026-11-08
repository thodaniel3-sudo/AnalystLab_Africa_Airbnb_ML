# Reviews Deep Inspection

**Project:** Airbnb Listing Price Prediction
**Author:** Daniel Thomas

## 1. Duplicate review_id investigation

The assignment brief states 160 repeated review IDs exist. Below is the evidence, produced directly from Reviews.csv.

- Duplicated review_id values: **160**
- Total rows involved: **320**

### Full list of duplicated review_id values

| review_id | occurrences | same_listing_id | distinct_listing_ids |
|---|---|---|---|
| 200088015 | 2 | NO (across listings) | 2 |
| 205182329 | 2 | NO (across listings) | 2 |
| 223954194 | 2 | NO (across listings) | 2 |
| 256197483 | 2 | NO (across listings) | 2 |
| 256940062 | 2 | NO (across listings) | 2 |
| 267670224 | 2 | NO (across listings) | 2 |
| 268584240 | 2 | NO (across listings) | 2 |
| 273473836 | 2 | NO (across listings) | 2 |
| 275984980 | 2 | NO (across listings) | 2 |
| 280843268 | 2 | NO (across listings) | 2 |
| 289753825 | 2 | NO (across listings) | 2 |
| 294423663 | 2 | NO (across listings) | 2 |
| 303293968 | 2 | NO (across listings) | 2 |
| 307840556 | 2 | NO (across listings) | 2 |
| 327085784 | 2 | NO (across listings) | 2 |
| 329562820 | 2 | NO (across listings) | 2 |
| 330884408 | 2 | NO (across listings) | 2 |
| 331068452 | 2 | NO (across listings) | 2 |
| 332797584 | 2 | NO (across listings) | 2 |
| 335456934 | 2 | NO (across listings) | 2 |
| 336397888 | 2 | NO (across listings) | 2 |
| 337697354 | 2 | NO (across listings) | 2 |
| 361328474 | 2 | NO (across listings) | 2 |
| 364789949 | 2 | NO (across listings) | 2 |
| 427811589 | 2 | NO (across listings) | 2 |
| 427955757 | 2 | NO (across listings) | 2 |
| 433288519 | 2 | NO (across listings) | 2 |
| 436153856 | 2 | NO (across listings) | 2 |
| 436497479 | 2 | NO (across listings) | 2 |
| 437102525 | 2 | NO (across listings) | 2 |
| 438705999 | 2 | NO (across listings) | 2 |
| 440793959 | 2 | NO (across listings) | 2 |
| 441537536 | 2 | NO (across listings) | 2 |
| 442312418 | 2 | NO (across listings) | 2 |
| 443507550 | 2 | NO (across listings) | 2 |
| 443650128 | 2 | NO (across listings) | 2 |
| 444763297 | 2 | NO (across listings) | 2 |
| 445491105 | 2 | NO (across listings) | 2 |
| 446379965 | 2 | NO (across listings) | 2 |
| 446724786 | 2 | NO (across listings) | 2 |
| 447868466 | 2 | NO (across listings) | 2 |
| 450028189 | 2 | NO (across listings) | 2 |
| 450104937 | 2 | NO (across listings) | 2 |
| 451121741 | 2 | NO (across listings) | 2 |
| 451122560 | 2 | NO (across listings) | 2 |
| 451736404 | 2 | NO (across listings) | 2 |
| 453232853 | 2 | NO (across listings) | 2 |
| 453624892 | 2 | NO (across listings) | 2 |
| 453946082 | 2 | NO (across listings) | 2 |
| 453963542 | 2 | NO (across listings) | 2 |
| 454441541 | 2 | NO (across listings) | 2 |
| 455177386 | 2 | NO (across listings) | 2 |
| 455738773 | 2 | NO (across listings) | 2 |
| 455861702 | 2 | NO (across listings) | 2 |
| 457149359 | 2 | NO (across listings) | 2 |
| 457429673 | 2 | NO (across listings) | 2 |
| 457495274 | 2 | NO (across listings) | 2 |
| 458593168 | 2 | NO (across listings) | 2 |
| 459349011 | 2 | NO (across listings) | 2 |
| 459964316 | 2 | NO (across listings) | 2 |
| 460455014 | 2 | NO (across listings) | 2 |
| 460786978 | 2 | NO (across listings) | 2 |
| 461782390 | 2 | NO (across listings) | 2 |
| 461797282 | 2 | NO (across listings) | 2 |
| 465151567 | 2 | NO (across listings) | 2 |
| 465644565 | 2 | NO (across listings) | 2 |
| 466427820 | 2 | NO (across listings) | 2 |
| 466432164 | 2 | NO (across listings) | 2 |
| 466487277 | 2 | NO (across listings) | 2 |
| 467261216 | 2 | NO (across listings) | 2 |
| 468403795 | 2 | NO (across listings) | 2 |
| 468721740 | 2 | NO (across listings) | 2 |
| 468808489 | 2 | NO (across listings) | 2 |
| 469689272 | 2 | NO (across listings) | 2 |
| 470333123 | 2 | NO (across listings) | 2 |
| 471366529 | 2 | NO (across listings) | 2 |
| 471852641 | 2 | NO (across listings) | 2 |
| 472179462 | 2 | NO (across listings) | 2 |
| 472640607 | 2 | NO (across listings) | 2 |
| 473725187 | 2 | NO (across listings) | 2 |
| 473748555 | 2 | NO (across listings) | 2 |
| 474335317 | 2 | NO (across listings) | 2 |
| 475326042 | 2 | NO (across listings) | 2 |
| 475368998 | 2 | NO (across listings) | 2 |
| 475928925 | 2 | NO (across listings) | 2 |
| 477885122 | 2 | NO (across listings) | 2 |
| 480151900 | 2 | NO (across listings) | 2 |
| 480158533 | 2 | NO (across listings) | 2 |
| 480623194 | 2 | NO (across listings) | 2 |
| 481191869 | 2 | NO (across listings) | 2 |
| 481590289 | 2 | NO (across listings) | 2 |
| 483309396 | 2 | NO (across listings) | 2 |
| 485708074 | 2 | NO (across listings) | 2 |
| 487704386 | 2 | NO (across listings) | 2 |
| 489200252 | 2 | NO (across listings) | 2 |
| 492219824 | 2 | NO (across listings) | 2 |
| 492424555 | 2 | NO (across listings) | 2 |
| 495485733 | 2 | NO (across listings) | 2 |
| 496166849 | 2 | NO (across listings) | 2 |
| 497197201 | 2 | NO (across listings) | 2 |
| 498663038 | 2 | NO (across listings) | 2 |
| 501085508 | 2 | NO (across listings) | 2 |
| 501156070 | 2 | NO (across listings) | 2 |
| 503536011 | 2 | NO (across listings) | 2 |
| 503606250 | 2 | NO (across listings) | 2 |
| 505376843 | 2 | NO (across listings) | 2 |
| 506327652 | 2 | NO (across listings) | 2 |
| 512330950 | 2 | NO (across listings) | 2 |
| 513313851 | 2 | NO (across listings) | 2 |
| 518019555 | 2 | NO (across listings) | 2 |
| 521000228 | 2 | NO (across listings) | 2 |
| 523688520 | 2 | NO (across listings) | 2 |
| 524621277 | 2 | NO (across listings) | 2 |
| 527231325 | 2 | NO (across listings) | 2 |
| 528097900 | 2 | NO (across listings) | 2 |
| 529009867 | 2 | NO (across listings) | 2 |
| 529904103 | 2 | NO (across listings) | 2 |
| 531677923 | 2 | NO (across listings) | 2 |
| 532583777 | 2 | NO (across listings) | 2 |
| 534812290 | 2 | NO (across listings) | 2 |
| 535332380 | 2 | NO (across listings) | 2 |
| 535734528 | 2 | NO (across listings) | 2 |
| 537250663 | 2 | NO (across listings) | 2 |
| 537746399 | 2 | NO (across listings) | 2 |
| 537749435 | 2 | NO (across listings) | 2 |
| 539487586 | 2 | NO (across listings) | 2 |
| 543106401 | 2 | NO (across listings) | 2 |
| 543229744 | 2 | NO (across listings) | 2 |
| 543276514 | 2 | NO (across listings) | 2 |
| 543747815 | 2 | NO (across listings) | 2 |
| 544938786 | 2 | NO (across listings) | 2 |
| 545322016 | 2 | NO (across listings) | 2 |
| 547445510 | 2 | NO (across listings) | 2 |
| 547504877 | 2 | NO (across listings) | 2 |
| 548201216 | 2 | NO (across listings) | 2 |
| 552716644 | 2 | NO (across listings) | 2 |
| 554917096 | 2 | NO (across listings) | 2 |
| 555057458 | 2 | NO (across listings) | 2 |
| 557112442 | 2 | NO (across listings) | 2 |
| 558639962 | 2 | NO (across listings) | 2 |
| 559469909 | 2 | NO (across listings) | 2 |
| 581612711 | 2 | NO (across listings) | 2 |
| 585084535 | 2 | NO (across listings) | 2 |
| 588812335 | 2 | NO (across listings) | 2 |
| 588888008 | 2 | NO (across listings) | 2 |
| 603395800 | 2 | NO (across listings) | 2 |
| 615579679 | 2 | NO (across listings) | 2 |
| 644132654 | 2 | NO (across listings) | 2 |
| 646267384 | 2 | NO (across listings) | 2 |
| 647645352 | 2 | NO (across listings) | 2 |
| 648325360 | 2 | NO (across listings) | 2 |
| 651230246 | 2 | NO (across listings) | 2 |
| 653050066 | 2 | NO (across listings) | 2 |
| 658988460 | 2 | NO (across listings) | 2 |
| 660888419 | 2 | NO (across listings) | 2 |
| 667013656 | 2 | NO (across listings) | 2 |
| 668349646 | 2 | NO (across listings) | 2 |
| 701468704 | 2 | NO (across listings) | 2 |
| 721876697 | 2 | NO (across listings) | 2 |
| 724818676 | 2 | NO (across listings) | 2 |

### Pattern summary

- Duplicate IDs appearing **only within one listing**: **0**
- Duplicate IDs appearing **across different listings**: **160**

**Decision:** Duplicates span listings — treat `(listing_id, review_id)` as the composite key. Do not delete; use DISTINCT when counting.

## 2. Review activity over time

| Year | Reviews |
|---|---|
| 2008 | 2 |
| 2009 | 115 |
| 2010 | 1,236 |
| 2011 | 6,253 |
| 2012 | 19,922 |
| 2013 | 50,522 |
| 2014 | 122,132 |
| 2015 | 280,332 |
| 2016 | 501,754 |
| 2017 | 777,200 |
| 2018 | 1,142,496 |
| 2019 | 1,633,546 |
| 2020 | 755,324 |
| 2021 | 82,309 |

## 3. Listings with the most reviews

Top 15 listings by review count:

| listing_id | review_count |
|---|---|
| 17222007 | 891 |
| 8637229 | 828 |
| 1249964 | 796 |
| 32011332 | 762 |
| 2399029 | 754 |
| 32678719 | 753 |
| 470817 | 717 |
| 24745583 | 710 |
| 865289 | 702 |
| 162163 | 700 |
| 28323996 | 685 |
| 2709785 | 684 |
| 8228690 | 676 |
| 1385250 | 672 |
| 7944819 | 671 |