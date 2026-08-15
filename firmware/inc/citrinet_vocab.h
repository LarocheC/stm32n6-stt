/* citrinet_vocab.h -- GENERATED, DO NOT EDIT.
 *
 * Source : tokenizer/vocab.txt (1025 lines, '<piece> <id>' per line)
 * Tool   : firmware/tools/gen_tokenizer.py
 *
 * Vocabulary of nvidia/stt_en_citrinet_256_gamma_0_25 (SentencePiece unigram,
 * 1024 pieces) plus the CTC blank at id 1024.
 *
 * The SentencePiece word-boundary marker U+2581 (present in 812 of 1025
 * pieces) has already been replaced by ASCII 0x20 here, so kPieces[] is pure
 * 7-bit ASCII and detokenisation is plain concatenation -- no UTF-8 decoding
 * and no SentencePiece runtime on the MCU.
 *
 * Flash cost:
 *   kPieces    6170 B  (NUL-separated blob)
 *   kOffset    2052 B  (uint16_t x 1026)
 *   total      8222 B  .rodata
 */

#ifndef CITRINET_VOCAB_H
#define CITRINET_VOCAB_H

#include <stdint.h>

#define CITRINET_VOCAB_SIZE     1025u   /* == stai output shape.data[1] */
#define CITRINET_BLANK_ID       1024u
#define CITRINET_UNK_ID         0u
#define CITRINET_PIECES_BYTES   6170u
#define CITRINET_MAX_PIECE_LEN  12u   /* excluding the NUL */

/* Include this header from exactly ONE translation unit (the CTC decoder);
   the tables are file-static so a second includer would duplicate 8 KB. */

static const char kPieces[CITRINET_PIECES_BYTES] =
  "<unk>\0s\0 the\0t\0 a\0 i\0'\0 and\0"   /* 0..7 */
  " to\0ed\0d\0e\0 of\0ing\0 in\0 it\0"   /* 8..15 */
  " you\0 that\0n\0er\0y\0m\0r\0 be\0"   /* 16..23 */
  " was\0 he\0 is\0 for\0 know\0re\0p\0ly\0"   /* 24..31 */
  " but\0 they\0g\0 so\0 yeah\0 have\0 we\0o\0"   /* 32..39 */
  "c\0 s\0 like\0or\0 on\0a\0i\0 re\0"   /* 40..47 */
  " with\0ll\0 do\0 not\0al\0 are\0b\0le\0"   /* 48..55 */
  "u\0ar\0 c\0es\0 this\0 as\0l\0en\0"   /* 56..63 */
  " uh\0ion\0 what\0in\0ve\0k\0 there\0 or\0"   /* 64..71 */
  " my\0 can\0 all\0ent\0f\0 his\0 me\0 just\0"   /* 72..79 */
  " at\0 de\0 f\0 don\0 um\0il\0w\0 had\0"   /* 80..87 */
  " no\0an\0 think\0it\0ri\0 she\0 one\0ra\0"   /* 88..95 */
  " go\0 if\0h\0 from\0ation\0on\0 e\0v\0"   /* 96..103 */
  " an\0 \0ce\0 t\0 would\0 right\0ch\0 were\0"   /* 104..111 */
  "ic\0 out\0 will\0ur\0 about\0 well\0ment\0 oh\0"   /* 112..119 */
  "ck\0 by\0 her\0 up\0 con\0 when\0 st\0cause\0"   /* 120..127 */
  "th\0ir\0 also\0 their\0 more\0 time\0 w\0 people\0"   /* 128..135 */
  " how\0 has\0 pa\0 mean\0li\0 g\0 some\0 get\0"   /* 136..143 */
  " d\0 really\0 ex\0ro\0ate\0 said\0 been\0ge\0"   /* 144..151 */
  "ne\0 who\0el\0 other\0 ba\0 bu\0ist\0 them\0"   /* 152..159 */
  "ut\0 now\0 work\0ol\0 co\0 po\0 un\0 your\0"   /* 160..167 */
  " him\0lo\0 good\0ver\0 could\0us\0te\0 pro\0"   /* 168..175 */
  " even\0 ra\0ies\0 see\0la\0est\0 fa\0ad\0"   /* 176..183 */
  " then\0 ro\0 mo\0at\0x\0as\0 any\0ow\0"   /* 184..191 */
  "ance\0 su\0 bo\0ul\0ive\0 two\0 p\0ng\0"   /* 192..199 */
  " la\0 say\0 o\0z\0 ma\0 over\0id\0age\0"   /* 200..207 */
  " very\0 which\0 did\0ry\0 our\0 want\0 after\0 new\0"   /* 208..215 */
  " v\0 ha\0 ah\0 where\0oo\0ity\0ke\0 ch\0"   /* 216..223 */
  " lot\0 into\0 lo\0 se\0 sp\0ure\0 kind\0 day\0"   /* 224..231 */
  " than\0 dis\0 these\0 li\0 take\0 going\0 di\0ence\0"   /* 232..239 */
  " k\0 make\0 look\0 too\0 got\0able\0 here\0 ho\0"   /* 240..247 */
  " much\0um\0 part\0 b\0ish\0is\0 man\0 mhm\0"   /* 248..255 */
  " something\0ant\0 way\0 com\0 us\0 pre\0qu\0ard\0"   /* 256..263 */
  "vi\0 sa\0 back\0 should\0ci\0per\0 le\0 only\0"   /* 264..271 */
  "ru\0ot\0 first\0les\0 little\0 school\0 hu\0 bi\0"   /* 272..279 */
  " th\0ff\0man\0 off\0ated\0 come\0 car\0im\0"   /* 280..287 */
  " down\0 m\0un\0 mi\0 okay\0 fi\0 things\0ac\0"   /* 288..295 */
  "ain\0 need\0 many\0 thing\0ta\0pt\0ight\0om\0"   /* 296..303 */
  " year\0 fl\0 sc\0 ri\0 te\0 ru\0co\0 never\0"   /* 304..311 */
  " da\0 still\0 al\0 most\0pe\0 hi\0 singapore\0ful\0"   /* 312..319 */
  " long\0 call\0ine\0 years\0ig\0huh\0 three\0ide\0"   /* 320..327 */
  " fe\0 those\0ec\0 live\0 before\0 sta\0ise\0 again\0"   /* 328..335 */
  "am\0ical\0 play\0 old\0ma\0 en\0 '\0mi\0"   /* 336..343 */
  "tic\0ction\0mo\0 comp\0 hear\0 friend\0ian\0 vi\0"   /* 344..351 */
  "mb\0und\0 gu\0 through\0em\0 sha\0 min\0ary\0"   /* 352..359 */
  " home\0 guess\0 give\0 talk\0 high\0 such\0 bl\0 help\0"   /* 360..367 */
  "ho\0ous\0po\0 pi\0 ga\0 may\0 didn\0ti\0"   /* 368..375 */
  "ia\0no\0 app\0 every\0pp\0ni\0 ta\0 own\0"   /* 376..383 */
  " life\0 made\0 r\0 mar\0ath\0 y\0 place\0ally\0"   /* 384..391 */
  " alway\0 pe\0 start\0 name\0 same\0 let\0ph\0 last\0"   /* 392..399 */
  " actually\0 ne\0 both\0iv\0 another\0 pri\0 yes\0 great\0"   /* 400..407 */
  " person\0 boy\0 thought\0 watch\0ward\0 money\0 went\0 show\0"   /* 408..415 */
  " feel\0 h\0 different\0 used\0 hand\0 gra\0 tra\0 big\0"   /* 416..423 */
  " n\0 real\0 wa\0 why\0 four\0 find\0for\0 five\0"   /* 424..431 */
  " being\0ud\0ze\0 read\0lic\0da\0 family\0 che\0"   /* 432..439 */
  " per\0 end\0 change\0ness\0 imp\0 put\0ca\0 under\0"   /* 440..447 */
  " came\0ick\0ag\0 seem\0 win\0 ten\0 jo\0 tri\0"   /* 448..455 */
  " happen\0 pu\0 around\0 plan\0 sure\0 house\0 far\0j\0"   /* 456..463 */
  " sh\0 while\0 cha\0 anything\0 war\0nna\0 use\0 love\0"   /* 464..471 */
  " world\0 stuff\0 better\0ster\0 ca\0 does\0 probably\0 hard\0"   /* 472..479 */
  "gg\0ctor\0 ja\0 must\0 doing\0ha\0 found\0 care\0"   /* 480..487 */
  " du\0row\0 state\0 point\0min\0ach\0ative\0 pretty\0"   /* 488..495 */
  " interest\0our\0ever\0led\0 ju\0 week\0 tell\0 cre\0"   /* 496..503 */
  "ft\0 might\0 ki\0 cl\0 inter\0 six\0ton\0ak\0"   /* 504..511 */
  "ite\0ling\0 acc\0land\0 business\0 ph\0 away\0 though\0"   /* 512..519 */
  "lu\0 each\0 pay\0 keep\0 everything\0fi\0av\0 children\0"   /* 520..527 */
  " public\0 pass\0com\0ual\0 em\0 ti\0 maybe\0 young\0"   /* 528..535 */
  "ough\0 u\0ities\0 few\0 month\0 ste\0 move\0 close\0"   /* 536..543 */
  " sea\0son\0 pl\0 turn\0 men\0 add\0ious\0ible\0"   /* 544..551 */
  "qui\0 am\0nch\0 grow\0 miss\0 cor\0 fun\0 done\0"   /* 552..559 */
  " government\0 ever\0 act\0 set\0 kids\0 na\0 eight\0 exact\0"   /* 560..567 */
  "way\0 open\0 getting\0 next\0ab\0 problem\0 er\0 cri\0"   /* 568..575 */
  " job\0 produc\0 cat\0 night\0 book\0 learn\0line\0 small\0"   /* 576..583 */
  "ating\0 without\0ial\0 course\0 country\0 area\0 during\0 once\0"   /* 584..591 */
  " best\0 bad\0 mister\0 trans\0 nothing\0 believe\0 head\0 seven\0"   /* 592..599 */
  " whole\0 face\0 water\0 service\0 cap\0 bro\0 later\0amp\0"   /* 600..607 */
  " took\0 va\0 hour\0 true\0 mu\0 since\0 nine\0op\0"   /* 608..615 */
  " large\0 city\0 doesn\0 having\0 sub\0 art\0ship\0 continue\0"   /* 616..623 */
  " market\0 seen\0 game\0 sit\0 case\0 law\0 lu\0 certain\0"   /* 624..631 */
  " je\0 tru\0 else\0 qua\0 enough\0 second\0 hope\0 between\0"   /* 632..639 */
  " fact\0 saw\0 food\0 left\0 company\0 stand\0 able\0 teach\0"   /* 640..647 */
  " expect\0 sometimes\0 yet\0 follow\0 main\0 told\0 girl\0 system\0"   /* 648..655 */
  " bri\0 walk\0 bra\0 near\0 word\0bb\0 group\0 less\0"   /* 656..663 */
  " number\0 consider\0 remain\0 develop\0 train\0 town\0 lead\0 free\0"   /* 664..671 */
  " idea\0 together\0 important\0 gen\0 agree\0 quite\0 nice\0 try\0"   /* 672..679 */
  " understand\0 reason\0 support\0 trying\0 low\0 stop\0 sort\0 hundred\0"   /* 680..687 */
  "ash\0 become\0 eat\0 today\0ready\0 remember\0 issue\0 side\0"   /* 688..695 */
  "bi\0 line\0wi\0 matter\0 question\0 asked\0 dollars\0nder\0"   /* 696..703 */
  " twenty\0 rest\0 upon\0 student\0 count\0 father\0 half\0 office\0"   /* 704..711 */
  " mother\0 won\0 class\0 almos\0 def\0 wait\0 cause\0bility\0"   /* 712..719 */
  " sound\0 least\0 either\0 sign\0 fire\0 light\0 several\0 whatever\0"   /* 720..727 */
  " himself\0 computer\0 room\0 report\0 america\0 minute\0 wh\0 general\0"   /* 728..735 */
  " price\0 child\0 bre\0 definitely\0 parents\0 suppose\0 direct\0 operat\0"   /* 736..743 */
  " among\0 couple\0 cost\0light\0 often\0 arm\0 appear\0 wow\0"   /* 744..751 */
  " term\0 return\0 speak\0 usually\0 meet\0 ski\0 power\0 elect\0"   /* 752..759 */
  " south\0 team\0 local\0 north\0 current\0 wonder\0 include\0 strong\0"   /* 760..767 */
  " shi\0 making\0 full\0 leave\0 everyone\0 somebody\0 provide\0 movie\0"   /* 768..775 */
  " member\0 experience\0 woman\0 possib\0 order\0rself\0 york\0 someone\0"   /* 776..783 */
  " tax\0 short\0 umhu\0 sport\0 companies\0 health\0 bank\0 until\0"   /* 784..791 */
  " answer\0 everybody\0 police\0 especial\0where\0 major\0 travel\0 knew\0"   /* 792..799 */
  " program\0 land\0 employ\0 type\0 along\0 college\0 manag\0 music\0"   /* 800..807 */
  " clear\0 safe\0 myself\0 education\0 result\0 bring\0 court\0 enter\0"   /* 808..815 */
  " enjoy\0 hold\0 involv\0 mode\0 wheth\0 husband\0came\0 million\0"   /* 816..823 */
  " black\0 difficult\0 social\0 morning\0ency\0 level\0 felt\0 married\0"   /* 824..831 */
  " taking\0 require\0 living\0 early\0 wife\0 white\0 present\0 visit\0"   /* 832..839 */
  " effect\0 serve\0 affect\0 further\0 brother\0 simp\0 shop\0 period\0"   /* 840..847 */
  " daughter\0 national\0 break\0 quick\0 common\0 view\0 situation\0 outside\0"   /* 848..855 */
  " east\0 community\0 rather\0 thousand\0 future\0 concern\0 success\0 nu\0"   /* 856..863 */
  " grand\0 complete\0 spend\0 began\0 topic\0 example\0 phone\0 increase\0"   /* 864..871 */
  " street\0 sense\0 final\0 locat\0 perform\0 charge\0 record\0 instead\0"   /* 872..879 */
  " women\0 information\0 countries\0 born\0lthough\0 acros\0 funny\0 security\0"   /* 880..887 */
  " cold\0 smoke\0 depend\0 treat\0 process\0 condition\0 themselves\0 recent\0"   /* 888..895 */
  " ground\0 behind\0ified\0 custom\0 gold\0 design\0 thirty\0 third\0"   /* 896..903 */
  " drink\0 strange\0 slow\0 particular\0 attack\0 project\0 hello\0 special\0"   /* 904..911 */
  " listen\0 wrong\0 space\0 story\0 improve\0 english\0 happy\0 value\0"   /* 912..919 */
  " voice\0 spoke\0 account\0 brought\0 green\0 private\0 control\0 media\0"   /* 920..927 */
  " author\0 figure\0 china\0 university\0 easy\0 language\0 foreign\0 includ\0"   /* 928..935 */
  " please\0 easi\0 according\0 decided\0 dream\0 sudden\0 society\0 subject\0"   /* 936..943 */
  "place\0 animal\0 horse\0 cook\0 serious\0 draw\0 opportunit\0 front\0"   /* 944..951 */
  " church\0 president\0 addition\0 fifty\0 worth\0 financial\0 hospital\0 difference\0"   /* 952..959 */
  " collect\0 study\0 individual\0 effort\0 laugh\0 industry\0 stock\0 position\0"   /* 960..967 */
  " basicall\0 similar\0 regular\0 connect\0 research\0 accept\0 restaurant\0 protect\0"   /* 968..975 */
  " amount\0 write\0 drop\0 john\0 history\0 threat\0 middle\0 political\0"   /* 976..983 */
  " cross\0 sleep\0 original\0 popular\0 immediate\0 engine\0 kept\0 drug\0"   /* 984..991 */
  " decision\0 economy\0 measure\0 california\0 video\0 challenge\0 surprise\0 entire\0"   /* 992..999 */
  " exercise\0 itself\0 available\0 benefit\0 character\0 economic\0 patient\0 despite\0"   /* 1000..1007 */
  " feature\0 absolute\0 picture\0 club\0 football\0 focus\0 discuss\0 alchemist\0"   /* 1008..1015 */
  " respect\0 perhaps\0 technology\0 natural\0 summer\0 observ\0 express\0q\0"   /* 1016..1023 */
  "<blk>\0"   /* 1024..1024 */
;

static const uint16_t kOffset[CITRINET_VOCAB_SIZE + 1] = {
      0,     6,     8,    13,    15,    18,    21,    23,    28,    32,    35,    37,
     39,    43,    47,    51,    55,    60,    66,    68,    71,    73,    75,    77,
     81,    86,    90,    94,    99,   105,   108,   110,   113,   118,   124,   126,
    130,   136,   142,   146,   148,   150,   153,   159,   162,   166,   168,   170,
    174,   180,   183,   187,   192,   195,   200,   202,   205,   207,   210,   213,
    216,   222,   226,   228,   231,   235,   239,   245,   248,   251,   253,   260,
    264,   268,   273,   278,   282,   284,   289,   293,   299,   303,   307,   310,
    315,   319,   322,   324,   329,   333,   336,   343,   346,   349,   354,   359,
    362,   366,   370,   372,   378,   384,   387,   390,   392,   396,   398,   401,
    404,   411,   418,   421,   427,   430,   435,   441,   444,   451,   457,   462,
    466,   469,   473,   478,   482,   487,   493,   497,   503,   506,   509,   515,
    522,   528,   534,   537,   545,   550,   555,   559,   565,   568,   571,   577,
    582,   585,   593,   597,   600,   604,   610,   616,   619,   622,   627,   630,
    637,   641,   645,   649,   655,   658,   663,   669,   672,   676,   680,   684,
    690,   695,   698,   704,   708,   715,   718,   721,   726,   732,   736,   740,
    745,   748,   752,   756,   759,   765,   769,   773,   776,   778,   781,   786,
    789,   794,   798,   802,   805,   809,   814,   817,   820,   824,   829,   832,
    834,   838,   844,   847,   851,   857,   864,   869,   872,   877,   883,   890,
    895,   898,   902,   906,   913,   916,   920,   923,   927,   932,   938,   942,
    946,   950,   954,   960,   965,   971,   976,   983,   987,   993,  1000,  1004,
   1009,  1012,  1018,  1024,  1029,  1034,  1039,  1045,  1049,  1055,  1058,  1064,
   1067,  1071,  1074,  1079,  1084,  1095,  1099,  1104,  1109,  1113,  1118,  1121,
   1125,  1128,  1132,  1138,  1146,  1149,  1153,  1157,  1163,  1166,  1169,  1176,
   1180,  1188,  1196,  1200,  1204,  1208,  1211,  1215,  1220,  1225,  1231,  1236,
   1239,  1245,  1248,  1251,  1255,  1261,  1265,  1273,  1276,  1280,  1286,  1292,
   1299,  1302,  1305,  1310,  1313,  1319,  1323,  1327,  1331,  1335,  1339,  1342,
   1349,  1353,  1360,  1364,  1370,  1373,  1377,  1388,  1392,  1398,  1404,  1408,
   1415,  1418,  1422,  1429,  1433,  1437,  1444,  1447,  1453,  1461,  1466,  1470,
   1477,  1480,  1485,  1491,  1496,  1499,  1503,  1506,  1509,  1513,  1519,  1522,
   1528,  1534,  1542,  1546,  1550,  1553,  1557,  1561,  1570,  1573,  1578,  1583,
   1587,  1593,  1600,  1606,  1612,  1618,  1624,  1628,  1634,  1637,  1641,  1644,
   1648,  1652,  1657,  1663,  1666,  1669,  1672,  1677,  1684,  1687,  1690,  1694,
   1699,  1705,  1711,  1714,  1719,  1723,  1726,  1733,  1738,  1745,  1749,  1756,
   1762,  1768,  1773,  1776,  1782,  1792,  1796,  1802,  1805,  1814,  1819,  1824,
   1831,  1839,  1844,  1853,  1860,  1865,  1872,  1878,  1884,  1890,  1893,  1904,
   1910,  1916,  1921,  1926,  1931,  1934,  1940,  1944,  1949,  1955,  1961,  1965,
   1971,  1978,  1981,  1984,  1990,  1994,  1997,  2005,  2010,  2015,  2020,  2028,
   2033,  2038,  2043,  2046,  2053,  2059,  2063,  2066,  2072,  2077,  2082,  2086,
   2091,  2099,  2103,  2111,  2117,  2123,  2130,  2135,  2137,  2141,  2148,  2153,
   2163,  2168,  2172,  2177,  2183,  2190,  2197,  2205,  2210,  2214,  2220,  2230,
   2236,  2239,  2244,  2248,  2254,  2261,  2264,  2271,  2277,  2281,  2285,  2292,
   2299,  2303,  2307,  2313,  2321,  2331,  2335,  2340,  2344,  2348,  2354,  2360,
   2365,  2368,  2375,  2379,  2383,  2390,  2395,  2399,  2402,  2406,  2411,  2416,
   2421,  2431,  2435,  2441,  2449,  2452,  2458,  2463,  2469,  2481,  2484,  2487,
   2497,  2505,  2511,  2515,  2519,  2523,  2527,  2534,  2541,  2546,  2549,  2555,
   2560,  2567,  2572,  2578,  2585,  2590,  2594,  2598,  2604,  2609,  2614,  2619,
   2624,  2628,  2632,  2636,  2642,  2648,  2653,  2658,  2664,  2676,  2682,  2687,
   2692,  2698,  2702,  2709,  2716,  2720,  2726,  2735,  2741,  2744,  2753,  2757,
   2762,  2767,  2775,  2780,  2787,  2793,  2800,  2805,  2812,  2818,  2827,  2831,
   2839,  2848,  2854,  2862,  2868,  2874,  2879,  2887,  2894,  2903,  2912,  2918,
   2925,  2932,  2938,  2945,  2954,  2959,  2964,  2971,  2975,  2981,  2985,  2991,
   2997,  3001,  3008,  3014,  3017,  3024,  3030,  3037,  3045,  3050,  3055,  3060,
   3070,  3078,  3084,  3090,  3095,  3101,  3106,  3110,  3119,  3123,  3128,  3134,
   3139,  3147,  3155,  3161,  3170,  3176,  3181,  3187,  3193,  3202,  3209,  3215,
   3222,  3230,  3241,  3246,  3254,  3260,  3266,  3272,  3280,  3285,  3291,  3296,
   3302,  3308,  3311,  3318,  3324,  3332,  3342,  3350,  3359,  3366,  3372,  3378,
   3384,  3390,  3400,  3411,  3416,  3423,  3430,  3436,  3441,  3453,  3461,  3470,
   3478,  3483,  3489,  3495,  3504,  3508,  3516,  3521,  3528,  3534,  3544,  3551,
   3557,  3560,  3566,  3569,  3577,  3587,  3594,  3603,  3608,  3616,  3622,  3628,
   3637,  3644,  3652,  3658,  3666,  3674,  3679,  3686,  3693,  3698,  3704,  3711,
   3718,  3725,  3732,  3740,  3746,  3752,  3759,  3768,  3778,  3787,  3797,  3803,
   3811,  3820,  3828,  3832,  3841,  3848,  3855,  3860,  3872,  3881,  3890,  3898,
   3906,  3913,  3921,  3927,  3933,  3940,  3945,  3953,  3958,  3964,  3972,  3979,
   3988,  3994,  3999,  4006,  4013,  4020,  4026,  4033,  4040,  4049,  4057,  4066,
   4074,  4079,  4087,  4093,  4100,  4110,  4120,  4129,  4136,  4144,  4156,  4163,
   4171,  4178,  4184,  4190,  4199,  4204,  4211,  4217,  4224,  4235,  4243,  4249,
   4256,  4264,  4275,  4283,  4293,  4299,  4306,  4314,  4320,  4329,  4335,  4343,
   4349,  4356,  4365,  4372,  4379,  4386,  4392,  4400,  4411,  4419,  4426,  4433,
   4440,  4447,  4453,  4461,  4467,  4474,  4483,  4488,  4497,  4504,  4515,  4523,
   4532,  4537,  4544,  4550,  4559,  4567,  4576,  4584,  4591,  4597,  4604,  4613,
   4620,  4628,  4635,  4643,  4652,  4661,  4667,  4673,  4681,  4691,  4701,  4708,
   4715,  4723,  4729,  4740,  4749,  4755,  4766,  4774,  4784,  4792,  4801,  4810,
   4814,  4821,  4831,  4838,  4845,  4852,  4861,  4868,  4878,  4886,  4893,  4900,
   4907,  4916,  4924,  4932,  4941,  4948,  4961,  4972,  4978,  4986,  4993,  5000,
   5010,  5016,  5023,  5031,  5038,  5047,  5058,  5070,  5078,  5086,  5094,  5100,
   5108,  5114,  5122,  5130,  5137,  5144,  5153,  5159,  5171,  5179,  5188,  5195,
   5204,  5212,  5219,  5226,  5233,  5242,  5251,  5258,  5265,  5272,  5279,  5288,
   5297,  5304,  5313,  5322,  5329,  5337,  5345,  5352,  5364,  5370,  5380,  5389,
   5397,  5405,  5411,  5422,  5431,  5438,  5446,  5455,  5464,  5470,  5478,  5485,
   5491,  5500,  5506,  5518,  5525,  5533,  5544,  5554,  5561,  5568,  5579,  5589,
   5601,  5610,  5617,  5629,  5637,  5644,  5654,  5661,  5671,  5681,  5690,  5699,
   5708,  5718,  5726,  5738,  5747,  5755,  5762,  5768,  5774,  5783,  5791,  5799,
   5810,  5817,  5824,  5834,  5843,  5854,  5862,  5868,  5874,  5884,  5893,  5902,
   5914,  5921,  5932,  5942,  5950,  5960,  5968,  5979,  5988,  5999,  6009,  6018,
   6027,  6036,  6046,  6055,  6061,  6071,  6078,  6087,  6098,  6107,  6116,  6128,
   6137,  6145,  6153,  6162,  6164,  6170,
};

/* Piece id -> NUL-terminated ASCII string. Valid for id < CITRINET_VOCAB_SIZE. */
static inline const char *citrinet_piece(uint32_t id)
{
  return &kPieces[kOffset[id]];
}

/* Length in bytes, excluding the NUL. */
static inline uint32_t citrinet_piece_len(uint32_t id)
{
  return (uint32_t)(kOffset[id + 1u] - kOffset[id]) - 1u;
}

#endif /* CITRINET_VOCAB_H */
