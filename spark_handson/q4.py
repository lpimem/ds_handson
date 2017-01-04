# coding=utf8
from __future__ import print_function
        

import re
import sys
from pyspark import SparkContext

"""
    This program solves the 4th problem of homework 4.
"""

# define a regular expression for delimiters
NON_WORDS_DELIMITER = re.compile(r'[^\w\d]+')

def parseFileAsInvertedIndex(f, sc):
    return sc.textFile(f).zipWithIndex() \
                         .flatMap(lambda x:[[i, [x[1]]] for i in NON_WORDS_DELIMITER.split(x[0])]) \
                         .filter(lambda x: len(x[0].strip())> 0) \
                         .reduceByKey(lambda x,y:x+y)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: pyspark q4.py <file1> <file2>\r\n \
        \te.g. pyspark q4.py file:///home/cloudera/file")
        exit(-1)

    sc = SparkContext(appName="HW4_Q4_CommonWords")
    try:
        idx = parseFileAsInvertedIndex(sys.argv[1], sc)
        idx.foreach(lambda x: print("    ", x))
    finally:
        # make sure sc is stopped.
        sc.stop()
