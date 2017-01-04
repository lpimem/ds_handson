# coding=utf8
from __future__ import print_function
        

import re
import sys
from pyspark import SparkContext

"""
    This program solves the 3rd problem of homework 4:
        Given two files, find all the common non-article 
        (not “a”, “an” and “the”) words between the files. 
        Create your own files to test your program. 
        You don’t need to submit these test files. 
        (30 points)
"""

# define a regular expression for delimiters
NON_WORDS_DELIMITER = re.compile(r'[^\w\d]+')

def parseFile(f, sc, ref=None):
    if ref is None:
        return sc.textFile(f).flatMap(lambda x:NON_WORDS_DELIMITER.split(x)) \
                         .filter(lambda w:  w.lower() not in ("a", "an", "the")) \
                         .map(lambda x:(x, 1)) \
                         .reduceByKey(lambda x,y:max(x,y)) \
                         .collectAsMap()
    else:
        return sc.textFile(f).flatMap(lambda x:x.split()) \
                         .map(lambda x: (x, 1) if x in ref else (x, 0)) \
                         .reduceByKey(lambda x,y:max(x,y)) \
                         .filter(lambda t: t[1]>0) \
                         .map(lambda x: x[0])


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: pyspark q3.py <file1> <file2>\r\n \
        \te.g. pyspark q3.py file:///home/cloudera/file file:///home/cloudera/another_file")
        exit(-1)

    sc = SparkContext(appName="HW4_Q3_CommonWords")
    try:
        wordsInFile1 = parseFile(sys.argv[1], sc)
        commonWords  = parseFile(sys.argv[2], sc, ref=wordsInFile1)
        print("=" * 20)
        print("    R E S U L T S    ")
        print("-" * 20)
        print("words in", sys.argv[1], ":", len(wordsInFile1))
        n = commonWords.count()
        print("found", n, " common words:")
        commonWords.foreach(lambda x: print("    ", x))
        print("=" * 20)
    finally:
        # make sure sc is stopped.
        sc.stop()
