from __future__ import print_function

import re
import sys
from pyspark import SparkContext


# define a regular expression for delimiters
NON_WORDS_DELIMITER = re.compile(r'[^\w\d]+')


def main():
    if len(sys.argv) < 2:
        print('''Usage: pyspark q2.py <file>
    e.g. pyspark q2.py file:///home/cloudera/test_file''')
        exit(-1)

    sc = SparkContext(appName="HW4_Q2_LC")
    try:
        n = sc.textFile(sys.argv[1]) \
              .filter(lambda x: len(NON_WORDS_DELIMITER.split(x)) > 10).count()
        print("=" * 20)
        print("    R E S U L T S    ")
        print("Lines with more than 10 words:", n)
        print("=" * 20)
    finally:
        sc.stop()
        

if __name__ == '__main__':
    main()