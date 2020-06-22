import re
import pytest
from random import randrange as rrange

from ml.analytics import SequenceRuleEngine
from ml import logging

logging.getLogger().setLevel('INFO')

def test_shoplifting():
    print()
    labels = [
        'standing',
        'sitting',
        'walking',
        'running',
        'grabbing',
        'inspection',
        'concealing',
        'lying_on_floor',
        'walking_items',
        'retrieving_items',
        'returning_items',
        'concealing_items',
    ]
    engine = SequenceRuleEngine(labels, '->')
    shoplifting = "retrieving_items<=2 -> (walking | walking_items)? -> concealing_items>=2"
    #shoplifting = "retrieving_items>=2 -> inspection? -> walking_items? -> concealing_items>=2"
    rule = engine.compile(shoplifting)
    logging.info(f"rule={shoplifting}")
    logging.info(f"compiled={rule}")

    input = ['retrieving_items','retrieving_items',  'concealing_items', 'concealing_items', 'concealing_items']
    encoded = engine.encode(*input)
    matches = re.findall(rule, encoded)
    matches = re.search(rule, encoded)
    logging.info(f"input={input}")
    logging.info(f"encoded={re.compile(encoded)}")
    logging.info(f"matches={matches}")

def test_falling():
    print()
    labels = [
        'standing',
        'sitting',
        'walking',
        'running',
        'grabbing',
        'inspection',
        'concealing',
        'lying_on_floor'
    ]
    falling = "(Standing | Sitting | Walking | Running)>=3 -> Lying_on_floor>=3"
    engine = SequenceRuleEngine(labels, '->')
    rule = engine.compile(falling)
    logging.info(f"rule={falling}")
    logging.info(f"compiled={rule}")

    for _ in range(10):
        precursor = [labels[rrange(0, len(labels))] for i in range(4)]
        precondition = [labels[rrange(0, 4)]] * rrange(3, 10)
        postcondition = [labels[-1]] * rrange(3, 10) 
        postcursor = [labels[rrange(0, len(labels))] for i in range(4)]
        input = precursor + precondition + postcondition + postcursor
        encoded = engine.encode(*input)
        matches = re.findall(rule, encoded)
        logging.info(f"input={input}")
        logging.info(f"encoded={re.compile(encoded)}")
        logging.info(f"matches={matches}")
        assert matches

    for _ in range(10):
        precursor = [labels[rrange(0, len(labels))] for i in range(4)]
        precondition = [labels[rrange(0, 4)]] * rrange(3, 10)
        postcondition = [labels[-1]] * rrange(3, 10) 
        postcursor = [labels[rrange(0, len(labels))] for i in range(4)]
        input = precursor + postcursor
        encoded = engine.encode(*input)
        matches = re.findall(rule, encoded)
        logging.info(f"input={input}")
        logging.info(f"encoded={re.compile(encoded)}")
        logging.info(f"matches={matches}")
        assert not matches

print(test_shoplifting())
#print(test_falling())