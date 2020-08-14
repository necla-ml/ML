import re
import pytest
from random import randrange as rrange

from ml.analytics import SequenceRuleEngine
from ml import logging

@pytest.fixture
def beer_run():
    return "retrieving_items>=2 -> (walking | 'walking items' | \"walking_items\" | nothing)* -> ('entering' | exiting)+"

@pytest.fixture
def shoplifting():
    return "retrieving_items>=2 -> (walking | walking_items)* -> concealing_items>=2"

@pytest.fixture
def falling():
    return "(Standing | Sitting | Walking | Running)>=3 -> Lying_on_floor>=3"

def test_beer_run(beer_run):
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
        'walking items',
        'walking_items',
        'retrieving_items',
        'returning_items',
        'concealing_items',
        'entering',
        'entering123',
        '123entering',
        'exiting',
        'nothing',
    ]
    engine = SequenceRuleEngine(labels, '->')
    rule = engine.compile(beer_run)
    logging.info(f"rule={beer_run}")
    logging.info(f"compiled={rule}")

def test_shoplifting(shoplifting):
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
    rule = engine.compile(shoplifting)
    logging.info(f"rule={shoplifting}")
    logging.info(f"compiled={rule}")

    for _ in range(10):
        precursor = [labels[rrange(0, len(labels))] for i in range(4)]
        retrieving_items = ['retrieving_items'] * rrange(2, 5)
        walking_items = ['walking_items'] * rrange(0, 1) + ['walking'] * rrange(0, 1)
        concealing_items = ['concealing_items'] * rrange(2, 5)
        postcursor = [labels[rrange(0, len(labels))] for i in range(4)]
        input = precursor + retrieving_items + walking_items + concealing_items + postcursor
        encoded = engine.encode(*input)
        matches = re.findall(rule, encoded)
        logging.info(f"input={input}")
        logging.info(f"encoded={re.compile(encoded)}")
        logging.info(f"matches={matches}")
        assert matches

    for _ in range(10):
        precursor = [labels[rrange(0, 9)] for i in range(4)]
        postcursor = [labels[rrange(0, 9)] for i in range(4)]
        input = precursor + postcursor
        encoded = engine.encode(*input)
        matches = re.findall(rule, encoded)
        logging.info(f"input={input}")
        logging.info(f"encoded={encoded}")
        logging.info(f"matches={matches}")
        assert not matches

def test_falling(falling):
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