import unittest

from zigent.actions import ThinkAct
from zigent.agents.agent_utils import act_match


class Test_act_match(unittest.TestCase):
    def test_1(self):
        act_name = "Think"
        assert act_match(act_name, ThinkAct)
