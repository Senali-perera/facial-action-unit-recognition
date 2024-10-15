from unittest import TestCase

from actionunits.action_unit_decision_maker import ActionUnitDecisionMaker
from au_types.action_unit import ActionUnitName


class AUDecisionMakerTests(TestCase):
    def test_decisions(self):
        self.decision_maker = ActionUnitDecisionMaker()

        view1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.9]
        view2 = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9]
        fusion = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        self.decision_maker.set_inputs(view1, view2, fusion)

        print("test", self.decision_maker.view1)

        self.decision_maker.best_representative()

        selected_au = self.decision_maker.get_activated_action_units()

        print(
            [ActionUnitName.get_action_unit_name(au) for au in selected_au.keys()]
        )


