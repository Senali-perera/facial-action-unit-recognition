from au_types.action_unit import ActionUnit, ActionUnitName


class ActionUnitDecisionMaker:

    def __init__(self):
        self.view1 = {}
        self.view2 = {}
        self.fusion = {}
        self.THRESHOLD = 0.2

    def set_inputs(self, view1: list, view2: list, fusion: list):
        self.view1 = self.map_list_to_dict(view1)
        self.view2 = self.map_list_to_dict(view2)
        self.fusion = self.map_list_to_dict(fusion)

    def map_list_to_dict(self, values: list):
        return {au.name: value for au, value in zip(ActionUnit, values)}

    def best_representative(self):
        best_scores = {}
        for au in self.view1.keys():
            best_scores[au] = max(self.view1[au], self.view2[au], self.fusion[au])
        return best_scores

    def get_activated_action_units(self):
        best_scores = self.best_representative()
        activated_aus = {au: score for au, score in best_scores.items() if self.__is_activated(score)}
        return activated_aus

    def get_activated_action_unit_names(self, selected_au):
        return [ActionUnitName.get_action_unit_name(au) for au in selected_au.keys()]

    def __is_activated(self, score):

        return score > self.THRESHOLD
