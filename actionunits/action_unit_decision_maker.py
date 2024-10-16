from au_types.action_unit import ActionUnit, ActionUnitName


class ActionUnitDecisionMaker:

    def __init__(self):
        self.view1 = {}
        self.view2 = {}
        self.fusion = {}
        self.THRESHOLD = 0
        self.thresholds = {ActionUnit.AU1.value: 0.08910267,
                           ActionUnit.AU2.value: 0.04149807,
                           ActionUnit.AU4.value: 0.18062165,
                           ActionUnit.AU5.value: 0.09666822,
                           ActionUnit.AU6.value: 0.065210536,
                           ActionUnit.AU9.value: 0.06184598,
                           ActionUnit.AU12.value: 0.31178287,
                           ActionUnit.AU17.value: 0.062417142,
                           ActionUnit.AU20.value: 0.04055041,
                           ActionUnit.AU25.value: 0.5772027,
                           ActionUnit.AU26.value: 0.029543918,
                           ActionUnit.AU43.value: 0.049267963}

    def set_inputs(self, view1: list, view2: list, fusion: list):
        self.view1 = self.map_list_to_dict(view1)
        self.view2 = self.map_list_to_dict(view2)
        self.fusion = self.map_list_to_dict(fusion)

    def map_list_to_dict(self, values: list):
        return {au.name: value for au, value in zip(ActionUnit, values)}

    def get_activated_action_units(self):
        best_scores = self.best_representative_dynamic_threshold()
        activated_aus = {au: score for au, score in best_scores.items() if self.__is_activated(score, au)}
        return activated_aus

    def get_activated_action_unit_names(self, selected_au):
        return [ActionUnitName.get_action_unit_name(au) for au in selected_au.keys()]

    def __is_activated(self, score, au):
        return score > self.thresholds.get(au)

    # alternative function for finding best representative AUs with confidence based dynamic threshold method
    def best_representative_dynamic_threshold(self, base_threshold=0.1, confidence_factor=0.01):
        best_scores = {}
        for au in self.view1.keys():
            average_score = (self.view1[au] + self.view2[au]) / 2
            confidence = abs(self.view1[au] - self.view2[au])
            dynamic_threshold = base_threshold + (confidence_factor * confidence)
            
            if average_score >= dynamic_threshold:
                best_scores[au] = average_score
            else:
                best_scores[au] = 0  # AU not activated
        return best_scores