from au_types.action_unit import ActionUnit, ActionUnitName


class Face:
    def __init__(self, face_id, face_name):
        pass

    def get_action_units(self):
        pass


class UpperFace(Face):
    def __init__(self, face_id, face_name, upperface_id, upperface_name):
        super().__init__(face_id, face_name)
        self.upperface_id = upperface_id
        self.upperface_name = upperface_name
        self.upperface_au = [ActionUnit.AU1.name, ActionUnit.AU2.name, ActionUnit.AU4.name, ActionUnit.AU5.name,
                             ActionUnit.AU6.name, ActionUnit.AU43.name]

    def get_action_units(self):
        return self.upperface_au


class LowerFace(Face):
    def __init__(self, face_id, face_name, lowerface_id, lowerface_name):
        super().__init__(face_id, face_name)
        self.lowerface_id = lowerface_id
        self.lowerface_name = lowerface_name
        self.lowerface_au = [ActionUnit.AU9.name, ActionUnit.AU12.name, ActionUnit.AU17.name, ActionUnit.AU20.name,
                             ActionUnit.AU25.name, ActionUnit.AU26.name]

    def get_action_units(self):
        return self.lowerface_au


class Mouth(LowerFace):
    def __init__(self, face_id, face_name, mouth_id, mouth_name):
        super().__init__(face_id, face_name, None, None)
        self.mouth_au = [ActionUnit.AU12.name, ActionUnit.AU17.name, ActionUnit.AU20.name, ActionUnit.AU25.name,
                         ActionUnit.AU26.name]

    def get_action_units(self):
        pass


class Eyes(UpperFace):
    def __init__(self, face_id, face_name, eyes_id, eyes_name):
        super().__init__(face_id, face_name, None, None)
        self.eyes_au = [ActionUnit.AU1.name, ActionUnit.AU2.name, ActionUnit.AU4.name, ActionUnit.AU5.name,
                        ActionUnit.AU6.name, ActionUnit.AU43.name]

    def get_action_units(self):
        return self.eyes_au


class Nose(LowerFace):
    def __init__(self, face_id, face_name, nose_id, nose_name):
        super().__init__(face_id, face_name, None, None)
        self.nose_au = [ActionUnit.AU9.name, ActionUnit.AU20.name, ActionUnit.AU25.name, ActionUnit.AU26.name]

    def get_action_units(self):
        pass
