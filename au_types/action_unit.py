from enum import Enum


class ActionUnit(Enum):
    AU1 = "AU1"
    AU2 = "AU2"
    AU4 = "AU4"
    AU5 = "AU5"
    AU6 = "AU6"
    AU9 = "AU9"
    AU12 = "AU12"
    AU17 = "AU17"
    AU20 = "AU20"
    AU25 = "AU25"
    AU26 = "AU26"
    AU43 = "AU43"


class ActionUnitName(Enum):
    AU1 = "Inner Brow Raiser"
    AU2 = "Outer Brow Raiser"
    AU4 = "Brow Lowerer"
    AU5 = "Upper Lid Raiser"
    AU6 = "Cheek Raiser"
    AU9 = "Nose Wrinkler"
    AU12 = "Lip Corner Puller"
    AU17 = "Chin Raiser"
    AU20 = "Lip Stretcher"
    AU25 = "Lips Part"
    AU26 = "Jaw Drop"
    AU43 = "Eyes Closed"

    @staticmethod
    def get_action_unit_name(au: str) -> str:
        try:
            return ActionUnitName[au].value
        except KeyError:
            return "Unknown Action Unit"


