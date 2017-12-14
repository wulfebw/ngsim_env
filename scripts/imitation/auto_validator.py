
from hgail.misc.validator import Validator

class AutoValidator(Validator):

    def __init__(self, *args, **kwargs):
        super(AutoValidator, self).__init__(*args, **kwargs)

    def validate(self, itr, objs):
        