from metrics.basic import Metrics


class Dummy(Metrics):
    """
    Dummy score to make Overal score positive and easy to read
    """
    def __init__(self, name=None, weight=1, value=5, if_use=True):
        super(Dummy, self).__init__('Dummy', weight, if_use)
        self.value = 5

    def calculate_metric(self):
        return self.value
