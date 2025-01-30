class Iterative():
    def __init__(self, scheduling):
        if isinstance(scheduling, list):
            self.scheduling = scheduling
            self.update_next()
        else:
            self.scheduling = None
            self.next_check = scheduling

    def update_next(self):
        if self.scheduling and len(self.scheduling) > 0:
            self.next_check = self.scheduling[0]
            self.scheduling = self.scheduling[1:]
        else:
            self.next_check = None
