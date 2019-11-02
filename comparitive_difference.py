class comp_diff(object):
    """The class comp_diff is the normalized comparison difference，
    apotential is the analytical potential of an object,
    apppotential is the multipole expansion approximation potential."""

    def __init__(self, apotential, apppotential):
        self.difference = (apppotential-apotential)/apotential
