""" Piece Class Definition """
class Piece():
    """ constructor """
    def __init__(self, top, left, bottom, right):
        self.t = top
        self.l = left
        self.b = bottom
        self.r = right
    
    """ rotate 90° to the right """
    def rotate_right(self):
        self.r, self.b, self.l, self.t = self.t, self.r, self.b, self.l
    
    """ rotate 90° to the left """
    def rotate_left(self):
        self.l, self.b, self.r, self.t = self.t, self.l, self.b, self.r
    
    """ toString() """
    def __repr__(self):
        string = "Piece {\n"
        string += "- Top: " + str(self.t) + "\n"
        string += "- Left: " + str(self.l) + "\n"
        string += "- Bottom: " + str(self.b) + "\n"
        string += "- Right: " + str(self.r) + "\n"
        return string + "}"


""" Main """
p = Piece("bord", "creux", "bosse", "bosse")
print(p)

print("\nRotated to right\n")
p.rotate_right()
print(p)

