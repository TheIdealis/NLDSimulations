from linear import Linear
from animator import Animator

RG = Linear()

RG.integrate()

anim = Animator(RG)
anim.start_animation()
