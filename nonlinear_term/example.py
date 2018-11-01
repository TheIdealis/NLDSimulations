from nonlinear import NonLinear
from animator import Animator

RG = NonLinear()

RG.integrate()

anim = Animator(RG)
anim.start_animation()
