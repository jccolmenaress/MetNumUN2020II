︠98ddb036-8a87-4b5a-85ec-0c0c0a63272c︠

%md
# SymPyLab

SymPy’s documentation
- https://docs.sympy.org/latest/index.html

︡917fc6c5-4c75-4254-a6be-d711aadf88f4︡{"done":true,"md":"# SymPyLab\n\nSymPy’s documentation\n- https://docs.sympy.org/latest/index.html"}
︠24f4d360-69e0-4af7-9a16-92f814e8c02c︠
%md
## SymPy’s polynomials
- https://docs.sympy.org/latest/modules/polys/basics.html#polynomials

- (x-1)(x-2)(x-3)(x-4)(x-5) = x^5 - 15 x^4  + 85 x^3 - 225 x^2 + 274 x - 120

- (x^5 - 15 x^4  + 85 x^3 - 225 x^2 + 274 x - 120) / (x-1) = x^4  - 14 x^3  + 71 x^2  - 154 x + 120
︡ec0b361e-bb5b-4909-8fe9-e2e97e075a69︡{"done":true,"md":"## SymPy’s polynomials \n- https://docs.sympy.org/latest/modules/polys/basics.html#polynomials \n\n- (x-1)(x-2)(x-3)(x-4)(x-5) = x^5 - 15 x^4  + 85 x^3 - 225 x^2 + 274 x - 120\n\n- (x^5 - 15 x^4  + 85 x^3 - 225 x^2 + 274 x - 120) / (x-1) = x^4  - 14 x^3  + 71 x^2  - 154 x + 120"}
︠23364a96-1c03-424f-8d5e-992a5a8c70cf︠
%md
<img src="https://raw.githubusercontent.com/gjhernandezp/algorithms/master/SymPyLab/sympylabwolfram1.jpg" />
%md
<img src="https://raw.githubusercontent.com/gjhernandezp/algorithms/master/SymPyLab/sympylabwolfram3.jpg" />
︡7ab13ba6-a044-44a5-94dc-b8aabec1c57b︡{"done":true,"md":"<img src=\"https://raw.githubusercontent.com/gjhernandezp/algorithms/master/SymPyLab/sympylabwolfram1.jpg\" /> \n%md\n<img src=\"https://raw.githubusercontent.com/gjhernandezp/algorithms/master/SymPyLab/sympylabwolfram3.jpg\" />"}
︠0eb491a0-abc5-47d1-a4f4-0fcbf21bea1es︠
from sympy import Symbol
from sympy import div

x = Symbol('x')

p = x**5 - 15*x**4  + 85*x**3 - 225*x**2 + 274*x - 120

p, r = div(p,  x-1)

print(p)
print(r)

p, r = div(p,  x-2)

print(p)
print(r)

p, r = div(p,  x-3)

print(p)
print(r)

p, r = div(p,  x-4)

print(p)
print(r)
︡1dcfec45-a93d-4d6c-bbbc-67af9c73c831︡{"stdout":"x**4 - 14*x**3 + 71*x**2 - 154*x + 120\n"}︡{"stdout":"0\n"}︡{"stdout":"x**3 - 12*x**2 + 47*x - 60\n"}︡{"stdout":"0\n"}︡{"stdout":"x**2 - 9*x + 20\n"}︡{"stdout":"0\n"}︡{"stdout":"x - 5\n"}︡{"stdout":"0\n"}︡{"done":true}
︠1af7eb25-c7cd-4485-a8e1-7ea8e3ce8796︠
%md
## SymPy’s polynomial simple univariate polynomial factorization
- https://docs.sympy.org/latest/modules/polys/wester.html#simple-univariate-polynomial-factorization
- factor(x\*\*5 - 15\*x\*\*4  + 85\*x\*\*3 - 225\*x\*\*2 + 274\*x - 120)
︡c24d4b7c-3a70-40ae-8475-4765faf1901c︡{"done":true,"md":"## SymPy’s polynomial simple univariate polynomial factorization\n- https://docs.sympy.org/latest/modules/polys/wester.html#simple-univariate-polynomial-factorization\n- factor(x\\*\\*5 - 15\\*x\\*\\*4  + 85\\*x\\*\\*3 - 225\\*x\\*\\*2 + 274\\*x - 120)"}
︠7d4217d7-d243-4bab-bd31-52ebb2325429︠
%md
<img src="https://raw.githubusercontent.com/gjhernandezp/algorithms/master/SymPyLab/sympylabwolfram4.jpg" />
︡867fb387-bfc7-4d09-82ad-00967c5f10cf︡{"done":true,"md":"<img src=\"https://raw.githubusercontent.com/gjhernandezp/algorithms/master/SymPyLab/sympylabwolfram4.jpg\" />"}
︠56acd18b-9360-4b46-b9fa-aabde1e3be11s︠
from sympy import *
x = Symbol('x')
factor(x**5 - 15*x**4  + 85*x**3 - 225*x**2 + 274*x - 120)
︡2439a755-e8ec-43f4-94d7-9dfe63c0bbd1︡{"stdout":"(x - 5)*(x - 4)*(x - 3)*(x - 2)*(x - 1)\n"}︡{"done":true}
︠4f76f112-8277-49dd-a144-ca480cb40682︠
%md
## SymPy’s solvers
- https://docs.sympy.org/latest/tutorial/solvers.html
- x\*\*5 - 15\*x\*\*4  + 85\*x\*\* 3 - 225\*x\*\* 2 + 274\*x - 120 = 0
︡d3bdb588-774d-43c0-ac72-aca99e267694︡{"done":true,"md":"## SymPy’s solvers\n- https://docs.sympy.org/latest/tutorial/solvers.html\n- x\\*\\*5 - 15\\*x\\*\\*4  + 85\\*x\\*\\* 3 - 225\\*x\\*\\* 2 + 274\\*x - 120 = 0"}
︠4f4ff513-a09c-4b64-8eab-ea6a5f6469f8︠
%md
<img src="https://raw.githubusercontent.com/gjhernandezp/algorithms/master/SymPyLab/sympylabwolfram5.jpg" />
︡2a187950-fce8-437d-8988-883fe6b01ebd︡{"done":true,"md":"<img src=\"https://raw.githubusercontent.com/gjhernandezp/algorithms/master/SymPyLab/sympylabwolfram5.jpg\" />"}
︠208890d9-5b2a-4626-8645-c5fb8ea18503s︠
from sympy import *
x = Symbol('x')
solveset(Eq(x**5 - 15*x**4  + 85*x**3 - 225*x**2 + 274*x - 120, 0), x)
︡e95c22c4-f920-4efb-933c-4b7946c40927︡{"stdout":"FiniteSet(1, 2, 3, 4, 5)"}︡{"stdout":"\n"}︡{"done":true}︡
︠a5b8a87d-07d5-483f-bd0c-b3753f6ce9f5︠
%md
## SymPy’s Symbolic and Numercical Complex Evaluations
- https://docs.sympy.org/latest/modules/evalf.html](https://)
- x = x1 + I*x2,y = y1 + I*y2, z = z1 + I*z2, x*y*z
︡05c5990c-fef4-401f-ac09-d353218e617d︡{"done":true,"md":"## SymPy’s Symbolic and Numercical Complex Evaluations\n- https://docs.sympy.org/latest/modules/evalf.html](https://)\n- x = x1 + I*x2,y = y1 + I*y2, z = z1 + I*z2, x*y*z"}
︠5a310e4e-2882-4ed8-a458-29070681d27a︠
%md
<img src="https://raw.githubusercontent.com/gjhernandezp/algorithms/master/SymPyLab/sympylabwolfram7.jpg" />
︡7ab51317-802f-44bb-a5ac-7cdf2ac3398c︡{"done":true,"md":"<img src=\"https://raw.githubusercontent.com/gjhernandezp/algorithms/master/SymPyLab/sympylabwolfram7.jpg\" />"}
︠5b1b1d1b-e767-400e-b50e-403d02546bcas︠
from sympy import *
x1, x2, y1, y2, z1, z2 = symbols("x1 x2 y1 y2 z1 z2", real=True)
x = x1 + I*x2
y = y1 + I*y2
z = z1 + I*z2

print(x*y*z)
print(expand(x*y*z))
print(expand((x*y)*z))
print(expand(x*(y*z)))

w=N(1/(pi + I), 20)
print('w=',w)
︡5eb67a7b-97bf-416b-bef4-0b2c2ee20b6a︡{"stdout":"(x1 + I*x2)*(y1 + I*y2)*(z1 + I*z2)\n"}︡{"stdout":"x1*y1*z1 + I*x1*y1*z2 + I*x1*y2*z1 - x1*y2*z2 + I*x2*y1*z1 - x2*y1*z2 - x2*y2*z1 - I*x2*y2*z2\n"}︡{"stdout":"x1*y1*z1 + I*x1*y1*z2 + I*x1*y2*z1 - x1*y2*z2 + I*x2*y1*z1 - x2*y1*z2 - x2*y2*z1 - I*x2*y2*z2\n"}︡{"stdout":"x1*y1*z1 + I*x1*y1*z2 + I*x1*y2*z1 - x1*y2*z2 + I*x2*y1*z1 - x2*y1*z2 - x2*y2*z1 - I*x2*y2*z2\n"}︡{"stdout":"w= 0.28902548222223624241 - 0.091999668350375232456*I\n"}︡{"done":true}
︠ade7bc04-05b3-4ec8-a368-314c446b76b3︠
%md
## SymPy’s integrals
- https://docs.sympy.org/latest/modules/integrals/integrals.html
- [risk-engineering.org](https://risk-engineering.org/notebook/monte-carlo-LHS.html)
︡3bfdae87-c8a8-4b81-913c-1c306ceed90e︡{"done":true,"md":"## SymPy’s integrals\n- https://docs.sympy.org/latest/modules/integrals/integrals.html\n- [risk-engineering.org](https://risk-engineering.org/notebook/monte-carlo-LHS.html)"}
︠bc4f974f-dbf9-4924-8260-ae71887f6e5b︠
%md

Let’s start with a simple integration problem in 1D,

$$\int_1^5 x^2 dx$$

This is easy to solve analytically, and we can use the SymPy library in case you’ve forgotten how to resolve simple integrals.
︡fdae48d0-2a1a-4d47-941e-aeb6718dbaa4︡{"done":true,"md":"\nLet’s start with a simple integration problem in 1D,\n\n$$\\int_1^5 x^2 dx$$\n \nThis is easy to solve analytically, and we can use the SymPy library in case you’ve forgotten how to resolve simple integrals."}
︠8ea6dfe1-c5ab-470c-bfff-983d6937af28s︠
import sympy
# we’ll save results using different methods in this data structure, called a dictionary
result = {}
x = sympy.Symbol("x")
i = sympy.integrate(x**2)
print(i)
result["analytical"] = float(i.subs(x, 5) - i.subs(x, 1))
print("Analytical result: {}".format(result["analytical"]))
︡9b195c6c-343f-464d-8d88-71d4719285d4︡{"stdout":"x**3/3\n"}︡{"stdout":"Analytical result: 41.333333333333336\n"}︡{"done":true}
︠eb8e63a6-1aae-409a-8f10-67b8e1cfb547︠
%md
**Integrating with Monte Carlo**
[risk-engineering.org](https://risk-engineering.org/notebook/monte-carlo-LHS.html)

We can estimate this integral using a standard Monte Carlo method, where we use the fact that the expectation of a random variable is related to its integral

$$\mathbb{E}(f(x)) = \int_I f(x) dx $$

We will sample a large number N of points in I and calculate their average, and multiply by the range over which we are integrating.
︡0fdabeac-115b-482e-9469-f5953ce2bfc4︡{"done":true,"md":"**Integrating with Monte Carlo**\n[risk-engineering.org](https://risk-engineering.org/notebook/monte-carlo-LHS.html)\n\nWe can estimate this integral using a standard Monte Carlo method, where we use the fact that the expectation of a random variable is related to its integral\n\n$$\\mathbb{E}(f(x)) = \\int_I f(x) dx $$\n\nWe will sample a large number N of points in I and calculate their average, and multiply by the range over which we are integrating."}
︠cd6c334c-4423-42a7-bae0-40c15f286c1es︠
import numpy
N = 100000
accum = 0
for i in range(N):
    x = numpy.random.uniform(1, 5)
    accum += x**2

volume = 5 - 1

result["MC"] = volume * accum / float(N)
print("Standard Monte Carlo result: {}".format(result["MC"]))
︡559c663d-a9a9-4691-985e-e2240bbfc724︡{"stdout":"Standard Monte Carlo result: 41.3564526580444\n"}︡{"done":true}
︠b3d811de-37fe-44a4-8163-017843ba4523︠
%md
- integrate(x\*\*2 * sin(x)\*\*3)

<img src="https://raw.githubusercontent.com/gjhernandezp/algorithms/master/SymPyLab/sympylabwolfram8.jpg" />
︡7fd981dd-dad4-4a54-adbd-afe2344414ac︡{"done":true,"md":"- integrate(x\\*\\*2 * sin(x)\\*\\*3)\n\n<img src=\"https://raw.githubusercontent.com/gjhernandezp/algorithms/master/SymPyLab/sympylabwolfram8.jpg\" />"}
︠89c92cd3-0929-419f-8ff2-615f51fed745s︠

import sympy
x = sympy.Symbol("x")
i = integrate(x**2 * sin(x)**3)
print(i)
print(float(i.subs(x, 5) - i.subs(x, 1)))
︡be849804-c1d0-47b0-8063-b60a9db15bc8︡{"stdout":"-x**2*sin(x)**2*cos(x) - 2*x**2*cos(x)**3/3 + 14*x*sin(x)**3/9 + 4*x*sin(x)*cos(x)**2/3 + 14*sin(x)**2*cos(x)/9 + 40*cos(x)**3/27\n"}︡{"stdout":"-15.42978215330555\n"}︡{"done":true}
︠dbf27167-e07f-4ea0-8668-1094aaafe054︠
︡bf079986-7f11-4b9b-83a3-18debfd452b4︡
︠9b0d22c3-1710-4039-86a1-60af37c31637s︠
import numpy
N = 100000
accum = 0
l =[]
for i in range(N):
    x = numpy.random.uniform(1, 5)
    accum += x**2 * sin(x)**3
volume = 5 - 1
result["MC"] = volume * accum / float(N)
print("Standard Monte Carlo result: {}".format(result["MC"]))
︡f1240556-8f20-40d8-8981-a37b9a5e3d8e︡{"stdout":"Standard Monte Carlo result: -15.4063276241183\n"}︡{"done":true}
︠d5e5716e-a498-4cc0-8354-0000a08a3d24︠
%md
**A higher dimensional integral** [risk-engineering.org](https://risk-engineering.org/notebook/monte-carlo-LHS.html)


Let us now analyze an integration problem in dimension 4, the Ishigami function. This is a well-known function in numerical optimization and stochastic analysis, because it is very highly non-linear.
︡9bbf061c-8c7e-4b29-9517-735b8fd70701︡{"done":true,"md":"**A higher dimensional integral** [risk-engineering.org](https://risk-engineering.org/notebook/monte-carlo-LHS.html) \n\n\nLet us now analyze an integration problem in dimension 4, the Ishigami function. This is a well-known function in numerical optimization and stochastic analysis, because it is very highly non-linear."}
︠a60a0ae9-4c20-417a-970d-6d54ba1dcc28s︠
import sympy

x1 = sympy.Symbol("x1")
x2 = sympy.Symbol("x2")
x3 = sympy.Symbol("x3")
expr = sympy.sin(x1) + 7*sympy.sin(x2)**2 + 0.1 * x3**4 * sympy.sin(x1)
res = sympy.integrate(expr,
                      (x1, -sympy.pi, sympy.pi),
                      (x2, -sympy.pi, sympy.pi),
                      (x3, -sympy.pi, sympy.pi))
# Note: we use float(res) to convert res from symbolic form to floating point form
result = {}
result["analytical"] = float(res)
print("Analytical result: {}".format(result["analytical"]))
︡70bbd4f6-cef2-458d-95ce-524572fbcd78︡{"stdout":"Analytical result: 868.175747048395\n"}︡{"done":true}
︠e3573eaf-c82b-4d9d-a886-9abba525b7a3s︠
N = 10_000
accum = 0
for i in range(N):
    xx1 = numpy.random.uniform(-numpy.pi, numpy.pi)
    xx2 = numpy.random.uniform(-numpy.pi, numpy.pi)
    xx3 = numpy.random.uniform(-numpy.pi, numpy.pi)
    accum += numpy.sin(xx1) + 7*numpy.sin(xx2)**2 + 0.1 * xx3**4 * numpy.sin(xx1)
volume = (2 * numpy.pi)**3
result = {}
result["MC"] = volume * accum / float(N)
print("Standard Monte Carlo result: {}".format(result["MC"]))
︡1a9c0980-ed2f-45f8-897a-0cca425e43b9︡{"stdout":"Standard Monte Carlo result: 866.8757784457939\n"}︡{"done":true}︡
︠86b0b6fd-890e-4a4c-9ac8-545c36bbd4e0s︠
import math
import numpy
import matplotlib.pyplot as plt
# adapted from https://mail.scipy.org/pipermail/scipy-user/2013-June/034744.html
def halton(dim: int, nbpts: int):
    h = numpy.full(nbpts * dim, numpy.nan)
    p = numpy.full(nbpts, numpy.nan)
    P = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    lognbpts = math.log(nbpts + 1)
    for i in range(dim):
        b = P[i]
        n = int(math.ceil(lognbpts / math.log(b)))
        for t in range(n):
            p[t] = pow(b, -(t + 1))

        for j in range(nbpts):
            d = j + 1
            sum_ = math.fmod(d, b) * p[0]
            for t in range(1, n):
                d = math.floor(d / b)
                sum_ += math.fmod(d, b) * p[t]

            h[j*dim + i] = sum_
    return h.reshape(nbpts, dim)
print("Ok")

︡c4737524-86eb-44c5-bd23-08c780f1155a︡{"stdout":"Ok\n"}︡{"done":true}
︠88361b92-25c2-460c-b21c-8a8d4da582d9s︠
import sys
import matplotlib.pyplot as plt
N = 1000
seq = halton(2, N)
plt.title("2D Halton sequence")
# Note: we use "alpha=0.5" in the scatterplot so that the plotted points are semi-transparent
# (alpha-transparency of 0.5 out of 1), so that we can see when any points are superimposed.
plt.scatter(seq[:,0], seq[:,1], marker=".", alpha=0.5);
plt.show()

︡5a5c78a4-5cf7-4ca0-9b8b-8fc741b9ddfc︡{"stdout":"Text(0.5,1,'2D Halton sequence')\n"}︡{"stdout":"<matplotlib.collections.PathCollection object at 0x7fb0a50442e8>\n"}︡{"file":{"filename":"46046acf-ac39-43f5-ac1a-113fa1a00255.svg","show":true,"text":null,"uuid":"1fbff1a3-ec5d-4b63-8ebb-243dbac6adef"},"once":false}︡{"done":true}
︠c78c6a17-8b98-4370-907b-ff7d209007f4s︠
from numpy import pi
N = 10000
import sympy as sym

seq = halton(3, N)
accum = 0
for i in range(N):
    xx1 = -numpy.pi + seq[i][0] * numpy.pi * 2
    xx2 = -numpy.pi + seq[i][1] * numpy.pi * 2
    xx3 = -numpy.pi + seq[i][2] * numpy.pi * 2
    accum += sympy.sin(xx1) + 7*sympy.sin(xx2)**2 + 0.1 * xx3**4 * sympy.sin(xx1)
volume = (2 * numpy.pi)**3
result = {}
result["MC"] = volume * accum / float(N)
print("Qausi Monte Carlo Halton Sequence result: {}".format(result["MC"]))
︡f9080387-e03e-47ef-9534-3cc5e541a955︡{"stdout":"Qausi Monte Carlo Halton Sequence result: 868.238928030592\n"}︡{"done":true}
︠fbab915a-3a92-42b8-ad29-0058169debbd︠
%md
## Wolfram alpha answers quastion in natural languaje
- What is the average temperature in Bogota in September?

<img src="https://raw.githubusercontent.com/gjhernandezp/algorithms/master/SymPyLab/sympylabwolfram6.jpg" />
︡db8480bb-43de-44aa-9ee4-cf63d59300c2︡{"done":true,"md":"## Wolfram alpha answers quastion in natural languaje\n- What is the average temperature in Bogota in September?\n\n<img src=\"https://raw.githubusercontent.com/gjhernandezp/algorithms/master/SymPyLab/sympylabwolfram6.jpg\" />"}
︠55bfe722-cf41-4e1c-9754-98ac76bccbfe︠
%md
**Todo el código previamente usado en este notebook, hace referencia al notebook elaborado por el profesor German Hernandez**
https://github.com/gjhernandezp/algorithms/tree/master/SymPyLab
︡f03ff9e9-0f7a-4911-b000-2185899a8068︡{"done":true,"md":"**Todo el código previamente usado en este notebook, hace referencia al notebook elaborado por el profesor German Hernandez**\nhttps://github.com/gjhernandezp/algorithms/tree/master/SymPyLab"}
︠386fcca8-3cf7-475c-b70a-e1c7761000c4︠
%md
## Laboratorio #5 - Uso de Sympy
︡8843e054-25d0-4af3-9f4e-c361117fc666︡{"done":true,"md":"## Laboratorio #5 - Uso de Sympy"}
︠ac8dabc8-6294-4335-b668-70f684352298︠
%md
1) Modify the code of SymPyLab form github to add cells and images from Wolfram Alpha for the polynomial (x-1)(x-2)(x-3)(x-4)(x-5)(x-6)(x-7)(x-8)(x-9)(x-10) and integrals of one dimensional an three dimensional functions that are product of a polynomial, log, sin, and e^. and the other cells create you own examples.
︡add7fd59-ed1b-41de-9909-c0fa67332a0e︡{"done":true,"md":"1) Modify the code of SymPyLab form github to add cells and images from Wolfram Alpha for the polynomial (x-1)(x-2)(x-3)(x-4)(x-5)(x-6)(x-7)(x-8)(x-9)(x-10) and integrals of one dimensional an three dimensional functions that are product of a polynomial, log, sin, and e^. and the other cells create you own examples."}
︠31a158f7-109b-4b89-9431-0c604e4c3584︠
%md
<img src="https://raw.githubusercontent.com/Ojtinjacar/AlgorithmsUN2020II/master/Lab5/wolf_img_1.jpg" />
︡91f1ebb5-436c-4c1e-b85d-f3103acc890c︡{"done":true,"md":"<img src=\"https://raw.githubusercontent.com/Ojtinjacar/AlgorithmsUN2020II/master/Lab5/wolf_img_1.jpg\" />"}
︠59dc4157-e215-46e2-aaa7-0b7f16e805c0︠
%md
## SymPy’s polynomials
- (x-1)(x-2)(x-3)(x-4)(x-5)(x-6)(x-7)(x-8)(x-9)(x-10) = x^10 - 55 x^9 + 1320 x^8 - 18150 x^7 + 157773 x^6 - 902055 x^5 + 3416930 x^4 - 8409500 x^3 + 12753576 x^2 - 10628640 x + 3628800
︡ae27f65c-3629-4142-801e-cdb2cf6745ce︡{"done":true,"md":"## SymPy’s polynomials \n- (x-1)(x-2)(x-3)(x-4)(x-5)(x-6)(x-7)(x-8)(x-9)(x-10) = x^10 - 55 x^9 + 1320 x^8 - 18150 x^7 + 157773 x^6 - 902055 x^5 + 3416930 x^4 - 8409500 x^3 + 12753576 x^2 - 10628640 x + 3628800"}
︠7c7ef85f-c50a-4f60-b2bf-4a3f22b87c24s︠
from sympy import Symbol
from sympy import div

x = Symbol('x')

p = x**10 - 55*x**9 + 1320*x**8 - 18150*x**7 + 157773*x**6 - 902055*x**5 + 3416930*x**4 - 8409500*x**3 + 12753576*x**2 - 10628640*x + 3628800
p, r = div(p,  x-1)

print(p)
print(r)

p, r = div(p,  x-2)

print(p)
print(r)

p, r = div(p,  x-3)

print(p)
print(r)

p, r = div(p,  x-4)

print(p)
print(r)

p, r = div(p,  x-5)

print(p)
print(r)

p, r = div(p,  x-6)

print(p)
print(r)

p, r = div(p,  x-7)

print(p)
print(r)

p, r = div(p,  x-8)

print(p)
print(r)

p, r = div(p,  x-9)

print(p)
print(r)
︡21b94f6b-61c5-457d-962c-6d7bfdc935d6︡{"stdout":"x**9 - 54*x**8 + 1266*x**7 - 16884*x**6 + 140889*x**5 - 761166*x**4 + 2655764*x**3 - 5753736*x**2 + 6999840*x - 3628800\n"}︡{"stdout":"0\n"}︡{"stdout":"x**8 - 52*x**7 + 1162*x**6 - 14560*x**5 + 111769*x**4 - 537628*x**3 + 1580508*x**2 - 2592720*x + 1814400\n"}︡{"stdout":"0\n"}︡{"stdout":"x**7 - 49*x**6 + 1015*x**5 - 11515*x**4 + 77224*x**3 - 305956*x**2 + 662640*x - 604800\n"}︡{"stdout":"0\n"}︡{"stdout":"x**6 - 45*x**5 + 835*x**4 - 8175*x**3 + 44524*x**2 - 127860*x + 151200\n"}︡{"stdout":"0\n"}︡{"stdout":"x**5 - 40*x**4 + 635*x**3 - 5000*x**2 + 19524*x - 30240\n"}︡{"stdout":"0\n"}︡{"stdout":"x**4 - 34*x**3 + 431*x**2 - 2414*x + 5040\n"}︡{"stdout":"0\n"}︡{"stdout":"x**3 - 27*x**2 + 242*x - 720\n"}︡{"stdout":"0\n"}︡{"stdout":"x**2 - 19*x + 90\n"}︡{"stdout":"0\n"}︡{"stdout":"x - 10\n"}︡{"stdout":"0\n"}︡{"done":true}︡{"stdout":"0\n"}︡{"stdout":"x**2 - 19*x + 90\n"}︡{"stdout":"0\n"}︡{"stdout":"x - 10\n"}︡{"stdout":"0\n"}︡{"done":true}︡︡{"stdout":"0\n"}︡{"stdout":"x**8 - 52*x**7 + 1162*x**6 - 14560*x**5 + 111769*x**4 - 537628*x**3 + 1580508*x**2 - 2592720*x + 1814400\n"}︡{"stdout":"0\n"}︡{"stdout":"x**7 - 49*x**6 + 1015*x**5 - 11515*x**4 + 77224*x**3 - 305956*x**2 + 662640*x - 604800\n"}︡{"stdout":"0\n"}︡{"stdout":"x**6 - 45*x**5 + 835*x**4 - 8175*x**3 + 44524*x**2 - 127860*x + 151200\n"}︡{"stdout":"0\n"}︡{"stdout":"x**5 - 40*x**4 + 635*x**3 - 5000*x**2 + 19524*x - 30240\n"}︡{"stdout":"0\n"}︡{"stdout":"x**4 - 34*x**3 + 431*x**2 - 2414*x + 5040\n"}︡{"stdout":"0\n"}︡{"stdout":"x**3 - 27*x**2 + 242*x - 720\n"}
︠95273f79-0500-427c-9b26-9a3cce807553︠
%md
## SymPy’s polynomial simple univariate polynomial factorization
- factor(x\*\*10 - 55\*x\*\*9 + 1320\*x\*\*8 - 18150\*x\*\*7 + 157773\*x\*\*6 - 902055\*x\*\*5 + 3416930\*x\*\*4 - 8409500\*x\*\*3 + 12753576\*x\*\*2 - 10628640\*x + 3628800)
︡954049b0-fc7c-4de8-ae02-f3b4401eb08b︡{"done":true,"md":"## SymPy’s polynomial simple univariate polynomial factorization\n- factor(x\\*\\*10 - 55\\*x\\*\\*9 + 1320\\*x\\*\\*8 - 18150\\*x\\*\\*7 + 157773\\*x\\*\\*6 - 902055\\*x\\*\\*5 + 3416930\\*x\\*\\*4 - 8409500\\*x\\*\\*3 + 12753576\\*x\\*\\*2 - 10628640\\*x + 3628800)"}
︠4d685aeb-51a7-4605-a832-c4c03ef62d95︠
%md
<img src="https://raw.githubusercontent.com/Ojtinjacar/AlgorithmsUN2020II/master/Lab5/wolf_img_2.jpg"/>
︡feee0a91-b557-401e-9d05-15d1c8f64b1f︡{"done":true,"md":"<img src=\"https://raw.githubusercontent.com/Ojtinjacar/AlgorithmsUN2020II/master/Lab5/wolf_img_2.jpg\"/>"}
︠f1365774-9d54-4715-82ae-7895aa3650fds︠
from sympy import *
x = Symbol('x')
factor(x**10 - 55*x**9 + 1320*x**8 - 18150*x**7 + 157773*x**6 - 902055*x**5 + 3416930*x**4 - 8409500*x**3 + 12753576*x**2 - 10628640*x + 3628800)
︡947d54a5-7e67-44e7-8c05-6aa44f3d791a︡{"stdout":"(x - 10)*(x - 9)*(x - 8)*(x - 7)*(x - 6)*(x - 5)*(x - 4)*(x - 3)*(x - 2)*(x - 1)\n"}︡{"done":true}
︠98e6a173-cb63-4c5d-bf70-489a3a1c6857︠
%md
## SymPy’s solvers
- x\*\*5 - 15\*x\*\*4  + 85\*x\*\* 3 - 225\*x\*\* 2 + 274\*x - 120 = 0
︡eeb8525c-b283-43e1-99cc-05f91c34ced3︡{"done":true,"md":"## SymPy’s solvers\n- x\\*\\*5 - 15\\*x\\*\\*4  + 85\\*x\\*\\* 3 - 225\\*x\\*\\* 2 + 274\\*x - 120 = 0"}
︠cd50e60f-5b8b-48b7-b615-32ab762e3493︠
%md
<img src="https://raw.githubusercontent.com/Ojtinjacar/AlgorithmsUN2020II/master/Lab5/wolf_img_3.jpg"/>
︡995306aa-b8f2-4875-8543-07b6951643be︡{"done":true,"md":"<img src=\"https://raw.githubusercontent.com/Ojtinjacar/AlgorithmsUN2020II/master/Lab5/wolf_img_3.jpg\"/>"}
︠c0d09178-4e58-4737-bb26-5dc77dd58e9bs︠
from sympy import *
x = Symbol('x')
solveset(Eq(x**10 - 55*x**9 + 1320*x**8 - 18150*x**7 + 157773*x**6 - 902055*x**5 + 3416930*x**4 - 8409500*x**3 + 12753576*x**2 - 10628640*x + 3628800,0), x)
︡6b5fa68d-7f1b-47fe-900b-eb5a4a2ad5c4︡{"stdout":"FiniteSet(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)\n"}︡{"done":true}
︠183d7698-b192-4a47-9d4b-8ba89e1878fd︠

%md
Vamos a trabajar con 3 integrales de una variable.

1)

$$\int_2^3 \frac{cos(x)}{x} dx$$
︡e26a73e9-843d-4b2c-a556-799acb478d8e︡{"done":true,"md":"Vamos a trabajar con 3 integrales de una variable.\n\n1)\n\n$$\\int_2^3 \\frac{cos(x)}{x} dx$$"}
︠134d968f-0104-4396-bd2d-c83d9dab29fes︠

import sympy
result = {}
x = sympy.Symbol("x")
i = sympy.integrate(sympy.cos(x)/x)
print(i)
result["analytical"] = float(i.subs(x, 3) - i.subs(x, 2))
print("Analytical result: {}".format(result["analytical"]))
︡ec776944-38dc-46bd-bd70-6f43bea9a4fd︡{"stdout":"-log(x) + log(x**2)/2 + Ci(x)\n"}︡{"stdout":"Analytical result: -0.30335104276686464\n"}︡{"done":true}
︠65580680-97c7-4af6-a1b4-946ee1909faf︠
%md
<img src="https://raw.githubusercontent.com/Ojtinjacar/AlgorithmsUN2020II/master/Lab5/wolf_img_4.jpg"/>
︡5899a0a6-4630-4a6b-a7eb-df8c978b6783︡{"done":true,"md":"<img src=\"https://raw.githubusercontent.com/Ojtinjacar/AlgorithmsUN2020II/master/Lab5/wolf_img_4.jpg\"/>"}
︠c4458807-1851-488f-b565-ba28cfc9d793︠
%md
2)
$$\int_0^{\pi/2} 2\pi cos^2(x) dx$$
︡3e2a8381-be40-4ee6-9239-b1bcf7100be9︡{"done":true,"md":"2) \n$$\\int_0^{\\pi/2} 2\\pi cos^2(x) dx$$"}
︠ac4dc67b-6b24-42ec-81ad-2a1a12dfdb19s︠
import sympy
result = {}
x = sympy.Symbol("x")
i = sympy.integrate(2 * sympy.pi * sympy.cos(x)**2)
print(i)
result["analytical"] = float(i.subs(x, sympy.pi/2) - i.subs(x, 0))
print("Analytical result: {}".format(result["analytical"]))
︡71192e09-5237-4364-b730-a03f38b42246︡{"stdout":"2*pi*(x/2 + sin(x)*cos(x)/2)\n"}︡{"stdout":"Analytical result: 4.934802200544679\n"}︡{"done":true}
︠b8c9ec3c-2a27-4afa-a1b7-e2d7c50b2868︠
%md
<img src="https://raw.githubusercontent.com/Ojtinjacar/AlgorithmsUN2020II/master/Lab5/wolf_img_5.jpg"/>
︡2482c05f-b141-4591-b561-ae4aff26f5eb︡{"done":true,"md":"<img src=\"https://raw.githubusercontent.com/Ojtinjacar/AlgorithmsUN2020II/master/Lab5/wolf_img_5.jpg\"/>"}
︠9b609fac-b5c6-40ec-b7d8-c76832b0b3ac︠
%md
3)
$$\int_\frac{3}{2}^3 e^x sen(x) cos(x) dx$$
︡f8de1903-7267-40e2-bc8f-a80e65be9924︡{"done":true,"md":"3)\n$$\\int_\\frac{3}{2}^3 e^x sen(x) cos(x) dx$$"}
︠19c4dc22-41d5-4824-a78a-15ddb2466bafs︠
import sympy
result = {}
x = sympy.Symbol("x")
i = sympy.integrate(sympy.exp(1)**x * sympy.sin(x) * sympy.cos(x))
print(i)
result["analytical"] = float(i.subs(x, 3) - i.subs(x, 3/2))
print("Analytical result: {}".format(result["analytical"]))
︡465d054c-2333-43a2-a38c-e9d436128771︡{"stdout":"exp(x)*sin(x)**2/5 + exp(x)*sin(x)*cos(x)/5 - exp(x)*cos(x)**2/5\n"}︡{"stdout":"Analytical result: -5.368941489730238\n"}︡{"done":true}
︠da461fa0-64d2-427e-b552-ecf3494bdafds︠
%md
<img src="https://raw.githubusercontent.com/Ojtinjacar/AlgorithmsUN2020II/master/Lab5/wolf_img_6.jpg"/>
︡e2898f07-6df8-4ee3-9d6d-89412c7080b8︡{"done":true,"md":"<img src=\"https://raw.githubusercontent.com/Ojtinjacar/AlgorithmsUN2020II/master/Lab5/wolf_img_6.jpg\"/>"}︡{"done":true}︡
︠65490e4c-1f88-4ff0-a337-b0aa1d18a795︠

%md
Vamos a trabajar con integrales en 2 variables

1)

$$\int_0^1 \int_0^{2-2x} (1+3x+y )  dydx$$
︡9067eac4-29e7-431e-85a3-9ff91d673665︡{"done":true,"md":"Vamos a trabajar con integrales en 2 variables\n\n1)\n\n$$\\int_0^1 \\int_0^{2-2x} (1+3x+y )  dydx$$"}
︠efaf8bfb-e731-444b-b251-6f477e6cbd3cs︠
import sympy
x = sympy.Symbol("x")
y = sympy.Symbol("y")
i = sympy.integrate(1 + 3*x + y,(y,0,2-2*x)) ##En los parámetros de "integrate", tenemos también una forma en la cuál
i2 = sympy.integrate(1 + 3*x + y,y)          ##se ingresa la función, y seguido de una coma (,), ingresamos la variable a integrar
print(i2)                                    ## y los límites de la misma.
t2 = sympy.integrate(i,x)                    ## A partir de acá y en los demás ejercicios, se trabajará con esos dos parámetros.
print(t2)
t = sympy.integrate(i,(x,0,1))
print("Analytical result: {}".format(t))
︡6be4ee46-ecb0-44e0-b64b-c17feda2491b︡{"stdout":"y**2/2 + y*(3*x + 1)\n"}︡{"stdout":"-4*x**3/3 + 4*x\n"}︡{"stdout":"Analytical result: 8/3\n"}︡{"done":true}
︠09b66ea3-e183-4bed-9fbb-5944481e04d8︠
%md
<img src="https://raw.githubusercontent.com/Ojtinjacar/AlgorithmsUN2020II/master/Lab5/wolf_img_7.jpg"/>
︡667b5d54-6536-4bcb-ae8c-ec2457f5986e︡{"done":true,"md":"<img src=\"https://raw.githubusercontent.com/Ojtinjacar/AlgorithmsUN2020II/master/Lab5/wolf_img_7.jpg\"/>"}
︠253eb0a6-d099-4d5b-855c-2d97333c3a40︠
%md
2)

$$\int_{-3}^{3} \int_1^{\sqrt{9-x^2}} e^xy  dydx$$
︡2d062ee0-0784-4577-a851-61cacf8825fb︡{"done":true,"md":"2) \n\n$$\\int_{-3}^{3} \\int_1^{\\sqrt{9-x^2}} e^xy  dydx$$"}
︠8647efd4-fb2f-4090-8888-44ebcce48aa8s︠
import math
import sympy

x = sympy.Symbol("x")
y = sympy.Symbol("y")
i = sympy.integrate((sympy.exp(1)**x)*y,(y,1,(9-x**2)**(1/2)))
print(i)
t2 = sympy.integrate(i,x)
print(t2)
t = sympy.integrate(i,(x,-3,3))
print("Analytical result: {}".format(t))
print(4.5*math.exp(-3)+1.5*math.exp(3)) ##Incorporamos la librería math, para dar un resultado numérico
︡62e6c422-10bb-498e-aa46-b502764932bb︡{"stdout":"(9 - x**2)*exp(x)/2 - exp(x)/2\n"}︡{"stdout":"(-x**2 + 2*x + 6)*exp(x)/2\n"}︡{"stdout":"Analytical result: 9*exp(-3)/2 + 3*exp(3)/2\n"}︡{"stdout":"30.3523471924369\n"}︡{"done":true}
︠16a56af9-4a91-4ac5-9a2a-02b68e275542︠
%md
A partir de ahora usamos una página parecida a Wolfram, pero su nombre es Symbolab.
Se realiza esto, porque cuando se intentó calcular algunas integrales dobles y triples, no encontrabamos soluciones.

︡0a92c431-1153-4344-8f64-904760e73729︡{"done":true,"md":"A partir de ahora usamos una página parecida a Wolfram, pero su nombre es Symbolab.\nSe realiza esto, porque cuando se intentó calcular algunas integrales dobles y triples, no encontrabamos soluciones."}
︠47b8d0b7-f2b0-40ac-b5c0-c25934dabd41s︠
︡17227f9a-e02a-4c56-9776-296444d5b208︡{"done":true}
︠bd983026-e86b-4afc-aaa7-9c76224d8a6c︠

%md

<img src="https://raw.githubusercontent.com/Ojtinjacar/AlgorithmsUN2020II/master/Lab5/symb_img_8.jpg"/>
︡07ee07ed-4a38-4013-a035-92f149d6637c︡{"done":true,"md":"\n<img src=\"https://raw.githubusercontent.com/Ojtinjacar/AlgorithmsUN2020II/master/Lab5/symb_img_8.jpg\"/>"}
︠06ba631e-9f14-4073-82c5-3a8b9d119cab︠

%md
3)

$$\int _{\frac{\pi }{4}}^{\frac{\pi }{2}}\:\int _2^{x^2}x^2ln\left(y\right)dydx$$
︡1f2d6088-5f8e-493e-8dbb-ee402ecac88a︡{"done":true,"md":"3)\n\n$$\\int _{\\frac{\\pi }{4}}^{\\frac{\\pi }{2}}\\:\\int _2^{x^2}x^2ln\\left(y\\right)dydx$$"}
︠eccea5f9-2559-421c-80b3-2e4e339c15d7s︠
import math
import sympy

x = sympy.Symbol("x")
y = sympy.Symbol("y")
i = sympy.integrate((x**2*sympy.ln(y)),(y,2,x**2))
print(i)
t2 = sympy.integrate(i,x)
print(t2)
t = sympy.integrate(i,(x,sympy.pi/4,sympy.pi/2))
print("Analytical result: {}".format(t))
print(-217*math.pi**5/25600 - math.pi**5*math.log(math.pi**2/16)/5120 + 7*math.pi**3*(-2*math.log(2)/3 + 2/3)/64 + math.pi**5*math.log(math.pi**2/4)/160)
︡ee50b418-a06a-433d-8ed7-a287aa52af1c︡{"stdout":"x**4*log(x**2) - x**4 - 2*x**2*log(2) + 2*x**2\n"}︡{"stdout":"x**5*log(x**2)/5 - 7*x**5/25 + x**3*(2/3 - 2*log(2)/3)\n"}︡{"stdout":"Analytical result: -217*pi**5/25600 - pi**5*log(pi**2/16)/5120 + 7*pi**3*(2/3 - 2*log(2)/3)/64 + pi**5*log(pi**2/4)/160\n"}︡{"stdout":"-0.14394799304112094\n"}︡{"done":true}
︠eb9e0549-74a1-4ddd-93e5-a7fae90dc060︠
%md
<img src="https://raw.githubusercontent.com/Ojtinjacar/AlgorithmsUN2020II/master/Lab5/symb_img_9.jpg"/>
︡5657b546-5e91-4b4f-8392-944568bee179︡{"done":true,"md":"<img src=\"https://raw.githubusercontent.com/Ojtinjacar/AlgorithmsUN2020II/master/Lab5/symb_img_9.jpg\"/>"}
︠91cfa57b-4154-46bf-a7d9-ec491bee2c74︠
%md
Y para finalizar, vamos a trabajar con integrales de 3 variables

1)

$$\int _0^2\int _0^{z^2}\int _0^{y-z}\left(2x-y\right)dxdydz\:\:$$
︡7ed4925f-e6d1-4ea1-982b-ec3c80faf8a4︡{"done":true,"md":"Y para finalizar, vamos a trabajar con integrales de 3 variables\n\n1)\n\n$$\\int _0^2\\int _0^{z^2}\\int _0^{y-z}\\left(2x-y\\right)dxdydz\\:\\:$$"}
︠01544151-293b-4a6f-91fd-b93224fbe90cs︠
import sympy

x = sympy.Symbol("x")
y = sympy.Symbol("y")
z = sympy.Symbol("z")

i = sympy.integrate(2*x-y,(x,0,y-z)) ##En este caso, tomamos el resultado de la integral "i" y la aplicamos en la siguiente integral
print(i)                             ## que tiene como resultado "t" e iteramos con las demás variables
t = sympy.integrate(i,(y,0,z**2))
print(t)
r = sympy.integrate(t,(z,0,2))
print(r)

print("Analytical result: {}".format(r))
︡eaac1c94-f147-4fc1-98ca-4737bfa6c2cf︡{"stdout":"-y*(y - z) + (y - z)**2\n"}︡{"stdout":"-z**5/2 + z**4\n"}︡{"stdout":"16/15\n"}︡{"stdout":"Analytical result: 16/15\n"}︡{"done":true}︡︡{"stdout":"-z**5/2 + z**4\n"}︡{"stdout":"16/15\n"}︡{"stdout":"Analytical result: 16/15\n"}︡{"done":true}
︠12cb59d5-aa48-472c-a42e-3d3cdcf0af2d︠
%md
<img src="https://raw.githubusercontent.com/Ojtinjacar/AlgorithmsUN2020II/master/Lab5/symb_img_10.jpg"/>
︡b44fbb38-83a2-473e-a0fc-037a09c7dfff︡{"done":true,"md":"<img src=\"https://raw.githubusercontent.com/Ojtinjacar/AlgorithmsUN2020II/master/Lab5/symb_img_10.jpg\"/>"}
︠0cba4088-99bc-4455-8172-306d787328e2︠
%md
2)

$$\int _0^{\sqrt{\pi }}\int _0^x\int _0^{xz}\left(x^2sin\left(y\right)\right)dydzdx$$
︡97f4fd61-9731-4473-a54e-2f4f2886fc50︡{"done":true,"md":"2)\n\n$$\\int _0^{\\sqrt{\\pi }}\\int _0^x\\int _0^{xz}\\left(x^2sin\\left(y\\right)\\right)dydzdx$$"}
︠97dab8df-8c3c-4f0c-93b0-213ab3d0c76cs︠
import sympy

x = sympy.Symbol("x")
y = sympy.Symbol("y")
z = sympy.Symbol("z")

i = sympy.integrate(x**2 * sympy.sin(y),(y,0,x*z))
print(i)
t = sympy.integrate(i,(z,0,x))
print(t)
r = sympy.integrate(t,(x,0,sympy.sqrt(sympy.pi)))
print(r)

print("Analytical result: {}".format(r))
︡0d9bf453-bd40-4037-9bbd-cf1dd86bf01b︡{"stdout":"-x**2*cos(x*z) + x**2\n"}︡{"stdout":"x**3 + Piecewise((-x*sin(x**2), (x > -oo) & (x < oo) & Ne(x, 0)), (-x**3, True))\n"}︡{"stdout":"-1 + pi**2/4\n"}︡{"stdout":"Analytical result: -1 + pi**2/4\n"}︡{"done":true}
︠ebc2b213-c50e-45b5-8ac4-172ec1eb5a17︠
%md
<img src="https://raw.githubusercontent.com/Ojtinjacar/AlgorithmsUN2020II/master/Lab5/symb_img_11.jpg"/>
︡8f005370-23f9-46cf-9fd4-459171445873︡{"done":true,"md":"<img src=\"https://raw.githubusercontent.com/Ojtinjacar/AlgorithmsUN2020II/master/Lab5/symb_img_11.jpg\"/>"}
︠2342a3df-cf12-4789-ba18-dfc1b6dd1f18︠
%md
3)

$$\int _0^{\pi }\int _0^{2\pi }\int _0^1e^{\left(x^2\right)^{\frac{3}{2}}}x^2sin\left(z\right)dxdydz$$
︡b34bb57c-c52a-45bc-bc8f-d86ca40ee83f︡{"done":true,"md":"3) \n\n$$\\int _0^{\\pi }\\int _0^{2\\pi }\\int _0^1e^{\\left(x^2\\right)^{\\frac{3}{2}}}x^2sin\\left(z\\right)dxdydz$$"}
︠833b0552-7cc6-4e32-896d-5124989b009as︠
import sympy
import math
x = sympy.Symbol("x")
y = sympy.Symbol("y")
z = sympy.Symbol("z")

i = sympy.integrate(sympy.exp(1)**(x**3) *x**2 * sympy.sin(z),(x,0,1))
print(i)
t = sympy.integrate(i,(y,0,2*sympy.pi))
print(t)
r = sympy.integrate(t,(z,0,sympy.pi))
print(r)

print("Analytical result: {}".format(r))
print(-2*math.pi*(-math.exp(1)/3 + 1/3) + 2*math.pi*(-1/3 + math.exp(1)/3))
︡e3e1e88c-2b97-4343-b172-b23ad5055dac︡{"stdout":"-sin(z)/3 + E*sin(z)/3\n"}︡{"stdout":"2*pi*(-sin(z)/3 + E*sin(z)/3)\n"}︡{"stdout":"-2*pi*(1/3 - E/3) + 2*pi*(-1/3 + E/3)\n"}︡{"stdout":"Analytical result: -2*pi*(1/3 - E/3) + 2*pi*(-1/3 + E/3)\n"}︡{"stdout":"Analytical result: -2*pi*(1/3 - E/3) + 2*pi*(-1/3 + E/3)\n"}︡{"stdout":"7.197522092111699\n"}︡{"stdout":"2*pi*(-sin(z)/3 + E*sin(z)/3)\n"}︡{"stdout":"-2*pi*(1/3 - E/3) + 2*pi*(-1/3 + E/3)\n"}︡{"stdout":"7.197522092111699\n"}︡{"done":true}︡
︠790cc9a9-3655-4f2f-817c-5b1c1d7a689f︠
%md
<img src="https://raw.githubusercontent.com/Ojtinjacar/AlgorithmsUN2020II/master/Lab5/symb_img_12.jpg"/>
︡94d3a3c7-3c66-49a7-805d-94a9602f8517︡{"done":true,"md":"<img src=\"https://raw.githubusercontent.com/Ojtinjacar/AlgorithmsUN2020II/master/Lab5/symb_img_12.jpg\"/>"}
︠fbe0d6da-bb28-4fe5-946c-cbb671ab6560︠









