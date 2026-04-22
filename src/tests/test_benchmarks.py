import pytest
import time

from sympy import Matrix, sqrt, symbols, exp
from sympy.core.random import verify_numerically

from chimera import QPoly, get_matrix_terms
from chimera.utils import get_term

s, sigma = symbols("s sigma")

mrg = Matrix([
    [sigma ** (-2 * sqrt(7) / 3 + 2 * sqrt(11) / 3) * (
                sigma ** (sqrt(3)) * (1 - 3 * s) + 9 * (-10 * s - 42) / sigma ** (sqrt(7)) + 9 * (
                    22 * s - 8) / sigma ** (sqrt(7)) + 2 * (25 * s + 27) / sigma ** (sqrt(11))),
     sigma ** (-2 * sqrt(7) / 3 + 2 * sqrt(11) / 3) * (
                 5 * sigma ** (sqrt(3)) + 18 / sigma ** (sqrt(7)) - 122 / sigma ** (sqrt(11))),
     sigma ** (-2 * sqrt(7) / 3 + 2 * sqrt(11) / 3) * (
                 -3 * sigma ** (sqrt(3)) + 963 / sigma ** (sqrt(7)) - 44 / sigma ** (sqrt(11)))],
    [sigma ** (-sqrt(11) / 3 + sqrt(7) / 3) * (
                5 * sigma ** (sqrt(3)) * (1 - 3 * s) / 2 + sigma ** (sqrt(7)) * (-s - 1) + 51 * (-10 * s - 42) / (
                    2 * sigma ** (sqrt(7))) + 31 * (22 * s - 8) / (2 * sigma ** (sqrt(7)))) + sigma ** (
                 -2 * sqrt(7) / 3 + 2 * sqrt(11) / 3) * (
                 -sigma ** (-sqrt(11) - sqrt(3) + 2 * sqrt(7)) * (-s - 1) / (1 - 3 * s) + 4 * exp(
             -sqrt(11) / 3 - sqrt(3) / 3 + 2 * sqrt(7) / 3) / (9 * (s - 1 / 3))) * (
                 sigma ** (sqrt(3)) * (1 - 3 * s) + 9 * (-10 * s - 42) / sigma ** (sqrt(7)) + 9 * (
                     22 * s - 8) / sigma ** (sqrt(7)) + 2 * (25 * s + 27) / sigma ** (sqrt(11))),
     sigma ** (-sqrt(11) / 3 + sqrt(7) / 3) * (
                 25 * sigma ** (sqrt(3)) / 2 + 2 * sigma ** (sqrt(7)) + 461 / sigma ** (sqrt(7))) + sigma ** (
                 -2 * sqrt(7) / 3 + 2 * sqrt(11) / 3) * (
                 -sigma ** (-sqrt(11) - sqrt(3) + 2 * sqrt(7)) * (-s - 1) / (1 - 3 * s) + 4 * exp(
             -sqrt(11) / 3 - sqrt(3) / 3 + 2 * sqrt(7) / 3) / (9 * (s - 1 / 3))) * (
                 5 * sigma ** (sqrt(3)) + 18 / sigma ** (sqrt(7)) - 122 / sigma ** (sqrt(11))),
     sigma ** (-sqrt(11) / 3 + sqrt(7) / 3) * (
                 -15 * sigma ** (sqrt(3)) / 2 + 5 * sigma ** (sqrt(7)) + 3997 / (2 * sigma ** (sqrt(7)))) + sigma ** (
                 -2 * sqrt(7) / 3 + 2 * sqrt(11) / 3) * (
                 -sigma ** (-sqrt(11) - sqrt(3) + 2 * sqrt(7)) * (-s - 1) / (1 - 3 * s) + 4 * exp(
             -sqrt(11) / 3 - sqrt(3) / 3 + 2 * sqrt(7) / 3) / (9 * (s - 1 / 3))) * (
                 -3 * sigma ** (sqrt(3)) + 963 / sigma ** (sqrt(7)) - 44 / sigma ** (sqrt(11)))],
    [sigma ** (-sqrt(11) / 3 + sqrt(7) / 3) * (
                sigma ** (sqrt(11)) * (-s - 3) + 9 * sigma ** (sqrt(3)) * (1 - 3 * s) / 31 + 47 * sigma ** (sqrt(7)) * (
                    -s - 1) / 31 + 315 * (-10 * s - 42) / (31 * sigma ** (sqrt(7)))) + sigma ** (
                 -2 * sqrt(7) / 3 + 2 * sqrt(11) / 3) * (
                 -sigma ** (-sqrt(3) + sqrt(7)) * (-s - 3) / (1 - 3 * s) - sigma ** (
                     -sqrt(11) - sqrt(3) + 2 * sqrt(7)) * (-47 * s / 31 - 47 / 31) / (1 - 3 * s) - (
                             -10 * exp(-sqrt(3) / 3 + sqrt(7) / 3) / 9 - 188 * exp(
                         -sqrt(11) / 3 - sqrt(3) / 3 + 2 * sqrt(7) / 3) / 279) / (s - 1 / 3)) * (
                 sigma ** (sqrt(3)) * (1 - 3 * s) + 9 * (-10 * s - 42) / sigma ** (sqrt(7)) + 9 * (
                     22 * s - 8) / sigma ** (sqrt(7)) + 2 * (25 * s + 27) / sigma ** (sqrt(11))) + sigma ** (
                 -2 * sqrt(7) / 3 + 2 * sqrt(11) / 3) * (
                 sigma ** (sqrt(3)) * (1 - 3 * s) + 9 * (-10 * s - 42) / sigma ** (sqrt(7)) + 9 * (
                     22 * s - 8) / sigma ** (sqrt(7)) + 2 * (25 * s + 27) / sigma ** (sqrt(11))) * (
                 -4 * sigma ** (-sqrt(7) + sqrt(11)) * (7 * s - 19) * exp(
             -sqrt(11) / 3 - sqrt(3) / 3 + 2 * sqrt(7) / 3) / (3 * (1 - 3 * s) * (s - 7)) - 1600 * sigma ** (
                             -5 * sqrt(7) + 2 * sqrt(3) + 3 * sqrt(11)) * (7 * s - 19) * exp(
             -sqrt(11) - sqrt(3) + 2 * sqrt(7)) / (
                             27 * (1 - 3 * s) * (s ** 3 - 21 * s ** 2 + 147 * s - 343)) - 80 * sigma ** (
                             -3 * sqrt(7) + sqrt(3) + 2 * sqrt(11)) * (19 - 7 * s) * exp(
             -2 * sqrt(11) / 3 - 2 * sqrt(3) / 3 + 4 * sqrt(7) / 3) / (9 * (1 - 3 * s) * (s ** 2 - 14 * s + 49)) - (
                             -(s - 7) ** 2 * (-4 * exp(-sqrt(11) / 3) * exp(-sqrt(3) / 3) * exp(2 * sqrt(7) / 3) * exp(
                         -7 * sqrt(7) + 7 * sqrt(11)) - 800 * (-5 * sqrt(7) + 2 * sqrt(3) + 3 * sqrt(11)) ** 2 * exp(
                         -sqrt(11)) * exp(-sqrt(3)) * exp(2 * sqrt(7)) * exp(
                         -35 * sqrt(7) + 14 * sqrt(3) + 21 * sqrt(11)) / 9 - 400 * (
                                                          -5 * sqrt(7) + 2 * sqrt(3) + 3 * sqrt(11)) * exp(
                         -sqrt(11)) * exp(-sqrt(3)) * exp(2 * sqrt(7)) * exp(
                         -35 * sqrt(7) + 14 * sqrt(3) + 21 * sqrt(11)) / 27 - 20 * exp(-2 * sqrt(11) / 3) * exp(
                         -2 * sqrt(3) / 3) * exp(4 * sqrt(7) / 3) * exp(
                         -sqrt(7) + sqrt(3) / 3 + 2 * sqrt(11) / 3) / 9 - 20 * exp(-sqrt(11)) * exp(-sqrt(3)) * exp(
                         2 * sqrt(7)) * exp(-5 * sqrt(7) / 3 + 2 * sqrt(3) / 3 + sqrt(11)) / 9 - 20 * exp(
                         -sqrt(11) / 3) * exp(-sqrt(3) / 3) * exp(2 * sqrt(7) / 3) * exp(
                         -sqrt(7) / 3 + sqrt(11) / 3) / 9 + 20 * exp(-sqrt(11)) * exp(-sqrt(3)) * exp(
                         2 * sqrt(7)) * exp(-35 * sqrt(7) + 14 * sqrt(3) + 21 * sqrt(11)) / 9 + 20 * exp(
                         -2 * sqrt(11) / 3) * exp(-2 * sqrt(3) / 3) * exp(4 * sqrt(7) / 3) * exp(
                         -21 * sqrt(7) + 7 * sqrt(3) + 14 * sqrt(11)) / 9 + 80 * (
                                                          -3 * sqrt(7) + sqrt(3) + 2 * sqrt(11)) * exp(
                         -2 * sqrt(11) / 3) * exp(-2 * sqrt(3) / 3) * exp(4 * sqrt(7) / 3) * exp(
                         -21 * sqrt(7) + 7 * sqrt(3) + 14 * sqrt(11)) / 3) / 2 - (s - 7) * (
                                         -800 * (-5 * sqrt(7) + 2 * sqrt(3) + 3 * sqrt(11)) * exp(-sqrt(11)) * exp(
                                     -sqrt(3)) * exp(2 * sqrt(7)) * exp(
                                     -35 * sqrt(7) + 14 * sqrt(3) + 21 * sqrt(11)) / 9 - 200 * exp(-sqrt(11)) * exp(
                                     -sqrt(3)) * exp(2 * sqrt(7)) * exp(
                                     -35 * sqrt(7) + 14 * sqrt(3) + 21 * sqrt(11)) / 27 + 200 * exp(
                                     -sqrt(11) / 3) * exp(-sqrt(3) / 3) * exp(2 * sqrt(7) / 3) * exp(
                                     -sqrt(7) / 3 + sqrt(11) / 3) / 27 + 200 * exp(-sqrt(11)) * exp(-sqrt(3)) * exp(
                                     2 * sqrt(7)) * exp(-5 * sqrt(7) / 3 + 2 * sqrt(3) / 3 + sqrt(11)) / 27 + 200 * exp(
                                     -2 * sqrt(11) / 3) * exp(-2 * sqrt(3) / 3) * exp(4 * sqrt(7) / 3) * exp(
                                     -sqrt(7) + sqrt(3) / 3 + 2 * sqrt(11) / 3) / 27 + 40 * exp(
                                     -2 * sqrt(11) / 3) * exp(-2 * sqrt(3) / 3) * exp(4 * sqrt(7) / 3) * exp(
                                     -21 * sqrt(7) + 7 * sqrt(3) + 14 * sqrt(11)) / 3) + 4000 * exp(-sqrt(11)) * exp(
                         -sqrt(3)) * exp(2 * sqrt(7)) * exp(
                         -5 * sqrt(7) / 3 + 2 * sqrt(3) / 3 + sqrt(11)) / 81 + 4000 * exp(-2 * sqrt(11) / 3) * exp(
                         -2 * sqrt(3) / 3) * exp(4 * sqrt(7) / 3) * exp(
                         -sqrt(7) + sqrt(3) / 3 + 2 * sqrt(11) / 3) / 81 + 4000 * exp(-sqrt(11) / 3) * exp(
                         -sqrt(3) / 3) * exp(2 * sqrt(7) / 3) * exp(-sqrt(7) / 3 + sqrt(11) / 3) / 81 + 800 * exp(
                         -sqrt(11)) * exp(-sqrt(3)) * exp(2 * sqrt(7)) * exp(
                         -35 * sqrt(7) + 14 * sqrt(3) + 21 * sqrt(11)) / 9 + (
                                         -80000 * exp(-2 * sqrt(11) / 3) * exp(-2 * sqrt(3) / 3) * exp(
                                     4 * sqrt(7) / 3) * exp(
                                     -sqrt(7) + sqrt(3) / 3 + 2 * sqrt(11) / 3) / 243 - 80000 * exp(
                                     -sqrt(11) / 3) * exp(-sqrt(3) / 3) * exp(2 * sqrt(7) / 3) * exp(
                                     -sqrt(7) / 3 + sqrt(11) / 3) / 243 - 80000 * exp(-sqrt(11)) * exp(-sqrt(3)) * exp(
                                     2 * sqrt(7)) * exp(-5 * sqrt(7) / 3 + 2 * sqrt(3) / 3 + sqrt(11)) / 243) / (
                                         s - 1 / 3)) / (s - 7) ** 3) + (sigma ** (-sqrt(11) / 3 + sqrt(7) / 3) * (
                5 * sigma ** (sqrt(3)) * (1 - 3 * s) / 2 + sigma ** (sqrt(7)) * (-s - 1) + 51 * (-10 * s - 42) / (
                    2 * sigma ** (sqrt(7))) + 31 * (22 * s - 8) / (2 * sigma ** (sqrt(7)))) + sigma ** (
                                                                                    -2 * sqrt(7) / 3 + 2 * sqrt(
                                                                                11) / 3) * (-sigma ** (
                -sqrt(11) - sqrt(3) + 2 * sqrt(7)) * (-s - 1) / (1 - 3 * s) + 4 * exp(
        -sqrt(11) / 3 - sqrt(3) / 3 + 2 * sqrt(7) / 3) / (9 * (s - 1 / 3))) * (sigma ** (sqrt(3)) * (1 - 3 * s) + 9 * (
                -10 * s - 42) / sigma ** (sqrt(7)) + 9 * (22 * s - 8) / sigma ** (sqrt(7)) + 2 * (
                                                                                           25 * s + 27) / sigma ** (
                                                                                   sqrt(11)))) * (
                 -sigma ** (-5 * sqrt(7) + 2 * sqrt(3) + 3 * sqrt(11)) * (3 * s - 1) * (
                     400 * (19 - 7 * s) * exp(-2 * sqrt(11) / 3 - 2 * sqrt(3) / 3 + 4 * sqrt(7) / 3) / (
                         9 * (3 * s ** 3 - 43 * s ** 2 + 161 * s - 49)) - 400 * (7 * s - 19) * exp(
                 -2 * sqrt(11) / 3 - 2 * sqrt(3) / 3 + 4 * sqrt(7) / 3) / (9 * (s - 7) * (3 * s ** 2 - 22 * s + 7))) / (
                             s - 7) - 20 * sigma ** (-3 * sqrt(7) + sqrt(3) + 2 * sqrt(11)) * (3 * s - 1) * (
                             7 * s - 19) * exp(-sqrt(11) / 3 - sqrt(3) / 3 + 2 * sqrt(7) / 3) / (
                             3 * (s - 7) * (3 * s ** 2 - 22 * s + 7)) - (-(s - 7) ** 4 * (
                     -44800 * (-5 * sqrt(7) + 2 * sqrt(3) + 3 * sqrt(11)) * exp(-2 * sqrt(11) / 3) * exp(
                 -2 * sqrt(3) / 3) * exp(4 * sqrt(7) / 3) * exp(
                 -35 * sqrt(7) + 14 * sqrt(3) + 21 * sqrt(11)) / 3 - 32000 * (
                                 -5 * sqrt(7) + 2 * sqrt(3) + 3 * sqrt(11)) ** 2 * exp(-2 * sqrt(11) / 3) * exp(
                 -2 * sqrt(3) / 3) * exp(4 * sqrt(7) / 3) * exp(
                 -35 * sqrt(7) + 14 * sqrt(3) + 21 * sqrt(11)) + 1120 * exp(-sqrt(11) / 3) * exp(-sqrt(3) / 3) * exp(
                 2 * sqrt(7) / 3) * exp(-21 * sqrt(7) + 7 * sqrt(3) + 14 * sqrt(11)) + 4800 * (
                                 -3 * sqrt(7) + sqrt(3) + 2 * sqrt(11)) * exp(-sqrt(11) / 3) * exp(-sqrt(3) / 3) * exp(
                 2 * sqrt(7) / 3) * exp(-21 * sqrt(7) + 7 * sqrt(3) + 14 * sqrt(11))) / 24 - (s - 7) ** 3 * (
                                                                                     -11200 * exp(
                                                                                 -2 * sqrt(11) / 3) * exp(
                                                                                 -2 * sqrt(3) / 3) * exp(
                                                                                 4 * sqrt(7) / 3) * exp(
                                                                                 -35 * sqrt(7) + 14 * sqrt(
                                                                                     3) + 21 * sqrt(11)) / 3 - 16000 * (
                                                                                                 -5 * sqrt(
                                                                                             7) + 2 * sqrt(
                                                                                             3) + 3 * sqrt(11)) * exp(
                                                                                 -2 * sqrt(11) / 3) * exp(
                                                                                 -2 * sqrt(3) / 3) * exp(
                                                                                 4 * sqrt(7) / 3) * exp(
                                                                                 -35 * sqrt(7) + 14 * sqrt(
                                                                                     3) + 21 * sqrt(11)) + 1200 * exp(
                                                                                 -sqrt(11) / 3) * exp(
                                                                                 -sqrt(3) / 3) * exp(
                                                                                 2 * sqrt(7) / 3) * exp(
                                                                                 -21 * sqrt(7) + 7 * sqrt(
                                                                                     3) + 14 * sqrt(11))) / 6 + 8000 * (
                                                                                     s - 7) ** 2 * exp(
             -2 * sqrt(11) / 3) * exp(-2 * sqrt(3) / 3) * exp(4 * sqrt(7) / 3) * exp(
             -35 * sqrt(7) + 14 * sqrt(3) + 21 * sqrt(11)) / 3) / (s - 7) ** 5) + (
                 sigma ** (-sqrt(11) / 3 + sqrt(7) / 3) * (
                     5 * sigma ** (sqrt(3)) * (1 - 3 * s) / 2 + sigma ** (sqrt(7)) * (-s - 1) + 51 * (-10 * s - 42) / (
                         2 * sigma ** (sqrt(7))) + 31 * (22 * s - 8) / (2 * sigma ** (sqrt(7)))) + sigma ** (
                             -2 * sqrt(7) / 3 + 2 * sqrt(11) / 3) * (
                             -sigma ** (-sqrt(11) - sqrt(3) + 2 * sqrt(7)) * (-s - 1) / (1 - 3 * s) + 4 * exp(
                         -sqrt(11) / 3 - sqrt(3) / 3 + 2 * sqrt(7) / 3) / (9 * (s - 1 / 3))) * (
                             sigma ** (sqrt(3)) * (1 - 3 * s) + 9 * (-10 * s - 42) / sigma ** (sqrt(7)) + 9 * (
                                 22 * s - 8) / sigma ** (sqrt(7)) + 2 * (25 * s + 27) / sigma ** (sqrt(11)))) * (
                 -sigma ** (-sqrt(7) + sqrt(11)) * (7 * s - 19) / (s - 7) - 400 * sigma ** (
                     -5 * sqrt(7) + 2 * sqrt(3) + 3 * sqrt(11)) * (7 * s - 19) * exp(
             -2 * sqrt(11) / 3 - 2 * sqrt(3) / 3 + 4 * sqrt(7) / 3) / (9 * (s - 7) ** 3) + 20 * sigma ** (
                             -3 * sqrt(7) + sqrt(3) + 2 * sqrt(11)) * (7 * s - 19) * exp(
             -sqrt(11) / 3 - sqrt(3) / 3 + 2 * sqrt(7) / 3) / (3 * (s - 7) ** 2) - (-(s - 7) ** 2 * (
                     -400 * (-3 * sqrt(7) + sqrt(3) + 2 * sqrt(11)) * exp(-sqrt(11) / 3) * exp(-sqrt(3) / 3) * exp(
                 2 * sqrt(7) / 3) * exp(-21 * sqrt(7) + 7 * sqrt(3) + 14 * sqrt(11)) - 280 * exp(-sqrt(11) / 3) * exp(
                 -sqrt(3) / 3) * exp(2 * sqrt(7) / 3) * exp(-21 * sqrt(7) + 7 * sqrt(3) + 14 * sqrt(11)) / 3 + 4000 * (
                                 -5 * sqrt(7) + 2 * sqrt(3) + 3 * sqrt(11)) ** 2 * exp(-2 * sqrt(11) / 3) * exp(
                 -2 * sqrt(3) / 3) * exp(4 * sqrt(7) / 3) * exp(
                 -35 * sqrt(7) + 14 * sqrt(3) + 21 * sqrt(11)) / 3 + 5600 * (
                                 -5 * sqrt(7) + 2 * sqrt(3) + 3 * sqrt(11)) * exp(-2 * sqrt(11) / 3) * exp(
                 -2 * sqrt(3) / 3) * exp(4 * sqrt(7) / 3) * exp(
                 -35 * sqrt(7) + 14 * sqrt(3) + 21 * sqrt(11)) / 9 + 60 * exp(-7 * sqrt(7) + 7 * sqrt(11))) / 2 - (
                                                                                                s - 7) * (-200 * exp(
             -sqrt(11) / 3) * exp(-sqrt(3) / 3) * exp(2 * sqrt(7) / 3) * exp(
             -21 * sqrt(7) + 7 * sqrt(3) + 14 * sqrt(11)) + 4000 * (-5 * sqrt(7) + 2 * sqrt(3) + 3 * sqrt(11)) * exp(
             -2 * sqrt(11) / 3) * exp(-2 * sqrt(3) / 3) * exp(4 * sqrt(7) / 3) * exp(
             -35 * sqrt(7) + 14 * sqrt(3) + 21 * sqrt(11)) / 3 + 2800 * exp(-2 * sqrt(11) / 3) * exp(
             -2 * sqrt(3) / 3) * exp(4 * sqrt(7) / 3) * exp(
             -35 * sqrt(7) + 14 * sqrt(3) + 21 * sqrt(11)) / 9) - 4000 * exp(-2 * sqrt(11) / 3) * exp(
             -2 * sqrt(3) / 3) * exp(4 * sqrt(7) / 3) * exp(-35 * sqrt(7) + 14 * sqrt(3) + 21 * sqrt(11)) / 3) / (
                             s - 7) ** 3), sigma ** (-sqrt(11) / 3 + sqrt(7) / 3) * (
                 4 * sigma ** (sqrt(11)) + 45 * sigma ** (sqrt(3)) / 31 + 94 * sigma ** (sqrt(7)) / 31 + 13545 / (
                     31 * sigma ** (sqrt(7)))) + sigma ** (-2 * sqrt(7) / 3 + 2 * sqrt(11) / 3) * (
                 5 * sigma ** (sqrt(3)) + 18 / sigma ** (sqrt(7)) - 122 / sigma ** (sqrt(11))) * (
                 -sigma ** (-sqrt(3) + sqrt(7)) * (-s - 3) / (1 - 3 * s) - sigma ** (
                     -sqrt(11) - sqrt(3) + 2 * sqrt(7)) * (-47 * s / 31 - 47 / 31) / (1 - 3 * s) - (
                             -10 * exp(-sqrt(3) / 3 + sqrt(7) / 3) / 9 - 188 * exp(
                         -sqrt(11) / 3 - sqrt(3) / 3 + 2 * sqrt(7) / 3) / 279) / (s - 1 / 3)) + sigma ** (
                 -2 * sqrt(7) / 3 + 2 * sqrt(11) / 3) * (
                 5 * sigma ** (sqrt(3)) + 18 / sigma ** (sqrt(7)) - 122 / sigma ** (sqrt(11))) * (
                 -4 * sigma ** (-sqrt(7) + sqrt(11)) * (7 * s - 19) * exp(
             -sqrt(11) / 3 - sqrt(3) / 3 + 2 * sqrt(7) / 3) / (3 * (1 - 3 * s) * (s - 7)) - 1600 * sigma ** (
                             -5 * sqrt(7) + 2 * sqrt(3) + 3 * sqrt(11)) * (7 * s - 19) * exp(
             -sqrt(11) - sqrt(3) + 2 * sqrt(7)) / (
                             27 * (1 - 3 * s) * (s ** 3 - 21 * s ** 2 + 147 * s - 343)) - 80 * sigma ** (
                             -3 * sqrt(7) + sqrt(3) + 2 * sqrt(11)) * (19 - 7 * s) * exp(
             -2 * sqrt(11) / 3 - 2 * sqrt(3) / 3 + 4 * sqrt(7) / 3) / (9 * (1 - 3 * s) * (s ** 2 - 14 * s + 49)) - (
                             -(s - 7) ** 2 * (-4 * exp(-sqrt(11) / 3) * exp(-sqrt(3) / 3) * exp(2 * sqrt(7) / 3) * exp(
                         -7 * sqrt(7) + 7 * sqrt(11)) - 800 * (-5 * sqrt(7) + 2 * sqrt(3) + 3 * sqrt(11)) ** 2 * exp(
                         -sqrt(11)) * exp(-sqrt(3)) * exp(2 * sqrt(7)) * exp(
                         -35 * sqrt(7) + 14 * sqrt(3) + 21 * sqrt(11)) / 9 - 400 * (
                                                          -5 * sqrt(7) + 2 * sqrt(3) + 3 * sqrt(11)) * exp(
                         -sqrt(11)) * exp(-sqrt(3)) * exp(2 * sqrt(7)) * exp(
                         -35 * sqrt(7) + 14 * sqrt(3) + 21 * sqrt(11)) / 27 - 20 * exp(-2 * sqrt(11) / 3) * exp(
                         -2 * sqrt(3) / 3) * exp(4 * sqrt(7) / 3) * exp(
                         -sqrt(7) + sqrt(3) / 3 + 2 * sqrt(11) / 3) / 9 - 20 * exp(-sqrt(11)) * exp(-sqrt(3)) * exp(
                         2 * sqrt(7)) * exp(-5 * sqrt(7) / 3 + 2 * sqrt(3) / 3 + sqrt(11)) / 9 - 20 * exp(
                         -sqrt(11) / 3) * exp(-sqrt(3) / 3) * exp(2 * sqrt(7) / 3) * exp(
                         -sqrt(7) / 3 + sqrt(11) / 3) / 9 + 20 * exp(-sqrt(11)) * exp(-sqrt(3)) * exp(
                         2 * sqrt(7)) * exp(-35 * sqrt(7) + 14 * sqrt(3) + 21 * sqrt(11)) / 9 + 20 * exp(
                         -2 * sqrt(11) / 3) * exp(-2 * sqrt(3) / 3) * exp(4 * sqrt(7) / 3) * exp(
                         -21 * sqrt(7) + 7 * sqrt(3) + 14 * sqrt(11)) / 9 + 80 * (
                                                          -3 * sqrt(7) + sqrt(3) + 2 * sqrt(11)) * exp(
                         -2 * sqrt(11) / 3) * exp(-2 * sqrt(3) / 3) * exp(4 * sqrt(7) / 3) * exp(
                         -21 * sqrt(7) + 7 * sqrt(3) + 14 * sqrt(11)) / 3) / 2 - (s - 7) * (
                                         -800 * (-5 * sqrt(7) + 2 * sqrt(3) + 3 * sqrt(11)) * exp(-sqrt(11)) * exp(
                                     -sqrt(3)) * exp(2 * sqrt(7)) * exp(
                                     -35 * sqrt(7) + 14 * sqrt(3) + 21 * sqrt(11)) / 9 - 200 * exp(-sqrt(11)) * exp(
                                     -sqrt(3)) * exp(2 * sqrt(7)) * exp(
                                     -35 * sqrt(7) + 14 * sqrt(3) + 21 * sqrt(11)) / 27 + 200 * exp(
                                     -sqrt(11) / 3) * exp(-sqrt(3) / 3) * exp(2 * sqrt(7) / 3) * exp(
                                     -sqrt(7) / 3 + sqrt(11) / 3) / 27 + 200 * exp(-sqrt(11)) * exp(-sqrt(3)) * exp(
                                     2 * sqrt(7)) * exp(-5 * sqrt(7) / 3 + 2 * sqrt(3) / 3 + sqrt(11)) / 27 + 200 * exp(
                                     -2 * sqrt(11) / 3) * exp(-2 * sqrt(3) / 3) * exp(4 * sqrt(7) / 3) * exp(
                                     -sqrt(7) + sqrt(3) / 3 + 2 * sqrt(11) / 3) / 27 + 40 * exp(
                                     -2 * sqrt(11) / 3) * exp(-2 * sqrt(3) / 3) * exp(4 * sqrt(7) / 3) * exp(
                                     -21 * sqrt(7) + 7 * sqrt(3) + 14 * sqrt(11)) / 3) + 4000 * exp(-sqrt(11)) * exp(
                         -sqrt(3)) * exp(2 * sqrt(7)) * exp(
                         -5 * sqrt(7) / 3 + 2 * sqrt(3) / 3 + sqrt(11)) / 81 + 4000 * exp(-2 * sqrt(11) / 3) * exp(
                         -2 * sqrt(3) / 3) * exp(4 * sqrt(7) / 3) * exp(
                         -sqrt(7) + sqrt(3) / 3 + 2 * sqrt(11) / 3) / 81 + 4000 * exp(-sqrt(11) / 3) * exp(
                         -sqrt(3) / 3) * exp(2 * sqrt(7) / 3) * exp(-sqrt(7) / 3 + sqrt(11) / 3) / 81 + 800 * exp(
                         -sqrt(11)) * exp(-sqrt(3)) * exp(2 * sqrt(7)) * exp(
                         -35 * sqrt(7) + 14 * sqrt(3) + 21 * sqrt(11)) / 9 + (
                                         -80000 * exp(-2 * sqrt(11) / 3) * exp(-2 * sqrt(3) / 3) * exp(
                                     4 * sqrt(7) / 3) * exp(
                                     -sqrt(7) + sqrt(3) / 3 + 2 * sqrt(11) / 3) / 243 - 80000 * exp(
                                     -sqrt(11) / 3) * exp(-sqrt(3) / 3) * exp(2 * sqrt(7) / 3) * exp(
                                     -sqrt(7) / 3 + sqrt(11) / 3) / 243 - 80000 * exp(-sqrt(11)) * exp(-sqrt(3)) * exp(
                                     2 * sqrt(7)) * exp(-5 * sqrt(7) / 3 + 2 * sqrt(3) / 3 + sqrt(11)) / 243) / (
                                         s - 1 / 3)) / (s - 7) ** 3) + (sigma ** (-sqrt(11) / 3 + sqrt(7) / 3) * (
                25 * sigma ** (sqrt(3)) / 2 + 2 * sigma ** (sqrt(7)) + 461 / sigma ** (sqrt(7))) + sigma ** (
                                                                                    -2 * sqrt(7) / 3 + 2 * sqrt(
                                                                                11) / 3) * (-sigma ** (
                -sqrt(11) - sqrt(3) + 2 * sqrt(7)) * (-s - 1) / (1 - 3 * s) + 4 * exp(
        -sqrt(11) / 3 - sqrt(3) / 3 + 2 * sqrt(7) / 3) / (9 * (s - 1 / 3))) * (5 * sigma ** (sqrt(3)) + 18 / sigma ** (
        sqrt(7)) - 122 / sigma ** (sqrt(11)))) * (
                 -sigma ** (-5 * sqrt(7) + 2 * sqrt(3) + 3 * sqrt(11)) * (3 * s - 1) * (
                     400 * (19 - 7 * s) * exp(-2 * sqrt(11) / 3 - 2 * sqrt(3) / 3 + 4 * sqrt(7) / 3) / (
                         9 * (3 * s ** 3 - 43 * s ** 2 + 161 * s - 49)) - 400 * (7 * s - 19) * exp(
                 -2 * sqrt(11) / 3 - 2 * sqrt(3) / 3 + 4 * sqrt(7) / 3) / (9 * (s - 7) * (3 * s ** 2 - 22 * s + 7))) / (
                             s - 7) - 20 * sigma ** (-3 * sqrt(7) + sqrt(3) + 2 * sqrt(11)) * (3 * s - 1) * (
                             7 * s - 19) * exp(-sqrt(11) / 3 - sqrt(3) / 3 + 2 * sqrt(7) / 3) / (
                             3 * (s - 7) * (3 * s ** 2 - 22 * s + 7)) - (-(s - 7) ** 4 * (
                     -44800 * (-5 * sqrt(7) + 2 * sqrt(3) + 3 * sqrt(11)) * exp(-2 * sqrt(11) / 3) * exp(
                 -2 * sqrt(3) / 3) * exp(4 * sqrt(7) / 3) * exp(
                 -35 * sqrt(7) + 14 * sqrt(3) + 21 * sqrt(11)) / 3 - 32000 * (
                                 -5 * sqrt(7) + 2 * sqrt(3) + 3 * sqrt(11)) ** 2 * exp(-2 * sqrt(11) / 3) * exp(
                 -2 * sqrt(3) / 3) * exp(4 * sqrt(7) / 3) * exp(
                 -35 * sqrt(7) + 14 * sqrt(3) + 21 * sqrt(11)) + 1120 * exp(-sqrt(11) / 3) * exp(-sqrt(3) / 3) * exp(
                 2 * sqrt(7) / 3) * exp(-21 * sqrt(7) + 7 * sqrt(3) + 14 * sqrt(11)) + 4800 * (
                                 -3 * sqrt(7) + sqrt(3) + 2 * sqrt(11)) * exp(-sqrt(11) / 3) * exp(-sqrt(3) / 3) * exp(
                 2 * sqrt(7) / 3) * exp(-21 * sqrt(7) + 7 * sqrt(3) + 14 * sqrt(11))) / 24 - (s - 7) ** 3 * (
                                                                                     -11200 * exp(
                                                                                 -2 * sqrt(11) / 3) * exp(
                                                                                 -2 * sqrt(3) / 3) * exp(
                                                                                 4 * sqrt(7) / 3) * exp(
                                                                                 -35 * sqrt(7) + 14 * sqrt(
                                                                                     3) + 21 * sqrt(11)) / 3 - 16000 * (
                                                                                                 -5 * sqrt(
                                                                                             7) + 2 * sqrt(
                                                                                             3) + 3 * sqrt(11)) * exp(
                                                                                 -2 * sqrt(11) / 3) * exp(
                                                                                 -2 * sqrt(3) / 3) * exp(
                                                                                 4 * sqrt(7) / 3) * exp(
                                                                                 -35 * sqrt(7) + 14 * sqrt(
                                                                                     3) + 21 * sqrt(11)) + 1200 * exp(
                                                                                 -sqrt(11) / 3) * exp(
                                                                                 -sqrt(3) / 3) * exp(
                                                                                 2 * sqrt(7) / 3) * exp(
                                                                                 -21 * sqrt(7) + 7 * sqrt(
                                                                                     3) + 14 * sqrt(11))) / 6 + 8000 * (
                                                                                     s - 7) ** 2 * exp(
             -2 * sqrt(11) / 3) * exp(-2 * sqrt(3) / 3) * exp(4 * sqrt(7) / 3) * exp(
             -35 * sqrt(7) + 14 * sqrt(3) + 21 * sqrt(11)) / 3) / (s - 7) ** 5) + (
                 sigma ** (-sqrt(11) / 3 + sqrt(7) / 3) * (
                     25 * sigma ** (sqrt(3)) / 2 + 2 * sigma ** (sqrt(7)) + 461 / sigma ** (sqrt(7))) + sigma ** (
                             -2 * sqrt(7) / 3 + 2 * sqrt(11) / 3) * (
                             -sigma ** (-sqrt(11) - sqrt(3) + 2 * sqrt(7)) * (-s - 1) / (1 - 3 * s) + 4 * exp(
                         -sqrt(11) / 3 - sqrt(3) / 3 + 2 * sqrt(7) / 3) / (9 * (s - 1 / 3))) * (
                             5 * sigma ** (sqrt(3)) + 18 / sigma ** (sqrt(7)) - 122 / sigma ** (sqrt(11)))) * (
                 -sigma ** (-sqrt(7) + sqrt(11)) * (7 * s - 19) / (s - 7) - 400 * sigma ** (
                     -5 * sqrt(7) + 2 * sqrt(3) + 3 * sqrt(11)) * (7 * s - 19) * exp(
             -2 * sqrt(11) / 3 - 2 * sqrt(3) / 3 + 4 * sqrt(7) / 3) / (9 * (s - 7) ** 3) + 20 * sigma ** (
                             -3 * sqrt(7) + sqrt(3) + 2 * sqrt(11)) * (7 * s - 19) * exp(
             -sqrt(11) / 3 - sqrt(3) / 3 + 2 * sqrt(7) / 3) / (3 * (s - 7) ** 2) - (-(s - 7) ** 2 * (
                     -400 * (-3 * sqrt(7) + sqrt(3) + 2 * sqrt(11)) * exp(-sqrt(11) / 3) * exp(-sqrt(3) / 3) * exp(
                 2 * sqrt(7) / 3) * exp(-21 * sqrt(7) + 7 * sqrt(3) + 14 * sqrt(11)) - 280 * exp(-sqrt(11) / 3) * exp(
                 -sqrt(3) / 3) * exp(2 * sqrt(7) / 3) * exp(-21 * sqrt(7) + 7 * sqrt(3) + 14 * sqrt(11)) / 3 + 4000 * (
                                 -5 * sqrt(7) + 2 * sqrt(3) + 3 * sqrt(11)) ** 2 * exp(-2 * sqrt(11) / 3) * exp(
                 -2 * sqrt(3) / 3) * exp(4 * sqrt(7) / 3) * exp(
                 -35 * sqrt(7) + 14 * sqrt(3) + 21 * sqrt(11)) / 3 + 5600 * (
                                 -5 * sqrt(7) + 2 * sqrt(3) + 3 * sqrt(11)) * exp(-2 * sqrt(11) / 3) * exp(
                 -2 * sqrt(3) / 3) * exp(4 * sqrt(7) / 3) * exp(
                 -35 * sqrt(7) + 14 * sqrt(3) + 21 * sqrt(11)) / 9 + 60 * exp(-7 * sqrt(7) + 7 * sqrt(11))) / 2 - (
                                                                                                s - 7) * (-200 * exp(
             -sqrt(11) / 3) * exp(-sqrt(3) / 3) * exp(2 * sqrt(7) / 3) * exp(
             -21 * sqrt(7) + 7 * sqrt(3) + 14 * sqrt(11)) + 4000 * (-5 * sqrt(7) + 2 * sqrt(3) + 3 * sqrt(11)) * exp(
             -2 * sqrt(11) / 3) * exp(-2 * sqrt(3) / 3) * exp(4 * sqrt(7) / 3) * exp(
             -35 * sqrt(7) + 14 * sqrt(3) + 21 * sqrt(11)) / 3 + 2800 * exp(-2 * sqrt(11) / 3) * exp(
             -2 * sqrt(3) / 3) * exp(4 * sqrt(7) / 3) * exp(
             -35 * sqrt(7) + 14 * sqrt(3) + 21 * sqrt(11)) / 9) - 4000 * exp(-2 * sqrt(11) / 3) * exp(
             -2 * sqrt(3) / 3) * exp(4 * sqrt(7) / 3) * exp(-35 * sqrt(7) + 14 * sqrt(3) + 21 * sqrt(11)) / 3) / (
                             s - 7) ** 3), sigma ** (-sqrt(11) / 3 + sqrt(7) / 3) * (
                 -2 * sigma ** (sqrt(11)) - 27 * sigma ** (sqrt(3)) / 31 + 235 * sigma ** (sqrt(7)) / 31 + 10710 / (
                     31 * sigma ** (sqrt(7)))) + sigma ** (-2 * sqrt(7) / 3 + 2 * sqrt(11) / 3) * (
                 -3 * sigma ** (sqrt(3)) + 963 / sigma ** (sqrt(7)) - 44 / sigma ** (sqrt(11))) * (
                 -sigma ** (-sqrt(3) + sqrt(7)) * (-s - 3) / (1 - 3 * s) - sigma ** (
                     -sqrt(11) - sqrt(3) + 2 * sqrt(7)) * (-47 * s / 31 - 47 / 31) / (1 - 3 * s) - (
                             -10 * exp(-sqrt(3) / 3 + sqrt(7) / 3) / 9 - 188 * exp(
                         -sqrt(11) / 3 - sqrt(3) / 3 + 2 * sqrt(7) / 3) / 279) / (s - 1 / 3)) + sigma ** (
                 -2 * sqrt(7) / 3 + 2 * sqrt(11) / 3) * (
                 -3 * sigma ** (sqrt(3)) + 963 / sigma ** (sqrt(7)) - 44 / sigma ** (sqrt(11))) * (
                 -4 * sigma ** (-sqrt(7) + sqrt(11)) * (7 * s - 19) * exp(
             -sqrt(11) / 3 - sqrt(3) / 3 + 2 * sqrt(7) / 3) / (3 * (1 - 3 * s) * (s - 7)) - 1600 * sigma ** (
                             -5 * sqrt(7) + 2 * sqrt(3) + 3 * sqrt(11)) * (7 * s - 19) * exp(
             -sqrt(11) - sqrt(3) + 2 * sqrt(7)) / (
                             27 * (1 - 3 * s) * (s ** 3 - 21 * s ** 2 + 147 * s - 343)) - 80 * sigma ** (
                             -3 * sqrt(7) + sqrt(3) + 2 * sqrt(11)) * (19 - 7 * s) * exp(
             -2 * sqrt(11) / 3 - 2 * sqrt(3) / 3 + 4 * sqrt(7) / 3) / (9 * (1 - 3 * s) * (s ** 2 - 14 * s + 49)) - (
                             -(s - 7) ** 2 * (-4 * exp(-sqrt(11) / 3) * exp(-sqrt(3) / 3) * exp(2 * sqrt(7) / 3) * exp(
                         -7 * sqrt(7) + 7 * sqrt(11)) - 800 * (-5 * sqrt(7) + 2 * sqrt(3) + 3 * sqrt(11)) ** 2 * exp(
                         -sqrt(11)) * exp(-sqrt(3)) * exp(2 * sqrt(7)) * exp(
                         -35 * sqrt(7) + 14 * sqrt(3) + 21 * sqrt(11)) / 9 - 400 * (
                                                          -5 * sqrt(7) + 2 * sqrt(3) + 3 * sqrt(11)) * exp(
                         -sqrt(11)) * exp(-sqrt(3)) * exp(2 * sqrt(7)) * exp(
                         -35 * sqrt(7) + 14 * sqrt(3) + 21 * sqrt(11)) / 27 - 20 * exp(-2 * sqrt(11) / 3) * exp(
                         -2 * sqrt(3) / 3) * exp(4 * sqrt(7) / 3) * exp(
                         -sqrt(7) + sqrt(3) / 3 + 2 * sqrt(11) / 3) / 9 - 20 * exp(-sqrt(11)) * exp(-sqrt(3)) * exp(
                         2 * sqrt(7)) * exp(-5 * sqrt(7) / 3 + 2 * sqrt(3) / 3 + sqrt(11)) / 9 - 20 * exp(
                         -sqrt(11) / 3) * exp(-sqrt(3) / 3) * exp(2 * sqrt(7) / 3) * exp(
                         -sqrt(7) / 3 + sqrt(11) / 3) / 9 + 20 * exp(-sqrt(11)) * exp(-sqrt(3)) * exp(
                         2 * sqrt(7)) * exp(-35 * sqrt(7) + 14 * sqrt(3) + 21 * sqrt(11)) / 9 + 20 * exp(
                         -2 * sqrt(11) / 3) * exp(-2 * sqrt(3) / 3) * exp(4 * sqrt(7) / 3) * exp(
                         -21 * sqrt(7) + 7 * sqrt(3) + 14 * sqrt(11)) / 9 + 80 * (
                                                          -3 * sqrt(7) + sqrt(3) + 2 * sqrt(11)) * exp(
                         -2 * sqrt(11) / 3) * exp(-2 * sqrt(3) / 3) * exp(4 * sqrt(7) / 3) * exp(
                         -21 * sqrt(7) + 7 * sqrt(3) + 14 * sqrt(11)) / 3) / 2 - (s - 7) * (
                                         -800 * (-5 * sqrt(7) + 2 * sqrt(3) + 3 * sqrt(11)) * exp(-sqrt(11)) * exp(
                                     -sqrt(3)) * exp(2 * sqrt(7)) * exp(
                                     -35 * sqrt(7) + 14 * sqrt(3) + 21 * sqrt(11)) / 9 - 200 * exp(-sqrt(11)) * exp(
                                     -sqrt(3)) * exp(2 * sqrt(7)) * exp(
                                     -35 * sqrt(7) + 14 * sqrt(3) + 21 * sqrt(11)) / 27 + 200 * exp(
                                     -sqrt(11) / 3) * exp(-sqrt(3) / 3) * exp(2 * sqrt(7) / 3) * exp(
                                     -sqrt(7) / 3 + sqrt(11) / 3) / 27 + 200 * exp(-sqrt(11)) * exp(-sqrt(3)) * exp(
                                     2 * sqrt(7)) * exp(-5 * sqrt(7) / 3 + 2 * sqrt(3) / 3 + sqrt(11)) / 27 + 200 * exp(
                                     -2 * sqrt(11) / 3) * exp(-2 * sqrt(3) / 3) * exp(4 * sqrt(7) / 3) * exp(
                                     -sqrt(7) + sqrt(3) / 3 + 2 * sqrt(11) / 3) / 27 + 40 * exp(
                                     -2 * sqrt(11) / 3) * exp(-2 * sqrt(3) / 3) * exp(4 * sqrt(7) / 3) * exp(
                                     -21 * sqrt(7) + 7 * sqrt(3) + 14 * sqrt(11)) / 3) + 4000 * exp(-sqrt(11)) * exp(
                         -sqrt(3)) * exp(2 * sqrt(7)) * exp(
                         -5 * sqrt(7) / 3 + 2 * sqrt(3) / 3 + sqrt(11)) / 81 + 4000 * exp(-2 * sqrt(11) / 3) * exp(
                         -2 * sqrt(3) / 3) * exp(4 * sqrt(7) / 3) * exp(
                         -sqrt(7) + sqrt(3) / 3 + 2 * sqrt(11) / 3) / 81 + 4000 * exp(-sqrt(11) / 3) * exp(
                         -sqrt(3) / 3) * exp(2 * sqrt(7) / 3) * exp(-sqrt(7) / 3 + sqrt(11) / 3) / 81 + 800 * exp(
                         -sqrt(11)) * exp(-sqrt(3)) * exp(2 * sqrt(7)) * exp(
                         -35 * sqrt(7) + 14 * sqrt(3) + 21 * sqrt(11)) / 9 + (
                                         -80000 * exp(-2 * sqrt(11) / 3) * exp(-2 * sqrt(3) / 3) * exp(
                                     4 * sqrt(7) / 3) * exp(
                                     -sqrt(7) + sqrt(3) / 3 + 2 * sqrt(11) / 3) / 243 - 80000 * exp(
                                     -sqrt(11) / 3) * exp(-sqrt(3) / 3) * exp(2 * sqrt(7) / 3) * exp(
                                     -sqrt(7) / 3 + sqrt(11) / 3) / 243 - 80000 * exp(-sqrt(11)) * exp(-sqrt(3)) * exp(
                                     2 * sqrt(7)) * exp(-5 * sqrt(7) / 3 + 2 * sqrt(3) / 3 + sqrt(11)) / 243) / (
                                         s - 1 / 3)) / (s - 7) ** 3) + (sigma ** (-sqrt(11) / 3 + sqrt(7) / 3) * (
                -15 * sigma ** (sqrt(3)) / 2 + 5 * sigma ** (sqrt(7)) + 3997 / (2 * sigma ** (sqrt(7)))) + sigma ** (
                                                                                    -2 * sqrt(7) / 3 + 2 * sqrt(
                                                                                11) / 3) * (-sigma ** (
                -sqrt(11) - sqrt(3) + 2 * sqrt(7)) * (-s - 1) / (1 - 3 * s) + 4 * exp(
        -sqrt(11) / 3 - sqrt(3) / 3 + 2 * sqrt(7) / 3) / (9 * (s - 1 / 3))) * (-3 * sigma ** (
        sqrt(3)) + 963 / sigma ** (sqrt(7)) - 44 / sigma ** (sqrt(11)))) * (
                 -sigma ** (-5 * sqrt(7) + 2 * sqrt(3) + 3 * sqrt(11)) * (3 * s - 1) * (
                     400 * (19 - 7 * s) * exp(-2 * sqrt(11) / 3 - 2 * sqrt(3) / 3 + 4 * sqrt(7) / 3) / (
                         9 * (3 * s ** 3 - 43 * s ** 2 + 161 * s - 49)) - 400 * (7 * s - 19) * exp(
                 -2 * sqrt(11) / 3 - 2 * sqrt(3) / 3 + 4 * sqrt(7) / 3) / (9 * (s - 7) * (3 * s ** 2 - 22 * s + 7))) / (
                             s - 7) - 20 * sigma ** (-3 * sqrt(7) + sqrt(3) + 2 * sqrt(11)) * (3 * s - 1) * (
                             7 * s - 19) * exp(-sqrt(11) / 3 - sqrt(3) / 3 + 2 * sqrt(7) / 3) / (
                             3 * (s - 7) * (3 * s ** 2 - 22 * s + 7)) - (-(s - 7) ** 4 * (
                     -44800 * (-5 * sqrt(7) + 2 * sqrt(3) + 3 * sqrt(11)) * exp(-2 * sqrt(11) / 3) * exp(
                 -2 * sqrt(3) / 3) * exp(4 * sqrt(7) / 3) * exp(
                 -35 * sqrt(7) + 14 * sqrt(3) + 21 * sqrt(11)) / 3 - 32000 * (
                                 -5 * sqrt(7) + 2 * sqrt(3) + 3 * sqrt(11)) ** 2 * exp(-2 * sqrt(11) / 3) * exp(
                 -2 * sqrt(3) / 3) * exp(4 * sqrt(7) / 3) * exp(
                 -35 * sqrt(7) + 14 * sqrt(3) + 21 * sqrt(11)) + 1120 * exp(-sqrt(11) / 3) * exp(-sqrt(3) / 3) * exp(
                 2 * sqrt(7) / 3) * exp(-21 * sqrt(7) + 7 * sqrt(3) + 14 * sqrt(11)) + 4800 * (
                                 -3 * sqrt(7) + sqrt(3) + 2 * sqrt(11)) * exp(-sqrt(11) / 3) * exp(-sqrt(3) / 3) * exp(
                 2 * sqrt(7) / 3) * exp(-21 * sqrt(7) + 7 * sqrt(3) + 14 * sqrt(11))) / 24 - (s - 7) ** 3 * (
                                                                                     -11200 * exp(
                                                                                 -2 * sqrt(11) / 3) * exp(
                                                                                 -2 * sqrt(3) / 3) * exp(
                                                                                 4 * sqrt(7) / 3) * exp(
                                                                                 -35 * sqrt(7) + 14 * sqrt(
                                                                                     3) + 21 * sqrt(11)) / 3 - 16000 * (
                                                                                                 -5 * sqrt(
                                                                                             7) + 2 * sqrt(
                                                                                             3) + 3 * sqrt(11)) * exp(
                                                                                 -2 * sqrt(11) / 3) * exp(
                                                                                 -2 * sqrt(3) / 3) * exp(
                                                                                 4 * sqrt(7) / 3) * exp(
                                                                                 -35 * sqrt(7) + 14 * sqrt(
                                                                                     3) + 21 * sqrt(11)) + 1200 * exp(
                                                                                 -sqrt(11) / 3) * exp(
                                                                                 -sqrt(3) / 3) * exp(
                                                                                 2 * sqrt(7) / 3) * exp(
                                                                                 -21 * sqrt(7) + 7 * sqrt(
                                                                                     3) + 14 * sqrt(11))) / 6 + 8000 * (
                                                                                     s - 7) ** 2 * exp(
             -2 * sqrt(11) / 3) * exp(-2 * sqrt(3) / 3) * exp(4 * sqrt(7) / 3) * exp(
             -35 * sqrt(7) + 14 * sqrt(3) + 21 * sqrt(11)) / 3) / (s - 7) ** 5) + (
                 sigma ** (-sqrt(11) / 3 + sqrt(7) / 3) * (
                     -15 * sigma ** (sqrt(3)) / 2 + 5 * sigma ** (sqrt(7)) + 3997 / (
                         2 * sigma ** (sqrt(7)))) + sigma ** (-2 * sqrt(7) / 3 + 2 * sqrt(11) / 3) * (
                             -sigma ** (-sqrt(11) - sqrt(3) + 2 * sqrt(7)) * (-s - 1) / (1 - 3 * s) + 4 * exp(
                         -sqrt(11) / 3 - sqrt(3) / 3 + 2 * sqrt(7) / 3) / (9 * (s - 1 / 3))) * (
                             -3 * sigma ** (sqrt(3)) + 963 / sigma ** (sqrt(7)) - 44 / sigma ** (sqrt(11)))) * (
                 -sigma ** (-sqrt(7) + sqrt(11)) * (7 * s - 19) / (s - 7) - 400 * sigma ** (
                     -5 * sqrt(7) + 2 * sqrt(3) + 3 * sqrt(11)) * (7 * s - 19) * exp(
             -2 * sqrt(11) / 3 - 2 * sqrt(3) / 3 + 4 * sqrt(7) / 3) / (9 * (s - 7) ** 3) + 20 * sigma ** (
                             -3 * sqrt(7) + sqrt(3) + 2 * sqrt(11)) * (7 * s - 19) * exp(
             -sqrt(11) / 3 - sqrt(3) / 3 + 2 * sqrt(7) / 3) / (3 * (s - 7) ** 2) - (-(s - 7) ** 2 * (
                     -400 * (-3 * sqrt(7) + sqrt(3) + 2 * sqrt(11)) * exp(-sqrt(11) / 3) * exp(-sqrt(3) / 3) * exp(
                 2 * sqrt(7) / 3) * exp(-21 * sqrt(7) + 7 * sqrt(3) + 14 * sqrt(11)) - 280 * exp(-sqrt(11) / 3) * exp(
                 -sqrt(3) / 3) * exp(2 * sqrt(7) / 3) * exp(-21 * sqrt(7) + 7 * sqrt(3) + 14 * sqrt(11)) / 3 + 4000 * (
                                 -5 * sqrt(7) + 2 * sqrt(3) + 3 * sqrt(11)) ** 2 * exp(-2 * sqrt(11) / 3) * exp(
                 -2 * sqrt(3) / 3) * exp(4 * sqrt(7) / 3) * exp(
                 -35 * sqrt(7) + 14 * sqrt(3) + 21 * sqrt(11)) / 3 + 5600 * (
                                 -5 * sqrt(7) + 2 * sqrt(3) + 3 * sqrt(11)) * exp(-2 * sqrt(11) / 3) * exp(
                 -2 * sqrt(3) / 3) * exp(4 * sqrt(7) / 3) * exp(
                 -35 * sqrt(7) + 14 * sqrt(3) + 21 * sqrt(11)) / 9 + 60 * exp(-7 * sqrt(7) + 7 * sqrt(11))) / 2 - (
                                                                                                s - 7) * (-200 * exp(
             -sqrt(11) / 3) * exp(-sqrt(3) / 3) * exp(2 * sqrt(7) / 3) * exp(
             -21 * sqrt(7) + 7 * sqrt(3) + 14 * sqrt(11)) + 4000 * (-5 * sqrt(7) + 2 * sqrt(3) + 3 * sqrt(11)) * exp(
             -2 * sqrt(11) / 3) * exp(-2 * sqrt(3) / 3) * exp(4 * sqrt(7) / 3) * exp(
             -35 * sqrt(7) + 14 * sqrt(3) + 21 * sqrt(11)) / 3 + 2800 * exp(-2 * sqrt(11) / 3) * exp(
             -2 * sqrt(3) / 3) * exp(4 * sqrt(7) / 3) * exp(
             -35 * sqrt(7) + 14 * sqrt(3) + 21 * sqrt(11)) / 9) - 4000 * exp(-2 * sqrt(11) / 3) * exp(
             -2 * sqrt(3) / 3) * exp(4 * sqrt(7) / 3) * exp(-35 * sqrt(7) + 14 * sqrt(3) + 21 * sqrt(11)) / 3) / (
                             s - 7) ** 3)]])


def test_qpoly_short():
    # parse QPolys from big expressions
    start = time.perf_counter()
    mat = mrg[:2, :]
    for idx, expr in mat.iter_items():
        print(idx)
        q = QPoly(expr, sigma)
        assert verify_numerically(expr, q.as_expr())
    total = time.perf_counter() - start
    print(f"Parsing a {mat.shape} matrix took {total} seconds.")


@pytest.mark.skip("Extensive Benchmark, takes quite long")
def test_qpoly_long():
    # parse QPolys from big expressions
    start = time.perf_counter()
    for idx, expr in mrg.iter_items():
        print(idx)
        q = QPoly(expr, sigma)
        assert verify_numerically(expr, q.as_expr())
    total = time.perf_counter() - start
    print(f"Parsing a 3x3 matrix took {total} seconds.")
