from __future__ import annotations

import pytest

from omnivault.dsa.algorithms.number_theory import (
    extended_gcd,
    fast_exponentiation,
    gcd,
    is_prime,
    lcm,
    miller_rabin,
    modular_inverse,
    pollard_rho,
    sieve_of_eratosthenes,
    trial_division,
)
from omnivault.dsa.core.errors import InvalidConfiguration


class TestPrimes:
    @pytest.mark.unit
    def test_sieve_basic(self) -> None:
        assert sieve_of_eratosthenes(30) == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

    @pytest.mark.unit
    def test_sieve_two(self) -> None:
        assert sieve_of_eratosthenes(2) == [2]

    @pytest.mark.unit
    def test_sieve_invalid(self) -> None:
        with pytest.raises(InvalidConfiguration):
            sieve_of_eratosthenes(1)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("n", "expected"),
        [
            (2, True),
            (3, True),
            (97, True),
            (1, False),
            (0, False),
            (-7, False),
            (4, False),
            (91, False),
            (1_000_003, True),
        ],
    )
    def test_is_prime(self, n: int, expected: bool) -> None:
        assert is_prime(n) is expected

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("n", "expected"),
        [
            (2, True),
            (3, True),
            (97, True),
            (561, False),
            (1105, False),
            (1729, False),
            (1, False),
            (0, False),
            (91, False),
            (1_000_003, True),
            (10**18 + 9, True),
        ],
    )
    def test_miller_rabin(self, n: int, expected: bool) -> None:
        assert miller_rabin(n) is expected


class TestModular:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("a", "b", "expected"),
        [(48, 18, 6), (0, 7, 7), (7, 0, 7), (-12, 8, 4), (17, 13, 1)],
    )
    def test_gcd(self, a: int, b: int, expected: int) -> None:
        assert gcd(a, b) == expected

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("a", "b", "expected"),
        [(4, 6, 12), (3, 5, 15), (0, 5, 0), (-4, 6, 12)],
    )
    def test_lcm(self, a: int, b: int, expected: int) -> None:
        assert lcm(a, b) == expected

    @pytest.mark.unit
    def test_extended_gcd(self) -> None:
        g, x, y = extended_gcd(30, 12)
        assert (g, x, y) == (6, 1, -2)
        assert 30 * x + 12 * y == g

    @pytest.mark.unit
    def test_modular_inverse(self) -> None:
        assert modular_inverse(3, 11) == 4
        assert (3 * modular_inverse(3, 11)) % 11 == 1

    @pytest.mark.unit
    def test_modular_inverse_no_inverse(self) -> None:
        with pytest.raises(InvalidConfiguration):
            modular_inverse(6, 9)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("base", "exp", "mod", "expected"),
        [(2, 10, 1000, 24), (2, 10, None, 1024), (5, 0, 7, 1), (7, 3, None, 343)],
    )
    def test_fast_exponentiation(self, base: int, exp: int, mod: int | None, expected: int) -> None:
        assert fast_exponentiation(base, exp, mod) == expected

    @pytest.mark.unit
    def test_fast_exponentiation_negative_not_invertible(self) -> None:
        with pytest.raises(InvalidConfiguration):
            fast_exponentiation(2, -1, None)


class TestFactorization:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("n", "expected"),
        [
            (60, [2, 2, 3, 5]),
            (2, [2]),
            (97, [97]),
            (1024, [2] * 10),
        ],
    )
    def test_trial_division(self, n: int, expected: list[int]) -> None:
        assert trial_division(n) == expected

    @pytest.mark.unit
    def test_trial_division_invalid(self) -> None:
        with pytest.raises(InvalidConfiguration):
            trial_division(1)

    @pytest.mark.unit
    def test_pollard_rho_finds_divisor(self) -> None:
        n = 8051
        divisor = pollard_rho(n)
        assert 1 < divisor < n
        assert n % divisor == 0

    @pytest.mark.unit
    def test_pollard_rho_prime_returns_n(self) -> None:
        assert pollard_rho(13) == 13
