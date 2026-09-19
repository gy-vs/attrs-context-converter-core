# SPDX-License-Identifier: MIT

"""
Tests for `attr.converters`.
"""


import pickle

import pytest

import attr

from attr import Converter, Factory, attrib
from attr._compat import _AnnotationExtractor
from attr.converters import default_if_none, optional, pipe, to_bool


class TestConverter:
    @pytest.mark.parametrize("takes_self", [True, False])
    @pytest.mark.parametrize("takes_field", [True, False])
    def test_pickle(self, takes_self, takes_field):
        """
        Wrapped converters can be pickled.
        """
        c = Converter(int, takes_self=takes_self, takes_field=takes_field)

        new_c = pickle.loads(pickle.dumps(c))

        assert c == new_c
        assert takes_self == new_c.takes_self
        assert takes_field == new_c.takes_field
        assert c.__call__.__name__ == new_c.__call__.__name__

    @pytest.mark.parametrize(
        "scenario",
        [
            ((False, False), "__attr_converter_le_name(le_value)"),
            (
                (True, True),
                "__attr_converter_le_name(le_value, self, attr_dict['le_name'])",
            ),
            (
                (True, False),
                "__attr_converter_le_name(le_value, self)",
            ),
            (
                (False, True),
                "__attr_converter_le_name(le_value, attr_dict['le_name'])",
            ),
        ],
    )
    def test_fmt_converter_call(self, scenario):
        """
        _fmt_converter_call determines the arguments to the wrapped converter
        according to `takes_self` and `takes_field`.
        """
        (takes_self, takes_field), expect = scenario

        c = Converter(None, takes_self=takes_self, takes_field=takes_field)

        assert expect == c._fmt_converter_call("le_name", "le_value")

    def test_works_as_adapter(self):
        """
        Converter instances work as adapters and pass the correct arguments to
        the wrapped converter callable.
        """
        taken = None
        instance = object()
        field = object()

        def save_args(*args):
            nonlocal taken
            taken = args
            return args[0]

        Converter(save_args)(42, instance, field)

        assert (42,) == taken

        Converter(save_args, takes_self=True)(42, instance, field)

        assert (42, instance) == taken

        Converter(save_args, takes_field=True)(42, instance, field)

        assert (42, field) == taken

        Converter(save_args, takes_self=True, takes_field=True)(
            42, instance, field
        )

        assert (42, instance, field) == taken

    def test_annotations_if_last_in_pipe(self):
        """
        If the wrapped converter has annotations, they are copied to the
        Converter __call__.
        """

        def wrapped(_, __, ___) -> float:
            pass

        c = Converter(wrapped)

        assert float is c.__call__.__annotations__["return"]

        # Doesn't overwrite globally.

        c2 = Converter(int)

        assert float is c.__call__.__annotations__["return"]
        assert None is c2.__call__.__annotations__.get("return")

    def test_falsey_converter(self):
        """
        Passing a false-y instance still produces a valid converter.
        """

        class MyConv:
            def __bool__(self):
                return False

            def __call__(self, value):
                return value * 2

        @attr.s
        class C:
            a = attrib(converter=MyConv())

        c = C(21)
        assert 42 == c.a

    def test_eq_hash_repr(self):
        """
        Converters with equal wrapped callables and flags compare equal, are
        hashable, and have a helpful repr.
        """
        c1 = Converter(int, takes_self=True)
        c2 = Converter(int, takes_self=True)
        c3 = Converter(int, takes_field=True)
        c4 = Converter(str, takes_self=True)

        assert c1 == c2
        assert hash(c1) == hash(c2)
        assert c1 != c3
        assert c1 != c4
        assert c1 != 42

        r = repr(c1)

        assert "takes_self=True" in r
        assert "takes_field=False" in r
        assert "converter=" in r


class TestOptional:
    """
    Tests for `optional`.
    """

    def test_success_with_type(self):
        """
        Wrapped converter is used as usual if value is not None.
        """
        c = optional(int)

        assert c("42") == 42

    def test_success_with_none(self):
        """
        Nothing happens if None.
        """
        c = optional(int)

        assert c(None) is None

    def test_fail(self):
        """
        Propagates the underlying conversion error when conversion fails.
        """
        c = optional(int)

        with pytest.raises(ValueError):
            c("not_an_int")

    def test_converter_instance(self):
        """
        Works when passed a Converter instance as argument.
        """
        c = optional(Converter(to_bool))

        assert True is c("yes", None, None)

    def test_converter_instance_context(self):
        """
        A Converter with takes_self / takes_field wrapped in optional receives
        the instance and field when used on a class.
        """
        taken = []

        def save_args(val, inst, field):
            taken.append((val, inst, field))
            return val

        @attr.s
        class C:
            x = attrib(
                converter=optional(
                    Converter(save_args, takes_self=True, takes_field=True)
                )
            )

        c = C(42)

        assert 42 == c.x
        assert 1 == len(taken)
        val, inst, field = taken[0]
        assert 42 == val
        assert c is inst
        assert "x" == field.name

        # None short-circuits the wrapped converter.
        taken.clear()

        assert C(None).x is None
        assert [] == taken


class TestDefaultIfNone:
    def test_missing_default(self):
        """
        Raises TypeError if neither default nor factory have been passed.
        """
        with pytest.raises(TypeError, match="Must pass either"):
            default_if_none()

    def test_too_many_defaults(self):
        """
        Raises TypeError if both default and factory are passed.
        """
        with pytest.raises(TypeError, match="but not both"):
            default_if_none(True, lambda: 42)

    def test_factory_takes_self(self):
        """
        Raises ValueError if passed Factory has takes_self=True.
        """
        with pytest.raises(ValueError, match="takes_self"):
            default_if_none(Factory(list, takes_self=True))

    @pytest.mark.parametrize("val", [1, 0, True, False, "foo", "", object()])
    def test_not_none(self, val):
        """
        If a non-None value is passed, it's handed down.
        """
        c = default_if_none("nope")

        assert val == c(val)

        c = default_if_none(factory=list)

        assert val == c(val)

    def test_none_value(self):
        """
        Default values are returned when a None is passed.
        """
        c = default_if_none(42)

        assert 42 == c(None)

    def test_none_factory(self):
        """
        Factories are used if None is passed.
        """
        c = default_if_none(factory=list)

        assert [] == c(None)

        c = default_if_none(default=Factory(list))

        assert [] == c(None)


class TestPipe:
    def test_success(self):
        """
        Succeeds if all wrapped converters succeed.
        """
        c = pipe(str, Converter(to_bool), bool)

        assert (
            True
            is c.converter("True", None, None)
            is c.converter(True, None, None)
        )

    def test_fail(self):
        """
        Fails if any wrapped converter fails.
        """
        c = pipe(str, to_bool)

        # First wrapped converter fails:
        with pytest.raises(ValueError):
            c(33)

        # Last wrapped converter fails:
        with pytest.raises(ValueError):
            c("33")

    def test_sugar(self):
        """
        `pipe(c1, c2, c3)` and `[c1, c2, c3]` are equivalent.
        """

        @attr.s
        class C:
            a1 = attrib(default="True", converter=pipe(str, to_bool, bool))
            a2 = attrib(default=True, converter=[str, to_bool, bool])

        c = C()
        assert True is c.a1 is c.a2

    def test_empty(self):
        """
        Empty pipe returns same value.
        """
        o = object()

        assert o is pipe()(o)

    def test_wrapped_annotation(self):
        """
        The return type of the wrapped converter is copied into its __call__
        and ultimately into pipe's wrapped converter.
        """

        def last(value) -> bool:
            return bool(value)

        @attr.s
        class C:
            x = attr.ib(converter=[Converter(int), Converter(last)])

        i = C(5)

        assert True is i.x
        assert (
            bool
            is _AnnotationExtractor(
                attr.fields(C).x.converter.__call__
            ).get_return_type()
        )

    def test_context_passed_to_wrapped_converters(self):
        """
        If a wrapped Converter asks for the instance and/or field, pipe
        forwards them; plain callables keep receiving only the value.
        """
        taken = []

        def takes_all(val, inst, field):
            taken.append((val, inst, field))
            return int(val) * 2

        @attr.s
        class C:
            x = attrib(
                converter=pipe(
                    str,
                    Converter(takes_all, takes_self=True, takes_field=True),
                    int,
                )
            )

        c = C("21")

        assert 42 == c.x
        assert 1 == len(taken)
        val, inst, field = taken[0]
        assert "21" == val
        assert c is inst
        assert "x" == field.name

    def test_error_not_hidden(self):
        """
        An exception raised by a wrapped converter propagates unchanged and
        the original callable shows up in the traceback.
        """

        def inner(val):
            raise ValueError("original error")

        c = pipe(str, Converter(inner))

        with pytest.raises(ValueError, match="original error") as ei:
            c("1", None, None)

        assert "inner" in {frame.name for frame in ei.traceback}


class TestOptionalPipe:
    def test_optional(self):
        """
        Nothing happens if None.
        """
        c = optional(pipe(str, Converter(to_bool), bool))

        assert None is c.converter(None, None, None)

    def test_pipe(self):
        """
        A value is given, run it through all wrapped converters.
        """
        c = optional(pipe(str, Converter(to_bool), bool))

        assert (
            True
            is c.converter("True", None, None)
            is c.converter(True, None, None)
        )

    def test_instance(self):
        """
        Should work when set as an attrib.
        """

        @attr.s
        class C:
            x = attrib(
                converter=optional(pipe(str, Converter(to_bool), bool)),
                default=None,
            )

        c1 = C()

        assert None is c1.x

        c2 = C("True")

        assert True is c2.x


class TestToBool:
    def test_unhashable(self):
        """
        Fails if value is unhashable.
        """
        with pytest.raises(ValueError, match="Cannot convert value to bool"):
            to_bool([])

    def test_truthy(self):
        """
        Fails if truthy values are incorrectly converted.
        """
        assert to_bool("t")
        assert to_bool("yes")
        assert to_bool("on")

    def test_falsy(self):
        """
        Fails if falsy values are incorrectly converted.
        """
        assert not to_bool("f")
        assert not to_bool("no")
        assert not to_bool("off")
